import { normalize, scale, type Vec3 } from '../core/math/vec3';
import { rpyToQuat } from '../core/math/mat4';
import { FOOT_MASS, type LimbDescription, type RobotDescription } from '../core/model/types';
import { assertValidRobot } from '../core/model/validate';

export interface BoxObstacle {
  name: string;
  /** Centre position, m. */
  pos: Vec3;
  /** Half extents, m. */
  halfSize: Vec3;
  yaw?: number;
}

export interface RangeRay {
  name: string;
  /** Mount point in the base frame. */
  pos: Vec3;
  /** Ray direction in the base frame (rangefinders measure along +Z of their site). */
  dir: Vec3;
}

export interface MjcfOptions {
  timestep?: number;
  /** Sliding friction applied to the floor and every robot geom. */
  friction?: number;
  /** Overrides the base mass (used for randomized trials). */
  baseMass?: number;
  floor?: boolean;
  obstacles?: BoxObstacle[];
  rangeRays?: RangeRay[];
}

/** Element names used in the generated MJCF; the runtime resolves ids from these. */
export const mjcfNames = {
  base: 'base',
  root: 'root',
  imuSite: 'imu',
  linkBody: (jointId: string) => `link_${jointId}`,
  joint: (jointId: string) => jointId,
  actuator: (jointId: string) => `act_${jointId}`,
  posSensor: (jointId: string) => `pos_${jointId}`,
  velSensor: (jointId: string) => `vel_${jointId}`,
  tipSite: (limbId: string) => `tip_${limbId}`,
  touchSensor: (limbId: string) => `touch_${limbId}`,
  footGeom: (limbId: string) => `foot_${limbId}`,
  rangeSite: (name: string) => `ray_${name}`,
  rangeSensor: (name: string) => `range_${name}`,
  imuQuat: 'imu_quat',
  imuGyro: 'imu_gyro',
  imuAcc: 'imu_acc',
} as const;

/** Mass of the foot sphere added at each leg tip, kg (defined with the mass model in core). */
export { FOOT_MASS };

function num(v: number): string {
  if (!Number.isFinite(v)) throw new RangeError('Non-finite number in MJCF generation');
  const s = Number(v.toPrecision(12));
  return Object.is(s, -0) ? '0' : String(s);
}

function nums(v: readonly number[]): string {
  return v.map(num).join(' ');
}

function esc(s: string): string {
  return s.replace(/[<>&"']/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' })[c]!);
}

function limbXml(limb: LimbDescription, indent: string): { xml: string; bodies: string[] } {
  const bodies: string[] = [];
  const open: string[] = [];
  const close: string[] = [];
  limb.joints.forEach((j, i) => {
    const pad = indent + '  '.repeat(i);
    const next: Vec3 = i + 1 < limb.joints.length ? limb.joints[i + 1]!.origin.xyz : limb.tip.xyz;
    const body = mjcfNames.linkBody(j.id);
    bodies.push(body);
    const quat = rpyToQuat(j.origin.rpy);
    const axis = normalize(j.axis);
    const range = j.type === 'continuous' ? '' : ` range="${num(j.limit.lower)} ${num(j.limit.upper)}"`;
    const type = j.type === 'prismatic' ? 'slide' : 'hinge';
    let geom: string;
    if (limb.role === 'wheel' && limb.wheel) {
      const a = scale(axis, limb.wheel.width / 2);
      geom = `<geom name="wheel_${esc(limb.id)}" type="cylinder" fromto="${nums(scale(a, -1))} ${nums(a)}" size="${num(limb.wheel.radius)}" mass="${num(j.link.mass)}"/>`;
    } else if (Math.hypot(...next) < 1e-6) {
      geom = `<geom type="sphere" size="${num(j.link.radius)}" mass="${num(j.link.mass)}"/>`;
    } else {
      geom = `<geom type="capsule" fromto="0 0 0 ${nums(next)}" size="${num(j.link.radius)}" mass="${num(j.link.mass)}"/>`;
    }
    let s = `${pad}<body name="${body}" pos="${nums(j.origin.xyz)}" quat="${nums(quat)}">\n`;
    s += `${pad}  <joint name="${mjcfNames.joint(j.id)}" type="${type}" axis="${nums(axis)}"${range} damping="${num(j.damping)}" armature="${num(j.armature)}"/>\n`;
    s += `${pad}  ${geom}\n`;
    if (i === limb.joints.length - 1 && limb.role !== 'wheel') {
      if (limb.tip.radius > 0) {
        s += `${pad}  <geom name="${mjcfNames.footGeom(limb.id)}" type="sphere" pos="${nums(limb.tip.xyz)}" size="${num(limb.tip.radius)}" mass="${num(FOOT_MASS)}"/>\n`;
      }
      s += `${pad}  <site name="${mjcfNames.tipSite(limb.id)}" pos="${nums(limb.tip.xyz)}" size="${num(Math.max(0.005, limb.tip.radius * 1.3))}"/>\n`;
    }
    open.push(s);
    close.unshift(`${pad}</body>\n`);
  });
  return { xml: open.join('') + close.join(''), bodies };
}

/** Converts a robot description into MJCF text. Pure; does not require WASM. */
export function generateMjcf(desc: RobotDescription, opts: MjcfOptions = {}): string {
  assertValidRobot(desc);
  const mu = opts.friction ?? 1;
  if (!(mu > 0)) throw new RangeError('Friction must be positive');
  const dt = opts.timestep ?? 0.002;
  const baseMass = opts.baseMass ?? desc.base.mass;
  if (!(baseMass > 0)) throw new RangeError('Base mass must be positive');
  const b = desc.base;
  const half = scale(b.size, 0.5);
  const lines: string[] = [];
  lines.push(`<mujoco model="${esc(desc.name)}">`);
  lines.push('  <compiler angle="radian" autolimits="true"/>');
  lines.push(`  <option timestep="${num(dt)}" integrator="implicitfast"/>`);
  lines.push('  <default>');
  lines.push(`    <geom condim="3" friction="${num(mu)} 0.005 0.0001"/>`);
  lines.push('  </default>');
  lines.push('  <worldbody>');
  if (opts.floor ?? true) lines.push(`    <geom name="floor" type="plane" size="0 0 0.05" friction="${num(mu)} 0.005 0.0001"/>`);
  for (const o of opts.obstacles ?? []) {
    lines.push(`    <geom name="obstacle_${esc(o.name)}" type="box" pos="${nums(o.pos)}" size="${nums(o.halfSize)}" quat="${nums(rpyToQuat([0, 0, o.yaw ?? 0]))}"/>`);
  }
  lines.push(`    <body name="${mjcfNames.base}" pos="${nums(b.pose.xyz)}" quat="${nums(rpyToQuat(b.pose.rpy))}">`);
  if (b.kind === 'floating') lines.push(`      <freejoint name="${mjcfNames.root}"/>`);
  lines.push(`      <geom name="base_geom" type="box" size="${nums(half)}" mass="${num(baseMass)}"/>`);
  lines.push(`      <site name="${mjcfNames.imuSite}" size="0.01"/>`);
  for (const r of opts.rangeRays ?? []) {
    // Orient the site so that its +Z axis points along the ray direction.
    const d = normalize(r.dir);
    const zaxis = nums(d);
    lines.push(`      <site name="${mjcfNames.rangeSite(r.name)}" pos="${nums(r.pos)}" zaxis="${zaxis}" size="0.005"/>`);
  }
  const limbBodies: string[] = [];
  for (const limb of desc.limbs) {
    const { xml, bodies } = limbXml(limb, '      ');
    lines.push(xml.trimEnd());
    limbBodies.push(...bodies);
  }
  lines.push('    </body>');
  lines.push('  </worldbody>');
  if (limbBodies.length > 0) {
    lines.push('  <contact>');
    for (const body of limbBodies) lines.push(`    <exclude body1="${mjcfNames.base}" body2="${body}"/>`);
    lines.push('  </contact>');
  }
  lines.push('  <actuator>');
  for (const limb of desc.limbs) {
    for (const j of limb.joints) {
      lines.push(`    <motor name="${mjcfNames.actuator(j.id)}" joint="${mjcfNames.joint(j.id)}" ctrlrange="${num(-j.limit.effort)} ${num(j.limit.effort)}"/>`);
    }
  }
  lines.push('  </actuator>');
  lines.push('  <sensor>');
  for (const limb of desc.limbs) {
    for (const j of limb.joints) {
      lines.push(`    <jointpos name="${mjcfNames.posSensor(j.id)}" joint="${mjcfNames.joint(j.id)}"/>`);
      lines.push(`    <jointvel name="${mjcfNames.velSensor(j.id)}" joint="${mjcfNames.joint(j.id)}"/>`);
    }
  }
  lines.push(`    <framequat name="${mjcfNames.imuQuat}" objtype="site" objname="${mjcfNames.imuSite}"/>`);
  lines.push(`    <gyro name="${mjcfNames.imuGyro}" site="${mjcfNames.imuSite}"/>`);
  lines.push(`    <accelerometer name="${mjcfNames.imuAcc}" site="${mjcfNames.imuSite}"/>`);
  for (const limb of desc.limbs) {
    if (limb.role === 'leg' && limb.tip.radius > 0) lines.push(`    <touch name="${mjcfNames.touchSensor(limb.id)}" site="${mjcfNames.tipSite(limb.id)}"/>`);
  }
  for (const r of opts.rangeRays ?? []) lines.push(`    <rangefinder name="${mjcfNames.rangeSensor(r.name)}" site="${mjcfNames.rangeSite(r.name)}"/>`);
  lines.push('  </sensor>');
  lines.push('</mujoco>');
  return lines.join('\n');
}
