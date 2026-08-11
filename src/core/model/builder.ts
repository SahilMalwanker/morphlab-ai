import { buildKinematicModel, forwardKinematics } from '../kinematics/fk';
import { identity } from '../math/mat4';
import type { Vec3 } from '../math/vec3';
import { homeConfiguration } from './catalog';
import { joint, serialChain, type ChainAxis } from './presets';
import type { JointDescription, JointType, LimbDescription, RobotDescription, ValidationIssue } from './types';
import { validateRobot } from './validate';

const neg = (x: number): number => (x === 0 ? 0 : -x);

/** Reflection of a point across the base XZ plane (y -> -y). */
export function mirrorPoint(v: Vec3): Vec3 {
  return [v[0], neg(v[1]), v[2]];
}

/**
 * Rotation axes are pseudovectors: conjugating a rotation about `a` by the
 * reflection M = diag(1, -1, 1) gives a rotation about -M a by the same angle.
 * With this axis the same joint angle produces the mirrored pose.
 */
export function mirrorRotationAxis(a: Vec3): Vec3 {
  return [neg(a[0]), a[1], neg(a[2])];
}

/** M R M for R = Rz(yaw) Ry(pitch) Rx(roll) equals Rz(-yaw) Ry(pitch) Rx(-roll). */
export function mirrorRpy(r: Vec3): Vec3 {
  return [neg(r[0]), r[1], neg(r[2])];
}

/** Mirror image of a limb across the base XZ plane, with the given ids and name. */
export function mirrorLimb(src: LimbDescription, id: string, name: string, jointIds: string[]): LimbDescription {
  if (jointIds.length !== src.joints.length) throw new RangeError('mirrorLimb needs one id per joint');
  const out: LimbDescription = {
    id,
    name,
    role: src.role,
    joints: src.joints.map((j, i) => ({
      ...structuredClone(j),
      id: jointIds[i]!,
      origin: { xyz: mirrorPoint(j.origin.xyz), rpy: mirrorRpy(j.origin.rpy) },
      axis: j.type === 'prismatic' ? mirrorPoint(j.axis) : mirrorRotationAxis(j.axis),
    })),
    tip: { xyz: mirrorPoint(src.tip.xyz), radius: src.tip.radius },
  };
  if (src.wheel) out.wheel = { ...src.wheel };
  return out;
}

/** The limb mounted at the mirror image of this limb's mount point, if any. */
export function mirrorPartner(desc: RobotDescription, limbId: string): string | null {
  const limb = desc.limbs.find((l) => l.id === limbId);
  const mount = limb?.joints[0]?.origin.xyz;
  if (!limb || !mount || Math.abs(mount[1]) < 1e-9) return null;
  const want = mirrorPoint(mount);
  const partner = desc.limbs.find((l) => {
    const m = l.joints[0]?.origin.xyz;
    return l !== limb && l.role === limb.role && m !== undefined && Math.hypot(m[0] - want[0], m[1] - want[1], m[2] - want[2]) < 1e-9;
  });
  return partner?.id ?? null;
}

function allJointIds(desc: RobotDescription, exceptLimb?: string): Set<string> {
  return new Set(desc.limbs.filter((l) => l.id !== exceptLimb).flatMap((l) => l.joints.map((j) => j.id)));
}

/** Keeps existing ids by position and creates unique ids for new joints. */
function jointIdsFor(desc: RobotDescription, limb: LimbDescription, n: number): string[] {
  const used = allJointIds(desc, limb.id);
  const ids: string[] = [];
  for (let i = 0; i < n; i++) {
    let id = limb.joints[i]?.id ?? `${limb.id}_j${i + 1}`;
    for (let k = 2; used.has(id); k++) id = `${limb.id}_j${i + 1}_${k}`;
    used.add(id);
    ids.push(id);
  }
  return ids;
}

function pruneStance(desc: RobotDescription): RobotDescription {
  if (!desc.stance) return desc;
  const ids = allJointIds(desc);
  const stance: Record<string, number> = {};
  for (const [k, v] of Object.entries(desc.stance)) if (ids.has(k)) stance[k] = v;
  return { ...desc, stance };
}

export interface JointPatch {
  name?: string;
  type?: JointType;
  xyz?: Vec3;
  rpy?: Vec3;
  axis?: Vec3;
  lower?: number;
  upper?: number;
  effort?: number;
  velocity?: number;
  radius?: number;
  mass?: number;
  damping?: number;
  armature?: number;
}

export function patchJoint(j: JointDescription, p: JointPatch): JointDescription {
  return {
    ...j,
    name: p.name ?? j.name,
    type: p.type ?? j.type,
    origin: { xyz: p.xyz ?? j.origin.xyz, rpy: p.rpy ?? j.origin.rpy },
    axis: p.axis ?? j.axis,
    limit: { lower: p.lower ?? j.limit.lower, upper: p.upper ?? j.limit.upper, effort: p.effort ?? j.limit.effort, velocity: p.velocity ?? j.limit.velocity },
    link: { radius: p.radius ?? j.link.radius, mass: p.mass ?? j.link.mass },
    damping: p.damping ?? j.damping,
    armature: p.armature ?? j.armature,
  };
}

/**
 * Applies `edit` to one limb. With `mirror`, the partner limb (if any) is
 * regenerated as the exact mirror image of the edited limb, keeping its own
 * id, name and joint ids, and its stance copies the edited limb's stance.
 */
export function editLimb(desc: RobotDescription, limbId: string, edit: (l: LimbDescription) => LimbDescription, mirror: boolean): RobotDescription {
  const idx = desc.limbs.findIndex((l) => l.id === limbId);
  if (idx < 0) throw new RangeError(`Unknown limb ${limbId}`);
  const partnerId = mirror ? mirrorPartner(desc, limbId) : null;
  const limbs = desc.limbs.slice();
  const edited = edit(structuredClone(desc.limbs[idx]!));
  limbs[idx] = edited;
  let stance = desc.stance ? { ...desc.stance } : undefined;
  if (partnerId) {
    const pIdx = limbs.findIndex((l) => l.id === partnerId);
    const partner = limbs[pIdx]!;
    const ids = jointIdsFor({ ...desc, limbs }, partner, edited.joints.length);
    limbs[pIdx] = mirrorLimb(edited, partner.id, partner.name, ids);
    if (stance) {
      for (const j of partner.joints) delete stance[j.id];
      edited.joints.forEach((j, i) => {
        const v = stance![j.id];
        if (v !== undefined) stance![ids[i]!] = v;
      });
    }
  }
  const out: RobotDescription = { ...desc, limbs };
  if (stance) out.stance = stance;
  return pruneStance(out);
}

/** Changes the number of joints in a limb, keeping the existing ones. */
export function setJointCount(desc: RobotDescription, limbId: string, n: number, mirror: boolean): RobotDescription {
  if (!Number.isInteger(n) || n < 1) throw new RangeError('A limb needs at least one joint');
  return editLimb(desc, limbId, (limb) => {
    if (limb.role === 'wheel') throw new RangeError('A wheel has exactly one joint');
    const joints = limb.joints.slice(0, n);
    let tip = limb.tip.xyz;
    if (n < limb.joints.length) tip = limb.joints[n]!.origin.xyz;
    const ids = jointIdsFor(desc, limb, n);
    while (joints.length < n) {
      const prev = joints[joints.length - 1]!;
      const i = joints.length;
      joints.push(
        patchJoint(structuredClone(prev), {
          name: `Joint ${i + 1}`,
          xyz: tip,
          rpy: [0, 0, 0],
          radius: Math.max(0.01, prev.link.radius * 0.85),
          mass: Math.max(0.02, prev.link.mass * 0.8),
        }),
      );
    }
    return { ...limb, joints: joints.map((j, i) => ({ ...j, id: ids[i]! })), tip: { ...limb.tip, xyz: tip } };
  }, mirror);
}

/** Removes a limb (and its mirror partner when `mirror` is set). */
export function removeLimb(desc: RobotDescription, limbId: string, mirror: boolean): RobotDescription {
  const partner = mirror ? mirrorPartner(desc, limbId) : null;
  const limbs = desc.limbs.filter((l) => l.id !== limbId && l.id !== partner);
  if (limbs.length === desc.limbs.length) throw new RangeError(`Unknown limb ${limbId}`);
  if (limbs.length === 0) throw new RangeError('A robot needs at least one limb');
  return pruneStance({ ...desc, limbs });
}

/**
 * Base height that puts the lowest point of the robot (link capsules, tip
 * spheres, wheels, base box) `clearance` metres above the ground at q.
 */
export function restingBaseHeight(desc: RobotDescription, q: number[], clearance = 0.002): number {
  const km = buildKinematicModel(desc);
  const fk = forwardKinematics(km, q, identity());
  let low = -desc.base.size[2] / 2;
  desc.limbs.forEach((limb, li) => {
    const start = km.limbs[li]!.start;
    limb.joints.forEach((j, ji) => {
      const T = fk.links[start + ji]!;
      const next = ji + 1 < limb.joints.length ? fk.links[start + ji + 1]! : fk.tips[li]!;
      low = Math.min(low, T[14]! - j.link.radius, next[14]! - j.link.radius);
    });
    const tip = fk.tips[li]!;
    low = Math.min(low, tip[14]! - (limb.wheel ? limb.wheel.radius : limb.tip.radius));
  });
  return -low + clearance;
}

/**
 * Sets the initial base height so the lowest point of the robot at its home
 * configuration is just above the ground (a fixed base then stands on the
 * ground or, for legged bodies, holds the legs clear like a test stand).
 */
export function placeOnGround(desc: RobotDescription): RobotDescription {
  const z = restingBaseHeight(desc, homeConfiguration(desc));
  const xyz = desc.base.pose.xyz;
  return { ...desc, base: { ...desc.base, pose: { ...desc.base.pose, xyz: [xyz[0], xyz[1], z] } } };
}

/**
 * The same body bolted to a test stand: fixed, level base raised by the
 * largest limb reach above its resting height, so no limb can touch the
 * ground in any configuration. Used for joint-level experiments where ground
 * contact would hide the joint dynamics.
 */
export function testStand(desc: RobotDescription): RobotDescription {
  const xyz = desc.base.pose.xyz;
  const fixed = placeOnGround({ ...desc, base: { ...desc.base, kind: 'fixed', pose: { xyz: [xyz[0], xyz[1], 0], rpy: [0, 0, 0] } } });
  const reach = Math.max(0, ...buildKinematicModel(fixed).limbs.map((l) => l.reachBound + Math.hypot(...(fixed.limbs.find((x) => x.id === l.id)!.joints[0]!.origin.xyz))));
  const p = fixed.base.pose.xyz;
  return { ...fixed, base: { ...fixed.base, pose: { ...fixed.base.pose, xyz: [p[0], p[1], p[2] + reach] } } };
}

export interface LeggedSpec {
  /** Leg pairs (left and right), 1 to 4. */
  pairs: number;
  /** Joints per leg, 1 to 4. */
  dofPerLeg: number;
  bodyLength: number;
  bodyWidth: number;
  bodyHeight: number;
  bodyMass: number;
  thigh: number;
  shank: number;
  effort: number;
}

export const DEFAULT_LEGGED: LeggedSpec = { pairs: 2, dofPerLeg: 3, bodyLength: 0.4, bodyWidth: 0.2, bodyHeight: 0.08, bodyMass: 4, thigh: 0.16, shank: 0.16, effort: 12 };

const PAIR_POSITIONS: Record<number, string[]> = {
  1: ['middle'],
  2: ['front', 'rear'],
  3: ['front', 'middle', 'rear'],
  4: ['front', 'front_middle', 'rear_middle', 'rear'],
};

const label = (id: string): string => id.replace(/_/g, ' ');
const cap = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

export function leggedSpecIssues(s: LeggedSpec): string[] {
  const out: string[] = [];
  if (!Number.isInteger(s.pairs) || s.pairs < 1 || s.pairs > 4) out.push('Leg pairs must be a whole number from 1 to 4.');
  if (!Number.isInteger(s.dofPerLeg) || s.dofPerLeg < 1 || s.dofPerLeg > 4) out.push('Joints per leg must be a whole number from 1 to 4.');
  const pos = (v: number, what: string, max: number) => {
    if (!Number.isFinite(v) || v <= 0 || v > max) out.push(`${what} must be greater than 0 and at most ${max}.`);
  };
  pos(s.bodyLength, 'Body length (m)', 3);
  pos(s.bodyWidth, 'Body width (m)', 3);
  pos(s.bodyHeight, 'Body height (m)', 3);
  pos(s.bodyMass, 'Body mass (kg)', 200);
  pos(s.thigh, 'Upper leg length (m)', 1);
  pos(s.shank, 'Lower leg length (m)', 1);
  pos(s.effort, 'Joint effort limit (N*m)', 1000);
  return out;
}

function leftLeg(pos: string, x: number, s: LeggedSpec): { limb: LimbDescription; stance: Record<string, number> } {
  const id = `${pos}_left`;
  const mount: Vec3 = [x, s.bodyWidth / 2, 0];
  const common = { effort: s.effort, velocity: 20 };
  const joints: JointDescription[] = [];
  const stance: Record<string, number> = {};
  const add = (key: string, name: string, xyz: Vec3, axis: Vec3, lower: number, upper: number, radius: number, mass: number, home: number) => {
    joints.push(joint({ id: `${id}_${key}`, name, xyz, axis, lower, upper, radius, mass, ...common }));
    stance[`${id}_${key}`] = home;
  };
  const first = (offset: Vec3): Vec3 => (joints.length === 0 ? mount : offset);
  if (s.dofPerLeg >= 4) add('hip_yaw', 'Hip yaw', first([0, 0, 0]), [0, 0, 1], -0.8, 0.8, 0.025, 0.1, 0);
  if (s.dofPerLeg >= 3) add('hip_roll', 'Hip roll', first([0, 0, -0.03]), [1, 0, 0], -0.7, 0.7, 0.025, 0.12, 0);
  if (s.dofPerLeg === 1) {
    add('hip_pitch', 'Hip pitch', first([0, 0.03, 0]), [0, 1, 0], -1.8, 1.8, 0.022, 1.5 * (s.thigh + s.shank) * 0.75, 0);
    return { limb: { id, name: `${cap(label(pos))} left leg`, role: 'leg', joints, tip: { xyz: [0, 0, -(s.thigh + s.shank)], radius: 0.022 } }, stance };
  }
  add('hip_pitch', 'Hip pitch', first([0, 0.03, 0]), [0, 1, 0], -1.8, 1.8, 0.022, 1.5 * s.thigh, 0.7);
  add('knee', 'Knee', [0, 0, -s.thigh], [0, 1, 0], -2.7, -0.15, 0.018, 0.75 * s.shank, -1.4);
  return { limb: { id, name: `${cap(label(pos))} left leg`, role: 'leg', joints, tip: { xyz: [0, 0, -s.shank], radius: 0.022 } }, stance };
}

/** Floating-base legged robot with mirrored left and right legs, placed on the ground. */
export function generateLegged(s: LeggedSpec): RobotDescription {
  const issues = leggedSpecIssues(s);
  if (issues.length) throw new RangeError(issues.join(' '));
  const names = PAIR_POSITIONS[s.pairs]!;
  const half = Math.max(0, s.bodyLength / 2 - 0.03);
  const limbs: LimbDescription[] = [];
  const stance: Record<string, number> = {};
  names.forEach((pos, i) => {
    const x = names.length === 1 ? 0 : half * (1 - (2 * i) / (names.length - 1));
    const left = leftLeg(pos, x, s);
    const rightId = `${pos}_right`;
    const rightIds = left.limb.joints.map((j) => j.id.replace(`${pos}_left_`, `${rightId}_`));
    const right = mirrorLimb(left.limb, rightId, `${cap(label(pos))} right leg`, rightIds);
    limbs.push(left.limb, right);
    Object.assign(stance, left.stance);
    left.limb.joints.forEach((j, k) => (stance[rightIds[k]!] = left.stance[j.id]!));
  });
  const desc: RobotDescription = {
    schemaVersion: 1,
    id: 'legged',
    name: `${2 * s.pairs}-legged robot (${s.dofPerLeg} DOF per leg)`,
    base: { kind: 'floating', size: [s.bodyLength, s.bodyWidth, s.bodyHeight], mass: s.bodyMass, pose: { xyz: [0, 0, 0.5], rpy: [0, 0, 0] } },
    limbs,
    stance,
  };
  return placeOnGround(desc);
}

const ARM_AXES: ChainAxis[] = ['z', 'y', 'y', 'x', 'y', 'x', 'y'];
const ARM_WEIGHTS = [0.2, 0.3, 0.25, 0.1, 0.08, 0.06, 0.05];

/** Fixed-base arm with 1 to 7 joints and the given total link length. */
export function generateArm(dof: number, reach: number): RobotDescription {
  if (!Number.isInteger(dof) || dof < 1 || dof > 7) throw new RangeError('Arm joints must be a whole number from 1 to 7.');
  if (!Number.isFinite(reach) || reach <= 0 || reach > 2) throw new RangeError('Arm reach must be greater than 0 and at most 2 m.');
  const axes: ChainAxis[] = dof === 1 ? ['y'] : dof === 2 ? ['y', 'y'] : ARM_AXES.slice(0, dof);
  const w = ARM_WEIGHTS.slice(0, dof);
  const sum = w.reduce((a, b) => a + b, 0);
  return { ...serialChain({ axes, lengths: w.map((v) => (v / sum) * reach) }), id: 'arm', name: `${dof}-joint arm` };
}

/** Human-readable form of a validation issue, naming the limb and joint. */
export function describeIssue(desc: RobotDescription, issue: ValidationIssue): string {
  const m = /^limbs\[(\d+)\](?:\.joints\[(\d+)\])?/.exec(issue.path);
  if (m) {
    const li = Number(m[1]);
    const limb = Array.isArray(desc.limbs) ? desc.limbs[li] : undefined;
    const limbName = limb?.name?.trim() || `Limb ${li + 1}`;
    if (m[2] !== undefined) {
      const ji = Number(m[2]);
      const jointName = (Array.isArray(limb?.joints) ? limb!.joints[ji]?.name?.trim() : undefined) || `joint ${ji + 1}`;
      return `${limbName}, ${jointName}: ${issue.message}.`;
    }
    return `${limbName}: ${issue.message}.`;
  }
  if (issue.path.startsWith('base')) return `Base: ${issue.message}.`;
  if (issue.path.startsWith('stance.')) {
    const id = issue.path.slice('stance.'.length);
    for (const limb of Array.isArray(desc.limbs) ? desc.limbs : []) {
      const j = Array.isArray(limb?.joints) ? limb.joints.find((x) => x?.id === id) : undefined;
      if (j) return `${limb.name}, ${j.name}: ${issue.message}.`;
    }
  }
  if (issue.path.startsWith('stance')) return `Stance: ${issue.message}.`;
  return `${issue.message}.`;
}

/** Clamps stance values into their joint limits; returns the robot and whether anything changed. */
export function clampStance(desc: RobotDescription): { robot: RobotDescription; changed: boolean } {
  if (!desc.stance) return { robot: desc, changed: false };
  const stance = { ...desc.stance };
  let changed = false;
  for (const limb of desc.limbs) {
    for (const j of limb.joints) {
      const v = stance[j.id];
      if (v === undefined || j.type === 'continuous' || !(j.limit.lower < j.limit.upper)) continue;
      const c = Math.min(j.limit.upper, Math.max(j.limit.lower, v));
      if (c !== v) {
        stance[j.id] = c;
        changed = true;
      }
    }
  }
  return { robot: changed ? { ...desc, stance } : desc, changed };
}

/** Validation messages for a candidate description, ready to show to the user. */
export function describeIssues(desc: RobotDescription): string[] {
  return validateRobot(desc).map((i) => describeIssue(desc, i));
}
