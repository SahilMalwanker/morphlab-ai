import { buildKinematicModel, forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { identity } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import { mirrorPoint, mirrorRotationAxis } from '../core/model/builder';
import { homeConfiguration } from '../core/model/catalog';
import type { RobotDescription } from '../core/model/types';
import { patternPhases, type GaitParams } from './gait';
import type { KeyframeMotion, SkillGoal } from './taught';

/**
 * Left-right mirror images of taught skills (D26). Robot learning reuses the
 * symmetry of a body so that what was learned for one side serves the other
 * (symmetry augmentation in RSL-RL, after Mittal et al., ICRA 2024; mirrored
 * motion clips in motion imitation). Here a skill for one side becomes the
 * draft of its counterpart: limbs swap with their mirror partners, joint
 * targets move to the partner joints, sideways motion and turns change sign,
 * and the goals follow. The body's symmetry is checked with forward
 * kinematics, not assumed. A mirrored skill is a new draft that is tried and
 * evaluated like any other: a symmetric drawing does not prove symmetric
 * physics.
 */

export interface MirrorMap {
  /** The mirror partner of each limb by index; a limb on the plane of symmetry is its own. */
  limbs: number[];
  /** For each joint in global order, the joint that mirrors it and the sign its value takes there. */
  joints: { index: number; sign: 1 | -1 }[];
}

export type MirrorMapResult = { ok: true; map: MirrorMap } | { ok: false; reason: string };

/** Agreement required between a pose and the mirror image of its mirror, m (or rad). */
const TOL = 1e-6;

const neg = (x: number): number => (x === 0 ? 0 : -x);
const close = (a: readonly number[], b: readonly number[]): boolean => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]!) <= TOL);

function tipPositions(km: KinematicModel, q: readonly number[]): Vec3[] {
  return forwardKinematics(km, q, identity()).tips.map((T) => [T[12]!, T[13]!, T[14]!] as Vec3);
}

export function mirrorQ(map: MirrorMap, q: readonly number[]): number[] {
  const out = q.map(() => 0);
  map.joints.forEach((m, j) => (out[m.index] = m.sign === 1 ? q[j]! : neg(q[j]!)));
  return out;
}

/**
 * Pairs each limb with the limb mounted at the mirror image of its mount
 * point, and each joint with the joint at the same place in the partner limb.
 * A joint value keeps its sign when the partner's axis is the mirror image of
 * this one's (the builder's mirrored legs), and flips it when the axis is the
 * opposite (a base yaw on the plane of symmetry). The map is then verified:
 * the home stance must be its own mirror image, and in test poses every limb
 * tip of the mirrored pose must sit at the mirror image of its partner's tip.
 */
export function mirrorMap(robot: RobotDescription): MirrorMapResult {
  const km = buildKinematicModel(robot);
  const limbs: number[] = [];
  for (const [i, limb] of robot.limbs.entries()) {
    const mount = limb.joints[0]?.origin.xyz ?? limb.tip.xyz;
    const want = mirrorPoint(mount);
    const p = robot.limbs.findIndex((l) => {
      const m = l.joints[0]?.origin.xyz ?? l.tip.xyz;
      return l.role === limb.role && l.joints.length === limb.joints.length && Math.hypot(m[0] - want[0], m[1] - want[1], m[2] - want[2]) < TOL;
    });
    if (p < 0) return { ok: false, reason: `${robot.name} is not mirror symmetric: the ${limb.name} has no partner on the other side.` };
    limbs.push(p);
    if (limbs[p] !== undefined && limbs[p] !== i) return { ok: false, reason: `${robot.name} is not mirror symmetric: the ${limb.name} and the ${robot.limbs[p]!.name} do not pair up.` };
  }
  const joints: MirrorMap['joints'] = [];
  for (const [i, limb] of km.limbs.entries()) {
    const partner = km.limbs[limbs[i]!]!;
    for (let k = 0; k < limb.count; k++) {
      const a = km.joints[limb.start + k]!;
      const index = partner.start + k;
      const b = km.joints[index]!;
      const image = a.type === 'prismatic' ? mirrorPoint(a.axis) : mirrorRotationAxis(a.axis);
      const sign = a.type !== b.type ? 0 : close(b.axis, image) ? 1 : close(b.axis, image.map(neg)) ? -1 : 0;
      if (sign === 0) return { ok: false, reason: `${robot.name} is not mirror symmetric: joint ${a.id} has no mirror image in the ${robot.limbs[limbs[i]!]!.name}.` };
      joints.push({ index, sign });
    }
  }
  const map: MirrorMap = { limbs, joints };
  const home = homeConfiguration(robot);
  if (!close(mirrorQ(map, home), home)) return { ok: false, reason: `${robot.name} is not mirror symmetric: its home stance differs between the left and right sides.` };
  // Away from the home stance too, every limb tip of the mirrored pose must be the mirror image of its partner's.
  const probe = home.map((v, j) => v + 0.1 * Math.sin(1.7 * j + 0.4));
  for (const q of [home, probe]) {
    const tips = tipPositions(km, q);
    const mirrored = tipPositions(km, mirrorQ(map, q));
    if (!tips.every((t, l) => close(mirrored[limbs[l]!]!, mirrorPoint(t)))) return { ok: false, reason: `${robot.name} is not mirror symmetric: its limbs do not move as mirror images of each other.` };
  }
  return { ok: true, map };
}

const NAME_SWAP: Record<string, string> = { left: 'right', right: 'left', clockwise: 'counterclockwise', counterclockwise: 'clockwise', 'counter-clockwise': 'clockwise', anticlockwise: 'clockwise' };

/** The name with left and right (and clockwise) swapped, or null when it names no side. */
export function mirrorName(name: string): string | null {
  let changed = false;
  const out = name.replace(/(?<![A-Za-z])(counter-clockwise|counterclockwise|anticlockwise|clockwise|left|right)(?![A-Za-z])/gi, (w) => {
    changed = true;
    const r = NAME_SWAP[w.toLowerCase()]!;
    return w[0] === w[0]!.toUpperCase() ? r[0]!.toUpperCase() + r.slice(1) : r;
  });
  return changed ? out : null;
}

function partnerId(robot: RobotDescription, map: MirrorMap, id: string): string {
  const i = robot.limbs.findIndex((l) => l.id === id);
  return i < 0 ? id : robot.limbs[map.limbs[i]!]!.id;
}

/** A goal as it reads for the mirror image: sides, sideways directions, turns and roll change sign. */
export function mirrorGoal(g: SkillGoal, robot: RobotDescription, map: MirrorMap): SkillGoal {
  switch (g.kind) {
    case 'travel':
      return { ...g, direction: g.direction === 'left' ? 'right' : g.direction === 'right' ? 'left' : g.direction };
    case 'turn':
      return { ...g, left: !g.left };
    case 'bodyAngle':
      return g.axis === 'roll' ? { ...g, minRad: neg(g.maxRad), maxRad: neg(g.minRad) } : { ...g };
    case 'tipHeight':
    case 'footDown':
    case 'tipSwing':
      return { ...g, limbId: partnerId(robot, map, g.limbId) };
    case 'tipReach':
      return { ...g, limbId: partnerId(robot, map, g.limbId), y: neg(g.y) };
    case 'airborne':
      return g.limbIds ? { ...g, limbIds: g.limbIds.map((id) => partnerId(robot, map, id)) } : { ...g };
    default:
      return { ...g };
  }
}

/** A gait's mirror image: sideways steps and turns change sign, and each leg takes its partner's phase. */
export function mirrorGait(g: GaitParams, robot: RobotDescription, map: MirrorMap): GaitParams {
  const own = patternPhases(robot, g.pattern);
  const phases = g.phases ?? own;
  const mirrored = robot.limbs.map((_, i) => phases[map.limbs[i]!] ?? 0);
  const out: GaitParams = { ...g, lateralM: neg(g.lateralM), turnRad: neg(g.turnRad) };
  delete out.phases;
  // Explicit phases only when they differ from the pattern's own (a trot mirrored starts with the other diagonal pair).
  if (!close(mirrored, own)) out.phases = mirrored;
  if (g.heldUp) out.heldUp = g.heldUp.map((id) => partnerId(robot, map, id));
  return out;
}

export interface MirrorSource {
  goals: readonly SkillGoal[];
  motion: KeyframeMotion | null;
  gait: GaitParams | null;
}

export type MirrorResult = { ok: true; goals: SkillGoal[]; motion: KeyframeMotion | null; gait: GaitParams | null } | { ok: false; reason: string };

/** The mirror image of a taught skill's keyframes or gait, and of its goals, on this robot. */
export function mirrorTaught(robot: RobotDescription, src: MirrorSource): MirrorResult {
  const r = mirrorMap(robot);
  if (!r.ok) return r;
  const map = r.map;
  let motion: KeyframeMotion | null = null;
  if (src.motion) {
    const km = buildKinematicModel(robot);
    const ids = km.joints.map((j) => j.id);
    if (src.motion.jointIds.join('\n') !== ids.join('\n')) return { ok: false, reason: 'its keyframes were posed on a different set of joints.' };
    const keyframes: KeyframeMotion['keyframes'] = [];
    for (const [n, k] of src.motion.keyframes.entries()) {
      const q = mirrorQ(map, k.q);
      const j = q.findIndex((v, i) => v < km.joints[i]!.lower - 1e-9 || v > km.joints[i]!.upper + 1e-9);
      if (j >= 0) return { ok: false, reason: `keyframe ${n + 1}: the mirrored target of joint ${ids[j]} (${q[j]!.toFixed(3)}) is outside its limits.` };
      keyframes.push({ t: k.t, q });
    }
    motion = { jointIds: ids, keyframes, endHoldS: src.motion.endHoldS };
  }
  return { ok: true, goals: src.goals.map((g) => mirrorGoal(g, robot, map)), motion, gait: src.gait ? mirrorGait(src.gait, robot, map) : null };
}
