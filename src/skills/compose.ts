import { buildKinematicModel, forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { solveIK } from '../core/kinematics/ik';
import { fromXyzRpy, identity, invertRigid, transformDirection, transformPoint } from '../core/math/mat4';
import { add, type Vec3 } from '../core/math/vec3';
import { homeConfiguration } from '../core/model/catalog';
import type { RobotDescription } from '../core/model/types';
import { hasFoot, homeTipClearance, restingHeight, type SkillGoal } from './taught';

/**
 * Poses described in words of the body instead of joint sliders (D24): the
 * body moved from its home stance while the feet stay where they stand,
 * limb tips moved from where they rest, and joint targets set directly.
 * Every pose is relative to the home stance; the kinematics core turns it
 * into joint targets (damped least-squares IK, always within joint limits).
 */

export interface BodyOffset {
  x?: number;
  y?: number;
  z?: number;
  roll?: number;
  pitch?: number;
  yaw?: number;
}

/** A limb tip moved from where it rests in this pose, in the home body frame (x forward, y left, z up), m. */
export interface TipOffset {
  limb: string;
  x?: number;
  y?: number;
  z?: number;
}

export interface JointTarget {
  joint: string;
  /** rad, or m for sliding joints */
  value: number;
}

export interface PoseSpec {
  body?: BodyOffset;
  tips?: TipOffset[];
  joints?: JointTarget[];
}

/** A pose and the time to reach it from the previous one (the first from the home stance), s. */
export interface ComposedFrame extends PoseSpec {
  seconds: number;
}

export const MAX_COMPOSED_KEYFRAMES = 8;
/** Accepted ranges of composed poses; the AI tool schema repeats them (checked by a test). */
export const COMPOSE_LIMITS = { bodyShiftM: 0.2, bodyAngleRad: 0.8, tipOffsetM: 0.5, secondsMin: 0.1, secondsMax: 10 } as const;

/** IK residuals above this are reported with the pose, m. */
const REPORT_RESIDUAL_M = 0.002;

export type ComposeResult = { ok: true; keyframes: { durationS: number; q: number[] }[]; notes: string[] } | { ok: false; problems: string[] };

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

interface Context {
  robot: RobotDescription;
  km: KinematicModel;
  home: number[];
  /** Tip positions in the base frame at home. */
  homeTips: Vec3[];
}

function context(robot: RobotDescription): Context {
  const km = buildKinematicModel(robot);
  const home = homeConfiguration(robot);
  const fk = forwardKinematics(km, home, identity());
  return { robot, km, home, homeTips: fk.tips.map((T) => [T[12]!, T[13]!, T[14]!] as Vec3) };
}

function bounded(where: string, v: unknown, max: number, unit: string, out: string[]): number {
  if (v === undefined) return 0;
  if (!finite(v) || Math.abs(v) > max) {
    out.push(`${where} must be a number from -${max} to ${max} ${unit}.`);
    return 0;
  }
  return v;
}

function composePose(c: Context, pose: PoseSpec, seed: number[], where: string): { q: number[]; notes: string[]; problems: string[] } {
  const { robot, km } = c;
  const q = c.home.slice();
  const problems: string[] = [];
  const notes: string[] = [];
  let T_B1_B0 = identity();
  let moved = false;
  if (pose.body) {
    const b = pose.body;
    if (robot.base.kind !== 'floating') {
      problems.push(`${where}.body: ${robot.name} has a fixed base, so its body cannot move; move limb tips instead.`);
    } else {
      const s = COMPOSE_LIMITS.bodyShiftM, a = COMPOSE_LIMITS.bodyAngleRad;
      const xyz: Vec3 = [bounded(`${where}.body.x`, b.x, s, 'm', problems), bounded(`${where}.body.y`, b.y, s, 'm', problems), bounded(`${where}.body.z`, b.z, s, 'm', problems)];
      const rpy: Vec3 = [bounded(`${where}.body.roll`, b.roll, a, 'rad', problems), bounded(`${where}.body.pitch`, b.pitch, a, 'rad', problems), bounded(`${where}.body.yaw`, b.yaw, a, 'rad', problems)];
      moved = [...xyz, ...rpy].some((v) => v !== 0);
      T_B1_B0 = invertRigid(fromXyzRpy(xyz, rpy));
    }
  }
  const offsets = new Map<number, Vec3>();
  (pose.tips ?? []).forEach((t, i) => {
    const at = `${where}.tips[${i}]`;
    const li = robot.limbs.findIndex((l) => l.id === t.limb);
    const limb = robot.limbs[li];
    if (!limb) {
      problems.push(`${at}.limb: ${robot.name} has no limb "${String(t.limb)}"; its limbs are ${robot.limbs.map((l) => l.id).join(', ')}.`);
      return;
    }
    if (limb.role === 'wheel') {
      problems.push(`${at}.limb: "${limb.id}" is a wheel; it has no tip to place.`);
      return;
    }
    if (offsets.has(li)) {
      problems.push(`${at}.limb: "${limb.id}" is listed twice in one pose.`);
      return;
    }
    const m = COMPOSE_LIMITS.tipOffsetM;
    offsets.set(li, [bounded(`${at}.x`, t.x, m, 'm', problems), bounded(`${at}.y`, t.y, m, 'm', problems), bounded(`${at}.z`, t.z, m, 'm', problems)]);
  });
  robot.limbs.forEach((limb, li) => {
    if (limb.role === 'wheel') return;
    const d = offsets.get(li);
    // Legs keep their feet where they stand while the body moves; other limbs move with the body.
    if (!d && !(moved && limb.role === 'leg')) return;
    const target = transformPoint(T_B1_B0, d ? add(c.homeTips[li]!, d) : c.homeTips[li]!);
    const r = solveIK(km, li, target, seed, { T_W_B: identity(), tolerance: 1e-4, restarts: 4, seed: 7 });
    const k = km.limbs[li]!;
    for (let j = k.start; j < k.start + k.count; j++) q[j] = r.q[j]!;
    if (r.status !== 'converged' && r.residual > REPORT_RESIDUAL_M) {
      notes.push(`${where}: the ${limb.name.toLowerCase()} tip stops ${(r.residual * 1000).toFixed(0)} mm from its target (${r.beyondReach ? 'out of its reach' : 'joint limits or the limb shape'})`);
    }
  });
  const seen = new Set<number>();
  (pose.joints ?? []).forEach((t, i) => {
    const at = `${where}.joints[${i}]`;
    const idx = km.joints.findIndex((j) => j.id === t.joint);
    const j = km.joints[idx];
    if (!j) {
      problems.push(`${at}.joint: ${robot.name} has no joint "${String(t.joint)}".`);
      return;
    }
    if (seen.has(idx)) {
      problems.push(`${at}.joint: "${j.id}" is listed twice in one pose.`);
      return;
    }
    seen.add(idx);
    const lo = Number.isFinite(j.lower) ? j.lower : -2 * Math.PI;
    const hi = Number.isFinite(j.upper) ? j.upper : 2 * Math.PI;
    const unit = j.type === 'prismatic' ? 'm' : 'rad';
    if (!finite(t.value) || t.value < lo || t.value > hi) {
      problems.push(`${at}.value: "${j.id}" takes ${lo.toFixed(2)} to ${hi.toFixed(2)} ${unit}, not ${String(t.value)}.`);
      return;
    }
    q[idx] = t.value;
  });
  return { q, notes, problems };
}

/**
 * Turns composed poses into keyframes of joint targets for a robot. Nothing
 * is clamped silently: out-of-range values, unknown limbs and joints are
 * problems; tips that IK cannot reach are reported with their distance.
 */
export function composeKeyframes(robot: RobotDescription, frames: readonly ComposedFrame[], goals: readonly SkillGoal[] = []): ComposeResult {
  if (!Array.isArray(frames) || frames.length === 0 || frames.length > MAX_COMPOSED_KEYFRAMES) {
    return { ok: false, problems: [`keyframes must hold 1 to ${MAX_COMPOSED_KEYFRAMES} poses.`] };
  }
  const c = context(robot);
  const keyframes: { durationS: number; q: number[] }[] = [];
  const notes: string[] = [];
  const problems: string[] = [];
  let seed = c.home;
  let lastBodyMove = -1;
  frames.forEach((f: ComposedFrame, i: number) => {
    const where = `keyframes[${i}]`;
    if (!finite(f.seconds) || f.seconds < COMPOSE_LIMITS.secondsMin || f.seconds > COMPOSE_LIMITS.secondsMax) {
      problems.push(`${where}.seconds must be from ${COMPOSE_LIMITS.secondsMin} to ${COMPOSE_LIMITS.secondsMax} s.`);
    }
    const r = composePose(c, f, seed, where);
    problems.push(...r.problems);
    notes.push(...r.notes);
    if (i > 0 && i < frames.length - 1 && !f.body && !f.tips?.length && !f.joints?.length && seed.some((value, index) => Math.abs(value - c.home[index]!) > 1e-6)) {
      notes.push(`${where}: a pose with only seconds returns to home, not a hold; repeat the previous body, tips and joints to hold that pose`);
    }
    if (f.body) {
      lastBodyMove = Object.values(f.body).some((value) => value !== undefined && value !== 0) ? i : -1;
    } else if (lastBodyMove >= 0 && f.tips?.some((tip) => (tip.z ?? 0) > 0 && robot.limbs.some((limb) => limb.id === tip.limb && hasFoot(limb)))) {
      notes.push(`${where}: the body offset from keyframes[${lastBodyMove}] is omitted while a foot is lifted; each pose starts from home, so repeat body in every lifted-foot frame that needs that weight shift`);
    } else if (!f.tips?.length && !f.joints?.length) {
      lastBodyMove = -1;
    }
    keyframes.push({ durationS: f.seconds, q: r.q });
    seed = r.q;
  });
  for (const goal of goals) {
    if (goal.kind !== 'bodyAngle') continue;
    const targets = frames.map((frame: ComposedFrame) => frame.body?.[goal.axis] ?? 0);
    const positive = goal.minRad > 0 && targets.every((value: number) => value <= 0) && targets.some((value: number) => value < 0);
    const negative = goal.maxRad < 0 && targets.every((value: number) => value >= 0) && targets.some((value: number) => value > 0);
    if (positive || negative) {
      const convention = goal.axis === 'pitch' ? 'positive pitch lowers the front (nose down)' : 'positive roll raises the left side';
      notes.unshift(`body.${goal.axis} sign conflict: the goal requires ${positive ? 'positive' : 'negative'} angles, but every nonzero body.${goal.axis} target has the opposite sign; ${convention}. Revise the signs and repeat the corrected pose for the requested hold before returning home; a failed sketch is not evidence of a physical limit`);
    }
  }
  return problems.length ? { ok: false, problems } : { ok: true, keyframes, notes };
}

const num = (v: number, digits = 3): string => {
  const s = v.toFixed(digits);
  return Number(s) === 0 ? (0).toFixed(digits) : s;
};

function axisText(a: Vec3): string {
  const names = ['x', 'y', 'z'];
  for (let k = 0; k < 3; k++) if (Math.abs(a[k]!) > 0.999) return `${a[k]! > 0 ? '+' : '-'}${names[k]}`;
  return `(${a.map((v) => num(v, 2)).join(', ')})`;
}

/**
 * The robot in numbers a language model can compose poses from: the body,
 * and per limb its tip at home (base frame) and its joints with axes in the
 * base frame at home, limits and home values.
 */
export function describeBody(robot: RobotDescription): string {
  const c = context(robot);
  const fk = forwardKinematics(c.km, c.home, identity());
  const b = robot.base;
  const size = b.size.map((v) => num(v, 2)).join(' x ');
  const rest = restingHeight(robot);
  const head =
    b.kind === 'floating' && rest !== null
      ? `Body: a floating base of ${size} m and ${b.mass} kg; in the home stance it rests ${num(rest)} m above the ground.`
      : `Body: a fixed base of ${size} m (it never moves).`;
  const limbs = robot.limbs.map((limb, li) => {
    const k = c.km.limbs[li]!;
    const joints: string[] = [];
    for (let j = k.start; j < k.start + k.count; j++) {
      const kj = c.km.joints[j]!;
      const axis = axisText(transformDirection(fk.links[j]!, kj.axis));
      const range = Number.isFinite(kj.lower) ? `${num(kj.lower, 2)} to ${num(kj.upper, 2)}` : 'continuous';
      joints.push(`${kj.id} (axis ${axis}, ${range}, home ${num(c.home[j]!, 2)})`);
    }
    const efforts = limb.joints.map((j) => j.limit.effort);
    const effort = Math.min(...efforts) === Math.max(...efforts) ? `${efforts[0]}` : `${Math.min(...efforts)} to ${Math.max(...efforts)}`;
    const role = limb.role === 'leg' ? (hasFoot(limb) ? 'leg with a foot' : 'leg') : limb.role;
    if (limb.role === 'wheel') return `${limb.id} (${limb.name}, wheel): joints ${joints.join('; ')}.`;
    const t = c.homeTips[li]!;
    const clearance = homeTipClearance(robot, li);
    return `${limb.id} (${limb.name}, ${role}): tip at home (${t.map((v) => num(v)).join(', ')}) m, ${num(clearance)} m above the ground; effort limit ${effort} N*m; joints ${joints.join('; ')}.`;
  });
  return `${head} Limbs, with tip positions in the body frame (x forward, y left, z up) and joint values in rad: ${limbs.join(' ')}`;
}
