import { buildKinematicModel, clampToLimits, forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { fromXyzRpy } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import { restingBaseHeight } from '../core/model/builder';
import { homeConfiguration } from '../core/model/catalog';
import type { RobotDescription } from '../core/model/types';
import type { SkillController } from './controllers';

/**
 * Skills the user teaches: a motion posed as keyframes, and goals that say
 * what success means. Goals are judged from simulator ground truth; the
 * motion is played open loop (it does not react to what happens).
 */

/** A pose to reach t seconds after the skill starts, as joint targets in global joint order (rad, or m for sliding joints). */
export interface Keyframe {
  t: number;
  q: number[];
}

/** Motion from the home stance through posed keyframes, holding the last pose for endHoldS. */
export interface KeyframeMotion {
  jointIds: string[];
  keyframes: Keyframe[];
  endHoldS: number;
}

export const MAX_KEYFRAMES = 24;
export const MAX_MOTION_S = 30;
export const MAX_END_HOLD_S = 10;
export const MAX_GOALS = 12;
/** Goals that judge the end of a run look at this final window, s. */
export const END_WINDOW_S = 0.5;

export type TravelDirection = 'forward' | 'backward' | 'left' | 'right';
export type BodyAxis = 'pitch' | 'roll';
/** Directions in the start heading frame: x forward, y left, z up. */
export type TipAxis = 'x' | 'y' | 'z';

export type SkillGoal =
  | { kind: 'upright'; maxTiltRad: number }
  | { kind: 'noFall'; minBaseHeightM: number }
  | { kind: 'tipHeight'; limbId: string; minHeightM: number; holdS: number }
  | { kind: 'baseHeight'; below: boolean; heightM: number; holdS: number }
  | { kind: 'airborne'; minS: number; limbIds?: string[] }
  | { kind: 'baseRise'; minRiseM: number }
  | { kind: 'travel'; direction: TravelDirection; minDistanceM: number; maxDistanceM?: number }
  | { kind: 'turn'; left: boolean; minAngleRad: number; maxAngleRad?: number }
  | { kind: 'standsAtEnd'; maxTiltRad: number; minBaseHeightM: number }
  | { kind: 'bodyAngle'; axis: BodyAxis; minRad: number; maxRad: number; holdS: number }
  | { kind: 'footDown'; limbId: string; minFraction: number }
  | { kind: 'tipSwing'; limbId: string; axis: TipAxis; minAmplitudeM: number; minSwings: number }
  | { kind: 'tipReach'; limbId: string; x: number; y: number; z: number; toleranceM: number; holdS: number };

export type SkillGoalKind = SkillGoal['kind'];

export const GOAL_LABELS: Record<SkillGoalKind, string> = {
  upright: 'Stays upright',
  noFall: 'Does not fall',
  tipHeight: 'Limb tip raised',
  baseHeight: 'Body height reached',
  airborne: 'Feet leave the ground',
  baseRise: 'Body rises',
  travel: 'Body moves',
  turn: 'Body turns',
  standsAtEnd: 'Ends standing',
  bodyAngle: 'Body angle held',
  footDown: 'Foot stays down',
  tipSwing: 'Limb tip swings',
  tipReach: 'Limb tip reaches a point',
};

/** Accepted range of every numeric goal parameter, inclusive (editor, import and storage). */
export const GOAL_BOUNDS: Record<string, [number, number]> = {
  maxTiltRad: [0.05, 1.6],
  minBaseHeightM: [0, 2],
  minHeightM: [0, 2],
  holdS: [0, 10],
  heightM: [0, 2],
  minS: [0.01, 2],
  minRiseM: [0.005, 1],
  minDistanceM: [0.01, 5],
  maxDistanceM: [0.01, 5],
  minAngleRad: [0.05, 6.3],
  maxAngleRad: [0.05, 6.3],
  minRad: [-1.6, 1.6],
  maxRad: [-1.6, 1.6],
  minFraction: [0, 1],
  minAmplitudeM: [0.005, 1],
  minSwings: [1, 20],
  x: [-2, 2],
  y: [-2, 2],
  z: [-2, 2],
  toleranceM: [0.005, 0.5],
};

/** A taught skill's name: letters, digits, spaces, hyphens, underscores and apostrophes. */
export const TAUGHT_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 _'-]{0,39}$/;

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

const NUMERIC_FIELDS: Record<SkillGoalKind, string[]> = {
  upright: ['maxTiltRad'],
  noFall: ['minBaseHeightM'],
  tipHeight: ['minHeightM', 'holdS'],
  baseHeight: ['heightM', 'holdS'],
  airborne: ['minS'],
  baseRise: ['minRiseM'],
  travel: ['minDistanceM'],
  turn: ['minAngleRad'],
  standsAtEnd: ['maxTiltRad', 'minBaseHeightM'],
  bodyAngle: ['minRad', 'maxRad', 'holdS'],
  footDown: ['minFraction'],
  tipSwing: ['minAmplitudeM', 'minSwings'],
  tipReach: ['x', 'y', 'z', 'toleranceM', 'holdS'],
};

/** Fields other than the numbers, per kind. */
const OTHER_FIELDS: Partial<Record<SkillGoalKind, string[]>> = {
  tipHeight: ['limbId'],
  baseHeight: ['below'],
  airborne: ['limbIds'],
  turn: ['left'],
  travel: ['direction'],
  bodyAngle: ['axis'],
  footDown: ['limbId'],
  tipSwing: ['limbId', 'axis'],
  tipReach: ['limbId'],
};

/** Optional upper bounds, with the lower bound each must not undercut. */
const OPTIONAL_MAX: Partial<Record<SkillGoalKind, [max: string, min: string]>> = {
  travel: ['maxDistanceM', 'minDistanceM'],
  turn: ['maxAngleRad', 'minAngleRad'],
};

/** Problems with one goal, independent of the robot; empty when valid. */
export function goalProblems(g: unknown, where = 'goal'): string[] {
  if (!isObject(g) || typeof g.kind !== 'string' || !(g.kind in NUMERIC_FIELDS)) return [`${where}.kind must be one of ${Object.keys(NUMERIC_FIELDS).join(', ')}.`];
  const kind = g.kind as SkillGoalKind;
  const out: string[] = [];
  for (const key of NUMERIC_FIELDS[kind]) {
    const [lo, hi] = GOAL_BOUNDS[key]!;
    const v = g[key];
    if (!finite(v) || v < lo || v > hi) out.push(`${where}.${key} must be a number from ${lo} to ${hi}.`);
  }
  const limbGoal = kind === 'tipHeight' || kind === 'footDown' || kind === 'tipSwing' || kind === 'tipReach';
  if (limbGoal && (typeof g.limbId !== 'string' || g.limbId.length === 0)) out.push(`${where}.limbId must name a limb.`);
  if (kind === 'airborne' && g.limbIds !== undefined) {
    const ids = g.limbIds;
    if (!Array.isArray(ids) || ids.length === 0 || ids.length > 8 || !ids.every((id) => typeof id === 'string' && id.length > 0) || new Set(ids).size !== ids.length) {
      out.push(`${where}.limbIds must list 1 to 8 different limbs when present.`);
    }
  }
  if (kind === 'baseHeight' && typeof g.below !== 'boolean') out.push(`${where}.below must be true or false.`);
  if (kind === 'turn' && typeof g.left !== 'boolean') out.push(`${where}.left must be true or false.`);
  if (kind === 'travel' && !['forward', 'backward', 'left', 'right'].includes(g.direction as string)) out.push(`${where}.direction must be forward, backward, left or right.`);
  if (kind === 'bodyAngle' && g.axis !== 'pitch' && g.axis !== 'roll') out.push(`${where}.axis must be pitch or roll.`);
  if (kind === 'bodyAngle' && finite(g.minRad) && finite(g.maxRad) && g.minRad > g.maxRad) out.push(`${where}.minRad must not be above maxRad.`);
  if (kind === 'tipSwing' && !['x', 'y', 'z'].includes(g.axis as string)) out.push(`${where}.axis must be x, y or z.`);
  if (kind === 'tipSwing' && finite(g.minSwings) && !Number.isInteger(g.minSwings)) out.push(`${where}.minSwings must be a whole number.`);
  const opt = OPTIONAL_MAX[kind];
  if (opt && g[opt[0]] !== undefined) {
    const [lo, hi] = GOAL_BOUNDS[opt[0]]!;
    const v = g[opt[0]];
    if (!finite(v) || v < lo || v > hi) out.push(`${where}.${opt[0]} must be a number from ${lo} to ${hi} when present.`);
    else if (finite(g[opt[1]]) && v < (g[opt[1]] as number)) out.push(`${where}.${opt[0]} must not be below ${opt[1]}.`);
  }
  const allowed = new Set(['kind', ...NUMERIC_FIELDS[kind], ...(OTHER_FIELDS[kind] ?? []), ...(opt ? [opt[0]] : [])]);
  const extra = Object.keys(g).filter((k) => !allowed.has(k));
  if (extra.length) out.push(`${where} has unknown fields: ${extra.join(', ')}.`);
  return out;
}

/** Problems with a goal list (1 to MAX_GOALS valid goals). */
export function goalsProblems(goals: unknown, where = 'goals'): string[] {
  if (!Array.isArray(goals) || goals.length === 0 || goals.length > MAX_GOALS) return [`${where} must list 1 to ${MAX_GOALS} goals.`];
  return goals.flatMap((g, i) => goalProblems(g, `${where}[${i}]`));
}

/** Problems with a keyframe motion, independent of the robot; empty when valid. */
export function motionProblems(m: unknown, where = 'motion'): string[] {
  if (!isObject(m)) return [`${where} must be an object.`];
  const out: string[] = [];
  const ids = m.jointIds;
  if (!Array.isArray(ids) || ids.length === 0 || ids.length > 64 || !ids.every((id) => typeof id === 'string' && id.length > 0)) out.push(`${where}.jointIds must list 1 to 64 joint ids.`);
  else if (new Set(ids).size !== ids.length) out.push(`${where}.jointIds must not repeat.`);
  const n = Array.isArray(ids) ? ids.length : -1;
  const ks = m.keyframes;
  if (!Array.isArray(ks) || ks.length === 0 || ks.length > MAX_KEYFRAMES) out.push(`${where}.keyframes must hold 1 to ${MAX_KEYFRAMES} keyframes.`);
  else {
    let prev = 0;
    ks.forEach((k, i) => {
      if (!isObject(k) || !finite(k.t) || !Array.isArray(k.q) || k.q.length !== n || !k.q.every(finite)) {
        out.push(`${where}.keyframes[${i}] must hold a time and one finite target per joint.`);
        return;
      }
      if (!(k.t > prev)) out.push(`${where}.keyframes[${i}].t must be later than the previous keyframe (${prev} s).`);
      prev = k.t;
    });
    if (prev > MAX_MOTION_S) out.push(`${where} is longer than ${MAX_MOTION_S} s.`);
  }
  if (!finite(m.endHoldS) || m.endHoldS < 0 || m.endHoldS > MAX_END_HOLD_S) out.push(`${where}.endHoldS must be from 0 to ${MAX_END_HOLD_S} s.`);
  return out;
}

/** Goal kinds a fixed base can be judged by: it never moves, so only its limb tips count. */
const FIXED_BASE_KINDS: readonly SkillGoalKind[] = ['tipHeight', 'tipSwing', 'tipReach'];

/** Problems with goals on a specific robot (limbs that must exist, feet that must exist, a body that can move). */
export function goalRobotProblems(goals: readonly SkillGoal[], robot: RobotDescription): string[] {
  const out: string[] = [];
  for (const g of goals) {
    if (robot.base.kind === 'fixed' && !FIXED_BASE_KINDS.includes(g.kind)) {
      out.push(`"${GOAL_LABELS[g.kind]}" measures the body, but ${robot.name} has a fixed base; use limb tip goals.`);
      continue;
    }
    if (g.kind === 'tipHeight' || g.kind === 'footDown' || g.kind === 'tipSwing' || g.kind === 'tipReach') {
      const l = robot.limbs.find((x) => x.id === g.limbId);
      if (!l) out.push(`The robot has no limb "${g.limbId}".`);
      else if (l.role === 'wheel') out.push(`"${l.name}" is a wheel; its tip is not measured.`);
      else if (g.kind === 'footDown' && !hasFoot(l)) out.push(`"${l.name}" has no foot with a touch sensor, so its ground contact cannot be measured.`);
    }
    if (g.kind === 'airborne' && !robot.limbs.some(hasFoot)) out.push(`${robot.name} has no feet with touch sensors, so leaving the ground cannot be measured.`);
    if (g.kind === 'airborne' && g.limbIds) {
      for (const id of g.limbIds) {
        const l = robot.limbs.find((x) => x.id === id);
        if (!l) out.push(`The robot has no limb "${id}".`);
        else if (!hasFoot(l)) out.push(`"${l.name}" has no foot with a touch sensor, so its leaving the ground cannot be measured.`);
      }
    }
  }
  return out;
}

/** Goal kinds that can be measured on a robot: a fixed base never moves, so only limb tips count there. */
export function goalKindsFor(robot: RobotDescription): SkillGoalKind[] {
  if (robot.base.kind === 'fixed') return [...FIXED_BASE_KINDS];
  const kinds = Object.keys(GOAL_LABELS) as SkillGoalKind[];
  return robot.limbs.some(hasFoot) ? kinds : kinds.filter((k) => k !== 'airborne' && k !== 'footDown');
}

/** Body height at rest in the home stance (floating bases), the reference for a rise; null for a fixed base. */
export function restingHeight(robot: RobotDescription): number | null {
  return robot.base.kind === 'floating' ? restingBaseHeight(robot, homeConfiguration(robot)) : null;
}

/** Height of a limb tip above the ground in the home stance, with a floating body at its resting height, m. */
export function homeTipClearance(robot: RobotDescription, limbIndex: number): number {
  const p = robot.base.pose;
  const T_W_B = fromXyzRpy([p.xyz[0], p.xyz[1], restingHeight(robot) ?? p.xyz[2]], p.rpy);
  const tip = forwardKinematics(buildKinematicModel(robot), homeConfiguration(robot), T_W_B).tips[limbIndex]!;
  return tip[14]! - robot.limbs[limbIndex]!.tip.radius;
}

export function hasFoot(l: RobotDescription['limbs'][number]): boolean {
  return l.role === 'leg' && l.tip.radius > 0;
}

/** Default goal of a kind for a robot, scaled to its home stance. */
export function defaultGoal(kind: SkillGoalKind, robot: RobotDescription): SkillGoal {
  const home = restingHeight(robot) ?? robot.base.pose.xyz[2];
  const round = (v: number, d = 2) => Number(v.toFixed(d));
  switch (kind) {
    case 'upright': return { kind, maxTiltRad: 0.5 };
    case 'noFall': return { kind, minBaseHeightM: round(0.5 * home) };
    case 'tipHeight': {
      const i = robot.limbs.findIndex((l) => l.role !== 'wheel');
      const rest = i < 0 ? 0 : Math.max(0, homeTipClearance(robot, i));
      return { kind, limbId: robot.limbs[i]?.id ?? '', minHeightM: round(Math.min(GOAL_BOUNDS.minHeightM![1], rest + 0.05)), holdS: 1 };
    }
    case 'baseHeight': return { kind, below: true, heightM: round(0.75 * home), holdS: 1 };
    case 'airborne': return { kind, minS: 0.05 };
    case 'baseRise': return { kind, minRiseM: 0.03 };
    case 'travel': return { kind, direction: 'forward', minDistanceM: 0.1 };
    case 'turn': return { kind, left: true, minAngleRad: 0.5 };
    case 'standsAtEnd': return { kind, maxTiltRad: 0.35, minBaseHeightM: round(0.5 * home) };
    case 'bodyAngle': return { kind, axis: 'pitch', minRad: -0.6, maxRad: -0.15, holdS: 1 };
    case 'footDown': return { kind, limbId: robot.limbs.find(hasFoot)?.id ?? '', minFraction: 0.9 };
    case 'tipSwing': return { kind, limbId: robot.limbs.find((l) => l.role !== 'wheel')?.id ?? '', axis: 'y', minAmplitudeM: 0.05, minSwings: 4 };
    case 'tipReach': return { kind, limbId: robot.limbs.find((l) => l.role !== 'wheel')?.id ?? '', x: 0.1, y: 0, z: 0.05, toleranceM: 0.03, holdS: 0.5 };
  }
}

const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(0)} deg`;
const TIP_AXIS_TEXT: Record<TipAxis, string> = { x: 'forward and back', y: 'left and right', z: 'up and down' };

/** Plain description of a goal. */
export function goalText(g: SkillGoal, limbName: (id: string) => string = (id) => id): string {
  switch (g.kind) {
    case 'upright': return `the body stays upright (tilt at most ${deg(g.maxTiltRad)})`;
    case 'noFall': return `the body stays above ${g.minBaseHeightM} m`;
    case 'tipHeight': return `the ${limbName(g.limbId)} tip is at least ${g.minHeightM} m above the ground${g.holdS > 0 ? ` for ${g.holdS} s` : ''}`;
    case 'baseHeight': return `the body is ${g.below ? 'below' : 'above'} ${g.heightM} m${g.holdS > 0 ? ` for ${g.holdS} s` : ''}`;
    case 'airborne':
      return g.limbIds
        ? `the feet of the ${g.limbIds.map(limbName).join(' and ')} leave the ground together (the others may stay down), the body upright, for ${g.minS} s`
        : `all feet leave the ground together, the body upright, for ${g.minS} s`;
    case 'baseRise': return `the body rises ${g.minRiseM} m above its resting height`;
    case 'travel': return `the body ends ${g.maxDistanceM !== undefined ? `${g.minDistanceM} to ${g.maxDistanceM}` : g.minDistanceM} m ${g.direction} of where it started`;
    case 'turn': return `the body turns ${g.maxAngleRad !== undefined ? `${deg(g.minAngleRad)} to ${deg(g.maxAngleRad)}` : deg(g.minAngleRad)} to the ${g.left ? 'left' : 'right'}`;
    case 'standsAtEnd': return `the body ends standing (tilt at most ${deg(g.maxTiltRad)}, body above ${g.minBaseHeightM} m in the last ${END_WINDOW_S} s)`;
    case 'bodyAngle':
      return `the body ${g.axis} stays between ${deg(g.minRad)} and ${deg(g.maxRad)} for ${g.holdS} s (${g.axis === 'pitch' ? 'positive lowers the front' : 'positive raises the left side'})`;
    case 'footDown': return `the ${limbName(g.limbId)} stays on the ground for at least ${Math.round(g.minFraction * 100)} % of the run`;
    case 'tipSwing': return `the ${limbName(g.limbId)} tip swings ${TIP_AXIS_TEXT[g.axis]} at least ${g.minSwings} times by ${g.minAmplitudeM} m or more`;
    case 'tipReach': return `the ${limbName(g.limbId)} tip comes within ${g.toleranceM} m of the point (${g.x}, ${g.y}, ${g.z}) m from where it started (x forward, y left, z up) for ${g.holdS} s`;
  }
}

export function motionDuration(m: KeyframeMotion): number {
  return (m.keyframes[m.keyframes.length - 1]?.t ?? 0) + m.endHoldS;
}

/**
 * Plays posed keyframes open loop: a smooth (cosine) blend from the home
 * stance to each pose in turn, then the last pose is held. Targets are
 * clamped to the joint limits; the motion does not use sensor readings.
 */
export class KeyframeController implements SkillController {
  private t = 0;
  readonly durationS: number;

  constructor(private readonly km: KinematicModel, private readonly stanceQ: number[], private readonly motion: KeyframeMotion) {
    if (motion.keyframes.length === 0) throw new RangeError('A keyframe motion needs at least one keyframe');
    if (motion.jointIds.length !== stanceQ.length || motion.jointIds.some((id, i) => id !== km.joints[i]?.id)) {
      throw new RangeError("The keyframes are for other joints than this robot's");
    }
    this.durationS = motionDuration(motion);
  }

  get phase(): string {
    if (this.finished) return 'done';
    const i = this.motion.keyframes.findIndex((k) => this.t < k.t);
    return i < 0 ? 'holding the last pose' : `moving to keyframe ${i + 1}`;
  }

  get finished(): boolean {
    return this.t > this.durationS;
  }

  update(t: number): number[] {
    this.t = t;
    let prevT = 0;
    let prevQ = this.stanceQ;
    for (const k of this.motion.keyframes) {
      if (t < k.t) {
        const s = Math.min(1, Math.max(0, (t - prevT) / (k.t - prevT)));
        const w = 0.5 - 0.5 * Math.cos(Math.PI * s);
        return clampToLimits(this.km, prevQ.map((v, j) => v + w * (k.q[j]! - v)));
      }
      prevT = k.t;
      prevQ = k.q;
    }
    return clampToLimits(this.km, prevQ);
  }
}

/** Ground truth of one physics step, used only to judge a run (controllers never see it). */
export interface JudgeSample {
  t: number;
  basePos: Vec3;
  tiltRad: number;
  /** Heading of the body's x axis in the world (rad, counterclockwise about +z). */
  yawRad: number;
  /** Body pitch, positive when the front is lower than the rear, rad. */
  pitchRad: number;
  /** Body roll, positive when the left side is higher than the right, rad. */
  rollRad: number;
  /** Per limb: tip height above the ground (tip z minus its radius), m; NaN where there is no tip. */
  tipClearanceM: number[];
  /** Per limb: tip position in the world, m. */
  tipPos: Vec3[];
  /** Per limb: its foot's touch sensor reports contact. */
  tipContact: boolean[];
  /** Parts other than the feet within 5 mm of the floor ("body", "front left leg knee"); left out where not measured. */
  groundParts?: string[];
  /** Centre of mass of the whole robot in the world, m; free-standing bodies only. */
  comPos?: Vec3;
}

export interface GoalResult {
  goal: SkillGoal;
  met: boolean;
  /** The quantity the goal is judged on, in `unit`. */
  measured: number;
  unit: string;
  /** What was measured, in plain words. */
  text: string;
  /** Progress toward the goal from 0 to 1 (1 when met); a reward term for improvement. */
  score: number;
}

/**
 * A check every run of a free-standing body must pass whatever its goals
 * say: what someone watching would call a failure even when every goal's
 * number came out right (a jump that lands on its knees, a stand that ends
 * on its back).
 */
export interface BodyCheck {
  kind: 'feetOnly' | 'noTipOver';
  met: boolean;
  text: string;
}

export interface GoalVerdict {
  success: boolean;
  goals: GoalResult[];
  /** Whole-run checks of a free-standing body; empty for a fixed base. */
  checks: BodyCheck[];
  maxTiltRad: number;
  minBaseHeightM: number;
}

/** A part other than a foot counts as touching the ground after this long, s (one grazing step does not). */
const GROUND_TOUCH_S = 0.02;
/** Past this tilt the body has tipped over, unless a goal asks for more. */
const TIP_OVER_RAD = Math.PI / 2;

const clamp01 = (v: number) => (Number.isNaN(v) ? 0 : Math.min(1, Math.max(0, v)));
const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
/** The body counts as come to rest once it is this close to its resting height, m. */
const SETTLE_TOL_M = 0.003;
/** Largest tilt at which every foot off the ground counts as airborne (a jump), rad; named feet off the ground allow up to tipping over (a stand on the hind legs leans back). */
const AIRBORNE_MAX_TILT_RAD = 0.6;

/** Tracks the longest continuous stretch in which a condition held. */
class Stretch {
  private start = -1;
  longest = 0;
  reached = false;
  add(t: number, on: boolean): void {
    if (!on) {
      this.start = -1;
      return;
    }
    this.reached = true;
    if (this.start < 0) this.start = t;
    this.longest = Math.max(this.longest, t - this.start);
  }
}

/** What a goal remembers between samples beyond its stretch. */
interface Track {
  /** Body angle: the range seen and the closest approach to the band, rad. */
  lo: number;
  hi: number;
  closest: number;
  /** Foot down: samples with contact. */
  down: number;
  /** Tip swing: where the current stroke started, its furthest point, its direction, strokes counted and the largest. */
  from: number;
  ext: number;
  dir: number;
  strokes: number;
  largest: number;
  /** Tip reach: the point in the world and the closest distance to it, m. */
  target: Vec3 | null;
  best: number;
}

const newTrack = (): Track => ({ lo: Infinity, hi: -Infinity, closest: Infinity, down: 0, from: Number.NaN, ext: Number.NaN, dir: 0, strokes: 0, largest: 0, target: null, best: Infinity });

/**
 * Judges a run against taught goals from ground-truth samples. Shared by
 * evaluation trials and live playback, so both apply the same rules.
 * Heights above the start (a rise, "body above") count only after the body
 * has first come to rest at its resting height: a body placed slightly above
 * the ground drops at the start, and that drop is not a rise.
 */
export class GoalJudge {
  private n = 0;
  private start: JudgeSample | null = null;
  private last: JudgeSample | null = null;
  private maxTilt = 0;
  private minZ = Infinity;
  private settled = false;
  private maxZSettled = -Infinity;
  private yawTotal = 0;
  private prevYaw = 0;
  private readonly stretches: Stretch[];
  private readonly maxTip: number[];
  private readonly tracks: Track[];
  private readonly tail: { t: number; tilt: number; z: number }[] = [];
  private readonly limbIndex: (id: string) => number;
  private readonly feet: number[];
  /** Per goal: the feet an airborne goal names (null: every foot). */
  private readonly goalFeet: (number[] | null)[];
  /** Per part off the feet: when its current touch of the ground began, and its longest touch, s. */
  private readonly touchFrom = new Map<string, number>();
  private readonly touchLongest = new Map<string, number>();
  /** Lying down is the one request that lets the body touch the ground. */
  private readonly groundAllowed: boolean;
  private readonly tipLimit: number;
  /**
   * Share of the goals that can be judged at one moment (heights, angles,
   * contacts, tilt) that hold in the last sample, a hold goal whose hold was
   * already met counting as holding; 0 when none can. The dense reward of a
   * feedback policy (D30): once a pose was held long enough, holding it
   * longer earns nothing, so the motion can go on.
   */
  instant = 0;

  /**
   * `restHeightM` is the body height at rest in the home stance, the
   * reference for a rise; without it the first sample's height is used.
   * `bodyChecks` adds the whole-run checks of a free-standing body.
   */
  constructor(
    readonly goals: readonly SkillGoal[],
    private readonly limbs: readonly { id: string; name: string; foot: boolean }[],
    private readonly restHeightM: number | null = null,
    private readonly bodyChecks = false,
  ) {
    this.stretches = goals.map(() => new Stretch());
    this.maxTip = goals.map(() => -Infinity);
    this.tracks = goals.map(newTrack);
    this.limbIndex = (id) => limbs.findIndex((l) => l.id === id);
    this.feet = limbs.map((l, i) => (l.foot ? i : -1)).filter((i) => i >= 0);
    this.goalFeet = goals.map((g) => (g.kind === 'airborne' && g.limbIds ? g.limbIds.map((id) => this.limbIndex(id)) : null));
    this.groundAllowed = goals.some((g) => g.kind === 'baseHeight' && g.below && restHeightM !== null && g.heightM <= 0.6 * restHeightM);
    this.tipLimit = Math.max(TIP_OVER_RAD, ...goals.map((g) => (g.kind === 'bodyAngle' ? Math.max(Math.abs(g.minRad), Math.abs(g.maxRad)) : 0)));
  }

  static forRobot(goals: readonly SkillGoal[], robot: RobotDescription): GoalJudge {
    return new GoalJudge(goals, robot.limbs.map((l) => ({ id: l.id, name: l.name, foot: hasFoot(l) })), restingHeight(robot), robot.base.kind === 'floating');
  }

  add(s: JudgeSample): void {
    if (!this.start) {
      this.start = s;
      this.prevYaw = s.yawRad;
    }
    this.n++;
    this.last = s;
    const z = s.basePos[2];
    this.maxTilt = Math.max(this.maxTilt, s.tiltRad);
    this.minZ = Math.min(this.minZ, z);
    if (!this.settled && z <= (this.restHeightM ?? this.start?.basePos[2] ?? z) + SETTLE_TOL_M) this.settled = true;
    if (this.settled) this.maxZSettled = Math.max(this.maxZSettled, z);
    this.yawTotal += wrap(s.yawRad - this.prevYaw);
    this.prevYaw = s.yawRad;
    this.tail.push({ t: s.t, tilt: s.tiltRad, z });
    while (this.tail.length > 1 && this.tail[0]!.t < s.t - END_WINDOW_S) this.tail.shift();
    const touching = s.groundParts ?? [];
    for (const p of touching) {
      const from = this.touchFrom.get(p) ?? s.t;
      this.touchFrom.set(p, from);
      this.touchLongest.set(p, Math.max(this.touchLongest.get(p) ?? 0, s.t - from));
    }
    for (const p of [...this.touchFrom.keys()]) if (!touching.includes(p)) this.touchFrom.delete(p);
    const feet = this.feet;
    const start = this.start ?? s;
    let held = 0, judged = 0;
    const now = (c: boolean, done = false): boolean => {
      judged++;
      if (c || done) held++;
      return c;
    };
    this.goals.forEach((g, i) => {
      const st = this.stretches[i]!;
      const met = (holdS: number): boolean => st.reached && st.longest >= holdS;
      switch (g.kind) {
        case 'upright':
          now(s.tiltRad <= g.maxTiltRad);
          break;
        case 'noFall':
          now(z >= g.minBaseHeightM);
          break;
        case 'tipHeight': {
          const c = s.tipClearanceM[this.limbIndex(g.limbId)] ?? Number.NaN;
          if (Number.isFinite(c)) this.maxTip[i] = Math.max(this.maxTip[i]!, c);
          st.add(s.t, now(c >= g.minHeightM, met(g.holdS)));
          break;
        }
        case 'baseHeight':
          st.add(s.t, now(g.below ? z <= g.heightM : this.settled && z >= g.heightM, met(g.holdS)));
          break;
        case 'airborne': {
          // A body on its back has every foot in the air too; only an upright body is jumping.
          const fs = this.goalFeet[i] ?? feet;
          const maxTilt = this.goalFeet[i] ? TIP_OVER_RAD : AIRBORNE_MAX_TILT_RAD;
          st.add(s.t, now(fs.length > 0 && s.tiltRad <= maxTilt && fs.every((f) => f >= 0 && !s.tipContact[f]), met(g.minS)));
          break;
        }
        case 'bodyAngle': {
          const v = g.axis === 'pitch' ? s.pitchRad : s.rollRad;
          const k = this.tracks[i]!;
          k.lo = Math.min(k.lo, v);
          k.hi = Math.max(k.hi, v);
          k.closest = Math.min(k.closest, v < g.minRad ? g.minRad - v : v > g.maxRad ? v - g.maxRad : 0);
          st.add(s.t, now(v >= g.minRad && v <= g.maxRad, met(g.holdS)));
          break;
        }
        case 'footDown':
          if (now(s.tipContact[this.limbIndex(g.limbId)] === true)) this.tracks[i]!.down++;
          break;
        case 'tipSwing': {
          const p = s.tipPos[this.limbIndex(g.limbId)];
          if (p) this.stroke(this.tracks[i]!, this.heading(p)[g.axis === 'x' ? 0 : g.axis === 'y' ? 1 : 2], g.minAmplitudeM);
          break;
        }
        case 'tipReach': {
          const li = this.limbIndex(g.limbId);
          const p = s.tipPos[li];
          const k = this.tracks[i]!;
          if (!p) break;
          if (!k.target) {
            const p0 = start.tipPos[li] ?? p;
            const c = Math.cos(start.yawRad), sn = Math.sin(start.yawRad);
            k.target = [p0[0] + g.x * c - g.y * sn, p0[1] + g.x * sn + g.y * c, p0[2] + g.z];
          }
          const d = Math.hypot(p[0] - k.target[0], p[1] - k.target[1], p[2] - k.target[2]);
          k.best = Math.min(k.best, d);
          st.add(s.t, now(d <= g.toleranceM, met(g.holdS)));
          break;
        }
        default:
          break;
      }
    });
    if (this.bodyChecks) {
      now(touching.length === 0 || this.groundAllowed);
      now(s.tiltRad <= this.tipLimit);
    }
    this.instant = judged > 0 ? held / judged : 0;
  }

  /** A world point in the start heading frame: x forward, y left, z up. */
  private heading(p: Vec3): Vec3 {
    const yaw = this.start?.yawRad ?? 0;
    const c = Math.cos(yaw), sn = Math.sin(yaw);
    return [p[0] * c + p[1] * sn, -p[0] * sn + p[1] * c, p[2]];
  }

  private limbText(id: string): string {
    return this.limbs[this.limbIndex(id)]?.name.toLowerCase() ?? id;
  }

  /**
   * Counts strokes: movements of at least `h` in one direction, each ending
   * where the motion turns back by `h`. The first stroke is the first time
   * the range seen spans `h`, so a swing centred on the start counts too.
   */
  private stroke(k: Track, v: number, h: number): void {
    if (k.dir === 0) {
      k.lo = Math.min(k.lo, v);
      k.hi = Math.max(k.hi, v);
      if (k.hi - k.lo >= h) {
        const up = v - k.lo >= k.hi - v;
        k.dir = up ? 1 : -1;
        k.from = up ? k.lo : k.hi;
        k.ext = v;
        k.strokes = 1;
        k.largest = Math.abs(v - k.from);
      }
      return;
    }
    if ((v - k.ext) * k.dir > 0) {
      k.ext = v;
    } else if (Math.abs(k.ext - v) >= h) {
      k.from = k.ext;
      k.ext = v;
      k.dir = -k.dir;
      k.strokes++;
    }
    k.largest = Math.max(k.largest, Math.abs(k.ext - k.from));
  }

  verdict(run: { unstable: boolean }): GoalVerdict {
    const results = this.goals.map((g, i) => this.result(g, i));
    const checks = this.bodyChecks ? this.checkResults() : [];
    return {
      success: !run.unstable && this.n > 0 && results.every((r) => r.met) && checks.every((c) => c.met),
      goals: results,
      checks,
      maxTiltRad: +this.maxTilt.toFixed(4),
      minBaseHeightM: +(Number.isFinite(this.minZ) ? this.minZ : 0).toFixed(4),
    };
  }

  private checkResults(): BodyCheck[] {
    const touched = [...this.touchLongest].filter(([, d]) => d >= GROUND_TOUCH_S).sort((a, b) => b[1] - a[1]);
    const feetOnly: BodyCheck = this.groundAllowed
      ? { kind: 'feetOnly', met: true, text: 'the body may touch the ground (a goal asks it to lie low)' }
      : touched.length === 0
        ? { kind: 'feetOnly', met: true, text: 'only the feet touched the ground' }
        : { kind: 'feetOnly', met: false, text: `the ${touched.slice(0, 3).map(([p, d]) => `${p} (${d.toFixed(2)} s)`).join(', ')} touched the ground; only the feet may` };
    const over = this.maxTilt > this.tipLimit;
    return [
      feetOnly,
      { kind: 'noTipOver', met: this.n > 0 && !over, text: over ? `tipped over (largest tilt ${deg(this.maxTilt)}, past ${deg(this.tipLimit)})` : `never tipped over (largest tilt ${deg(this.maxTilt)})` },
    ];
  }

  private result(g: SkillGoal, i: number): GoalResult {
    const st = this.stretches[i]!;
    const s0 = this.start;
    const s1 = this.last;
    const startZ = s0?.basePos[2] ?? 0;
    const r = (met: boolean, measured: number, unit: string, text: string, progress: number): GoalResult => ({
      goal: g, met, measured: +measured.toFixed(4), unit, text, score: met ? 1 : clamp01(progress),
    });
    switch (g.kind) {
      case 'upright': {
        const met = this.n > 0 && this.maxTilt <= g.maxTiltRad;
        return r(met, this.maxTilt, 'rad', `largest tilt ${deg(this.maxTilt)} (limit ${deg(g.maxTiltRad)})`, g.maxTiltRad / this.maxTilt);
      }
      case 'noFall': {
        const z = Number.isFinite(this.minZ) ? this.minZ : 0;
        return r(this.n > 0 && z >= g.minBaseHeightM, z, 'm', `lowest body height ${z.toFixed(3)} m (minimum ${g.minBaseHeightM} m)`, g.minBaseHeightM > 0 ? z / g.minBaseHeightM : 1);
      }
      case 'tipHeight': {
        const best = Number.isFinite(this.maxTip[i]!) ? this.maxTip[i]! : 0;
        const met = st.reached && st.longest >= g.holdS;
        const reach = g.minHeightM > 0 ? best / g.minHeightM : st.reached ? 1 : 0;
        const hold = g.holdS > 0 ? st.longest / g.holdS : st.reached ? 1 : 0;
        const name = this.limbs[this.limbIndex(g.limbId)]?.name.toLowerCase() ?? g.limbId;
        return r(met, st.longest, 's', `${name} tip at least ${g.minHeightM} m up for ${st.longest.toFixed(2)} s of ${g.holdS} s required (highest ${best.toFixed(3)} m)`, 0.5 * clamp01(reach) + 0.5 * clamp01(hold));
      }
      case 'baseHeight': {
        const ref = this.restHeightM ?? startZ;
        const extreme = g.below ? (Number.isFinite(this.minZ) ? this.minZ : startZ) : Number.isFinite(this.maxZSettled) ? this.maxZSettled : ref;
        const reach = g.below ? (startZ > g.heightM ? (startZ - extreme) / (startZ - g.heightM) : 1) : ref < g.heightM ? (extreme - ref) / (g.heightM - ref) : 1;
        const hold = g.holdS > 0 ? st.longest / g.holdS : st.reached ? 1 : 0;
        return r(st.reached && st.longest >= g.holdS, st.longest, 's', `body ${g.below ? 'below' : 'above'} ${g.heightM} m for ${st.longest.toFixed(2)} s of ${g.holdS} s required (${g.below ? 'lowest' : 'highest'} ${extreme.toFixed(3)} m)`, 0.5 * clamp01(reach) + 0.5 * clamp01(hold));
      }
      case 'airborne': {
        const which = g.limbIds ? `the feet of the ${g.limbIds.map((id) => this.limbText(id)).join(' and ')}` : 'every foot';
        return r(st.longest >= g.minS, st.longest, 's', `${which} off the ground with the body upright for at most ${st.longest.toFixed(3)} s at once (needed ${g.minS} s)`, st.longest / g.minS);
      }
      case 'bodyAngle': {
        const k = this.tracks[i]!;
        const hold = g.holdS > 0 ? st.longest / g.holdS : st.reached ? 1 : 0;
        const seen = Number.isFinite(k.lo) ? `${deg(k.lo)} to ${deg(k.hi)}` : 'not measured';
        return r(
          st.reached && st.longest >= g.holdS, st.longest, 's',
          `body ${g.axis} between ${deg(g.minRad)} and ${deg(g.maxRad)} for ${st.longest.toFixed(2)} s of ${g.holdS} s required (it ranged ${seen})`,
          0.5 * clamp01(Number.isFinite(k.closest) ? 1 - k.closest / 0.5 : 0) + 0.5 * clamp01(hold),
        );
      }
      case 'footDown': {
        const f = this.n > 0 ? this.tracks[i]!.down / this.n : 0;
        return r(this.n > 0 && f >= g.minFraction, f, 'fraction', `${this.limbText(g.limbId)} on the ground for ${Math.round(f * 100)} % of the run (needed ${Math.round(g.minFraction * 100)} %)`, g.minFraction > 0 ? f / g.minFraction : 1);
      }
      case 'tipSwing': {
        const k = this.tracks[i]!;
        // Before the first stroke the largest movement is the range seen so far.
        const largest = k.dir !== 0 ? k.largest : Number.isFinite(k.hi - k.lo) ? k.hi - k.lo : 0;
        return r(
          k.strokes >= g.minSwings, k.strokes, 'count',
          `${this.limbText(g.limbId)} tip swung ${TIP_AXIS_TEXT[g.axis]} ${k.strokes} times by at least ${g.minAmplitudeM} m (needed ${g.minSwings}; largest stroke ${largest.toFixed(3)} m)`,
          0.7 * (k.strokes / g.minSwings) + 0.3 * clamp01(largest / g.minAmplitudeM),
        );
      }
      case 'tipReach': {
        const k = this.tracks[i]!;
        const hold = g.holdS > 0 ? st.longest / g.holdS : st.reached ? 1 : 0;
        const closest = Number.isFinite(k.best) ? `came within ${k.best.toFixed(3)} m` : 'was never measured';
        return r(
          st.reached && st.longest >= g.holdS, Number.isFinite(k.best) ? k.best : 0, 'm',
          `${this.limbText(g.limbId)} tip ${closest} of the point (tolerance ${g.toleranceM} m), inside it for ${st.longest.toFixed(2)} s of ${g.holdS} s required`,
          0.5 * clamp01(Number.isFinite(k.best) ? 1 - Math.max(0, k.best - g.toleranceM) / 0.3 : 0) + 0.5 * clamp01(hold),
        );
      }
      case 'baseRise': {
        const ref = this.restHeightM ?? startZ;
        if (!this.settled) return r(false, 0, 'm', `the body never came to rest at its resting height of ${ref.toFixed(3)} m, so no rise was measured`, 0);
        const rise = this.maxZSettled - ref;
        return r(rise >= g.minRiseM, rise, 'm', `body rose ${rise.toFixed(3)} m above its resting height of ${ref.toFixed(3)} m (needed ${g.minRiseM} m)`, rise / g.minRiseM);
      }
      case 'travel': {
        let d = 0;
        if (s0 && s1) {
          const dx = s1.basePos[0] - s0.basePos[0], dy = s1.basePos[1] - s0.basePos[1];
          const c = Math.cos(s0.yawRad), s = Math.sin(s0.yawRad);
          const fwd = dx * c + dy * s, left = -dx * s + dy * c;
          d = g.direction === 'forward' ? fwd : g.direction === 'backward' ? -fwd : g.direction === 'left' ? left : -left;
        }
        const max = g.maxDistanceM;
        const needed = max !== undefined ? `${g.minDistanceM} to ${max} m` : `${g.minDistanceM} m`;
        return r(d >= g.minDistanceM && (max === undefined || d <= max), d, 'm', `body ended ${d.toFixed(3)} m ${g.direction} of its start (needed ${needed})`, max !== undefined && d > max ? max / d : d / g.minDistanceM);
      }
      case 'turn': {
        const a = g.left ? this.yawTotal : -this.yawTotal;
        const max = g.maxAngleRad;
        const needed = max !== undefined ? `${deg(g.minAngleRad)} to ${deg(max)}` : deg(g.minAngleRad);
        return r(a >= g.minAngleRad && (max === undefined || a <= max), a, 'rad', `body turned ${deg(a)} to the ${g.left ? 'left' : 'right'} (needed ${needed})`, max !== undefined && a > max ? max / a : a / g.minAngleRad);
      }
      case 'standsAtEnd': {
        const tilt = this.tail.length ? Math.max(...this.tail.map((x) => x.tilt)) : Infinity;
        const z = this.tail.length ? Math.min(...this.tail.map((x) => x.z)) : 0;
        const met = this.n > 0 && tilt <= g.maxTiltRad && z >= g.minBaseHeightM;
        const progress = 0.5 * clamp01(g.maxTiltRad / tilt) + 0.5 * clamp01(g.minBaseHeightM > 0 ? z / g.minBaseHeightM : 1);
        return r(met, Number.isFinite(tilt) ? tilt : 0, 'rad', `in the last ${END_WINDOW_S} s: largest tilt ${Number.isFinite(tilt) ? deg(tilt) : 'not measured'} (limit ${deg(g.maxTiltRad)}), lowest body height ${z.toFixed(3)} m (minimum ${g.minBaseHeightM} m)`, progress);
      }
    }
  }
}

/**
 * Scalar reward of one judged run: the mean goal progress and the worst
 * goal's progress in equal parts (success needs every goal, so improvement
 * must not buy progress on some goals by giving up another, such as staying
 * upright), a bonus for success, and a penalty for an unstable run.
 */
export function goalReward(v: { success: boolean; goals: readonly GoalResult[]; checks?: readonly BodyCheck[] }, unstable: boolean): number {
  const scores = [...v.goals.map((g) => g.score), ...(v.checks ?? []).map((c) => (c.met ? 1 : 0))];
  const mean = scores.length ? scores.reduce((s, x) => s + x, 0) / scores.length : 0;
  const worst = scores.length ? Math.min(...scores) : 0;
  return 0.5 * mean + 0.5 * worst + (v.success ? 1 : 0) - (unstable ? 1 : 0);
}

/** Lowercase words of a skill name or command, for matching. */
export function normalizeSkillName(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9']+/g, ' ').replace(/\s+/g, ' ').trim();
}

const LEAD = /^(please|kindly|now|then|ok|okay|hey|robot|can you|could you|would you|will you|go and|do|perform|run|play|execute|show me|show|try|start|the skill|skill|your|a|an|the)\s+/;
const TRAIL = /\s+(please|now|again|for me|once)$/;

/**
 * The taught skill a command names: its name exactly, optionally with
 * leading words such as "please", "do the" or "run" and trailing ones such
 * as "now" or "again". Returns null when no name matches.
 */
export function matchTaughtSkill(text: string, names: readonly string[]): string | null {
  let t = normalizeSkillName(text);
  for (let i = 0; i < 8; i++) {
    const hit = names.find((n) => normalizeSkillName(n) === t);
    if (hit) return hit;
    const next = t.replace(LEAD, '').replace(TRAIL, '');
    if (next === t) return null;
    t = next;
  }
  return null;
}
