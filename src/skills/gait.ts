import { buildKinematicModel, clampToLimits, forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { solveIK } from '../core/kinematics/ik';
import { identity, quatToRotation } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import { homeConfiguration } from '../core/model/catalog';
import type { RobotDescription } from '../core/model/types';
import type { SkillController } from './controllers';
import type { FootLiftObservation } from './footLift';
import type { SkillGoal } from './taught';

/**
 * Legged locomotion from a contact pattern (D25): a trajectory generator
 * moves each foot through stance (on the ground, sliding back relative to the
 * body) and swing (an arc forward) with per-leg phase offsets, the way
 * SayTap uses foot contact patterns as the interface between language and a
 * locomotion controller. Foot targets become joint targets by leg IK. An
 * optional feedback term reads the IMU (sensor only) and raises or lowers
 * feet against body roll and pitch, in the spirit of Policies Modulating
 * Trajectory Generators; its gains can be tuned by evolution strategies.
 */

export type GaitPattern = 'trot' | 'walk' | 'pace' | 'bound' | 'pronk' | 'wave' | 'stand';
export const GAIT_PATTERNS: readonly GaitPattern[] = ['trot', 'walk', 'pace', 'bound', 'pronk', 'wave', 'stand'];

export interface GaitParams {
  pattern: GaitPattern;
  /** Phase offset per limb in limb order, fractions of a cycle; replaces the pattern's offsets when given. */
  phases?: number[];
  /** Legs held off the ground for the whole skill (a three-legged walk). */
  heldUp?: string[];
  /** Fraction of a cycle each foot is on the ground. */
  dutyFactor: number;
  cycleS: number;
  /** Distance the body moves per cycle along its x axis, m (negative walks backward). */
  strideM: number;
  /** Distance per cycle along its y axis, m (positive steps left). */
  lateralM: number;
  /** Heading change per cycle, rad (positive turns left). */
  turnRad: number;
  stepHeightM: number;
  /** Body height change from the home stance, m. */
  bodyHeightM: number;
  durationS: number;
  /** IMU attitude feedback: foot height change per rad of roll or pitch, m/rad; 0 is open loop. */
  feedbackKp: number;
  /** Foot height change per rad/s of roll or pitch rate, m*s/rad. */
  feedbackKd: number;
}

/** Accepted ranges, inclusive (tool schema, editor, import and storage). */
export const GAIT_BOUNDS = {
  dutyFactor: [0.2, 1],
  cycleS: [0.25, 3],
  strideM: [-0.3, 0.3],
  lateralM: [-0.2, 0.2],
  turnRad: [-0.8, 0.8],
  stepHeightM: [0, 0.15],
  bodyHeightM: [-0.12, 0.06],
  durationS: [0.5, 20],
  feedbackKp: [0, 0.5],
  feedbackKd: [0, 0.05],
} as const;

export const DEFAULT_GAIT: GaitParams = {
  pattern: 'trot', dutyFactor: 0.6, cycleS: 0.5, strideM: 0.06, lateralM: 0, turnRad: 0, stepHeightM: 0.04, bodyHeightM: 0, durationS: 4, feedbackKp: 0.1, feedbackKd: 0.005,
};

/** Largest change of a foot's height made by attitude feedback, m. */
const MAX_FEEDBACK_M = 0.06;
/** The stepping fades in and out over at most this long, s, so the robot starts and ends standing. */
const MAX_FADE_S = 0.5;

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const KNOWN = new Set(['pattern', 'phases', 'heldUp', ...Object.keys(GAIT_BOUNDS)]);

export function legIndices(robot: RobotDescription): number[] {
  return robot.limbs.map((l, i) => (l.role === 'leg' ? i : -1)).filter((i) => i >= 0);
}

/** Problems with gait parameters, on a robot when one is given; empty when valid. */
export function gaitProblems(g: unknown, robot?: RobotDescription, where = 'gait'): string[] {
  if (!isObject(g)) return [`${where} must be an object.`];
  const out: string[] = [];
  if (!GAIT_PATTERNS.includes(g.pattern as GaitPattern)) out.push(`${where}.pattern must be one of ${GAIT_PATTERNS.join(', ')}.`);
  for (const [key, [lo, hi]] of Object.entries(GAIT_BOUNDS)) {
    const v = g[key];
    if (!finite(v) || v < lo || v > hi) out.push(`${where}.${key} must be a number from ${lo} to ${hi}.`);
  }
  const extra = Object.keys(g).filter((k) => !KNOWN.has(k));
  if (extra.length) out.push(`${where} has unknown fields: ${extra.join(', ')}.`);
  if (g.phases !== undefined && (!Array.isArray(g.phases) || !g.phases.every((p) => finite(p) && p >= 0 && p <= 1))) out.push(`${where}.phases must list numbers from 0 to 1.`);
  if (g.heldUp !== undefined && (!Array.isArray(g.heldUp) || !g.heldUp.every((id) => typeof id === 'string'))) out.push(`${where}.heldUp must list limb ids.`);
  if (!robot) return out;
  const legs = legIndices(robot);
  if (legs.length < 2) out.push(`${robot.name} has fewer than two legs, so it cannot step.`);
  // A walk or wave lifts one leg at a time: below this duty factor two legs are up together and the body loses its support.
  if ((g.pattern === 'walk' || g.pattern === 'wave') && g.phases === undefined && finite(g.dutyFactor) && legs.length >= 3) {
    const min = 1 - 1 / legs.length;
    if (g.dutyFactor < min - 1e-9) out.push(`${where}.dutyFactor must be at least ${min.toFixed(2)} for a ${g.pattern} on ${legs.length} legs: it lifts one leg at a time, so each foot must be down for ${Math.round(min * 100)} % of the cycle. A trot is faster.`);
  }
  if (Array.isArray(g.phases) && g.phases.length !== robot.limbs.length) out.push(`${where}.phases must give one value per limb (${robot.limbs.length}).`);
  if (Array.isArray(g.heldUp)) {
    for (const id of g.heldUp) {
      const l = robot.limbs.find((x) => x.id === id);
      if (!l || l.role !== 'leg') out.push(`${where}.heldUp: ${robot.name} has no leg "${String(id)}".`);
    }
    if (legs.length - g.heldUp.length < 3) out.push(`${where}.heldUp leaves fewer than three legs to stand on.`);
  }
  return out;
}

/** Home tip positions in the body frame, per limb. */
function homeTips(km: KinematicModel, stanceQ: readonly number[]): Vec3[] {
  return forwardKinematics(km, stanceQ, identity()).tips.map((T) => [T[12]!, T[13]!, T[14]!] as Vec3);
}

/**
 * Phase offsets per limb for a named pattern, from where each foot stands:
 * legs are sorted front to rear on each side (left: y > 0). Trot moves
 * diagonal legs together (a tripod on six legs), pace the legs of one side,
 * bound the front and rear halves, pronk all legs, walk steps one leg at a
 * time from the rear forward on each side, and wave one leg at a time.
 */
export function patternPhases(robot: RobotDescription, pattern: GaitPattern, stanceQ?: readonly number[]): number[] {
  const km = buildKinematicModel(robot);
  const tips = homeTips(km, stanceQ ?? homeConfiguration(robot));
  const legs = legIndices(robot);
  const phases = robot.limbs.map(() => 0);
  const left = legs.filter((i) => tips[i]![1] >= 0).sort((a, b) => tips[b]![0] - tips[a]![0]);
  const right = legs.filter((i) => tips[i]![1] < 0).sort((a, b) => tips[b]![0] - tips[a]![0]);
  const side = (list: number[], f: (rank: number, n: number) => number) => list.forEach((li, rank) => (phases[li] = f(rank, list.length)));
  switch (pattern) {
    case 'trot':
      side(left, (r) => (r % 2 === 0 ? 0 : 0.5));
      side(right, (r) => (r % 2 === 0 ? 0.5 : 0));
      break;
    case 'pace':
      side(left, () => 0);
      side(right, () => 0.5);
      break;
    case 'bound':
      side(left, (r, n) => (r < n / 2 ? 0 : 0.5));
      side(right, (r, n) => (r < n / 2 ? 0 : 0.5));
      break;
    case 'walk':
      // Rear to front on each side, the right side half a cycle later.
      side(left, (r, n) => ((n - 1 - r) / (2 * n)) % 1);
      side(right, (r, n) => (0.5 + (n - 1 - r) / (2 * n)) % 1);
      break;
    case 'wave': {
      const order = [...left.slice().reverse(), ...right.slice().reverse()];
      order.forEach((li, k) => (phases[li] = k / order.length));
      break;
    }
    case 'pronk':
    case 'stand':
      break;
  }
  return phases;
}

const smooth = (x: number): number => (x <= 0 ? 0 : x >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * x));

/** How far the stepping has faded in at time t: 0 at the start and the end, 1 while stepping. */
export function gaitEnvelope(g: GaitParams, t: number): number {
  const fade = Math.min(g.cycleS, MAX_FADE_S);
  return smooth(t / fade) * smooth((g.durationS - t) / fade);
}

/**
 * Plays a gait: foot targets in the body frame from the trajectory generator,
 * optionally corrected by IMU attitude feedback, turned into joint targets by
 * leg IK. Inputs are sensor readings only (the IMU orientation).
 */
export class GaitController implements SkillController {
  private t = 0;
  private q: number[];
  private prev: { t: number; roll: number; pitch: number } | null = null;
  private readonly tips: Vec3[];
  private readonly legs: number[];
  private readonly offsets: number[];
  private readonly held: Set<number>;
  private readonly lever: { x: number; y: number };
  private readonly gait: GaitParams;
  readonly durationS: number;

  constructor(private readonly km: KinematicModel, stanceQ: number[], gait: GaitParams) {
    const robot = km.description;
    const problems = gaitProblems(gait, robot);
    if (problems.length) throw new RangeError(problems[0]!);
    this.gait = { ...gait };
    this.q = stanceQ.slice();
    this.tips = homeTips(km, stanceQ);
    this.legs = legIndices(robot);
    this.offsets = gait.phases ?? patternPhases(robot, gait.pattern, stanceQ);
    this.held = new Set((gait.heldUp ?? []).map((id) => robot.limbs.findIndex((l) => l.id === id)));
    const xs = this.legs.map((i) => Math.abs(this.tips[i]![0]));
    const ys = this.legs.map((i) => Math.abs(this.tips[i]![1]));
    this.lever = { x: Math.max(1e-3, ...xs), y: Math.max(1e-3, ...ys) };
    this.durationS = gait.durationS;
  }

  get phase(): string {
    if (this.finished) return 'done';
    const fade = Math.min(this.gait.cycleS, MAX_FADE_S);
    return this.t < fade ? 'starting to step' : this.t > this.durationS - fade ? 'coming to a stop' : 'stepping';
  }

  get finished(): boolean {
    return this.t > this.durationS;
  }

  /** Changes the stride, side step and turn per cycle while stepping (a walking policy's command), within the bounds; non-finite values are ignored. */
  setMotion(strideM: number, lateralM: number, turnRad: number): void {
    if (![strideM, lateralM, turnRad].every(Number.isFinite)) return;
    const clamp = (v: number, [lo, hi]: readonly [number, number]) => Math.min(hi, Math.max(lo, v));
    this.gait.strideM = clamp(strideM, GAIT_BOUNDS.strideM);
    this.gait.lateralM = clamp(lateralM, GAIT_BOUNDS.lateralM);
    this.gait.turnRad = clamp(turnRad, GAIT_BOUNDS.turnRad);
  }

  update(t: number, obs: FootLiftObservation): number[] {
    this.t = t;
    const g = this.gait;
    // Fade the step in at the start and out at the end, so the robot starts and ends standing.
    const fade = Math.min(g.cycleS, MAX_FADE_S);
    const a = gaitEnvelope(g, t);
    const R = quatToRotation(obs.imuQuat);
    const pitch = -Math.asin(Math.max(-1, Math.min(1, R[6]!)));
    const roll = Math.asin(Math.max(-1, Math.min(1, R[7]!)));
    const dt = this.prev ? t - this.prev.t : 0;
    const pitchRate = dt > 0 ? (pitch - this.prev!.pitch) / dt : 0;
    const rollRate = dt > 0 ? (roll - this.prev!.roll) / dt : 0;
    this.prev = { t, roll, pitch };
    const cycles = t / g.cycleS;
    for (const li of this.legs) {
      const p0 = this.tips[li]!;
      let u = 0;
      let lift = 0;
      if (this.held.has(li)) {
        lift = Math.max(g.stepHeightM, 0.04) * smooth(t / fade);
      } else if (g.dutyFactor < 1) {
        const phi = (((cycles + this.offsets[li]!) % 1) + 1) % 1;
        if (phi < g.dutyFactor) {
          u = 0.5 - phi / g.dutyFactor;
        } else {
          const s = (phi - g.dutyFactor) / (1 - g.dutyFactor);
          u = -0.5 + smooth(s);
          lift = g.stepHeightM * Math.sin(Math.PI * s);
        }
      }
      const th = a * g.turnRad * u;
      const c = Math.cos(th), sn = Math.sin(th);
      const xi = p0[0] / this.lever.x, eta = p0[1] / this.lever.y;
      const fb = -g.feedbackKp * (pitch * xi - roll * eta) - g.feedbackKd * (pitchRate * xi - rollRate * eta);
      const target: Vec3 = [
        p0[0] * c - p0[1] * sn + a * g.strideM * u,
        p0[0] * sn + p0[1] * c + a * g.lateralM * u,
        p0[2] - smooth(t / fade) * g.bodyHeightM + a * lift + (this.held.has(li) ? lift * (1 - a) : 0) + Math.max(-MAX_FEEDBACK_M, Math.min(MAX_FEEDBACK_M, fb)),
      ];
      const r = solveIK(this.km, li, target, this.q, { T_W_B: identity(), maxIterations: 25, tolerance: 1e-4 });
      const k = this.km.limbs[li]!;
      for (let j = k.start; j < k.start + k.count; j++) this.q[j] = r.q[j]!;
    }
    return clampToLimits(this.km, this.q);
  }
}

/**
 * Goals a gait cannot meet because it steps the other way (a right turn
 * commanded as a left one), as reasons; empty when the directions agree.
 */
export function gaitGoalConflicts(g: GaitParams, goals: readonly SkillGoal[]): string[] {
  const out: string[] = [];
  for (const goal of goals) {
    if (goal.kind === 'turn' && (goal.left ? g.turnRad <= 0 : g.turnRad >= 0)) {
      out.push(`the goal turns ${goal.left ? 'left, so turnRad must be positive' : 'right, so turnRad must be negative'} (positive turns left).`);
    }
    if (goal.kind === 'travel') {
      const along = goal.direction === 'forward' || goal.direction === 'backward';
      const v = goal.direction === 'forward' ? g.strideM : goal.direction === 'backward' ? -g.strideM : goal.direction === 'left' ? g.lateralM : -g.lateralM;
      if (v <= 0) out.push(`the goal travels ${goal.direction}, so ${along ? 'strideM' : 'lateralM'} must be ${goal.direction === 'forward' || goal.direction === 'left' ? 'positive' : 'negative'}.`);
    }
  }
  // Travel the goals do not ask for goes unmeasured: a side step with a long stride walks diagonally and still passes.
  const sideways = goals.some((x) => x.kind === 'travel' && (x.direction === 'left' || x.direction === 'right'));
  const lengthwise = goals.some((x) => x.kind === 'travel' && (x.direction === 'forward' || x.direction === 'backward'));
  if (sideways && !lengthwise && Math.abs(g.strideM) > Math.abs(g.lateralM)) {
    out.push(`the goals ask for sideways travel only, but strideM (${g.strideM} m per cycle) moves the body further forward or backward than lateralM (${g.lateralM} m) moves it sideways; keep strideM near 0, or add a forward or backward travel goal.`);
  }
  if (lengthwise && !sideways && Math.abs(g.lateralM) > Math.abs(g.strideM)) {
    out.push(`the goals ask for forward or backward travel only, but lateralM (${g.lateralM} m per cycle) moves the body further sideways than strideM (${g.strideM} m) moves it; keep lateralM near 0, or add a sideways travel goal.`);
  }
  return out;
}

/** A short description of a gait for people and for the AI. */
export function gaitText(g: GaitParams): string {
  const moves: string[] = [];
  if (g.strideM !== 0) moves.push(`${Math.abs(g.strideM)} m ${g.strideM > 0 ? 'forward' : 'backward'} per cycle`);
  if (g.lateralM !== 0) moves.push(`${Math.abs(g.lateralM)} m to the ${g.lateralM > 0 ? 'left' : 'right'} per cycle`);
  if (g.turnRad !== 0) moves.push(`turning ${Math.abs(g.turnRad)} rad to the ${g.turnRad > 0 ? 'left' : 'right'} per cycle`);
  const feedback = g.feedbackKp > 0 || g.feedbackKd > 0 ? `IMU feedback kp ${g.feedbackKp} m/rad, kd ${g.feedbackKd} m*s/rad` : 'no feedback (open loop)';
  return `${g.phases ? 'custom contact pattern' : `${g.pattern} pattern`}, ${g.cycleS} s cycle, feet down ${Math.round(g.dutyFactor * 100)} % of it, steps ${g.stepHeightM} m high, ${moves.length ? moves.join(', ') : 'in place'}, body ${g.bodyHeightM} m from its home height, ${g.durationS} s${g.heldUp?.length ? `, holding up ${g.heldUp.join(', ')}` : ''}; ${feedback}`;
}
