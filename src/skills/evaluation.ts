import type { MainModule } from '@mujoco/mujoco';
import { TargetRateLimiter } from '../core/control/limits';
import { pdControl, type PdGains } from '../core/control/pd';
import { buildKinematicModel } from '../core/kinematics/fk';
import { gaussian, mulberry32 } from '../core/math/rng';
import type { RobotDescription } from '../core/model/types';
import { MujocoRuntime } from '../physics/runtime';
import { footLiftDuration, footLiftFeatures, makeController, skillDuration, type SkillSpec, type Trajectory } from './controllers';
import type { FootLiftParams } from './footLift';
import { GoalJudge, type BodyCheck, type GoalResult, type GoalVerdict, type JudgeSample, type SkillGoal } from './taught';

export interface FootLiftCriteria {
  minClearanceM: number;
  minHoldS: number;
  maxTiltRad: number;
  minBaseHeightM: number;
  /** Fraction of lifted samples in which every support foot reports contact. */
  minSupportContactFraction: number;
}

export const DEFAULT_CRITERIA: FootLiftCriteria = { minClearanceM: 0.03, minHoldS: 2.0, maxTiltRad: 0.35, minBaseHeightM: 0.12, minSupportContactFraction: 0.95 };

/** A taught skill is judged by the goals the user set. */
export interface GoalsJudging {
  goals: SkillGoal[];
}

/** How runs are judged: the foot-lift criteria, or a taught skill's goals. */
export type Judging = FootLiftCriteria | GoalsJudging;

export const isGoals = (j: Judging): j is GoalsJudging => 'goals' in j;

export interface TrialConditions {
  friction: number;
  baseMassKg: number;
  jointNoiseRad: number;
}

export type TrialFailure = 'fell' | 'tilted' | 'insufficient-clearance' | 'hold-too-short' | 'lost-support-contact' | 'aborted' | 'numerical-instability' | 'goal-not-met' | 'body-touched-ground' | 'tipped-over';

export interface TrialResult {
  trial: number;
  conditions: TrialConditions;
  success: boolean;
  failures: TrialFailure[];
  longestClearHoldS: number;
  maxClearanceM: number;
  maxTiltRad: number;
  minBaseHeightM: number;
  supportContactFraction: number;
  finalPhase: string;
  /** Per-goal results of a taught skill (the foot-lift fields above are then zero, except tilt and height). */
  goals?: GoalResult[];
  /** Whole-run checks of a free-standing body (only the feet touch the ground, no tipping over). */
  checks?: BodyCheck[];
}

export interface EvaluationOptions {
  trials: number;
  seed: number;
  frictionRange: [number, number];
  baseMassRange: [number, number];
  jointNoiseRad: number;
  gains?: PdGains;
}

/** Joint PD gains used when an evaluation does not specify any. */
export const DEFAULT_SKILL_GAINS: PdGains = { kp: 40, kd: 1 };

export const DEFAULT_EVALUATION: Omit<EvaluationOptions, 'seed'> = { trials: 20, frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02 };

/** Body mass is randomized within this fraction of the robot's own base mass. */
export const BASE_MASS_SPREAD = 0.125;

/** Evaluation options for a robot: mass randomized around its own base mass, the given gains and seed. */
export function evaluationOptionsFor(desc: RobotDescription, seed: number, gains: PdGains): EvaluationOptions {
  const m = desc.base.mass;
  return { ...DEFAULT_EVALUATION, seed, gains, baseMassRange: [+(m * (1 - BASE_MASS_SPREAD)).toFixed(4), +(m * (1 + BASE_MASS_SPREAD)).toFixed(4)] };
}

/** Everything needed to repeat an evaluation exactly; stored with the skill. */
export interface EvaluationContext {
  seed: number;
  /** Success thresholds of a foot-lift evaluation. */
  criteria?: FootLiftCriteria;
  /** Goals a taught skill was judged by. */
  goals?: SkillGoal[];
  gains: PdGains;
  /** Home joint positions the trials start from (before the random offsets), rad or m. */
  stanceQ: number[];
  /** Half-width of the random initial joint offsets, rad. */
  jointNoiseRad: number;
  frictionRange: [number, number];
  baseMassRange: [number, number];
  nominalBaseMassKg: number;
  limbId: string;
  timestepS: number;
  controlPeriodS: number;
  /** Joint targets were moved toward the controller's targets at no more than each joint's velocity limit. */
  velocityLimited: boolean;
  simulator: string;
}

export interface EvaluationSummary {
  trials: number;
  successes: number;
  successRate: number;
  wilson95: [number, number];
  conditions: { friction: [number, number]; baseMassKg: [number, number]; jointNoiseRad: [number, number] };
  results: TrialResult[];
  simulator: string;
  evaluatedAt: string;
  /** Always set by evaluateSkill; optional only for summaries built elsewhere (tests, older code). */
  context?: EvaluationContext;
}

export function wilson95(successes: number, n: number): [number, number] {
  if (n === 0) return [0, 1];
  const z = 1.959964, p = successes / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** One physics step as seen by the referee: ground truth plus the support feet's touch sensors. */
export interface RefereeSample {
  t: number;
  clearanceM: number;
  tiltRad: number;
  baseHeightM: number;
  /** Every support foot reports contact at this step. */
  supportContact: boolean;
}

export interface RefereeVerdict {
  success: boolean;
  failures: TrialFailure[];
  longestClearHoldS: number;
  maxClearanceM: number;
  maxTiltRad: number;
  minBaseHeightM: number;
  supportContactFraction: number;
}

/**
 * Success judge for one foot-lift run, from ground-truth samples. Shared by
 * evaluation trials and live playback, so a live run is called successful by
 * exactly the rules the evaluation used (with the requested hold time).
 */
export class FootLiftReferee {
  private clearStart = -1;
  private longest = 0;
  private maxClear = 0;
  private maxTilt = 0;
  private minHeight = Infinity;
  private lifted = 0;
  private liftedContact = 0;

  constructor(readonly criteria: FootLiftCriteria, readonly requiredHoldS: number = criteria.minHoldS) {}

  add(s: RefereeSample): void {
    this.maxClear = Math.max(this.maxClear, s.clearanceM);
    this.maxTilt = Math.max(this.maxTilt, s.tiltRad);
    this.minHeight = Math.min(this.minHeight, s.baseHeightM);
    if (s.clearanceM >= this.criteria.minClearanceM) {
      if (this.clearStart < 0) this.clearStart = s.t;
      this.longest = Math.max(this.longest, s.t - this.clearStart);
      this.lifted++;
      if (s.supportContact) this.liftedContact++;
    } else {
      this.clearStart = -1;
    }
  }

  get longestClearHoldS(): number {
    return this.longest;
  }

  verdict(run: { unstable: boolean; aborted: boolean }): RefereeVerdict {
    const c = this.criteria;
    const supportFraction = this.lifted > 0 ? this.liftedContact / this.lifted : 0;
    const failures: TrialFailure[] = [];
    if (run.unstable) failures.push('numerical-instability');
    if (run.aborted) failures.push('aborted');
    if (this.minHeight < c.minBaseHeightM) failures.push('fell');
    if (this.maxTilt > c.maxTiltRad) failures.push('tilted');
    if (this.maxClear < c.minClearanceM) failures.push('insufficient-clearance');
    else if (this.longest < this.requiredHoldS) failures.push('hold-too-short');
    if (this.maxClear >= c.minClearanceM && supportFraction < c.minSupportContactFraction) failures.push('lost-support-contact');
    return {
      success: failures.length === 0,
      failures,
      longestClearHoldS: +this.longest.toFixed(3),
      maxClearanceM: +this.maxClear.toFixed(4),
      maxTiltRad: +this.maxTilt.toFixed(4),
      minBaseHeightM: +(Number.isFinite(this.minHeight) ? this.minHeight : 0).toFixed(4),
      supportContactFraction: +supportFraction.toFixed(3),
    };
  }
}

/** Controller update period: every 5 physics steps (100 Hz at a 2 ms step). */
export const CONTROL_DECIMATION = 5;
export const SIMULATOR_LABEL = 'MuJoCo 3.14.0 (WASM)';

/** Optional recording of what the controller saw and commanded, at the control rate. */
export interface TrialRecording {
  /** Policy features (see footLiftFeatures). */
  features: number[][];
  /** Commanded joint targets. */
  qCmd: number[][];
}

export interface TrialInput {
  desc: RobotDescription;
  limbIndex: number;
  stanceQ: number[];
  spec: SkillSpec;
  criteria: Judging;
  conditions: TrialConditions;
  trial: number;
  noiseSeed: number;
  gains?: PdGains;
  /** Horizon used to normalize the time feature when recording. */
  featureHorizonS?: number;
}

/**
 * One randomized trial. The controller sees sensors only (encoders, IMU,
 * foot touch); ground truth is used only by the referee to score.
 */
export function runSkillTrial(mj: MainModule, input: TrialInput, recording?: TrialRecording): TrialResult {
  const { desc, limbIndex, stanceQ, spec, criteria, conditions, trial, noiseSeed } = input;
  const gains = input.gains ?? DEFAULT_SKILL_GAINS;
  const rng = mulberry32(noiseSeed);
  const q0 = stanceQ.map((v) => v + conditions.jointNoiseRad * Math.max(-1, Math.min(1, gaussian(rng) / 2)));
  const km = buildKinematicModel(desc);
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q0, friction: conditions.friction, baseMass: conditions.baseMassKg });
  try {
    const ctl = makeController(km, limbIndex, stanceQ, spec);
    const g = stanceQ.map(() => gains);
    const joints = desc.limbs.flatMap((l) => l.joints);
    const continuous = joints.map((j) => j.type === 'continuous');
    const limiter = new TargetRateLimiter(joints.map((j) => j.limit.velocity), continuous, q0);
    const judge = isGoals(criteria) ? GoalJudge.forRobot(criteria.goals, desc) : null;
    const referee = isGoals(criteria) ? null : new FootLiftReferee(criteria);
    const tipRadius = referee ? desc.limbs[limbIndex]!.tip.radius : 0;
    const support = referee ? supportLegs(desc, limbIndex) : [];
    const horizon = input.featureHorizonS ?? skillDuration(spec);
    let qTarget = stanceQ.slice();
    let lastTouch: number[] = [];
    let unstable = false, doneAt = -1;
    const maxSteps = Math.ceil((skillDuration(spec) + 2) / rt.timestep);
    for (let k = 0; k < maxSteps; k++) {
      const t = k * rt.timestep;
      rt.controlStep((s) => {
        if (k % CONTROL_DECIMATION === 0) {
          const obs = { q: s.q, qd: s.qd, touch: s.touch, imuQuat: s.imuQuat, gyro: s.gyro };
          if (recording) recording.features.push(footLiftFeatures(t, obs, stanceQ, horizon));
          qTarget = ctl.update(t, obs);
          if (recording) recording.qCmd.push(qTarget.slice());
        }
        limiter.step(qTarget, rt.timestep);
        lastTouch = s.touch;
        // Position regulation: the bounded target feeds the P term; the D term damps measured velocity only.
        return pdControl(g, rt.layout.effort, { qCmd: limiter.qCmd, qMeas: s.q, qdMeas: s.qd, continuous }).tauApplied;
      });
      if (rt.hasNonFiniteState()) {
        unstable = true;
        break;
      }
      // Referee (ground truth).
      if (judge) judge.add(judgeSample(rt, t, lastTouch));
      else {
        const b = rt.groundTruthBaseState();
        referee!.add({ t, clearanceM: rt.groundTruthTipPositions()[limbIndex]![2] - tipRadius, tiltRad: b.tilt, baseHeightM: b.pos[2], supportContact: support.every((i) => (lastTouch[i] ?? 0) > 0) });
      }
      if (ctl.finished && doneAt < 0) doneAt = t;
      if (doneAt >= 0 && t - doneAt > 0.5) break;
    }
    if (judge) {
      const v = judge.verdict({ unstable });
      const failures: TrialFailure[] = unstable ? ['numerical-instability'] : v.success ? [] : verdictFailures(v);
      return { trial, conditions, success: v.success, failures, longestClearHoldS: 0, maxClearanceM: 0, maxTiltRad: v.maxTiltRad, minBaseHeightM: v.minBaseHeightM, supportContactFraction: 0, finalPhase: ctl.phase, goals: v.goals, checks: v.checks };
    }
    const v = referee!.verdict({ unstable, aborted: ctl.phase === 'aborted' });
    return { trial, conditions, ...v, finalPhase: ctl.phase };
  } finally {
    rt.dispose();
  }
}

/** Why a judged run failed: a missed goal, and each whole-run body check it did not pass. */
export function verdictFailures(v: GoalVerdict): TrialFailure[] {
  const out: TrialFailure[] = v.goals.every((g) => g.met) ? [] : ['goal-not-met'];
  for (const c of v.checks) if (!c.met) out.push(c.kind === 'feetOnly' ? 'body-touched-ground' : 'tipped-over');
  return out;
}

/** Ground truth of the current physics state for a goal judge; controllers never see it. */
export function judgeSample(rt: MujocoRuntime, t: number, touch: readonly number[]): JudgeSample {
  const T = rt.groundTruthBaseTransform();
  const limbs = rt.description.limbs;
  const tips = rt.groundTruthTipPositions();
  const asin = (v: number) => Math.asin(Math.max(-1, Math.min(1, v)));
  return {
    t,
    basePos: [T[12]!, T[13]!, T[14]!],
    tiltRad: Math.acos(Math.max(-1, Math.min(1, T[10]!))),
    yawRad: Math.atan2(T[1]!, T[0]!),
    // The body x axis pointing down means the front is low (positive pitch); the y axis pointing up means the left side is high.
    pitchRad: -asin(T[2]!),
    rollRad: asin(T[6]!),
    tipClearanceM: tips.map((p, i) => p[2] - limbs[i]!.tip.radius),
    tipPos: tips,
    tipContact: limbs.map((_, i) => (touch[i] ?? 0) > 0),
    ...(rt.description.base.kind === 'floating' ? { groundParts: rt.groundTruthTouchingParts(), comPos: rt.groundTruthCenterOfMass() } : {}),
  };
}

/** Limbs that must keep their feet on the ground while `limbIndex` lifts. */
export function supportLegs(desc: RobotDescription, limbIndex: number): number[] {
  return desc.limbs.map((_, i) => i).filter((i) => i !== limbIndex && desc.limbs[i]!.role === 'leg');
}

/** Deterministic randomized conditions for trial i of an evaluation with this seed. */
export function trialConditions(opts: Pick<EvaluationOptions, 'seed' | 'frictionRange' | 'baseMassRange' | 'jointNoiseRad'>, i: number): TrialConditions {
  const rng = mulberry32(opts.seed * 7919 + i);
  return {
    friction: +(opts.frictionRange[0] + (opts.frictionRange[1] - opts.frictionRange[0]) * rng()).toFixed(4),
    baseMassKg: +(opts.baseMassRange[0] + (opts.baseMassRange[1] - opts.baseMassRange[0]) * rng()).toFixed(4),
    jointNoiseRad: opts.jointNoiseRad,
  };
}

/** Runs `opts.trials` randomized trials of a skill and summarizes them. Returning false from onTrial stops early. */
export function evaluateSkill(
  mj: MainModule,
  desc: RobotDescription,
  limbIndex: number,
  stanceQ: number[],
  spec: SkillSpec,
  opts: EvaluationOptions,
  criteria: Judging = DEFAULT_CRITERIA,
  onTrial?: (r: TrialResult) => boolean | void,
): EvaluationSummary {
  const results: TrialResult[] = [];
  for (let i = 0; i < opts.trials; i++) {
    const r = runSkillTrial(mj, { desc, limbIndex, stanceQ, spec, criteria, conditions: trialConditions(opts, i), trial: i, noiseSeed: opts.seed * 104729 + i, gains: opts.gains });
    results.push(r);
    if (onTrial?.(r) === false) break;
  }
  const successes = results.filter((r) => r.success).length;
  const gains = opts.gains ?? DEFAULT_SKILL_GAINS;
  return {
    trials: results.length,
    successes,
    successRate: results.length ? successes / results.length : 0,
    wilson95: wilson95(successes, results.length),
    conditions: { friction: opts.frictionRange, baseMassKg: opts.baseMassRange, jointNoiseRad: [-opts.jointNoiseRad, opts.jointNoiseRad] },
    results,
    simulator: SIMULATOR_LABEL,
    evaluatedAt: new Date().toISOString(),
    context: {
      seed: opts.seed,
      ...(isGoals(criteria) ? { goals: criteria.goals.map((x) => ({ ...x })) } : { criteria: { ...criteria } }),
      gains: { ...gains },
      stanceQ: stanceQ.slice(),
      jointNoiseRad: opts.jointNoiseRad,
      frictionRange: [opts.frictionRange[0], opts.frictionRange[1]],
      baseMassRange: [opts.baseMassRange[0], opts.baseMassRange[1]],
      nominalBaseMassKg: desc.base.mass,
      limbId: desc.limbs[limbIndex]?.id ?? '',
      timestepS: 0.002,
      controlPeriodS: CONTROL_DECIMATION * 0.002,
      velocityLimited: true,
      simulator: SIMULATOR_LABEL,
    },
  };
}

/** The Phase 1 entry point: the hand-designed state machine. */
export function evaluateFootLift(
  mj: MainModule,
  desc: RobotDescription,
  limbIndex: number,
  stanceQ: number[],
  params: FootLiftParams,
  opts: EvaluationOptions,
  criteria: FootLiftCriteria = DEFAULT_CRITERIA,
  onTrial?: (r: TrialResult) => void,
): EvaluationSummary {
  return evaluateSkill(mj, desc, limbIndex, stanceQ, { kind: 'state-machine', params }, opts, criteria, onTrial);
}

/** Records the state-machine expert once under nominal conditions as a replayable demonstration. */
export function recordExpertDemonstration(mj: MainModule, desc: RobotDescription, limbIndex: number, stanceQ: number[], params: FootLiftParams, gains?: PdGains): Trajectory {
  const rec: TrialRecording = { features: [], qCmd: [] };
  const km = buildKinematicModel(desc);
  runSkillTrial(mj, { desc, limbIndex, stanceQ, spec: { kind: 'state-machine', params }, criteria: DEFAULT_CRITERIA, conditions: { friction: 1, baseMassKg: desc.base.mass, jointNoiseRad: 0 }, trial: 0, noiseSeed: 1, gains }, rec);
  const dt = CONTROL_DECIMATION * 0.002;
  const n = Math.min(rec.qCmd.length, Math.round(footLiftDuration(params) / dt) + 1);
  return { dt, jointIds: km.joints.map((j) => j.id), frames: rec.qCmd.slice(0, n) };
}
