import type { PdGains } from '../core/control/pd';
import { mulberry32 } from '../core/math/rng';
import type { Vec3 } from '../core/math/vec3';
import type { ParseResult } from '../core/model/parse';
import type { RobotDescription } from '../core/model/types';
import { augmentedRandomSearchBatched, RunningStats, type ArsProgress } from '../learning/ars';
import type { PolicyJson } from './controllers';
import { trialConditions, wilson95, type TrialConditions } from './evaluation';
import { DEFAULT_GAIT, GAIT_BOUNDS, gaitEnvelope, gaitProblems, type GaitParams } from './gait';
import type { ConditionRanges } from './learning';
import { gaitCommand } from './policyGait';
import type { RolloutJob, RolloutOutcome, RolloutRunner, RolloutSetup } from './rollouts';
import { WALK_COMMAND_OUTPUTS, WALK_POLICY_INPUTS, walkPolicyProblem } from './walkPolicy';

/**
 * A learned walking controller per body (D28, physical-AI plan Phase 2; D29).
 * The gait generator is driven by a speed command. A calibration learned from
 * simulated runs (system identification) maps the speed asked for to the
 * generator's command, including the drift one direction causes in another,
 * and a feedback policy trained by random search over random commands reads
 * the IMU and a sensor-only estimate of where the body is against where the
 * command leads, and corrects the generator's command and every joint. A
 * language model commands it with speeds and a duration, or with a target
 * the tool turns into both; it never sets joint targets.
 */

/** A walking command in the body frame. */
export interface WalkCommand {
  /** Forward speed, m/s (negative walks backward). */
  forwardMps: number;
  /** Leftward speed, m/s (negative steps right). */
  leftMps: number;
  /** Yaw rate, rad/s (positive turns left). */
  turnRadps: number;
}

export interface WalkRanges {
  forwardMps: [number, number];
  leftMps: [number, number];
  turnRadps: [number, number];
}

export const DEFAULT_WALK_RANGES: WalkRanges = { forwardMps: [-0.12, 0.15], leftMps: [-0.08, 0.08], turnRadps: [-0.6, 0.6] };
/** Range of a walk's duration, s. */
export const WALK_DURATION_S: [number, number] = [1, 10];
/** Largest tilt of a walk that counts as upright, rad. */
const WALK_MAX_TILT_RAD = 0.5;
const RANGE_LIMITS: Record<keyof WalkRanges, number> = { forwardMps: 0.5, leftMps: 0.3, turnRadps: 1.5 };

export const vecOf = (c: WalkCommand): Vec3 => [c.forwardMps, c.leftMps, c.turnRadps];
const cmdOf = (v: Readonly<Vec3>): WalkCommand => ({ forwardMps: v[0], leftMps: v[1], turnRadps: v[2] });

/** Calibration features of a generator command: the positive and the negative part of each axis. */
function parts(n: Readonly<Vec3>): number[] {
  return [Math.max(n[0], 0), Math.min(n[0], 0), Math.max(n[1], 0), Math.min(n[1], 0), Math.max(n[2], 0), Math.min(n[2], 0)];
}

/** Solves A x = b by Gaussian elimination with partial pivoting; null when A is singular. */
function solve(A: readonly (readonly number[])[], b: readonly number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    if (!(Math.abs(M[p]![c]!) > 1e-12)) return null;
    [M[c], M[p]] = [M[p]!, M[c]!];
    for (let r = c + 1; r < n; r++) {
      const f = M[r]![c]! / M[c]![c]!;
      for (let k = c; k <= n; k++) M[r]![k]! -= f * M[c]![k]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r]![n]!;
    for (let k = r + 1; k < n; k++) s -= M[r]![k]! * x[k]!;
    x[r] = s / M[r]![r]!;
  }
  return x;
}

/** Velocity the calibration (3 rows of 6 coefficients) predicts for a generator command. */
export function predictVelocity(cal: readonly number[], nominal: Readonly<Vec3>): Vec3 {
  const f = parts(nominal);
  const row = (i: number) => f.reduce((s, x, j) => s + cal[6 * i + j]! * x, 0);
  return [row(0), row(1), row(2)];
}

/** Least-squares calibration from generator commands and the velocities measured with them. */
export function fitCalibration(samples: readonly { nominal: Vec3; measured: Vec3 }[]): number[] {
  const F = samples.map((s) => parts(s.nominal));
  const G = Array.from({ length: 6 }, (_, a) => Array.from({ length: 6 }, (_, b) => F.reduce((s, f) => s + f[a]! * f[b]!, 0) + (a === b ? 1e-9 : 0)));
  const out: number[] = [];
  for (let i = 0; i < 3; i++) {
    const rhs = Array.from({ length: 6 }, (_, a) => samples.reduce((s, smp, k) => s + F[k]![a]! * smp.measured[i]!, 0));
    out.push(...(solve(G, rhs) ?? new Array<number>(6).fill(0)));
  }
  return out;
}

/** The generator command whose predicted velocity is `desired` (the smallest, if several), or null when none is. */
export function nominalFor(cal: readonly number[], desired: Readonly<Vec3>): Vec3 | null {
  let best: Vec3 | null = null;
  for (let mask = 0; mask < 8; mask++) {
    const s = [mask & 1 ? -1 : 1, mask & 2 ? -1 : 1, mask & 4 ? -1 : 1];
    const A = [0, 1, 2].map((i) => [0, 1, 2].map((j) => cal[6 * i + 2 * j + (s[j]! > 0 ? 0 : 1)]!));
    const n = solve(A, [desired[0], desired[1], desired[2]]);
    if (!n || !n.every((v, j) => v * s[j]! >= -1e-12)) continue;
    const cand: Vec3 = [n[0]!, n[1]!, n[2]!];
    if (!best || Math.hypot(...cand) < Math.hypot(...best)) best = cand;
  }
  return best;
}

export interface WalkGait {
  gait: GaitParams;
  /** The calibration gave the generator command (false: the command was used as given). */
  calibrated: boolean;
  /** The generator's stride, side step or turn per cycle hit its bounds. */
  clamped: boolean;
}

/** The gait that walks a command: the generator command from the calibration (if any), as stride, side step and turn per cycle. */
export function walkGait(base: GaitParams, calibration: readonly number[] | null, cmd: WalkCommand, durationS: number): WalkGait {
  const desired = vecOf(cmd);
  const fromCal = calibration ? nominalFor(calibration, desired) : null;
  const nominal = fromCal ?? desired;
  let clamped = false;
  const perCycle = (v: number, [lo, hi]: readonly [number, number]) => {
    const x = v * base.cycleS;
    if (x < lo || x > hi) clamped = true;
    return Number(Math.min(hi, Math.max(lo, x)).toFixed(5));
  };
  const gait: GaitParams = { ...base, strideM: perCycle(nominal[0], GAIT_BOUNDS.strideM), lateralM: perCycle(nominal[1], GAIT_BOUNDS.lateralM), turnRad: perCycle(nominal[2], GAIT_BOUNDS.turnRad), durationS };
  return { gait, calibrated: fromCal !== null, clamped };
}

/**
 * Where a command takes the body if followed exactly, faded in and out like
 * the gait: forward and left, m, and heading change, rad, in the start frame.
 */
export function expectedPose(cmd: WalkCommand, gait: Pick<GaitParams, 'cycleS' | 'durationS'>): Vec3 {
  const steps = Math.max(1, Math.ceil(gait.durationS / 0.002));
  const dt = gait.durationS / steps;
  const g = gait as GaitParams;
  let x = 0, y = 0, psi = 0;
  for (let k = 0; k < steps; k++) {
    const a = gaitEnvelope(g, (k + 0.5) * dt);
    const mid = psi + 0.5 * a * cmd.turnRadps * dt;
    x += a * (cmd.forwardMps * Math.cos(mid) - cmd.leftMps * Math.sin(mid)) * dt;
    y += a * (cmd.forwardMps * Math.sin(mid) + cmd.leftMps * Math.cos(mid)) * dt;
    psi += a * cmd.turnRadps * dt;
  }
  return [x, y, psi];
}

/** Share of each trained range a walk to a target may use. */
export const TARGET_SPEED_FRACTION = 0.8;

/**
 * Constant body-frame speeds that end at `target` (forward and left, m, and heading change, rad, in the start frame) after the
 * gait's duration, fades included. For a fixed turn rate the end position is linear in the two speeds, so they solve a 2 x 2 system
 * whose determinant is the squared length of the faded heading average (positive for turns under a full circle).
 */
export function commandForTarget(target: Readonly<Vec3>, gait: Pick<GaitParams, 'cycleS' | 'durationS'>): WalkCommand {
  const turnRadps = target[2] / expectedPose({ forwardMps: 0, leftMps: 0, turnRadps: 1 }, gait)[2];
  const a = expectedPose({ forwardMps: 1, leftMps: 0, turnRadps }, gait);
  const b = expectedPose({ forwardMps: 0, leftMps: 1, turnRadps }, gait);
  const det = a[0] * b[1] - b[0] * a[1];
  return { forwardMps: (target[0] * b[1] - b[0] * target[1]) / det, leftMps: (a[0] * target[1] - target[0] * a[1]) / det, turnRadps };
}

/** The shortest walk (in 0.1 s steps) that reaches `target` with speeds within TARGET_SPEED_FRACTION of the ranges, or why there is none. */
export function walkForTarget(target: Readonly<Vec3>, ranges: WalkRanges, cycleS: number): { command: WalkCommand; durationS: number } | { problem: string } {
  if (!target.every(Number.isFinite)) return { problem: 'the target must be finite.' };
  const keys = ['forwardMps', 'leftMps', 'turnRadps'] as const;
  for (let tenths = WALK_DURATION_S[0] * 10; tenths <= WALK_DURATION_S[1] * 10; tenths++) {
    const durationS = tenths / 10;
    const raw = commandForTarget(target, { cycleS, durationS });
    const command = { forwardMps: Number(raw.forwardMps.toFixed(4)), leftMps: Number(raw.leftMps.toFixed(4)), turnRadps: Number(raw.turnRadps.toFixed(4)) };
    if (keys.every((k) => command[k] >= TARGET_SPEED_FRACTION * ranges[k][0] && command[k] <= TARGET_SPEED_FRACTION * ranges[k][1])) return { command, durationS };
  }
  return { problem: `reaching it takes longer than ${WALK_DURATION_S[1]} s at the speeds this controller was trained for; walk part of the way and call locomote again from there.` };
}

export interface WalkJudgement {
  expected: Vec3;
  measured: Vec3;
  positionErrorM: number;
  headingErrorRad: number;
  toleranceM: number;
  toleranceRad: number;
  upright: boolean;
  success: boolean;
}

/** A walk succeeds when the body stays up and ends within a quarter of the commanded travel and turn (at least 5 cm and 0.15 rad) of where the command leads. */
export function judgeWalk(expected: Readonly<Vec3>, measured: Readonly<Vec3>, upright: boolean): WalkJudgement {
  const positionErrorM = Math.hypot(measured[0] - expected[0], measured[1] - expected[1]);
  const headingErrorRad = Math.abs(measured[2] - expected[2]);
  const toleranceM = Math.max(0.05, 0.25 * Math.hypot(expected[0], expected[1]));
  const toleranceRad = Math.max(0.15, 0.25 * Math.abs(expected[2]));
  return {
    expected: [expected[0], expected[1], expected[2]], measured: [measured[0], measured[1], measured[2]], positionErrorM, headingErrorRad, toleranceM, toleranceRad, upright,
    success: upright && positionErrorM <= toleranceM && headingErrorRad <= toleranceRad,
  };
}

/** The generator settings a body walks with: a trot (a tripod on six legs), 0.5 s cycles, IMU feedback. */
export function locomotionBase(desc: RobotDescription, durationS = 3): GaitParams {
  const g: GaitParams = { ...DEFAULT_GAIT, strideM: 0, lateralM: 0, turnRad: 0, durationS };
  const problems = gaitProblems(g, desc);
  if (problems.length) throw new RangeError(problems[0]!);
  return g;
}

export function rangeProblem(c: WalkCommand, r: WalkRanges, durationS: number): string | null {
  for (const k of ['forwardMps', 'leftMps', 'turnRadps'] as const) {
    const [lo, hi] = r[k];
    if (!(c[k] >= lo && c[k] <= hi)) return `${k} must be from ${lo} to ${hi} (the range this controller was trained for).`;
  }
  if (!(durationS >= WALK_DURATION_S[0] && durationS <= WALK_DURATION_S[1])) return `durationS must be from ${WALK_DURATION_S[0]} to ${WALK_DURATION_S[1]} s.`;
  return null;
}

/** Random command k of a seeded set, uniform within the ranges. */
export function sampleCommand(r: WalkRanges, seed: number, k: number): WalkCommand {
  const rng = mulberry32(seed * 7919 + k * 104729 + 13);
  const pick = ([lo, hi]: readonly [number, number]) => Number((lo + (hi - lo) * rng()).toFixed(3));
  return { forwardMps: pick(r.forwardMps), leftMps: pick(r.leftMps), turnRadps: pick(r.turnRadps) };
}

/** Evaluation command k: the six single-axis commands at three quarters of each range first, then random ones. */
export function evaluationCommand(r: WalkRanges, seed: number, k: number): WalkCommand {
  const q = (v: number) => Number((0.75 * v).toFixed(3));
  const single: WalkCommand[] = [
    { forwardMps: q(r.forwardMps[1]), leftMps: 0, turnRadps: 0 }, { forwardMps: q(r.forwardMps[0]), leftMps: 0, turnRadps: 0 },
    { forwardMps: 0, leftMps: q(r.leftMps[1]), turnRadps: 0 }, { forwardMps: 0, leftMps: q(r.leftMps[0]), turnRadps: 0 },
    { forwardMps: 0, leftMps: 0, turnRadps: q(r.turnRadps[1]) }, { forwardMps: 0, leftMps: 0, turnRadps: q(r.turnRadps[0]) },
  ];
  return single[k] ?? sampleCommand(r, seed, k);
}

/** Generator commands measured for the calibration, from the ranges: each axis both ways at two sizes, and pairs of axes. */
function calibrationCommands(r: WalkRanges): Vec3[] {
  const f = 0.8;
  const [f0, f1] = r.forwardMps, [l0, l1] = r.leftMps, [t0, t1] = r.turnRadps;
  return [
    [f * f1, 0, 0], [f * f0, 0, 0], [0, f * l1, 0], [0, f * l0, 0], [0, 0, f * t1], [0, 0, f * t0],
    [0.4 * f1, 0, 0], [0.4 * f0, 0, 0], [0, 0.4 * l1, 0], [0, 0.4 * l0, 0], [0, 0, 0.4 * t1], [0, 0, 0.4 * t0],
    [f * f1, f * l1, 0], [f * f0, f * l0, 0], [f * f1, 0, f * t1], [f * f0, 0, f * t0], [0, f * l1, f * t0], [0, f * l0, f * t1],
  ];
}

export interface LocomotionConfig extends ConditionRanges {
  ranges: WalkRanges;
  /** Gait length of each training, comparison and evaluation run, s. */
  episodeS: number;
  iterations: number;
  directions: number;
  top: number;
  stepSize: number;
  noise: number;
  /** Random commands every candidate policy is scored on per iteration. */
  commandsPerCandidate: number;
  heldOutTrials: number;
  evaluationTrials: number;
  seed: number;
  /** The policy also corrects every joint target (otherwise it changes the generator's command only). */
  jointCorrections: boolean;
}

export const DEFAULT_LOCOMOTION: LocomotionConfig = {
  ranges: DEFAULT_WALK_RANGES, episodeS: 3, iterations: 80, directions: 8, top: 4, stepSize: 0.02, noise: 0.03, commandsPerCandidate: 2, heldOutTrials: 12, evaluationTrials: 40, seed: 23, jointCorrections: false,
  frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02,
};

export interface LocomotionScore {
  meanReturn: number;
  trackingErrorMps: number;
  yawRateErrorRadps: number;
  falls: number;
  /** Where the runs ended against where their commands lead. */
  meanPositionErrorM: number;
  meanHeadingErrorRad: number;
}

export interface WalkScore {
  trials: number;
  successes: number;
  falls: number;
  meanPositionErrorM: number;
  meanHeadingErrorRad: number;
}

export interface LocomotionEvaluation {
  seed: number;
  trials: number;
  durationS: number;
  /** The controller as saved: calibrated generator, with the policy when it was kept. */
  final: WalkScore & { wilson95: [number, number] };
  /** The generator with the learned calibration and no policy. */
  calibrated: WalkScore;
  /** The generator given the command as it is, nothing learned. */
  uncalibrated: WalkScore;
  /** How far the controller's sensor-only estimate (leg odometry and IMU heading) ended from ground truth, over the saved controller's runs. */
  odometry?: { meanPositionErrorM: number; meanHeadingErrorRad: number };
}

export interface LocomotionResult {
  base: GaitParams;
  ranges: WalkRanges;
  calibration: number[];
  calibrationRuns: { nominal: Vec3; measured: Vec3 }[];
  /** The feedback policy, or null when it did not beat the calibrated generator on held-out runs. */
  policy: PolicyJson | null;
  heldOut: { trials: number; generator: LocomotionScore; learned: LocomotionScore };
  evaluation: LocomotionEvaluation;
  history: ArsProgress[];
  rollouts: number;
  workers: number;
  parameters: number;
}

const midpoint = ([lo, hi]: readonly [number, number]) => (lo + hi) / 2;

/** A linear walking policy (no hidden layer) from a parameter vector and input normalization: the command changes, after one correction per joint when `jointCorrections`. */
export function walkPolicy(theta: ArrayLike<number>, joints: number, mean: ArrayLike<number>, std: ArrayLike<number>, jointCorrections = false): PolicyJson {
  const outputs = (jointCorrections ? joints : 0) + WALK_COMMAND_OUTPUTS;
  if (theta.length !== outputs * (WALK_POLICY_INPUTS + 1)) throw new RangeError(`Expected ${outputs * (WALK_POLICY_INPUTS + 1)} parameters, got ${theta.length}`);
  return { layers: [WALK_POLICY_INPUTS, outputs], activation: 'tanh', weights: Array.from(theta), inputMean: Array.from(mean), inputStd: Array.from(std) };
}

function scoreOf(outs: readonly RolloutOutcome[], expected: readonly Vec3[]): LocomotionScore {
  const avg = (f: (o: RolloutOutcome, i: number) => number) => outs.reduce((s, o, i) => s + f(o, i), 0) / Math.max(1, outs.length);
  const judged = outs.map((o, i) => judgeWalk(expected[i]!, [o.rollout.forwardM, o.rollout.leftM, o.rollout.turnRad], true));
  return {
    meanReturn: avg((o) => o.rollout.ret),
    trackingErrorMps: avg((o) => o.rollout.trackingErrorMps),
    yawRateErrorRadps: avg((o) => o.rollout.yawRateErrorRadps),
    falls: outs.filter((o) => o.rollout.fell || o.rollout.unstable).length,
    meanPositionErrorM: avg((_, i) => judged[i]!.positionErrorM),
    meanHeadingErrorRad: avg((_, i) => judged[i]!.headingErrorRad),
  };
}

/**
 * Learns a walking controller for a body: measures the gait generator under
 * a set of commands and fits the calibration, trains the feedback policy by
 * augmented random search with every candidate scored on random commands and
 * conditions, keeps the policy only if it beats the calibrated generator on
 * held-out runs, and evaluates the result against the generator with and
 * without the calibration. All runs go through `runner` (a worker pool in the
 * app); the result does not depend on the number of workers.
 */
export async function trainLocomotion(
  runner: RolloutRunner,
  desc: RobotDescription,
  stanceQ: number[],
  cfg: LocomotionConfig,
  onProgress?: (fraction: number, detail: string) => void,
): Promise<LocomotionResult> {
  const base = locomotionBase(desc, cfg.episodeS);
  const setup: RolloutSetup = { desc, stanceQ, ...(cfg.gains ? { gains: cfg.gains } : {}) };
  const joints = stanceQ.length;
  const nParams = ((cfg.jointCorrections ? joints : 0) + WALK_COMMAND_OUTPUTS) * (WALK_POLICY_INPUTS + 1);
  let rollouts = 0;
  const run = (jobs: RolloutJob[]) => {
    rollouts += jobs.length;
    return runner.run(setup, jobs);
  };
  const gaitFor = (cmd: WalkCommand, cal: readonly number[] | null) => walkGait(base, cal, cmd, cfg.episodeS).gait;

  onProgress?.(0, 'Measuring how the gait generator moves this body');
  const calibrationNominal = calibrationCommands(cfg.ranges);
  const nominal: TrialConditions = { friction: midpoint(cfg.frictionRange), baseMassKg: midpoint(cfg.baseMassRange), jointNoiseRad: 0 };
  const calibrationGaits = calibrationNominal.map((n) => gaitFor(cmdOf(n), null));
  const measured = await run(calibrationGaits.map((gait, i) => ({ gait, policy: null, conditions: nominal, noiseSeed: cfg.seed * 1000 + 500 + i, walkCommand: calibrationNominal[i]!, collectStats: true })));
  const stats = new RunningStats(WALK_POLICY_INPUTS);
  for (const o of measured) if (o.stats) stats.merge(RunningStats.fromParts(o.stats));
  const calibrationRuns = calibrationGaits.map((g, i) => ({ nominal: gaitCommand(g), measured: measured[i]!.rollout.meanVelocity }));
  const calibration = fitCalibration(calibrationRuns);

  let mean = Float64Array.from(stats.mean);
  let std = stats.std();
  let fresh = new RunningStats(WALK_POLICY_INPUTS);
  const K = cfg.commandsPerCandidate;
  const { theta, history } = await augmentedRandomSearchBatched(
    new Float64Array(nParams),
    async (candidates, it) => {
      const seed = cfg.seed * 1000 + it;
      const tasks = Array.from({ length: K }, (_, k) => {
        const cmd = sampleCommand(cfg.ranges, seed, k);
        return { cmd, gait: gaitFor(cmd, calibration), conditions: trialConditions({ ...cfg, seed }, k), noiseSeed: seed * 104729 + k };
      });
      const jobs = candidates.flatMap((th) => {
        const policy = walkPolicy(th, joints, mean, std, cfg.jointCorrections);
        return tasks.map((t): RolloutJob => ({ gait: t.gait, policy, conditions: t.conditions, noiseSeed: t.noiseSeed, target: vecOf(t.cmd), walkCommand: vecOf(t.cmd), training: true, trackPose: true, collectStats: true }));
      });
      const out = await run(jobs);
      for (const o of out) if (o.stats) fresh.merge(RunningStats.fromParts(o.stats));
      return candidates.map((_, c) => out.slice(c * K, (c + 1) * K).reduce((s, o) => s + o.rollout.ret, 0) / K);
    },
    { directions: cfg.directions, top: cfg.top, stepSize: cfg.stepSize, noise: cfg.noise, iterations: cfg.iterations, seed: cfg.seed },
    (p) => {
      stats.merge(fresh);
      fresh = new RunningStats(WALK_POLICY_INPUTS);
      mean = Float64Array.from(stats.mean);
      std = stats.std();
      onProgress?.(0.05 + (0.8 * (p.iteration + 1)) / cfg.iterations, `Iteration ${p.iteration + 1} of ${cfg.iterations}: mean return ${p.meanReturn.toFixed(3)} over random commands (${rollouts} runs)`);
    },
  );
  const policy = walkPolicy(theta, joints, mean, std, cfg.jointCorrections);

  onProgress?.(0.87, 'Comparing the policy with the calibrated generator on held-out commands');
  const hs = cfg.seed * 1000 + 999;
  const held = Array.from({ length: cfg.heldOutTrials }, (_, i) => {
    const cmd = sampleCommand(cfg.ranges, hs, i);
    return { cmd, gait: gaitFor(cmd, calibration), conditions: trialConditions({ ...cfg, seed: hs }, i), noiseSeed: hs * 104729 + i };
  });
  const heldJobs = (p: PolicyJson | null) => held.map((t): RolloutJob => ({ gait: t.gait, policy: p, conditions: t.conditions, noiseSeed: t.noiseSeed, target: vecOf(t.cmd), walkCommand: vecOf(t.cmd), trackPose: true }));
  const heldOut = await run([...heldJobs(null), ...heldJobs(policy)]);
  const heldExpected = held.map((t) => expectedPose(t.cmd, base));
  const generator = scoreOf(heldOut.slice(0, held.length), heldExpected);
  const learned = scoreOf(heldOut.slice(held.length), heldExpected);
  // A better return must not come from falling more often or from ending further from where the commands lead.
  const kept =
    learned.meanReturn > generator.meanReturn + 1e-9 && learned.falls <= generator.falls &&
    learned.meanPositionErrorM <= generator.meanPositionErrorM && learned.meanHeadingErrorRad <= generator.meanHeadingErrorRad;

  onProgress?.(0.92, `Evaluating over ${cfg.evaluationTrials} randomized commands`);
  const es = cfg.seed * 1000 + 2026;
  const trials = Array.from({ length: cfg.evaluationTrials }, (_, i) => ({ cmd: evaluationCommand(cfg.ranges, es, i), conditions: trialConditions({ ...cfg, seed: es }, i), noiseSeed: es * 104729 + i }));
  const variants: { cal: readonly number[] | null; policy: PolicyJson | null }[] = [{ cal: null, policy: null }, { cal: calibration, policy: null }, ...(kept ? [{ cal: calibration, policy }] : [])];
  const evalOut = await run(variants.flatMap((v) => trials.map((t): RolloutJob => ({ gait: gaitFor(t.cmd, v.cal), policy: v.policy, conditions: t.conditions, noiseSeed: t.noiseSeed, target: vecOf(t.cmd), walkCommand: vecOf(t.cmd) }))));
  const walkScore = (outs: readonly RolloutOutcome[]): WalkScore => {
    const judged = outs.map((o, i) => {
      const r = o.rollout;
      return judgeWalk(expectedPose(trials[i]!.cmd, base), [r.forwardM, r.leftM, r.turnRad], !r.fell && !r.unstable && r.maxTiltRad <= WALK_MAX_TILT_RAD);
    });
    const avg = (f: (j: WalkJudgement) => number) => judged.reduce((s, j) => s + f(j), 0) / Math.max(1, judged.length);
    return {
      trials: judged.length, successes: judged.filter((j) => j.success).length, falls: judged.filter((j) => !j.upright).length,
      meanPositionErrorM: avg((j) => j.positionErrorM), meanHeadingErrorRad: avg((j) => j.headingErrorRad),
    };
  };
  const n = trials.length;
  const uncalibrated = walkScore(evalOut.slice(0, n));
  const calibrated = walkScore(evalOut.slice(n, 2 * n));
  const finalRuns = kept ? evalOut.slice(2 * n, 3 * n) : evalOut.slice(n, 2 * n);
  const finalScore = kept ? walkScore(finalRuns) : calibrated;
  const odo = finalRuns.filter((o) => o.rollout.odometry !== undefined);
  const odometry = odo.length
    ? {
        meanPositionErrorM: odo.reduce((s, o) => s + Math.hypot(o.rollout.odometry![0] - o.rollout.forwardM, o.rollout.odometry![1] - o.rollout.leftM), 0) / odo.length,
        meanHeadingErrorRad: odo.reduce((s, o) => s + Math.abs(o.rollout.odometry![2] - o.rollout.turnRad), 0) / odo.length,
      }
    : undefined;
  onProgress?.(1, 'Done');
  return {
    base, ranges: cfg.ranges, calibration, calibrationRuns, policy: kept ? policy : null,
    heldOut: { trials: held.length, generator, learned },
    evaluation: { seed: es, trials: n, durationS: cfg.episodeS, final: { ...finalScore, wilson95: wilson95(finalScore.successes, n) }, calibrated, uncalibrated, ...(odometry ? { odometry } : {}) },
    history, rollouts, workers: runner.workers, parameters: nParams,
  };
}

/** The walking controller as stored, one per morphology. Schema 1 (D28) had a gait policy; it is accepted only without one. */
export interface LocomotionController {
  kind: 'morphlab.locomotion';
  schemaVersion: 1 | 2;
  robot: { robotId: string; morphologyHash: string };
  createdAt: string;
  base: GaitParams;
  ranges: WalkRanges;
  /** Generator command to measured velocity: 3 rows (forward, left, yaw rate) of 6 (positive and negative part of each axis). */
  calibration: number[];
  policy: PolicyJson | null;
  training: {
    algorithm: string;
    rollouts: number;
    workers: number;
    iterations: number;
    parameters: number;
    calibrationRuns: number;
    wallTimeS: number;
    seed: number;
    heldOut: LocomotionResult['heldOut'];
  };
  evaluation: LocomotionEvaluation & { conditions: { frictionRange: [number, number]; baseMassRange: [number, number]; jointNoiseRad: number; gains?: PdGains } };
}

export function locomotionController(r: LocomotionResult, robot: { robotId: string; morphologyHash: string }, cfg: LocomotionConfig, wallTimeS: number, now = new Date().toISOString()): LocomotionController {
  return {
    kind: 'morphlab.locomotion', schemaVersion: 2, robot: { ...robot }, createdAt: now, base: { ...r.base }, ranges: structuredClone(r.ranges), calibration: r.calibration.slice(), policy: r.policy,
    training: {
      algorithm: 'calibration by least squares on measured runs; feedback policy (sensor-only observation v2 with leg odometry) by augmented random search over random commands',
      rollouts: r.rollouts, workers: r.workers, iterations: r.history.length, parameters: r.parameters, calibrationRuns: r.calibrationRuns.length, wallTimeS: Number(wallTimeS.toFixed(1)), seed: cfg.seed, heldOut: r.heldOut,
    },
    evaluation: {
      ...r.evaluation,
      conditions: {
        frictionRange: [...cfg.frictionRange],
        baseMassRange: [...cfg.baseMassRange],
        jointNoiseRad: cfg.jointNoiseRad,
        ...(cfg.gains ? { gains: { kp: cfg.gains.kp, kd: cfg.gains.kd } } : {}),
      },
    },
  };
}

/** Why a stored controller cannot walk this robot, or null. */
export function locomotionProblem(c: LocomotionController, robot: RobotDescription, morphologyHash: string): string | null {
  if (c.robot.morphologyHash !== morphologyHash) return `the walking controller was trained on morphology ${c.robot.morphologyHash.slice(0, 12)}, not this body (${morphologyHash.slice(0, 12)}); train one for this body.`;
  const g = gaitProblems(c.base, robot);
  if (g.length) return `its gait does not fit this robot: ${g[0]}`;
  if (c.policy) return walkPolicyProblem(c.policy, robot.limbs.reduce((n, l) => n + l.joints.length, 0));
  return null;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const finiteList = (v: unknown, n?: number): v is number[] => Array.isArray(v) && (n === undefined || v.length === n) && v.every(finite);

function scoreProblems(s: unknown, where: string, trials: number): string[] {
  if (!isObject(s) || !Number.isInteger(s.successes) || (s.successes as number) < 0 || (s.successes as number) > trials || !finite(s.meanPositionErrorM) || !finite(s.meanHeadingErrorRad)) {
    return [`${where} must give successes (0 to ${trials}) and finite mean errors.`];
  }
  return [];
}

/** Validation of a stored walking controller (storage is untrusted): structure, sizes and finite numbers. Nothing is repaired. */
export function parseLocomotionController(raw: unknown): ParseResult<LocomotionController> {
  const errors: string[] = [];
  const o = isObject(raw) ? raw : {};
  if (o.kind !== 'morphlab.locomotion' || (o.schemaVersion !== 1 && o.schemaVersion !== 2)) errors.push('locomotion: kind must be "morphlab.locomotion" with schemaVersion 1 or 2.');
  const r = o.robot;
  if (!isObject(r) || typeof r.robotId !== 'string' || typeof r.morphologyHash !== 'string' || !/^[0-9a-f]{64}$/.test(r.morphologyHash)) errors.push('locomotion.robot must contain robotId and a 64-character hex morphologyHash.');
  if (typeof o.createdAt !== 'string') errors.push('locomotion.createdAt must be a string.');
  errors.push(...gaitProblems(o.base, undefined, 'locomotion.base'));
  const ranges = o.ranges;
  if (!isObject(ranges)) errors.push('locomotion.ranges must be an object.');
  else {
    for (const k of ['forwardMps', 'leftMps', 'turnRadps'] as const) {
      const v = ranges[k];
      if (!finiteList(v, 2) || !(v[0]! < v[1]!) || Math.abs(v[0]!) > RANGE_LIMITS[k] || Math.abs(v[1]!) > RANGE_LIMITS[k]) errors.push(`locomotion.ranges.${k} must be [low, high] with low < high within ${RANGE_LIMITS[k]}.`);
    }
  }
  if (!finiteList(o.calibration, 18)) errors.push('locomotion.calibration must be 18 finite numbers.');
  const p = o.policy;
  if (p !== null && o.schemaVersion === 1) errors.push('locomotion.policy of a schema 1 controller was trained before the walking policy of D29; train the walking controller again.');
  else if (p !== null) {
    const layers = isObject(p) && finiteList(p.layers, 2) && p.layers.every((n) => Number.isInteger(n) && n >= 1 && n <= 128) ? p.layers : null;
    if (!isObject(p) || !layers || layers[0] !== WALK_POLICY_INPUTS || p.activation !== 'tanh' || !finiteList(p.weights, layers[1]! * (layers[0]! + 1)) || !finiteList(p.inputMean, WALK_POLICY_INPUTS) || !finiteList(p.inputStd, WALK_POLICY_INPUTS) || !p.inputStd.every((s) => s > 0)) {
      errors.push(`locomotion.policy must be null or a linear policy from ${WALK_POLICY_INPUTS} inputs with finite weights and positive input scales.`);
    }
  }
  const t = o.training;
  if (!isObject(t) || typeof t.algorithm !== 'string' || ![t.rollouts, t.workers, t.iterations, t.parameters, t.calibrationRuns, t.wallTimeS, t.seed].every(finite)) errors.push('locomotion.training must name the algorithm and give finite counts.');
  const e = o.evaluation;
  if (!isObject(e) || !Number.isInteger(e.trials) || (e.trials as number) < 1 || !finite(e.durationS) || !finite(e.seed)) errors.push('locomotion.evaluation must give the trials, duration and seed.');
  else {
    const n = e.trials as number;
    errors.push(...scoreProblems(e.final, 'locomotion.evaluation.final', n), ...scoreProblems(e.calibrated, 'locomotion.evaluation.calibrated', n), ...scoreProblems(e.uncalibrated, 'locomotion.evaluation.uncalibrated', n));
    const odo = e.odometry;
    if (odo !== undefined && (!isObject(odo) || !finite(odo.meanPositionErrorM) || !finite(odo.meanHeadingErrorRad))) errors.push('locomotion.evaluation.odometry must give finite mean errors.');
    if (isObject(e.final) && Number.isInteger(e.final.successes)) {
      const w = wilson95(e.final.successes as number, n);
      if (!finiteList(e.final.wilson95, 2) || Math.abs(e.final.wilson95[0]! - w[0]) > 1e-6 || Math.abs(e.final.wilson95[1]! - w[1]) > 1e-6) errors.push('locomotion.evaluation.final.wilson95 must be the Wilson 95 % interval of its successes.');
    }
    const cond = e.conditions;
    if (isObject(cond) && cond.gains !== undefined) {
      const g = cond.gains;
      if (!isObject(g) || !finite(g.kp) || !finite(g.kd) || g.kp < 0 || g.kd < 0) errors.push('locomotion.evaluation.conditions.gains must give non-negative finite kp and kd.');
    }
  }
  if (errors.length) return { ok: false, errors };
  const keys = ['kind', 'schemaVersion', 'robot', 'createdAt', 'base', 'ranges', 'calibration', 'policy', 'training', 'evaluation'] as const;
  const copy: Record<string, unknown> = {};
  for (const k of keys) copy[k] = o[k];
  return { ok: true, value: structuredClone(copy) as unknown as LocomotionController };
}
