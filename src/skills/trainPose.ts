import type { MainModule } from '@mujoco/mujoco';
import { TargetRateLimiter } from '../core/control/limits';
import { pdControl } from '../core/control/pd';
import { buildKinematicModel } from '../core/kinematics/fk';
import { gaussian, mulberry32 } from '../core/math/rng';
import type { Vec3 } from '../core/math/vec3';
import type { RobotDescription } from '../core/model/types';
import { augmentedRandomSearchBatched, RunningStats, type ArsProgress, type RunningStatsParts } from '../learning/ars';
import { Mlp } from '../learning/mlp';
import { MujocoRuntime } from '../physics/runtime';
import type { PolicyJson, SkillController } from './controllers';
import { CONTROL_DECIMATION, DEFAULT_SKILL_GAINS, judgeSample, trialConditions, type TrialConditions } from './evaluation';
import type { ConditionRanges } from './learning';
import { LearnedPoseController, policyFeet, posePolicyInputs, posePolicyObservation, posePolicyWithoutFeedback } from './posePolicy';
import type { RolloutSetup } from './rollouts';
import { goalReward, GoalJudge, KeyframeController, motionDuration, restingHeight, type KeyframeMotion, type SkillGoal } from './taught';

/**
 * Reinforcement learning of a feedback policy for a keyframe skill (D30):
 * Augmented Random Search on a linear policy that corrects the keyframes'
 * joint targets from the IMU, the foot contacts and the joint encoders. Every training run is
 * randomized (friction, body mass, joint offsets) and the body is pushed at
 * a random moment, so only a policy that reacts to how the body moves earns
 * a higher return. The return comes from simulator ground truth, which the
 * policy never sees: the share of the skill's goals that hold at each moment,
 * the goals' verdict at the end, and penalties for spinning, jerky
 * corrections and falling. The policy is kept only if, on held-out pushed
 * runs, it meets every goal more often than the keyframes alone.
 */

/** A horizontal push on the body: start time (s), world-frame force (N) and duration (s). */
export interface Push {
  tS: number;
  forceN: Vec3;
  durationS: number;
}

/** One randomized run of a keyframe skill, as sent to a worker (plain data). */
export interface PoseRolloutJob {
  motion: KeyframeMotion;
  /** Null runs the keyframes alone (open loop). */
  policy: PolicyJson | null;
  conditions: TrialConditions;
  noiseSeed: number;
  goals: SkillGoal[];
  push?: Push;
  /** A training run ends at a fall and judges the goals at the control rate. */
  training?: boolean;
  /** Return the policy's input statistics of this run. */
  collectStats?: boolean;
}

export interface PoseRollout {
  ret: number;
  /** Every goal of the skill was met. */
  success: boolean;
  goalReward: number;
  fell: boolean;
  unstable: boolean;
  maxTiltRad: number;
  /** Mean share of the goals that held at each moment. */
  instantMean: number;
}

export interface PoseOutcome {
  rollout: PoseRollout;
  stats: RunningStatsParts | null;
}

/** Return weights: goals holding now (per second), standing as an end goal asks once the motion is over (per second), the verdict at the end, spin (per (rad/s)^2 s), correction rate. */
const W_INSTANT = 1;
const W_END = 4;
const W_GOAL = 2;
const W_SPIN = 0.02;
const W_ACTION_RATE = 0.01;
const FALL_PENALTY = 2;
/** The body counts as fallen past this tilt (on its side or back) or below this share of its resting height. */
const FALL_TILT_RAD = 2;
const FALL_HEIGHT_SHARE = 0.25;
/** Pushes last this long, s. */
const PUSH_S = 0.1;

/** One randomized run of keyframes, with the learned policy when one is given, scored by the goals and the dense return. */
export function posePolicyRollout(mj: MainModule, setup: RolloutSetup, job: PoseRolloutJob): PoseOutcome {
  const { desc, stanceQ } = setup;
  const gains = setup.gains ?? DEFAULT_SKILL_GAINS;
  const rng = mulberry32(job.noiseSeed);
  const q0 = stanceQ.map((v) => v + job.conditions.jointNoiseRad * Math.max(-1, Math.min(1, gaussian(rng) / 2)));
  const km = buildKinematicModel(desc);
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q0, friction: job.conditions.friction, baseMass: job.conditions.baseMassKg });
  try {
    const learned = job.policy ? new LearnedPoseController(km, stanceQ, job.motion, job.policy) : null;
    const ctl: SkillController = learned ?? new KeyframeController(km, stanceQ, job.motion);
    const durationS = motionDuration(job.motion);
    const feet = policyFeet(desc);
    const obsBuf = new Float64Array(posePolicyInputs(desc));
    const stats = job.collectStats ? new RunningStats(obsBuf.length) : null;
    const judge = GoalJudge.forRobot(job.goals, desc);
    const endGoals = job.goals.filter((x): x is Extract<SkillGoal, { kind: 'standsAtEnd' }> => x.kind === 'standsAtEnd');
    const g = stanceQ.map(() => gains);
    const joints = desc.limbs.flatMap((l) => l.joints);
    const continuous = joints.map((j) => j.type === 'continuous');
    const limiter = new TargetRateLimiter(joints.map((j) => j.limit.velocity), continuous, q0);
    const dtc = CONTROL_DECIMATION * rt.timestep;
    const rest = restingHeight(desc);
    const prevAction = new Float64Array(stanceQ.length);
    const pushStart = job.push ? Math.round(job.push.tS / rt.timestep) : -1;
    const pushEnd = job.push ? pushStart + Math.max(1, Math.round(job.push.durationS / rt.timestep)) : -1;
    let qTarget = stanceQ.slice();
    let lastTouch: number[] = [];
    let gyro: Vec3 = [0, 0, 0];
    let ret = 0, instSum = 0, instN = 0;
    let fell = false, unstable = false, doneAt = -1;
    const maxSteps = Math.ceil((durationS + 1) / rt.timestep);
    for (let k = 0; k < maxSteps; k++) {
      const t = k * rt.timestep;
      if (k === pushStart && job.push) rt.setBodyWrench(rt.layout.baseBodyId, job.push.forceN);
      if (k === pushEnd) rt.clearWrenches();
      rt.controlStep((s) => {
        if (k % CONTROL_DECIMATION === 0) {
          const obs = { q: s.q, qd: s.qd, touch: s.touch, imuQuat: s.imuQuat, gyro: s.gyro };
          qTarget = ctl.update(t, obs);
          // Without a policy the targets are the sketch itself.
          if (stats && !ctl.finished) stats.push(learned ? learned.observation : posePolicyObservation(t, obs, durationS, feet, qTarget, obsBuf));
          gyro = s.gyro;
        }
        limiter.step(qTarget, rt.timestep);
        lastTouch = s.touch;
        return pdControl(g, rt.layout.effort, { qCmd: limiter.qCmd, qMeas: s.q, qdMeas: s.qd, continuous }).tauApplied;
      });
      if (rt.hasNonFiniteState()) {
        unstable = true;
        break;
      }
      const controlEnd = k % CONTROL_DECIMATION === CONTROL_DECIMATION - 1;
      // Training judges at the control rate (cheaper); other runs every step, like an evaluation trial.
      if (!job.training || controlEnd) judge.add(judgeSample(rt, t, lastTouch));
      if (controlEnd) {
        let rate = 0;
        if (learned) {
          for (let i = 0; i < prevAction.length; i++) rate += (learned.action[i]! - prevAction[i]!) ** 2;
          prevAction.set(learned.action);
        }
        const spin = gyro[0] ** 2 + gyro[1] ** 2 + gyro[2] ** 2;
        const b = rt.groundTruthBaseState();
        // The judge sees an end goal only in the verdict; once the motion is over, standing counts at every step.
        const standing = ctl.finished && endGoals.length ? endGoals.filter((e) => b.tilt <= e.maxTiltRad && b.pos[2] >= e.minBaseHeightM).length / endGoals.length : 0;
        ret += dtc * (W_INSTANT * judge.instant + W_END * standing - W_SPIN * spin - W_ACTION_RATE * rate);
        instSum += judge.instant;
        instN++;
        if (!fell && (b.tilt > FALL_TILT_RAD || (rest !== null && b.pos[2] < FALL_HEIGHT_SHARE * rest))) {
          fell = true;
          ret -= FALL_PENALTY;
          if (job.training) break;
        }
      }
      if (ctl.finished && doneAt < 0) doneAt = t;
      if (doneAt >= 0 && t - doneAt > 0.5) break;
    }
    rt.clearWrenches();
    if (unstable) ret -= FALL_PENALTY;
    const v = judge.verdict({ unstable });
    const goalR = goalReward(v, unstable);
    return {
      rollout: { ret: ret + W_GOAL * goalR, success: v.success, goalReward: goalR, fell, unstable, maxTiltRad: v.maxTiltRad, instantMean: instN ? instSum / instN : 0 },
      stats: stats ? stats.toParts() : null,
    };
  } finally {
    rt.dispose();
  }
}

/** Runs batches of keyframe rollouts, in this thread or across workers; the outcomes keep the order of the jobs. */
export interface PoseRunner {
  readonly workers: number;
  runPose(setup: RolloutSetup, jobs: PoseRolloutJob[]): Promise<PoseOutcome[]>;
}

export interface PosePolicyConfig extends ConditionRanges {
  architecture?: 'linear' | 'mlp';
  incumbentPolicy?: PolicyJson;
  iterations: number;
  /** Directions per iteration (two candidates each). */
  directions: number;
  top: number;
  stepSize: number;
  noise: number;
  /** Randomized, pushed runs every candidate is scored on per iteration. */
  trialsPerCandidate: number;
  heldOutTrials: number;
  seed: number;
  /** Size of the push in every training and held-out run, N*s, applied sideways or lengthwise over 0.1 s. */
  pushImpulseNs: [number, number];
  /** When the push comes, as fractions of the motion's duration. */
  pushWindow: [number, number];
}

export const DEFAULT_POSE_POLICY: PosePolicyConfig = {
  iterations: 100, directions: 12, top: 6, stepSize: 0.02, noise: 0.03, trialsPerCandidate: 2, heldOutTrials: 12, seed: 37,
  pushImpulseNs: [0.2, 0.8], pushWindow: [0.2, 0.9],
  frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02,
};

export interface PoseScore {
  trials: number;
  /** Runs meeting every goal. */
  successes: number;
  falls: number;
  meanReturn: number;
  meanGoalReward: number;
  /** Mean share of the goals that held at each moment. */
  meanInstant: number;
}

export interface PosePolicyResult {
  policy: PolicyJson;
  incumbent?: PoseScore;
  feedbackAblation?: PoseScore;
  /** On the held-out pushed runs: no more falls and no fewer runs meeting every goal than the keyframes alone, and more such runs or a goal reward higher by more than 0.05. */
  improved: boolean;
  /** The keyframes alone on the held-out pushed runs. */
  baseline: PoseScore;
  /** The keyframes with the policy on the same runs. */
  result: PoseScore;
  heldOutTrials: number;
  pushImpulseNs: [number, number];
  history: ArsProgress[];
  rollouts: number;
  parameters: number;
  workers: number;
}

/** A linear pose policy (no hidden layer) from a parameter vector and input normalization. */
export function linearPosePolicy(theta: ArrayLike<number>, robot: RobotDescription, mean: ArrayLike<number>, std: ArrayLike<number>): PolicyJson {
  const inputs = posePolicyInputs(robot);
  const joints = robot.limbs.reduce((n, l) => n + l.joints.length, 0);
  if (theta.length !== joints * (inputs + 1)) throw new RangeError(`Expected ${joints * (inputs + 1)} parameters, got ${theta.length}`);
  return { layers: [inputs, joints], activation: 'tanh', weights: Array.from(theta), inputMean: Array.from(mean), inputStd: Array.from(std) };
}

/** A push of random size and horizontal direction at a random moment in the window, for run k of a seeded set. */
export function randomPush(cfg: Pick<PosePolicyConfig, 'pushImpulseNs' | 'pushWindow'>, durationS: number, seed: number, k: number): Push {
  const rng = mulberry32(seed * 6271 + k * 3571 + 5);
  const impulse = cfg.pushImpulseNs[0] + (cfg.pushImpulseNs[1] - cfg.pushImpulseNs[0]) * rng();
  const angle = 2 * Math.PI * rng();
  const tS = durationS * (cfg.pushWindow[0] + (cfg.pushWindow[1] - cfg.pushWindow[0]) * rng());
  const f = impulse / PUSH_S;
  return { tS: +tS.toFixed(3), forceN: [+(f * Math.cos(angle)).toFixed(3), +(f * Math.sin(angle)).toFixed(3), 0], durationS: PUSH_S };
}

/** The randomized conditions, joint-offset seed and push of run k of a seeded set. */
export function poseTask(cfg: PosePolicyConfig, durationS: number, seed: number, k: number): Pick<PoseRolloutJob, 'conditions' | 'noiseSeed' | 'push'> {
  return { conditions: trialConditions({ ...cfg, seed }, k), noiseSeed: seed * 104729 + k, push: randomPush(cfg, durationS, seed, k) };
}

/**
 * The policy's input normalization: running statistics, except the foot
 * contact flags, which map to -1 and 1. A flag that rarely changes in
 * training would otherwise get a tiny spread and blow up when it does.
 */
function normalization(stats: RunningStats, feet: number): { mean: Float64Array; std: Float64Array } {
  const mean = Float64Array.from(stats.mean);
  const std = stats.std();
  for (let k = 5; k < 5 + feet; k++) {
    mean[k] = 0.5;
    std[k] = 0.5;
  }
  return { mean, std };
}

function scoreOf(outs: readonly PoseOutcome[]): PoseScore {
  const n = Math.max(1, outs.length);
  const sum = (f: (r: PoseRollout) => number) => outs.reduce((s, o) => s + f(o.rollout), 0);
  return {
    trials: outs.length,
    successes: outs.filter((o) => o.rollout.success).length,
    falls: outs.filter((o) => o.rollout.fell || o.rollout.unstable).length,
    meanReturn: sum((r) => r.ret) / n,
    meanGoalReward: sum((r) => r.goalReward) / n,
    meanInstant: sum((r) => r.instantMean) / n,
  };
}

/** Trains a feedback policy for keyframes by augmented random search in randomized, pushed runs, and compares it with the keyframes alone on held-out pushed runs. */
export async function trainPosePolicy(
  runner: PoseRunner,
  desc: RobotDescription,
  stanceQ: number[],
  motion: KeyframeMotion,
  goals: SkillGoal[],
  cfg: PosePolicyConfig,
  onProgress?: (fraction: number, detail: string) => void,
): Promise<PosePolicyResult> {
  if (cfg.architecture !== undefined && cfg.architecture !== 'linear' && cfg.architecture !== 'mlp') throw new RangeError('Unknown pose policy architecture');
  const setup: RolloutSetup = { desc, stanceQ, ...(cfg.gains ? { gains: cfg.gains } : {}) };
  const inputs = posePolicyInputs(desc);
  const feet = policyFeet(desc).length;
  const layers = cfg.architecture === 'mlp' ? [inputs, 16, stanceQ.length] : [inputs, stanceQ.length];
  const initial = cfg.architecture === 'mlp' ? Mlp.initialized(layers, mulberry32(cfg.seed), 0).params : new Float64Array(Mlp.paramCount(layers));
  const nParams = initial.length;
  const durationS = motionDuration(motion);
  let rollouts = 0;
  const run = (jobs: PoseRolloutJob[]) => {
    rollouts += jobs.length;
    return runner.runPose(setup, jobs);
  };
  const task = (seed: number, k: number) => poseTask(cfg, durationS, seed, k);

  onProgress?.(0, 'Measuring the keyframes alone');
  const ss = cfg.seed * 1000 + 997;
  const first = await run(Array.from({ length: 4 }, (_, k) => ({ motion, policy: null, goals, ...task(ss, k), collectStats: true })));
  const stats = new RunningStats(inputs);
  for (const o of first) if (o.stats) stats.merge(RunningStats.fromParts(o.stats));
  let { mean, std } = normalization(stats, feet);
  let fresh = new RunningStats(inputs);
  const policyFor = (weights: ArrayLike<number>): PolicyJson => ({ layers: layers.slice(), activation: 'tanh', weights: Array.from(weights), inputMean: Array.from(mean), inputStd: Array.from(std) });
  const K = cfg.trialsPerCandidate;
  const { theta, history } = await augmentedRandomSearchBatched(
    initial,
    async (candidates, it) => {
      const seed = cfg.seed * 1000 + it;
      const tasks = Array.from({ length: K }, (_, k) => task(seed, k));
      const jobs = candidates.flatMap((th) => {
        const policy = policyFor(th);
        return tasks.map((t): PoseRolloutJob => ({ motion, policy, goals, ...t, training: true, collectStats: true }));
      });
      const out = await run(jobs);
      for (const o of out) if (o.stats) fresh.merge(RunningStats.fromParts(o.stats));
      return candidates.map((_, c) => out.slice(c * K, (c + 1) * K).reduce((s, o) => s + o.rollout.ret, 0) / K);
    },
    { directions: cfg.directions, top: cfg.top, stepSize: cfg.stepSize, noise: cfg.noise, iterations: cfg.iterations, seed: cfg.seed },
    (p) => {
      stats.merge(fresh);
      fresh = new RunningStats(inputs);
      ({ mean, std } = normalization(stats, feet));
      onProgress?.(0.05 + (0.85 * (p.iteration + 1)) / cfg.iterations, `Iteration ${p.iteration + 1} of ${cfg.iterations}: mean return ${p.meanReturn.toFixed(3)} over pushed runs (${rollouts} runs)`);
    },
  );
  const policy = policyFor(theta);

  onProgress?.(0.92, 'Comparing the policy with the keyframes alone on held-out pushed runs');
  const hs = cfg.seed * 1000 + 999;
  const held = Array.from({ length: cfg.heldOutTrials }, (_, k) => task(hs, k));
  const incumbentJobs = cfg.incumbentPolicy ? held.map((task): PoseRolloutJob => ({ motion, policy: cfg.incumbentPolicy!, goals, ...task })) : [];
  const ablated = cfg.architecture === 'mlp' ? posePolicyWithoutFeedback(policy, desc) : null;
  const ablationJobs = ablated ? held.map((task): PoseRolloutJob => ({ motion, policy: ablated, goals, ...task })) : [];
  const out = await run([...held.map((t): PoseRolloutJob => ({ motion, policy: null, goals, ...t })), ...held.map((t): PoseRolloutJob => ({ motion, policy, goals, ...t })), ...incumbentJobs, ...ablationJobs]);
  const baseline = scoreOf(out.slice(0, held.length));
  const result = scoreOf(out.slice(held.length, 2 * held.length));
  const incumbent = incumbentJobs.length ? scoreOf(out.slice(2 * held.length, 2 * held.length + incumbentJobs.length)) : null;
  const feedbackAblation = ablationJobs.length ? scoreOf(out.slice(2 * held.length + incumbentJobs.length)) : null;
  const improved = [baseline, ...(incumbent ? [incumbent] : [])].every((reference) =>
    result.falls <= reference.falls && result.successes >= reference.successes &&
    (result.successes > reference.successes || result.meanGoalReward > reference.meanGoalReward + 0.05),
  );
  onProgress?.(1, 'Done');
  return { policy, improved, baseline, result, ...(incumbent ? { incumbent } : {}), ...(feedbackAblation ? { feedbackAblation } : {}), heldOutTrials: held.length, pushImpulseNs: [...cfg.pushImpulseNs], history, rollouts, parameters: nParams, workers: runner.workers };
}
