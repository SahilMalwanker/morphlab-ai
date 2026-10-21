import type { MainModule } from '@mujoco/mujoco';
import { TargetRateLimiter } from '../core/control/limits';
import { pdControl, type PdGains } from '../core/control/pd';
import { buildKinematicModel } from '../core/kinematics/fk';
import { gaussian, mulberry32 } from '../core/math/rng';
import type { Vec3 } from '../core/math/vec3';
import type { RobotDescription } from '../core/model/types';
import { augmentedRandomSearch, RunningStats, type ArsProgress } from '../learning/ars';
import { MujocoRuntime } from '../physics/runtime';
import type { PolicyJson } from './controllers';
import { CONTROL_DECIMATION, DEFAULT_SKILL_GAINS, judgeSample, trialConditions, type TrialConditions } from './evaluation';
import { GaitController, gaitEnvelope, type GaitParams } from './gait';
import type { ConditionRanges } from './learning';
import { GAIT_POLICY_INPUTS, gaitCommand, gaitPolicyObservation, LearnedGaitController } from './policyGait';
import { goalReward, GoalJudge, restingHeight, type SkillGoal } from './taught';
import { WalkPolicyController } from './walkPolicy';

/**
 * Training of a learned feedback policy for a gait (D27): Augmented Random
 * Search (V2: observation normalization, top directions) on a linear policy
 * that adds bounded corrections to the gait generator's joint targets. The
 * return rewards following the gait's own speed along the directions it
 * commands and no motion along the others (drift), penalizes tilt, bouncing
 * and jerky corrections, and adds the skill's goal reward. It is computed
 * from simulator ground truth; the policy sees sensors only. The policy is
 * kept only if it beats the gait alone on held-out randomized conditions
 * without more falls, fewer goal successes or a lower goal reward.
 */

export interface GaitPolicyConfig extends ConditionRanges {
  iterations: number;
  /** Directions per iteration (two rollouts each). */
  directions: number;
  top: number;
  stepSize: number;
  noise: number;
  heldOutTrials: number;
  seed: number;
}

export const DEFAULT_GAIT_POLICY: GaitPolicyConfig = {
  iterations: 30, directions: 8, top: 4, stepSize: 0.02, noise: 0.03, heldOutTrials: 8, seed: 11,
  frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02,
};

/** Reward weights and widths (after the velocity-tracking rewards of legged_gym and MuJoCo Playground). */
const TRACK_LIN_SIGMA = 0.01; // (m/s)^2
const TRACK_YAW_SIGMA = 0.05; // (rad/s)^2
const W_TRACK_LIN = 1;
const W_TRACK_YAW = 0.5;
const W_TILT = 2;
const W_VZ = 0.5;
const W_ACTION_RATE = 0.01;
const FALL_PENALTY = 2;
const FALL_TILT_RAD = 1;
/** Weight of the skill's goal reward (about -1 to 2 per run) in the return. */
const W_GOAL = 2;
/** A kept policy may lower the mean held-out goal reward by at most this much. */
const GOAL_TOLERANCE = 0.01;
/** Pose tracking (walking controller): distance, m^2, and heading, rad^2, from where the commanded velocity leads; about 5 cm and 0.1 rad, the scale of the walk tolerance. */
const POSE_LIN_SIGMA = 0.0025;
const POSE_YAW_SIGMA = 0.01;
const W_POSE_LIN = 1;
const W_POSE_YAW = 1;

/**
 * The body velocity the reward asks for (forward m/s, left m/s, yaw rate
 * rad/s): along each direction the gait commands, the speed the gait alone
 * was measured to reach (its goals were set for that motion), unless it went
 * the other way; zero along the directions it does not command, so drift is
 * penalized.
 */
export function trackingTarget(command: Readonly<Vec3>, measured: Readonly<Vec3>): Vec3 {
  const t = (i: 0 | 1 | 2): number => (command[i] === 0 ? 0 : Math.sign(measured[i]) === Math.sign(command[i]) ? measured[i] : command[i]);
  return [t(0), t(1), t(2)];
}

export interface GaitRolloutInput {
  desc: RobotDescription;
  stanceQ: number[];
  gait: GaitParams;
  /** Null runs the gait generator alone. */
  policy: PolicyJson | null;
  conditions: TrialConditions;
  noiseSeed: number;
  gains?: PdGains;
  /** Receives the policy's raw observations (for input normalization). */
  stats?: RunningStats;
  /** Velocity the reward asks for while stepping (see trackingTarget); the gait's nominal command when absent. */
  target?: Vec3;
  /** The skill's goals: judged on ground truth, and their reward is added to the return. */
  goals?: SkillGoal[];
  /** A training run ends at a fall and judges the goals at the control rate; otherwise the goals are judged like an evaluation trial. */
  training?: boolean;
  /** Also reward staying on the path the target velocity traces from the start (position and heading), not only its speed. */
  trackPose?: boolean;
  /** Walk this command (forward m/s, left m/s, turn rad/s) with the walking controller (D29); the policy, if any, is a walking policy, and the statistics are its observations. */
  walkCommand?: Vec3;
}

export interface GaitRollout {
  /** Return: the time integral of the step reward, minus a penalty for a fall, plus the weighted goal reward when goals were given. */
  ret: number;
  fell: boolean;
  unstable: boolean;
  /** Root mean square error between the measured body velocity and the target while stepping, m/s. */
  trackingErrorMps: number;
  /** Root mean square error of the yaw rate while stepping, rad/s. */
  yawRateErrorRadps: number;
  /** Net displacement in the starting heading frame (m) and net turn (rad, positive left). */
  forwardM: number;
  leftM: number;
  turnRad: number;
  maxTiltRad: number;
  /** Velocity at full stepping: forward m/s, left m/s, yaw rate rad/s, fitted as v(t) = e(t) * v over the run (e the gait's fade envelope). */
  meanVelocity: Vec3;
  /** Goal verdict and goal reward when goals were given. */
  goalSuccess?: boolean;
  goalReward?: number;
  /** The walking controller's sensor-only estimate at the end (forward m, left m, turn rad, start frame), when it walked. */
  odometry?: Vec3;
}

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** One randomized run of a gait with or without a learned policy, scored by the dense reward and the skill's goals. */
export function gaitPolicyRollout(mj: MainModule, input: GaitRolloutInput): GaitRollout {
  const { desc, stanceQ, gait, policy, conditions, noiseSeed, stats, goals, training = false, trackPose = false } = input;
  const gains = input.gains ?? DEFAULT_SKILL_GAINS;
  const rng = mulberry32(noiseSeed);
  const q0 = stanceQ.map((v) => v + conditions.jointNoiseRad * Math.max(-1, Math.min(1, gaussian(rng) / 2)));
  const km = buildKinematicModel(desc);
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q0, friction: conditions.friction, baseMass: conditions.baseMassKg });
  try {
    const walker = input.walkCommand ? new WalkPolicyController(km, stanceQ, gait, input.walkCommand, policy) : null;
    const learned = policy && !walker ? new LearnedGaitController(km, stanceQ, gait, policy) : null;
    const ctl = walker ?? learned ?? new GaitController(km, stanceQ, gait);
    const actor = walker ? (policy ? walker : null) : learned;
    const g = stanceQ.map(() => gains);
    const joints = desc.limbs.flatMap((l) => l.joints);
    const continuous = joints.map((j) => j.type === 'continuous');
    const limiter = new TargetRateLimiter(joints.map((j) => j.limit.velocity), continuous, q0);
    const judge = goals ? GoalJudge.forRobot(goals, desc) : null;
    const rest = restingHeight(desc);
    const target = input.target ?? gaitCommand(gait);
    const obsBuf = new Float64Array(GAIT_POLICY_INPUTS);
    const prevAction = new Float64Array(actor ? actor.action.length : 0);
    const dtc = CONTROL_DECIMATION * rt.timestep;
    const T0 = rt.groundTruthBaseTransform();
    const p0 = [T0[12]!, T0[13]!];
    const yaw0 = Math.atan2(T0[1]!, T0[0]!);
    let yawPrev = yaw0;
    let turn = 0;
    let ret = 0;
    let fell = false;
    let unstable = false;
    let maxTilt = 0;
    let linSq = 0, yawSq = 0, tracked = 0;
    const velA: Vec3 = [0, 0, 0];
    let aSq = 0;
    let pathX = 0, pathY = 0, pathYaw = 0;
    let qTarget = stanceQ.slice();
    let lastTouch: number[] = [];
    let doneAt = -1;
    const maxSteps = Math.ceil((gait.durationS + 2) / rt.timestep);
    for (let k = 0; k < maxSteps; k++) {
      const t = k * rt.timestep;
      rt.controlStep((s) => {
        if (k % CONTROL_DECIMATION === 0) {
          const obs = { q: s.q, qd: s.qd, touch: s.touch, imuQuat: s.imuQuat, gyro: s.gyro };
          if (stats && !walker && !ctl.finished) stats.push(gaitPolicyObservation(t, obs, gait, obsBuf));
          qTarget = ctl.update(t, obs);
          if (stats && walker && !walker.finished) stats.push(walker.observation);
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
      // Training judges at the control rate (cheaper); other runs are judged every step, like an evaluation trial.
      if (judge && (!training || controlEnd)) judge.add(judgeSample(rt, t, lastTouch));
      // Reward (ground truth) at the end of each control period.
      if (controlEnd) {
        const tEnd = t + rt.timestep;
        const T = rt.groundTruthBaseTransform();
        const b = rt.groundTruthBaseState();
        const yaw = Math.atan2(T[1]!, T[0]!);
        const yawRate = wrap(yaw - yawPrev) / dtc;
        turn += wrap(yaw - yawPrev);
        yawPrev = yaw;
        const c = Math.cos(yaw), sn = Math.sin(yaw);
        const vFwd = c * b.linVel[0] + sn * b.linVel[1];
        const vLeft = -sn * b.linVel[0] + c * b.linVel[1];
        const a = gaitEnvelope(gait, Math.min(tEnd, gait.durationS));
        const eLin = (vFwd - a * target[0]) ** 2 + (vLeft - a * target[1]) ** 2;
        const eYaw = (yawRate - a * target[2]) ** 2;
        if (a > 0.5) {
          linSq += eLin;
          yawSq += eYaw;
          tracked++;
        }
        velA[0] += a * vFwd;
        velA[1] += a * vLeft;
        velA[2] += a * yawRate;
        aSq += a * a;
        let rate = 0;
        if (actor) {
          for (let i = 0; i < prevAction.length; i++) rate += (actor.action[i]! - prevAction[i]!) ** 2;
          prevAction.set(actor.action);
        }
        const tiltSq = T[2]! ** 2 + T[6]! ** 2;
        ret += dtc * (W_TRACK_LIN * Math.exp(-eLin / TRACK_LIN_SIGMA) + W_TRACK_YAW * Math.exp(-eYaw / TRACK_YAW_SIGMA) - W_TILT * tiltSq - W_VZ * b.linVel[2] ** 2 - W_ACTION_RATE * rate);
        if (trackPose) {
          const mid = pathYaw + 0.5 * a * target[2] * dtc;
          pathX += a * (target[0] * Math.cos(mid) - target[1] * Math.sin(mid)) * dtc;
          pathY += a * (target[0] * Math.sin(mid) + target[1] * Math.cos(mid)) * dtc;
          pathYaw += a * target[2] * dtc;
          const dx = T[12]! - p0[0]!, dy = T[13]! - p0[1]!;
          const ex = Math.cos(yaw0) * dx + Math.sin(yaw0) * dy - pathX;
          const ey = -Math.sin(yaw0) * dx + Math.cos(yaw0) * dy - pathY;
          ret += dtc * (W_POSE_LIN * Math.exp(-(ex * ex + ey * ey) / POSE_LIN_SIGMA) + W_POSE_YAW * Math.exp(-((turn - pathYaw) ** 2) / POSE_YAW_SIGMA));
        }
        maxTilt = Math.max(maxTilt, b.tilt);
        if (!fell && (b.tilt > FALL_TILT_RAD || (rest !== null && b.pos[2] < 0.5 * rest))) {
          fell = true;
          ret -= FALL_PENALTY;
          // Training stops at a fall; other runs continue like an evaluation trial.
          if (training) break;
        }
      }
      if (ctl.finished && doneAt < 0) doneAt = t;
      if (doneAt >= 0 && t - doneAt > 0.5) break;
    }
    if (unstable) ret -= FALL_PENALTY;
    const T = rt.groundTruthBaseTransform();
    const dx = T[12]! - p0[0]!, dy = T[13]! - p0[1]!;
    const out: GaitRollout = {
      ret,
      fell,
      unstable,
      trackingErrorMps: tracked ? Math.sqrt(linSq / tracked) : 0,
      yawRateErrorRadps: tracked ? Math.sqrt(yawSq / tracked) : 0,
      forwardM: Math.cos(yaw0) * dx + Math.sin(yaw0) * dy,
      leftM: -Math.sin(yaw0) * dx + Math.cos(yaw0) * dy,
      turnRad: turn,
      maxTiltRad: maxTilt,
      meanVelocity: aSq > 0 ? [velA[0] / aSq, velA[1] / aSq, velA[2] / aSq] : [0, 0, 0],
      ...(walker ? { odometry: walker.estimate } : {}),
    };
    if (judge) {
      const v = judge.verdict({ unstable });
      out.goalSuccess = v.success;
      out.goalReward = goalReward({ success: v.success, goals: v.goals }, unstable);
      out.ret += W_GOAL * out.goalReward;
    }
    return out;
  } finally {
    rt.dispose();
  }
}

/** A linear policy (no hidden layer) from a parameter vector and input normalization. */
export function linearGaitPolicy(theta: ArrayLike<number>, joints: number, mean: ArrayLike<number>, std: ArrayLike<number>): PolicyJson {
  if (theta.length !== joints * (GAIT_POLICY_INPUTS + 1)) throw new RangeError(`Expected ${joints * (GAIT_POLICY_INPUTS + 1)} parameters, got ${theta.length}`);
  return { layers: [GAIT_POLICY_INPUTS, joints], activation: 'tanh', weights: Array.from(theta), inputMean: Array.from(mean), inputStd: Array.from(std) };
}

export interface GaitPolicyScore {
  meanReturn: number;
  trackingErrorMps: number;
  yawRateErrorRadps: number;
  falls: number;
  /** Runs meeting every goal of the skill. */
  successes: number;
  /** Mean goal reward of the skill's goals (about -1 to 2; 1 or more when every goal is met). */
  meanGoalReward: number;
  meanForwardM: number;
  meanLeftM: number;
  meanTurnRad: number;
}

export interface GaitPolicyResult {
  policy: PolicyJson;
  /** True when the policy beat the gait alone on held-out conditions without more falls, fewer goal successes or a lower goal reward. */
  improved: boolean;
  baseline: GaitPolicyScore;
  result: GaitPolicyScore;
  /** Velocity the reward asked for while stepping: forward m/s, left m/s, yaw rate rad/s. */
  target: Vec3;
  heldOutTrials: number;
  history: ArsProgress[];
  rollouts: number;
  parameters: number;
}

/** Trains a learned feedback policy for a gait with augmented random search in randomized trials. */
export function trainGaitPolicy(
  mj: MainModule,
  desc: RobotDescription,
  stanceQ: number[],
  gait: GaitParams,
  goals: SkillGoal[],
  cfg: GaitPolicyConfig,
  onProgress?: (p: ArsProgress) => boolean | void,
): GaitPolicyResult {
  const joints = stanceQ.length;
  const nParams = joints * (GAIT_POLICY_INPUTS + 1);
  let rollouts = 0;
  const run = (policy: PolicyJson | null, seed: number, i: number, extra: Partial<GaitRolloutInput> = {}) => {
    rollouts++;
    return gaitPolicyRollout(mj, { desc, stanceQ, gait, policy, conditions: trialConditions({ ...cfg, seed }, i), noiseSeed: seed * 104729 + i, gains: cfg.gains, ...extra });
  };
  // Input statistics and the speed to follow come from one run of the gait alone; the statistics then grow with every iteration's runs.
  const stats = new RunningStats(GAIT_POLICY_INPUTS);
  const target = trackingTarget(gaitCommand(gait), run(null, cfg.seed * 1000 + 998, 0, { stats }).meanVelocity);
  let mean = Float64Array.from(stats.mean);
  let std = stats.std();
  let fresh = new RunningStats(GAIT_POLICY_INPUTS);
  const { theta, history } = augmentedRandomSearch(
    new Float64Array(nParams),
    (th, it) => run(linearGaitPolicy(th, joints, mean, std), cfg.seed * 1000 + it, 0, { stats: fresh, target, goals, training: true }).ret,
    { directions: cfg.directions, top: cfg.top, stepSize: cfg.stepSize, noise: cfg.noise, iterations: cfg.iterations, seed: cfg.seed },
    (p) => {
      stats.merge(fresh);
      fresh = new RunningStats(GAIT_POLICY_INPUTS);
      mean = Float64Array.from(stats.mean);
      std = stats.std();
      return onProgress?.({ ...p, rollouts });
    },
  );
  const policy = linearGaitPolicy(theta, joints, mean, std);
  const heldOut = cfg.seed * 1000 + 999;
  const score = (p: PolicyJson | null): GaitPolicyScore => {
    const runs = Array.from({ length: cfg.heldOutTrials }, (_, i) => run(p, heldOut, i, { goals, target }));
    const avg = (f: (r: GaitRollout) => number) => runs.reduce((s, r) => s + f(r), 0) / Math.max(1, runs.length);
    return {
      meanReturn: avg((r) => r.ret),
      trackingErrorMps: avg((r) => r.trackingErrorMps),
      yawRateErrorRadps: avg((r) => r.yawRateErrorRadps),
      falls: runs.filter((r) => r.fell || r.unstable).length,
      successes: runs.filter((r) => r.goalSuccess).length,
      meanGoalReward: avg((r) => r.goalReward ?? 0),
      meanForwardM: avg((r) => r.forwardM),
      meanLeftM: avg((r) => r.leftM),
      meanTurnRad: avg((r) => r.turnRad),
    };
  };
  const baseline = score(null);
  const result = score(policy);
  // A better return must not come from falling more often or from giving up on the skill's goals.
  const improved =
    result.meanReturn > baseline.meanReturn + 1e-9 && result.falls <= baseline.falls && result.successes >= baseline.successes && result.meanGoalReward >= baseline.meanGoalReward - GOAL_TOLERANCE;
  return { policy, improved, baseline, result, target, heldOutTrials: cfg.heldOutTrials, history, rollouts, parameters: nParams };
}
