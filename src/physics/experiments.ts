import type { MainModule } from '@mujoco/mujoco';
import { TargetRateLimiter, wrapToPi } from '../core/control/limits';
import { pdControl, secondOrderParameters, type PdGains } from '../core/control/pd';
import { analyticStepMetrics, linearResponse, secondOrderStep, stepMetrics, type StepMetrics } from '../core/control/stepMetrics';
import { gravityTorques } from '../core/dynamics/gravity';
import { buildKinematicModel, forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { solveIK } from '../core/kinematics/ik';
import type { Vec3 } from '../core/math/vec3';
import { testStand } from '../core/model/builder';
import { homeConfiguration } from '../core/model/catalog';
import type { RobotDescription } from '../core/model/types';
import { MujocoRuntime, type SensorReadings } from './runtime';

/**
 * A long-running simulation expressed as a generator that yields its
 * progress (0..1) every few steps, so the UI can run it in time slices.
 * Always run it to completion or call `return()`; the runtime is disposed in
 * a `finally` block.
 */
export type Run<T> = Generator<number, T, void>;

export function runToEnd<T>(run: Run<T>): T {
  for (;;) {
    const r = run.next();
    if (r.done) return r.value;
  }
}

const YIELD_EVERY = 50;

export interface ControllerSpec {
  gains: PdGains;
  /** Adds the model-based gravity torque at the measured configuration (fixed bases only). */
  gravityCompensation: boolean;
}

/** Fixed-base robots run as they are; floating robots are bolted to a test stand. */
export function experimentRobot(desc: RobotDescription): { robot: RobotDescription; onStand: boolean } {
  return desc.base.kind === 'floating' ? { robot: testStand(desc), onStand: true } : { robot: desc, onStand: false };
}

function controller(km: KinematicModel, spec: ControllerSpec, effort: number[], n: number) {
  const gains = new Array<PdGains>(n).fill(spec.gains);
  const fixed = km.description.base.kind === 'fixed';
  const continuous = km.joints.map((j) => j.type === 'continuous');
  return (s: SensorReadings, qCmd: ArrayLike<number>) =>
    pdControl(gains, effort, { qCmd, qMeas: s.q, qdMeas: s.qd, continuous, tauFeedforward: spec.gravityCompensation && fixed ? gravityTorques(km, s.q) : undefined });
}

/** Bounds commanded targets by each joint's declared velocity limit, as live control does. */
function rateLimiter(km: KinematicModel, q0: number[]): TargetRateLimiter {
  const vmax = km.description.limbs.flatMap((l) => l.joints.map((j) => j.limit.velocity));
  return new TargetRateLimiter(vmax, km.joints.map((j) => j.type === 'continuous'), q0);
}

/**
 * How the step is commanded. 'rate-limited' moves the target at the joint's
 * velocity limit, as live control does. 'ideal-step' jumps instantly: an
 * analytical reference that real actuators cannot follow.
 */
export type StepCommand = 'rate-limited' | 'ideal-step';

export interface StepTestConfig {
  joint: number;
  /** Step size added to the home position, rad (m for prismatic joints). */
  step: number;
  /** Time held at the home pose before the step, s. */
  settleS: number;
  /** Time recorded after the step, s. */
  durationS: number;
  controller: ControllerSpec;
  /** Default 'rate-limited'. */
  command?: StepCommand;
}

export interface StepTestResult {
  onStand: boolean;
  command: StepCommand;
  /** Time since the step, s. */
  t: number[];
  /** Commanded target after rate limiting (rad or m). */
  qCmd: number[];
  qMeas: number[];
  tauCmd: number[];
  tauApplied: number[];
  effort: number;
  /** Declared velocity limit of the joint, rad/s (m/s for prismatic joints). */
  velocityLimit: number;
  /** Time after the step during which the command was held back by the velocity limit, s. */
  rateLimitedS: number;
  y0: number;
  target: number;
  /** True when the requested step went past a joint limit and was clamped. */
  clamped: boolean;
  metrics: StepMetrics;
  saturationFraction: number;
  /** Diagonal joint inertia at the home pose (MuJoCo mass matrix), kg*m^2 (kg for prismatic joints). */
  inertia: number;
  wn: number;
  zeta: number;
  /** Linear second-order prediction for the same command (no gravity, coupling or saturation). */
  linear: StepMetrics | null;
  /** The linear prediction sampled at `t`. */
  linearQ: number[] | null;
  unstable: boolean;
}

/**
 * Step response of one joint under the given controller; every other joint
 * holds its home position. By default the commanded target moves at the
 * joint's velocity limit; 'ideal-step' jumps instantly (analytical reference).
 */
export function* stepTest(mj: MainModule, desc: RobotDescription, cfg: StepTestConfig): Run<StepTestResult> {
  const { robot, onStand } = experimentRobot(desc);
  const km = buildKinematicModel(robot);
  const q0 = homeConfiguration(robot);
  if (cfg.joint < 0 || cfg.joint >= q0.length) throw new RangeError('Joint index out of range');
  if (!Number.isFinite(cfg.step) || cfg.step === 0) throw new RangeError('Step size must be a non-zero finite number');
  const command: StepCommand = cfg.command ?? 'rate-limited';
  const rt = MujocoRuntime.create(mj, robot, { initialQ: q0 });
  try {
    const j = cfg.joint;
    const kj = km.joints[j]!;
    const requested = q0[j]! + cfg.step;
    const target = Math.min(kj.upper, Math.max(kj.lower, requested));
    const inertia = rt.jointInertias()[j]!;
    const effort = rt.layout.effort;
    const law = controller(km, cfg.controller, effort, q0.length);
    const limiter = rateLimiter(km, q0);
    const velocityLimit = robot.limbs.flatMap((l) => l.joints)[j]!.limit.velocity;
    const qReq = q0.slice();
    const h = rt.timestep;
    const settle = Math.round(cfg.settleS / h);
    const total = settle + Math.round(cfg.durationS / h);
    const out = { t: [] as number[], qCmd: [] as number[], qMeas: [] as number[], tauCmd: [] as number[], tauApplied: [] as number[] };
    let y0 = Number.NaN;
    let saturated = 0, limitedSteps = 0;
    let unstable = false;
    for (let k = 0; k < total; k++) {
      if (k === settle) qReq[j] = target;
      let qCmd: ArrayLike<number> = qReq;
      if (command === 'rate-limited') {
        if (limiter.step(qReq, h) && k >= settle) limitedSteps++;
        qCmd = limiter.qCmd;
      }
      rt.controlStep((s) => {
        const r = law(s, qCmd);
        if (k >= settle) {
          if (k === settle) y0 = s.q[j]!;
          out.t.push((k - settle) * h);
          out.qCmd.push(qCmd[j]!);
          out.qMeas.push(s.q[j]!);
          out.tauCmd.push(r.tauCmd[j]!);
          out.tauApplied.push(r.tauApplied[j]!);
          if (r.saturated[j]) saturated++;
        }
        return r.tauApplied;
      });
      if (rt.hasNonFiniteState()) {
        unstable = true;
        break;
      }
      if (k % YIELD_EVERY === 0) yield k / total;
    }
    const { wn, zeta } = secondOrderParameters(cfg.controller.gains, inertia);
    const step = target - y0;
    const predictable = wn > 0 && Number.isFinite(zeta) && out.t.length > 1 && Number.isFinite(y0) && step !== 0;
    let linear: StepMetrics | null = null;
    let linearQ: number[] | null = null;
    if (predictable && command === 'ideal-step') {
      linear = analyticStepMetrics(wn, zeta, cfg.durationS);
      linearQ = out.t.map((t) => y0 + step * secondOrderStep(t, wn, zeta));
    } else if (predictable) {
      const y = linearResponse(cfg.controller.gains.kp, cfg.controller.gains.kd, inertia, h, out.qCmd, y0);
      linear = stepMetrics(out.t, y, y0, target);
      linearQ = Array.from(y);
    }
    return {
      onStand,
      command,
      ...out,
      effort: effort[j]!,
      velocityLimit,
      rateLimitedS: limitedSteps * h,
      y0,
      target,
      clamped: target !== requested,
      metrics: !unstable && out.t.length > 1 && step !== 0 ? stepMetrics(out.t, out.qMeas, y0, target) : { riseTimeS: null, peakTimeS: null, overshootPct: Number.NaN, settlingTimeS: null, steadyStateError: Number.NaN },
      saturationFraction: out.t.length ? saturated / out.t.length : 0,
      inertia,
      wn,
      zeta,
      linear,
      linearQ,
      unstable,
    };
  } finally {
    rt.dispose();
  }
}

export type ChallengeId = 'reach' | 'hold' | 'push';

/** Impulse of the push challenge, N*s, applied sideways over 50 ms. */
export const PUSH_IMPULSE_NS = 4;

/**
 * Hold-challenge limits. Prismatic limits are the revolute ones times a
 * 0.25 m lever arm, so a sliding joint is held to the same tip-scale accuracy.
 */
export const HOLD_LIMITS = { angleRad: 0.02, angularSpeedRadS: 0.05, positionM: 0.005, linearSpeedMS: 0.0125 } as const;

export interface Criterion {
  label: string;
  measured: number;
  threshold: number;
  unit: string;
  /** Pass when measured < threshold ('below') or > threshold ('above'). */
  pass: 'below' | 'above';
  passed: boolean;
  /** Where the measured value comes from. */
  source: 'encoders' | 'IMU' | 'ground truth';
}

export interface ChallengeResult {
  id: ChallengeId;
  applicable: boolean;
  /** Why the challenge does not apply to this robot. */
  reason: string | null;
  passed: boolean;
  criteria: Criterion[];
  onStand: boolean;
  t: number[];
  /** Named traces for plotting. */
  series: Record<string, number[]>;
  unstable: boolean;
}

export const CHALLENGES: { id: ChallengeId; title: string; goal: string }[] = [
  { id: 'reach', title: 'Reach a target', goal: 'Move the arm tip to a target 0.2 s after start; within 2.0 s the tip, computed from encoder readings, must stay within 10 mm of the target until the end (3 s).' },
  { id: 'hold', title: 'Hold a pose', goal: 'Hold the home pose against gravity for 3 s; during the last second every revolute joint must stay within 0.02 rad of its command and below 0.05 rad/s (prismatic joints: within 5 mm and below 12.5 mm/s).' },
  { id: 'push', title: 'Survive a push', goal: `Stand for 1 s, take a ${PUSH_IMPULSE_NS} N*s sideways push over 50 ms, and recover: tilt below 0.35 rad, base height above 60 % of the pre-push height, and tilt back below 0.05 rad within 1.5 s.` },
];

function criterion(label: string, measured: number, threshold: number, unit: string, pass: 'below' | 'above', source: Criterion['source']): Criterion {
  const passed = Number.isFinite(measured) && (pass === 'below' ? measured < threshold : measured > threshold);
  return { label, measured, threshold, unit, pass, passed, source };
}

function notApplicable(id: ChallengeId, reason: string): ChallengeResult {
  return { id, applicable: false, reason, passed: false, criteria: [], onStand: false, t: [], series: {}, unstable: false };
}

function tiltFromImu(q: readonly number[]): number {
  // Angle between the body z axis and world z: R[2][2] = 1 - 2 (x^2 + y^2) for q = [w, x, y, z].
  const c = 1 - 2 * (q[1]! * q[1]! + q[2]! * q[2]!);
  return Math.acos(Math.min(1, Math.max(-1, c)));
}

function* reach(mj: MainModule, desc: RobotDescription, spec: ControllerSpec): Run<ChallengeResult> {
  if (desc.base.kind !== 'fixed') return notApplicable('reach', 'Reaching needs a fixed-base arm; this robot has a floating base.');
  const limb = desc.limbs.findIndex((l) => l.role === 'arm' && l.joints.length >= 2);
  if (limb < 0) return notApplicable('reach', 'Reaching needs an arm with at least two joints.');
  const km = buildKinematicModel(desc);
  const q0 = homeConfiguration(desc);
  const L = km.limbs[limb]!;
  const qGoal = q0.slice();
  for (let i = 0; i < L.count; i++) {
    const kj = km.joints[L.start + i]!;
    const v = q0[L.start + i]! + (i === 0 ? 0.6 : i % 2 ? -0.35 : 0.3);
    qGoal[L.start + i] = Math.min(kj.upper, Math.max(kj.lower, v));
  }
  const tipOf = (q: ArrayLike<number>): Vec3 => {
    const T = forwardKinematics(km, q).tips[limb]!;
    return [T[12]!, T[13]!, T[14]!];
  };
  const target = tipOf(qGoal);
  const ik = solveIK(km, limb, target, q0, { restarts: 2 });
  if (ik.status !== 'converged') return notApplicable('reach', `The target could not be solved by IK (${ik.message}).`);
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q0 });
  try {
    const law = controller(km, spec, rt.layout.effort, q0.length);
    const limiter = rateLimiter(km, q0);
    const h = rt.timestep;
    const total = Math.round(3 / h);
    const start = Math.round(0.2 / h);
    const t: number[] = [], err: number[] = [];
    let qReq = q0.slice();
    let unstable = false;
    for (let k = 0; k < total; k++) {
      if (k === start) qReq = ik.q.slice();
      // The target moves toward the IK solution no faster than each joint's velocity limit.
      limiter.step(qReq, h);
      rt.controlStep((s) => {
        const p = tipOf(s.q);
        t.push(k * h);
        err.push(Math.hypot(p[0] - target[0], p[1] - target[1], p[2] - target[2]));
        return law(s, limiter.qCmd).tauApplied;
      });
      if (rt.hasNonFiniteState()) {
        unstable = true;
        break;
      }
      if (k % YIELD_EVERY === 0) yield k / total;
    }
    const tol = 0.01;
    let lastOutside = -1;
    err.forEach((e, i) => {
      if (e > tol) lastOutside = i;
    });
    const settle = lastOutside === err.length - 1 ? Number.POSITIVE_INFINITY : (t[lastOutside + 1] ?? 0) - 0.2;
    const finalWindow = err.filter((_, i) => t[i]! >= 2.5);
    const criteria = [
      criterion('Tip error over the last 0.5 s (max)', unstable ? Number.NaN : Math.max(...finalWindow) * 1000, 10, 'mm', 'below', 'encoders'),
      criterion('Time to stay within 10 mm', unstable ? Number.NaN : settle, 2.0, 's', 'below', 'encoders'),
    ];
    return { id: 'reach', applicable: true, reason: null, passed: criteria.every((c) => c.passed), criteria, onStand: false, t, series: { 'tip error (mm)': err.map((e) => e * 1000) }, unstable };
  } finally {
    rt.dispose();
  }
}

function* hold(mj: MainModule, desc: RobotDescription, spec: ControllerSpec): Run<ChallengeResult> {
  const { robot, onStand } = experimentRobot(desc);
  const km = buildKinematicModel(robot);
  const q0 = homeConfiguration(robot);
  const rt = MujocoRuntime.create(mj, robot, { initialQ: q0 });
  try {
    const law = controller(km, spec, rt.layout.effort, q0.length);
    const h = rt.timestep;
    const total = Math.round(3 / h);
    const linear = km.joints.map((j) => j.type === 'prismatic');
    const hasAngular = linear.some((l) => !l), hasLinear = linear.some((l) => l);
    const t: number[] = [], worstErr: number[] = [];
    // Worst errors and speeds over the last second, per unit: [angular (rad, rad/s), linear (m, m/s)].
    const maxErr = [0, 0], maxVel = [0, 0];
    let unstable = false;
    for (let k = 0; k < total; k++) {
      rt.controlStep((s) => {
        const e = [0, 0], v = [0, 0];
        for (let i = 0; i < q0.length; i++) {
          const u = linear[i] ? 1 : 0;
          const raw = q0[i]! - s.q[i]!;
          e[u] = Math.max(e[u]!, Math.abs(km.joints[i]!.type === 'continuous' ? wrapToPi(raw) : raw));
          v[u] = Math.max(v[u]!, Math.abs(s.qd[i]!));
        }
        t.push(k * h);
        worstErr.push(e[0]!);
        if (k * h >= 2) {
          for (const u of [0, 1]) {
            maxErr[u] = Math.max(maxErr[u]!, e[u]!);
            maxVel[u] = Math.max(maxVel[u]!, v[u]!);
          }
        }
        return law(s, q0).tauApplied;
      });
      if (rt.hasNonFiniteState()) {
        unstable = true;
        break;
      }
      if (k % YIELD_EVERY === 0) yield k / total;
    }
    const m = (x: number) => (unstable ? Number.NaN : x);
    const criteria: Criterion[] = [];
    if (hasAngular) {
      criteria.push(
        criterion('Largest joint error in the last second', m(maxErr[0]!), HOLD_LIMITS.angleRad, 'rad', 'below', 'encoders'),
        criterion('Largest joint speed in the last second', m(maxVel[0]!), HOLD_LIMITS.angularSpeedRadS, 'rad/s', 'below', 'encoders'),
      );
    }
    if (hasLinear) {
      criteria.push(
        criterion('Largest prismatic joint error in the last second', m(maxErr[1]!), HOLD_LIMITS.positionM, 'm', 'below', 'encoders'),
        criterion('Largest prismatic joint speed in the last second', m(maxVel[1]!), HOLD_LIMITS.linearSpeedMS, 'm/s', 'below', 'encoders'),
      );
    }
    return { id: 'hold', applicable: true, reason: null, passed: criteria.every((c) => c.passed), criteria, onStand, t, series: { 'largest revolute joint error (rad)': worstErr }, unstable };
  } finally {
    rt.dispose();
  }
}

function* push(mj: MainModule, desc: RobotDescription, spec: ControllerSpec): Run<ChallengeResult> {
  if (desc.base.kind !== 'floating') return notApplicable('push', 'The push test needs a free-standing robot; this robot is fixed to the ground.');
  const km = buildKinematicModel(desc);
  const q0 = homeConfiguration(desc);
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q0 });
  try {
    const law = controller(km, spec, rt.layout.effort, q0.length);
    const h = rt.timestep;
    const pushStart = Math.round(1 / h), pushEnd = pushStart + Math.round(0.05 / h), total = Math.round(3.5 / h);
    const force = PUSH_IMPULSE_NS / ((pushEnd - pushStart) * h);
    const t: number[] = [], tilt: number[] = [], height: number[] = [];
    let preHeight = Number.NaN, maxTilt = 0, minHeight = Number.POSITIVE_INFINITY, recovered = Number.NaN;
    let unstable = false;
    for (let k = 0; k < total; k++) {
      if (k === pushStart) {
        preHeight = rt.groundTruthBaseState().pos[2];
        rt.setBodyWrench(rt.layout.baseBodyId, [0, force, 0]);
      }
      if (k === pushEnd) rt.clearWrenches();
      rt.controlStep((s) => {
        const a = tiltFromImu(s.imuQuat);
        t.push(k * h);
        tilt.push(a);
        return law(s, q0).tauApplied;
      });
      const z = rt.groundTruthBaseState().pos[2];
      height.push(z);
      if (k >= pushStart) {
        maxTilt = Math.max(maxTilt, tilt[tilt.length - 1]!);
        minHeight = Math.min(minHeight, z);
      }
      if (rt.hasNonFiniteState()) {
        unstable = true;
        break;
      }
      if (k % YIELD_EVERY === 0) yield k / total;
    }
    // Recovery: first time after the push from which tilt stays below 0.05 rad.
    let lastAbove = -1;
    tilt.forEach((a, i) => {
      if (i >= pushStart && a >= 0.05) lastAbove = i;
    });
    if (!unstable) recovered = lastAbove < 0 ? 0 : lastAbove === tilt.length - 1 ? Number.POSITIVE_INFINITY : (lastAbove + 1 - pushStart) * h;
    const criteria = [
      criterion('Largest tilt after the push', unstable ? Number.NaN : maxTilt, 0.35, 'rad', 'below', 'IMU'),
      criterion('Lowest base height / pre-push height', unstable ? Number.NaN : (minHeight / preHeight) * 100, 60, '%', 'above', 'ground truth'),
      criterion('Time until tilt stays below 0.05 rad', recovered, 1.5, 's', 'below', 'IMU'),
    ];
    return { id: 'push', applicable: true, reason: null, passed: criteria.every((c) => c.passed), criteria, onStand: false, t, series: { 'tilt (rad)': tilt, 'base height (m)': height }, unstable };
  } finally {
    rt.dispose();
  }
}

export function runChallenge(mj: MainModule, desc: RobotDescription, id: ChallengeId, spec: ControllerSpec): Run<ChallengeResult> {
  return id === 'reach' ? reach(mj, desc, spec) : id === 'hold' ? hold(mj, desc, spec) : push(mj, desc, spec);
}
