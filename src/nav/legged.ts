import { TargetRateLimiter } from '../core/control/limits';
import { pdControl, type PdGains } from '../core/control/pd';
import { buildKinematicModel, clampToLimits, type KinematicModel } from '../core/kinematics/fk';
import type { Vec3 } from '../core/math/vec3';
import type { RobotDescription } from '../core/model/types';
import { Mlp } from '../learning/mlp';
import type { SensorReadings } from '../physics/runtime';
import { DEFAULT_SKILL_GAINS } from '../skills/evaluation';
import type { FootLiftObservation } from '../skills/footLift';
import { GAIT_BOUNDS, GaitController, gaitEnvelope, legIndices, type GaitParams } from '../skills/gait';
import { nominalFor, type LocomotionController } from '../skills/locomotion';
import { LegOdometry } from '../skills/odometry';
import { RESIDUAL_SCALE_RAD } from '../skills/policyGait';
import { WALK_COMMAND_OUTPUTS, WALK_COMMAND_SCALE, WALK_COMMAND_TAU_S, WALK_POLICY_INPUTS, walkObservation, walkPolicyProblem } from '../skills/walkPolicy';
import type { LeggedDrive } from './drive';
import { wrapAngle, type Pose2 } from './follower';
import type { PoseEstimator } from './stack';

/** The gait updates every this many physics steps: 100 Hz at the 2 ms step, as in the skill evaluations. */
export const GAIT_DECIMATION = 5;
/** Joint gains of every walking measurement so far (the skill evaluations and gait spikes). */
export const LEG_GAINS: PdGains = DEFAULT_SKILL_GAINS;
/** Stride and turn per cycle change by at most this much per second, so planted feet are not dragged. */
const STRIDE_SLEW = 0.15;
const TURN_SLEW = 0.5;
/** The policy was trained on short walks with small errors; beyond these the reference is pulled to the estimate, m and rad. */
const MAX_PATH_ERROR_M = 0.1;
const MAX_HEADING_ERROR_RAD = 0.3;

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** The generator's stride, side step and turn per cycle that a walking controller's calibration predicts walk velocity cmd (m/s, m/s, rad/s), within the gait bounds. */
export function calibratedMotion(calibration: readonly number[], cmd: Readonly<Vec3>, cycleS: number): Vec3 {
  const nominal = nominalFor(calibration, cmd) ?? cmd;
  const perCycle = (v: number, [lo, hi]: readonly [number, number]) => clamp(v * cycleS, lo, hi);
  return [perCycle(nominal[0], GAIT_BOUNDS.strideM), perCycle(nominal[1], GAIT_BOUNDS.lateralM), perCycle(nominal[2], GAIT_BOUNDS.turnRad)];
}

/**
 * The learned parts of a walking controller (D29) for a walk whose command
 * changes: the calibration turns the commanded velocity into the
 * generator's stride, side step and turn per cycle, and the feedback policy
 * corrects them (and every joint, when it has joint outputs) from the IMU
 * and how far its own leg odometry is from where the commands lead. Its
 * observation is the one it was trained on. Reset at every start from
 * standing, like the walks it was trained on.
 */
class LearnedFeedback {
  private readonly net: Mlp | null;
  private readonly x = new Float64Array(WALK_POLICY_INPUTS);
  readonly observation = new Float64Array(WALK_POLICY_INPUTS);
  readonly action: Float64Array;
  private odometry: LegOdometry;
  private readonly path: Vec3 = [0, 0, 0];
  private readonly delta: Vec3 = [0, 0, 0];
  private tPrev: number | null = null;
  fault: string | null = null;

  constructor(private readonly km: KinematicModel, readonly controller: LocomotionController, joints: number) {
    const p = controller.policy;
    const problem = p ? walkPolicyProblem(p, joints) : null;
    if (problem) throw new RangeError(problem.charAt(0).toUpperCase() + problem.slice(1));
    this.net = p ? Mlp.fromJSON({ sizes: p.layers, activation: p.activation, params: p.weights }) : null;
    this.action = new Float64Array(this.net ? this.net.outputSize : WALK_COMMAND_OUTPUTS);
    this.odometry = new LegOdometry(km, legIndices(km.description));
  }

  reset(): void {
    this.odometry = new LegOdometry(this.km, legIndices(this.km.description));
    for (let k = 0; k < 3; k++) this.path[k] = this.delta[k] = 0;
    this.action.fill(0);
    this.tPrev = null;
    this.fault = null;
  }

  /** Stride, side step and turn per cycle for the commanded velocity cmd (m/s, m/s, rad/s) at time t; `local` is the gait clock. */
  motion(t: number, local: number, gait: GaitParams, obs: FootLiftObservation, cmd: Readonly<Vec3>): Vec3 {
    const dt = this.tPrev === null ? 0 : Math.max(0, t - this.tPrev);
    this.tPrev = t;
    const odo = this.odometry;
    odo.update(t, obs);
    if (dt > 0) {
      const a = gaitEnvelope(gait, Math.max(0, local - 0.5 * dt));
      const mid = this.path[2] + 0.5 * a * cmd[2] * dt;
      this.path[0] += a * (cmd[0] * Math.cos(mid) - cmd[1] * Math.sin(mid)) * dt;
      this.path[1] += a * (cmd[0] * Math.sin(mid) + cmd[1] * Math.cos(mid)) * dt;
      this.path[2] += a * cmd[2] * dt;
    }
    const ex = this.path[0] - odo.x, ey = this.path[1] - odo.y, e = Math.hypot(ex, ey);
    if (e > MAX_PATH_ERROR_M) {
      this.path[0] = odo.x + (ex * MAX_PATH_ERROR_M) / e;
      this.path[1] = odo.y + (ey * MAX_PATH_ERROR_M) / e;
    }
    const eh = this.path[2] - odo.heading;
    if (Math.abs(eh) > MAX_HEADING_ERROR_RAD) this.path[2] = odo.heading + Math.sign(eh) * MAX_HEADING_ERROR_RAD;
    walkObservation(this.observation, local, gait.cycleS, gaitEnvelope(gait, local), obs, cmd, this.path, odo);
    const base = calibratedMotion(this.controller.calibration, cmd, gait.cycleS);
    const p = this.controller.policy;
    if (!this.net || !p || this.fault) return base;
    for (let i = 0; i < WALK_POLICY_INPUTS; i++) this.x[i] = (this.observation[i]! - p.inputMean[i]!) / p.inputStd[i]!;
    const y = this.net.forward(this.x);
    if (!y.every(Number.isFinite)) {
      // A broken policy must not move the body: the calibrated generator walks on, and the walker reports it.
      this.fault = 'the policy output was not finite';
      this.action.fill(0);
      return base;
    }
    for (let i = 0; i < y.length; i++) this.action[i] = clamp(y[i]!, -1, 1);
    const corrections = this.action.length - WALK_COMMAND_OUTPUTS;
    const alpha = dt / (WALK_COMMAND_TAU_S + dt);
    for (let k = 0; k < WALK_COMMAND_OUTPUTS; k++) this.delta[k] = this.delta[k]! + alpha * (this.action[corrections + k]! - this.delta[k]!);
    return [base[0] + WALK_COMMAND_SCALE[0] * this.delta[0], base[1] + WALK_COMMAND_SCALE[1] * this.delta[1], base[2] + WALK_COMMAND_SCALE[2] * this.delta[2]];
  }

  /** The policy's joint corrections (when it has them), faded with the steps by a. */
  correct(q: number[], a: number): number[] {
    const corrections = this.action.length - WALK_COMMAND_OUTPUTS;
    if (!this.net || this.fault || corrections === 0) return q;
    for (let i = 0; i < corrections; i++) q[i] = q[i]! + a * RESIDUAL_SCALE_RAD * this.action[i]!;
    return clampToLimits(this.km, q);
  }
}

/**
 * Walks a legged body by a body command (forward speed and turn rate): the
 * gait generator (a trot; a tripod on six legs) with its stride and turn
 * per cycle set from the command, or, with a walking controller trained in
 * the AI Playground, by that controller's calibration and learned feedback
 * policy. The generator plays a gait of fixed length that fades in and
 * out; for an open-ended walk its clock is kept inside the steady stepping
 * window by moving it by whole cycles, which leaves every foot's phase
 * unchanged. A zero command first slows stride and turn to zero, then fades
 * the stepping out, so the body ends in its stance. Inputs are sensor
 * readings only.
 */
export class LeggedWalker {
  private ctl: GaitController | null = null;
  private t0 = 0;
  private stopping = false;
  private stride = 0;
  private turn = 0;
  private want = { stride: 0, turn: 0 };
  private lastT: number | null = null;
  private readonly gait: GaitParams;
  private readonly learned: LearnedFeedback | null;

  constructor(private readonly km: KinematicModel, private readonly stanceQ: readonly number[], gait: GaitParams, controller: LocomotionController | null = null) {
    this.gait = controller ? { ...controller.base, strideM: 0, lateralM: 0, turnRad: 0, durationS: GAIT_BOUNDS.durationS[1] } : gait;
    this.learned = controller ? new LearnedFeedback(km, controller, stanceQ.length) : null;
  }

  /** The walking controller in use, or null for the gait generator alone. */
  get controller(): LocomotionController | null {
    return this.learned?.controller ?? null;
  }

  /** Why the learned policy stopped correcting the walk, or null. */
  get fault(): string | null {
    return this.learned?.fault ?? null;
  }

  /** Forward speed (m/s) and turn rate (rad/s), as stride and turn per gait cycle; a walking controller's ranges bound them. Non-finite values count as zero. */
  setCommand(v: number, omega: number): void {
    const c = this.gait.cycleS;
    const r = this.learned?.controller.ranges;
    const vv = Number.isFinite(v) ? (r ? clamp(v, r.forwardMps[0], r.forwardMps[1]) : v) : 0;
    const ww = Number.isFinite(omega) ? (r ? clamp(omega, r.turnRadps[0], r.turnRadps[1]) : omega) : 0;
    this.want = { stride: vv * c, turn: ww * c };
  }

  /** True while the feet step (including the fade in and out). */
  get stepping(): boolean {
    return this.ctl !== null;
  }

  /** Joint targets at time t. */
  update(t: number, obs: FootLiftObservation): number[] {
    const cyc = this.gait.cycleS, span = this.gait.durationS;
    const dt = this.lastT === null ? 0 : Math.max(0, t - this.lastT);
    this.lastT = t;
    const wantsMotion = Math.abs(this.want.stride) > 1e-4 || Math.abs(this.want.turn) > 1e-4;
    let ctl = this.ctl;
    if (!ctl) {
      if (!wantsMotion) return this.stanceQ.slice();
      ctl = this.ctl = new GaitController(this.km, this.stanceQ.slice(), this.gait);
      this.t0 = t;
      this.stopping = false;
      this.stride = 0;
      this.turn = 0;
      this.learned?.reset();
    }
    if (!this.stopping) {
      const ws = wantsMotion ? this.want.stride : 0, wt = wantsMotion ? this.want.turn : 0;
      this.stride += clamp(ws - this.stride, -STRIDE_SLEW * dt, STRIDE_SLEW * dt);
      this.turn += clamp(wt - this.turn, -TURN_SLEW * dt, TURN_SLEW * dt);
    }
    // The fade lasts at most one cycle, so [cycle, span - cycle] is steady stepping.
    const local = t - this.t0;
    if (!this.stopping && !wantsMotion && this.stride === 0 && this.turn === 0 && local >= cyc) {
      // Stepping in place: move the clock forward by whole cycles to just before the fade-out.
      this.t0 -= Math.max(0, Math.floor((span - cyc - local) / cyc)) * cyc;
      this.stopping = true;
    } else if (!this.stopping && local > span - 3 * cyc) {
      this.t0 += Math.floor((local - cyc) / cyc) * cyc;
    }
    const now = t - this.t0;
    const motion: Vec3 = this.learned ? this.learned.motion(t, now, this.gait, obs, [this.stride / cyc, 0, this.turn / cyc]) : [this.stride, 0, this.turn];
    ctl.setMotion(motion[0], motion[1], motion[2]);
    let q = ctl.update(now, obs);
    if (this.learned) q = this.learned.correct(q, gaitEnvelope(this.gait, now));
    if (ctl.finished) {
      this.ctl = null;
      return this.stanceQ.slice();
    }
    return q;
  }
}

/**
 * Actuator torques for a legged body at every physics step: the walker's
 * joint targets (updated at 100 Hz), bounded by each joint's velocity limit,
 * tracked by PD with saturation at the effort limit. The same loop as the
 * skill evaluations. A walking controller given later takes over the next
 * time the body stands.
 */
export class LeggedDriver {
  private walker: LeggedWalker;
  private readonly km: KinematicModel;
  private readonly limiter: TargetRateLimiter;
  private readonly defaultGains: PdGains;
  private gains: PdGains[];
  private readonly continuous: boolean[];
  private qTarget: number[];
  private steps = 0;
  private command: [number, number] = [0, 0];
  private pending: LocomotionController | null | undefined = undefined;

  constructor(robot: RobotDescription, private readonly drive: LeggedDrive, gains: PdGains = LEG_GAINS, controller: LocomotionController | null = null) {
    this.km = buildKinematicModel(robot);
    this.walker = new LeggedWalker(this.km, drive.stanceQ, drive.gait, controller);
    const joints = robot.limbs.flatMap((l) => l.joints);
    this.continuous = joints.map((j) => j.type === 'continuous');
    this.limiter = new TargetRateLimiter(joints.map((j) => j.limit.velocity), this.continuous, drive.stanceQ);
    this.defaultGains = gains;
    const g = controller?.evaluation.conditions.gains ?? gains;
    this.gains = joints.map(() => g);
    this.qTarget = drive.stanceQ.slice();
  }

  /** The walking controller that walks now, or null for the gait generator alone. */
  get controller(): LocomotionController | null {
    return this.walker.controller;
  }

  /** The joint PD gains in use: the walking controller's trained gains when recorded, otherwise the default. */
  get jointGains(): PdGains {
    return this.gains[0] ?? this.defaultGains;
  }

  get fault(): string | null {
    return this.walker.fault;
  }

  /** Walks with this walking controller (null: the gait generator) from the next time the body stands. */
  setController(c: LocomotionController | null): void {
    this.pending = c;
  }

  setCommand(v: number, omega: number): void {
    this.command = [v, omega];
    this.walker.setCommand(v, omega);
  }

  /** Torques for one physics step of length h starting at time t, from sensor readings only. */
  torques(t: number, h: number, s: SensorReadings, effort: ArrayLike<number>): Float64Array {
    if (this.steps++ % GAIT_DECIMATION === 0) {
      if (this.pending !== undefined && !this.walker.stepping) {
        const g = this.pending?.evaluation.conditions.gains ?? this.defaultGains;
        this.gains = this.gains.map(() => g);
        this.walker = new LeggedWalker(this.km, this.drive.stanceQ, this.drive.gait, this.pending);
        this.walker.setCommand(this.command[0], this.command[1]);
        this.pending = undefined;
      }
      this.qTarget = this.walker.update(t, s);
    }
    this.limiter.step(this.qTarget, h);
    return pdControl(this.gains, effort, { qCmd: this.limiter.qCmd, qMeas: s.q, qdMeas: s.qd, continuous: this.continuous }).tauApplied;
  }
}

/**
 * Sensor-only pose in the map frame from leg odometry (feet on the ground by
 * touch sensor, forward kinematics of the encoders, IMU heading), started at
 * the declared start pose. The feet roll as the legs pitch through each
 * stance, so the contact point is tracked rather than the foot center.
 * Slipping feet make it drift.
 */
export function legOdometry(start: Pose2, km: KinematicModel, legs: readonly number[]): PoseEstimator {
  const odometry = new LegOdometry(km, legs, true);
  const c = Math.cos(start.yaw), s = Math.sin(start.yaw);
  return {
    update: (input) => {
      odometry.update(input.t, input.sensors);
      return { x: start.x + c * odometry.x - s * odometry.y, y: start.y + s * odometry.x + c * odometry.y, yaw: wrapAngle(start.yaw + odometry.heading) };
    },
  };
}
