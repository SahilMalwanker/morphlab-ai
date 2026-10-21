import { clampToLimits, type KinematicModel } from '../core/kinematics/fk';
import { quatToRotation } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import { Mlp } from '../learning/mlp';
import type { PolicyJson, SkillController } from './controllers';
import type { FootLiftObservation } from './footLift';
import { GaitController, gaitEnvelope, legIndices, type GaitParams } from './gait';
import { LegOdometry } from './odometry';
import { RESIDUAL_SCALE_RAD } from './policyGait';

/**
 * The learned feedback policy of a walking controller (D29). Like the gait
 * policy of D27 it adds bounded corrections to every joint target of the gait
 * generator, and it also changes the generator's stride, side step and turn
 * per cycle, the way Policies Modulating Trajectory Generators (Iscen et al.
 * 2018) modulate the generator's parameters. It reads sensors only: the IMU,
 * the gait phase, the walking command, and how far a sensor-only estimate of
 * the body's position and heading (LegOdometry) is from where the command
 * leads. A missing policy runs the generator alone and still computes the
 * observation, for the input statistics.
 */

/**
 * Observation v2: gravity direction x and y in the body frame (2), gyro (3),
 * gait phase and its second harmonic as sine and cosine (4), walking command
 * (3), heading error (1), position error in the body frame (2), velocity
 * error in the body frame (2), stepping envelope (1).
 */
export const WALK_POLICY_INPUTS = 18;
export const WALK_POLICY_OBSERVATION_VERSION = 2;
/** The last outputs: changes of stride, side step and turn per cycle (after one correction per joint, when the policy has them). */
export const WALK_COMMAND_OUTPUTS = 3;
/** Largest change the policy makes to the stride (m), side step (m) and turn (rad) per cycle. */
export const WALK_COMMAND_SCALE: Vec3 = [0.05, 0.03, 0.15];
/** Time constant of the low-pass filter on the command changes, s. */
export const WALK_COMMAND_TAU_S = 0.1;

/** Why a policy cannot be this robot's walking policy, or null. It changes the command only (3 outputs) or also corrects every joint (joints + 3). */
export function walkPolicyProblem(policy: PolicyJson, joints: number): string | null {
  const l = policy.layers;
  const out = l[l.length - 1];
  if (l[0] !== WALK_POLICY_INPUTS || (out !== WALK_COMMAND_OUTPUTS && out !== joints + WALK_COMMAND_OUTPUTS)) {
    return `the policy maps ${l[0]} inputs to ${out} outputs; a walking policy on this robot needs ${WALK_POLICY_INPUTS} inputs and ${WALK_COMMAND_OUTPUTS} or ${joints + WALK_COMMAND_OUTPUTS} outputs.`;
  }
  if (policy.inputMean.length !== WALK_POLICY_INPUTS || policy.inputStd.length !== WALK_POLICY_INPUTS || !policy.inputStd.every((s) => s > 0)) return 'the policy normalization does not match its inputs.';
  return null;
}

/** Walks a command with the gait generator, corrected by a learned policy when one is given. */
export class WalkPolicyController implements SkillController {
  private readonly tg: GaitController;
  private readonly net: Mlp | null;
  private readonly odometry: LegOdometry;
  private readonly x = new Float64Array(WALK_POLICY_INPUTS);
  /** The last raw observation (before normalization). */
  readonly observation = new Float64Array(WALK_POLICY_INPUTS);
  /** The last policy output clipped to [-1, 1]: joint corrections (when the policy has them), then the command changes; zero without a policy. */
  readonly action: Float64Array;
  private readonly delta: Vec3 = [0, 0, 0];
  private readonly path: Vec3 = [0, 0, 0];
  private tPrev = 0;
  private fault: string | null = null;

  constructor(private readonly km: KinematicModel, stanceQ: number[], private readonly gait: GaitParams, private readonly command: Readonly<Vec3>, private readonly policy: PolicyJson | null) {
    if (policy) {
      const problem = walkPolicyProblem(policy, stanceQ.length);
      if (problem) throw new RangeError(problem.charAt(0).toUpperCase() + problem.slice(1));
    }
    this.tg = new GaitController(km, stanceQ, gait);
    this.net = policy ? Mlp.fromJSON({ sizes: policy.layers, activation: policy.activation, params: policy.weights }) : null;
    this.odometry = new LegOdometry(km, legIndices(km.description));
    this.action = new Float64Array(this.net ? this.net.outputSize : WALK_COMMAND_OUTPUTS);
  }

  get phase(): string {
    if (this.tg.finished) return 'done';
    if (!this.net) return this.tg.phase;
    return this.fault ? `gait generator only (${this.fault})` : `${this.tg.phase} with learned feedback`;
  }

  get finished(): boolean {
    return this.tg.finished;
  }

  /** The sensor-only estimate: forward and left of the start, m, and heading change, rad. */
  get estimate(): Vec3 {
    return this.odometry.pose;
  }

  update(t: number, obs: FootLiftObservation): number[] {
    const dt = Math.max(0, t - this.tPrev);
    this.tPrev = t;
    this.odometry.update(t, obs);
    this.advancePath(t, dt);
    this.observe(t, obs);
    const corrections = this.action.length - WALK_COMMAND_OUTPUTS;
    const p = this.policy;
    if (this.net && p && !this.fault && !this.tg.finished) {
      for (let i = 0; i < WALK_POLICY_INPUTS; i++) this.x[i] = (this.observation[i]! - p.inputMean[i]!) / p.inputStd[i]!;
      const y = this.net.forward(this.x);
      if (y.every(Number.isFinite)) {
        for (let i = 0; i < y.length; i++) this.action[i] = Math.max(-1, Math.min(1, y[i]!));
        const alpha = dt / (WALK_COMMAND_TAU_S + dt);
        for (let k = 0; k < WALK_COMMAND_OUTPUTS; k++) this.delta[k] = this.delta[k]! + alpha * (this.action[corrections + k]! - this.delta[k]!);
        const g = this.gait;
        this.tg.setMotion(g.strideM + WALK_COMMAND_SCALE[0] * this.delta[0], g.lateralM + WALK_COMMAND_SCALE[1] * this.delta[1], g.turnRad + WALK_COMMAND_SCALE[2] * this.delta[2]);
      } else {
        // A broken policy must not move the body; the generator carries on with its own command and the phase says so.
        this.fault = 'the policy output was not finite';
        this.action.fill(0);
        this.tg.setMotion(this.gait.strideM, this.gait.lateralM, this.gait.turnRad);
      }
    }
    const base = this.tg.update(t, obs);
    if (!this.net || this.fault || this.tg.finished || corrections === 0) return base;
    const a = gaitEnvelope(this.gait, t);
    for (let i = 0; i < corrections; i++) base[i] = base[i]! + a * RESIDUAL_SCALE_RAD * this.action[i]!;
    return clampToLimits(this.km, base);
  }

  /** Where the command leads by time t, integrated at interval midpoints with the stepping envelope like expectedPose. */
  private advancePath(t: number, dt: number): void {
    if (dt <= 0) return;
    const a = gaitEnvelope(this.gait, Math.min(t - 0.5 * dt, this.gait.durationS));
    const c = this.command;
    const mid = this.path[2] + 0.5 * a * c[2] * dt;
    this.path[0] += a * (c[0] * Math.cos(mid) - c[1] * Math.sin(mid)) * dt;
    this.path[1] += a * (c[0] * Math.sin(mid) + c[1] * Math.cos(mid)) * dt;
    this.path[2] += a * c[2] * dt;
  }

  private observe(t: number, obs: FootLiftObservation): void {
    walkObservation(this.observation, t, this.gait.cycleS, gaitEnvelope(this.gait, Math.min(t, this.gait.durationS)), obs, this.command, this.path, this.odometry);
  }
}

/** The leg odometry fields the observation reads. */
export interface WalkEstimate {
  x: number;
  y: number;
  heading: number;
  vx: number;
  vy: number;
}

/** Observation v2 into `o`: t is the gait clock, a the stepping envelope, path where the command leads, odo the sensor-only estimate. */
export function walkObservation(o: Float64Array, t: number, cycleS: number, a: number, obs: FootLiftObservation, command: Readonly<Vec3>, path: Readonly<Vec3>, odo: WalkEstimate): void {
  const R = quatToRotation(obs.imuQuat);
  const w: Vec3 = obs.gyro ?? [0, 0, 0];
  const phi = (2 * Math.PI * t) / cycleS;
  const ex = path[0] - odo.x, ey = path[1] - odo.y;
  const c = Math.cos(odo.heading), s = Math.sin(odo.heading);
  o[0] = R[6]!;
  o[1] = R[7]!;
  o[2] = w[0];
  o[3] = w[1];
  o[4] = w[2];
  o[5] = Math.sin(phi);
  o[6] = Math.cos(phi);
  o[7] = Math.sin(2 * phi);
  o[8] = Math.cos(2 * phi);
  o[9] = command[0];
  o[10] = command[1];
  o[11] = command[2];
  o[12] = path[2] - odo.heading;
  o[13] = c * ex + s * ey;
  o[14] = -s * ex + c * ey;
  o[15] = a * command[0] - odo.vx;
  o[16] = a * command[1] - odo.vy;
  o[17] = a;
}
