import { clampToLimits, type KinematicModel } from '../core/kinematics/fk';
import { quatToRotation } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import { Mlp } from '../learning/mlp';
import type { PolicyJson, SkillController } from './controllers';
import type { FootLiftObservation } from './footLift';
import { GaitController, gaitEnvelope, type GaitParams } from './gait';

/**
 * A learned feedback policy in the joints of a gait (D27, after Policies
 * Modulating Trajectory Generators, Iscen et al. 2018): the gait generator
 * gives joint targets, and a network trained by random search adds a bounded
 * correction to every joint at every control step. It reads sensors only:
 * the gravity direction and angular rate from the IMU, the gait phase and the
 * commanded velocity. A zero policy reproduces the gait exactly; the
 * correction fades in and out with the steps.
 */

/** Observation v1: gravity direction x and y in the body frame (2), gyro (3), gait phase and its second harmonic as sine and cosine (4), command (3). */
export const GAIT_POLICY_INPUTS = 12;
export const GAIT_POLICY_OBSERVATION_VERSION = 1;
/** Largest correction the policy adds to a joint target, rad. */
export const RESIDUAL_SCALE_RAD = 0.15;

/** Commanded body velocity of a gait at full stride: forward and leftward m/s, yaw rate rad/s (positive turns left). */
export function gaitCommand(g: GaitParams): [number, number, number] {
  return [g.strideM / g.cycleS, g.lateralM / g.cycleS, g.turnRad / g.cycleS];
}

/** The policy's observation at time t since the gait started. */
export function gaitPolicyObservation(t: number, obs: FootLiftObservation, g: GaitParams, out: Float64Array = new Float64Array(GAIT_POLICY_INPUTS)): Float64Array {
  const R = quatToRotation(obs.imuQuat);
  const w: Vec3 = obs.gyro ?? [0, 0, 0];
  const phi = (2 * Math.PI * t) / g.cycleS;
  const c = gaitCommand(g);
  out[0] = R[6]!;
  out[1] = R[7]!;
  out[2] = w[0];
  out[3] = w[1];
  out[4] = w[2];
  out[5] = Math.sin(phi);
  out[6] = Math.cos(phi);
  out[7] = Math.sin(2 * phi);
  out[8] = Math.cos(2 * phi);
  out[9] = c[0];
  out[10] = c[1];
  out[11] = c[2];
  return out;
}

/** Why a policy cannot drive this robot's joints, or null. */
export function gaitPolicyProblem(policy: PolicyJson, joints: number): string | null {
  const l = policy.layers;
  if (l[0] !== GAIT_POLICY_INPUTS || l[l.length - 1] !== joints) return `the policy maps ${l[0]} inputs to ${l[l.length - 1]} outputs; a learned gait on this robot needs ${GAIT_POLICY_INPUTS} inputs and ${joints} outputs.`;
  if (policy.inputMean.length !== GAIT_POLICY_INPUTS || policy.inputStd.length !== GAIT_POLICY_INPUTS || !policy.inputStd.every((s) => s > 0)) return 'the policy normalization does not match its inputs.';
  return null;
}

/** The gait generator's targets plus the policy's bounded correction. */
export class LearnedGaitController implements SkillController {
  private readonly tg: GaitController;
  private readonly net: Mlp;
  private readonly x = new Float64Array(GAIT_POLICY_INPUTS);
  private fault: string | null = null;
  /** The last policy output, clipped to [-1, 1] (zero before the first step). */
  readonly action: Float64Array;

  constructor(private readonly km: KinematicModel, stanceQ: number[], private readonly gait: GaitParams, private readonly policy: PolicyJson) {
    const problem = gaitPolicyProblem(policy, stanceQ.length);
    if (problem) throw new RangeError(problem.charAt(0).toUpperCase() + problem.slice(1));
    this.tg = new GaitController(km, stanceQ, gait);
    this.net = Mlp.fromJSON({ sizes: policy.layers, activation: policy.activation, params: policy.weights });
    this.action = new Float64Array(stanceQ.length);
  }

  get phase(): string {
    if (this.fault) return `gait generator only (${this.fault})`;
    return this.tg.finished ? 'done' : `${this.tg.phase} with learned feedback`;
  }

  get finished(): boolean {
    return this.tg.finished;
  }

  update(t: number, obs: FootLiftObservation): number[] {
    const base = this.tg.update(t, obs);
    if (this.fault || this.tg.finished) return base;
    const x = gaitPolicyObservation(t, obs, this.gait, this.x);
    for (let i = 0; i < x.length; i++) x[i] = (x[i]! - this.policy.inputMean[i]!) / this.policy.inputStd[i]!;
    const y = this.net.forward(x);
    if (!y.every(Number.isFinite)) {
      // A broken policy must not move the joints; the gait carries on alone and the phase says so.
      this.fault = 'the policy output was not finite';
      this.action.fill(0);
      return base;
    }
    const a = gaitEnvelope(this.gait, t);
    for (let i = 0; i < base.length; i++) {
      this.action[i] = Math.max(-1, Math.min(1, y[i]!));
      base[i] = base[i]! + a * RESIDUAL_SCALE_RAD * this.action[i]!;
    }
    return clampToLimits(this.km, base);
  }
}
