import { clampToLimits, type KinematicModel } from '../core/kinematics/fk';
import { quatToRotation } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import type { RobotDescription } from '../core/model/types';
import { Mlp } from '../learning/mlp';
import type { PolicyJson, SkillController } from './controllers';
import type { FootLiftObservation } from './footLift';
import { hasFoot, KeyframeController, motionDuration, type KeyframeMotion } from './taught';

/**
 * A learned feedback policy in the joints of a keyframe skill (D30): the
 * keyframes only sketch the motion (a crouch and a jump, rearing up onto the
 * hind legs), and a network trained by reinforcement learning in randomized,
 * pushed simulations moves every joint target at every control step. It
 * reads sensors only: the gravity direction and angular rate from the IMU,
 * the foot contacts, the joint encoders against the sketch and the progress
 * of the motion. Open-loop keyframes cannot hold an unstable pose such as
 * standing on two legs; the policy reacts to how the body actually moves.
 */

/** Largest change the policy makes to a joint target, rad. */
export const POSE_RESIDUAL_RAD = 0.5;
export const POSE_POLICY_OBSERVATION_VERSION = 4;

/** Limbs whose foot contacts the policy reads, in limb order. */
export function policyFeet(robot: RobotDescription): number[] {
  return robot.limbs.map((l, i) => (hasFoot(l) ? i : -1)).filter((i) => i >= 0);
}

/** Observation v4: gravity direction x and y in the body frame (2), gyro (3), one contact flag per foot, motion progress and its sine (2), each joint's encoder minus the sketch's target (one per joint). */
export function posePolicyInputs(robot: RobotDescription): number {
  return 7 + policyFeet(robot).length + robot.limbs.reduce((n, l) => n + l.joints.length, 0);
}

/** The policy's observation at time t of a motion lasting durationS, with the sketch's joint targets at t. */
export function posePolicyObservation(t: number, obs: FootLiftObservation, durationS: number, feet: readonly number[], sketch: readonly number[], out: Float64Array): Float64Array {
  const R = quatToRotation(obs.imuQuat);
  const w: Vec3 = obs.gyro ?? [0, 0, 0];
  const p = Math.min(1, Math.max(0, t / Math.max(durationS, 1e-6)));
  out[0] = R[6]!;
  out[1] = R[7]!;
  out[2] = w[0];
  out[3] = w[1];
  out[4] = w[2];
  feet.forEach((li, k) => (out[5 + k] = (obs.touch[li] ?? 0) > 1 ? 1 : 0));
  out[5 + feet.length] = p;
  out[6 + feet.length] = Math.sin(Math.PI * p);
  sketch.forEach((q, j) => (out[7 + feet.length + j] = (obs.q[j] ?? q) - q));
  return out;
}

/** Why a policy cannot correct this robot's keyframe skill, or null. */
export function posePolicyProblem(policy: PolicyJson, robot: RobotDescription): string | null {
  const inputs = posePolicyInputs(robot);
  const joints = robot.limbs.reduce((n, l) => n + l.joints.length, 0);
  const l = policy.layers;
  if (l[0] !== inputs || l[l.length - 1] !== joints) return `the policy maps ${l[0]} inputs to ${l[l.length - 1]} outputs; a keyframe policy on ${robot.name} needs ${inputs} inputs and ${joints} outputs.`;
  if (policy.inputMean.length !== inputs || policy.inputStd.length !== inputs || !policy.inputStd.every((s) => s > 0)) return 'the policy normalization does not match its inputs.';
  return null;
}

export function posePolicyWithoutFeedback(policy: PolicyJson, robot: RobotDescription): PolicyJson {
  const problem = posePolicyProblem(policy, robot);
  if (problem) throw new RangeError(problem);
  Mlp.fromJSON({ sizes: policy.layers, activation: policy.activation, params: policy.weights });
  const weights = policy.weights.slice();
  const inputs = posePolicyInputs(robot);
  const progressIndex = 5 + policyFeet(robot).length;
  for (let output = 0; output < policy.layers[1]!; output++) {
    for (let input = 0; input < inputs; input++) {
      if (input !== progressIndex && input !== progressIndex + 1) weights[output * inputs + input] = 0;
    }
  }
  return { ...policy, layers: policy.layers.slice(), weights, inputMean: policy.inputMean.slice(), inputStd: policy.inputStd.slice() };
}

/** The keyframes' targets plus the policy's bounded correction; a zero policy is the keyframes exactly. */
export class LearnedPoseController implements SkillController {
  private readonly reference: KeyframeController;
  private readonly net: Mlp;
  private readonly feet: number[];
  private readonly x: Float64Array;
  /** The last raw observation (before normalization). */
  readonly observation: Float64Array;
  /** The last policy output clipped to [-1, 1] (zero before the first step and after a fault). */
  readonly action: Float64Array;
  readonly durationS: number;
  private fault: string | null = null;

  constructor(private readonly km: KinematicModel, stanceQ: number[], motion: KeyframeMotion, private readonly policy: PolicyJson) {
    const robot = km.description;
    const problem = posePolicyProblem(policy, robot);
    if (problem) throw new RangeError(problem.charAt(0).toUpperCase() + problem.slice(1));
    this.reference = new KeyframeController(km, stanceQ, motion);
    this.net = Mlp.fromJSON({ sizes: policy.layers, activation: policy.activation, params: policy.weights });
    this.feet = policyFeet(robot);
    this.observation = new Float64Array(posePolicyInputs(robot));
    this.x = new Float64Array(this.observation.length);
    this.action = new Float64Array(stanceQ.length);
    this.durationS = motionDuration(motion);
  }

  get phase(): string {
    if (this.reference.finished) return 'done';
    return this.fault ? `keyframes only (${this.fault})` : `${this.reference.phase} with learned feedback`;
  }

  get finished(): boolean {
    return this.reference.finished;
  }

  update(t: number, obs: FootLiftObservation): number[] {
    const base = this.reference.update(t);
    posePolicyObservation(t, obs, this.durationS, this.feet, base, this.observation);
    if (this.fault || this.reference.finished) return base;
    for (let i = 0; i < this.x.length; i++) this.x[i] = (this.observation[i]! - this.policy.inputMean[i]!) / this.policy.inputStd[i]!;
    const y = this.net.forward(this.x);
    if (!y.every(Number.isFinite)) {
      // A broken policy must not move the joints; the keyframes carry on alone and the phase says so.
      this.fault = 'the policy output was not finite';
      this.action.fill(0);
      return base;
    }
    for (let i = 0; i < base.length; i++) {
      this.action[i] = Math.max(-1, Math.min(1, y[i]!));
      base[i] = base[i]! + POSE_RESIDUAL_RAD * this.action[i]!;
    }
    return clampToLimits(this.km, base);
  }
}
