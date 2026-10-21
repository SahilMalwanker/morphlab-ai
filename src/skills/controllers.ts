import { buildKinematicModel, clampToLimits, forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { solveIK } from '../core/kinematics/ik';
import { identity, quatToRotation } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import type { RobotDescription } from '../core/model/types';
import { Mlp } from '../learning/mlp';
import { FootLiftController, type FootLiftObservation, type FootLiftParams } from './footLift';
import { GaitController, type GaitParams } from './gait';
import { LearnedGaitController } from './policyGait';
import { LearnedPoseController } from './posePolicy';
import { KeyframeController, motionDuration, type KeyframeMotion } from './taught';
import { WalkPolicyController } from './walkPolicy';

/** Common interface of every skill implementation. Inputs are sensor readings only. */
export interface SkillController {
  /** Joint position targets (rad) for time t since the skill started. */
  update(t: number, obs: FootLiftObservation): number[];
  readonly phase: string;
  readonly finished: boolean;
}

/** Time-indexed joint targets recorded from a demonstration. */
export interface Trajectory {
  /** Sample period, s. */
  dt: number;
  jointIds: string[];
  /** One joint-target vector per sample. */
  frames: number[][];
}

export interface PolicyJson {
  layers: number[];
  activation: 'tanh';
  weights: number[];
  inputMean: number[];
  inputStd: number[];
}

/** Serializable description of a controller, used across the worker boundary. */
export type SkillSpec =
  | { kind: 'state-machine' | 'tuned-parameters' | 'es-improved'; params: FootLiftParams }
  | { kind: 'replay'; trajectory: Trajectory }
  | { kind: 'imitation-mlp'; policy: PolicyJson; durationS: number }
  | { kind: 'keyframes'; motion: KeyframeMotion }
  | { kind: 'gait'; gait: GaitParams }
  /** With `walk`, the walking controller (D29) runs the policy toward that command (forward m/s, left m/s, turn rad/s). */
  | { kind: 'gait-learned'; gait: GaitParams; policy: PolicyJson; walk?: { command: Vec3 } }
  | { kind: 'keyframes-learned'; motion: KeyframeMotion; policy: PolicyJson };

export type SkillMethod = SkillSpec['kind'];

/** Plain-language label for each method, shown wherever a skill appears. */
export const METHOD_LABELS: Record<SkillMethod, string> = {
  'state-machine': 'State machine (hand-designed)',
  'tuned-parameters': 'State machine with tuned parameters',
  replay: 'Replay of a recorded demonstration (open loop)',
  'imitation-mlp': 'Imitation-learned neural policy',
  'es-improved': 'State machine improved by reward (evolution strategies)',
  keyframes: 'Keyframe motion you posed (open loop)',
  gait: 'Gait from a contact pattern (closed loop with IMU feedback, open loop without)',
  'gait-learned': 'Gait with a learned feedback policy (random search, closed loop)',
  'keyframes-learned': 'Keyframes with a learned feedback policy (reinforcement learning under pushes, closed loop)',
};

/** Open-loop playback of recorded joint targets with linear interpolation. */
export class ReplayController implements SkillController {
  private t = 0;
  constructor(private readonly traj: Trajectory) {
    if (!(traj.dt > 0) || traj.frames.length === 0) throw new RangeError('A replay needs a positive sample period and at least one frame');
  }
  get durationS(): number {
    return (this.traj.frames.length - 1) * this.traj.dt;
  }
  get phase(): string {
    return this.finished ? 'done' : 'replay';
  }
  get finished(): boolean {
    return this.t > this.durationS;
  }
  update(t: number): number[] {
    this.t = t;
    const f = this.traj.frames;
    const x = Math.min(Math.max(t / this.traj.dt, 0), f.length - 1);
    const i = Math.min(Math.floor(x), f.length - 2);
    if (i < 0) return f[0]!.slice();
    const a = x - i;
    return f[i]!.map((v, k) => v + a * (f[i + 1]![k]! - v));
  }
}

/**
 * Observation features for learned foot-lift policies, from sensors only:
 * joint offsets from the stance, IMU roll-pitch (from the gravity direction in
 * the body frame), binary foot contact, and the elapsed fraction of the skill.
 */
export function footLiftFeatures(t: number, obs: FootLiftObservation, stanceQ: readonly number[], horizonS: number): number[] {
  const R = quatToRotation(obs.imuQuat);
  // World z expressed in the body frame: third row of R (row-major).
  const gx = R[6]!, gy = R[7]!;
  const f: number[] = [];
  for (let i = 0; i < stanceQ.length; i++) f.push(obs.q[i]! - stanceQ[i]!);
  f.push(gx, gy);
  for (const c of obs.touch) f.push(Number.isFinite(c) && c > 1 ? 1 : 0);
  f.push(Math.min(1, Math.max(0, t / horizonS)));
  return f;
}

/** Neural policy: features to joint offsets from the stance, clamped to joint limits. */
export class MlpController implements SkillController {
  private readonly net: Mlp;
  private t = 0;
  constructor(private readonly km: KinematicModel, private readonly stanceQ: number[], private readonly policy: PolicyJson, readonly durationS: number) {
    this.net = Mlp.fromJSON({ sizes: policy.layers, activation: policy.activation, params: policy.weights });
    if (policy.inputMean.length !== this.net.inputSize || policy.inputStd.length !== this.net.inputSize) throw new RangeError('Policy normalization does not match its input size');
    if (this.net.outputSize !== stanceQ.length) throw new RangeError('Policy output size does not match the robot');
  }
  get phase(): string {
    return this.finished ? 'done' : 'policy';
  }
  get finished(): boolean {
    return this.t > this.durationS;
  }
  update(t: number, obs: FootLiftObservation): number[] {
    this.t = t;
    const x = footLiftFeatures(t, obs, this.stanceQ, this.durationS).map((v, i) => (v - this.policy.inputMean[i]!) / this.policy.inputStd[i]!);
    if (x.length !== this.net.inputSize) throw new RangeError('Observation size does not match the policy');
    const y = this.net.forward(x);
    return clampToLimits(this.km, this.stanceQ.map((s, i) => s + y[i]!));
  }
}

/** Total scripted duration of the state machine (without the touchdown wait). */
export function footLiftDuration(p: FootLiftParams): number {
  return p.settleS + p.shiftS + p.liftS + p.holdS + p.lowerS + p.unshiftS;
}

/** Duration a skill needs before it reports finished, s. */
export function skillDuration(spec: SkillSpec): number {
  if (spec.kind === 'replay') return (spec.trajectory.frames.length - 1) * spec.trajectory.dt;
  if (spec.kind === 'imitation-mlp') return spec.durationS;
  if (spec.kind === 'keyframes' || spec.kind === 'keyframes-learned') return motionDuration(spec.motion);
  if (spec.kind === 'gait' || spec.kind === 'gait-learned') return spec.gait.durationS;
  return footLiftDuration(spec.params) + 0.5;
}

export function makeController(km: KinematicModel, limbIndex: number, stanceQ: number[], spec: SkillSpec): SkillController {
  switch (spec.kind) {
    case 'replay':
      if (spec.trajectory.jointIds.length !== stanceQ.length || spec.trajectory.jointIds.some((id, i) => id !== km.joints[i]!.id)) {
        throw new RangeError('The recorded trajectory does not match this robot\'s joints');
      }
      return new ReplayController(spec.trajectory);
    case 'imitation-mlp':
      return new MlpController(km, stanceQ, spec.policy, spec.durationS);
    case 'keyframes':
      return new KeyframeController(km, stanceQ, spec.motion);
    case 'keyframes-learned':
      return new LearnedPoseController(km, stanceQ, spec.motion, spec.policy);
    case 'gait':
      return new GaitController(km, stanceQ, spec.gait);
    case 'gait-learned':
      return spec.walk ? new WalkPolicyController(km, stanceQ, spec.gait, spec.walk.command, spec.policy) : new LearnedGaitController(km, stanceQ, spec.gait, spec.policy);
    default: {
      const c = new FootLiftController(km, limbIndex, stanceQ, spec.params);
      return {
        update: (t, obs) => c.update(t, obs),
        get phase() { return c.currentPhase; },
        get finished() { return c.currentPhase === 'done' || c.currentPhase === 'aborted'; },
      };
    }
  }
}

/**
 * Task-space teleoperation for demonstrations: a body-shift fraction (0..1)
 * toward the support legs and a lift height for one foot, mapped to joint
 * targets with leg IK in the base frame (the same mapping the state machine
 * uses).
 */
export class FootLiftTeleop {
  private readonly km: KinematicModel;
  private readonly feet: Vec3[];
  private readonly shiftDir: Vec3;
  private targets: number[];
  constructor(desc: RobotDescription, readonly limbIndex: number, private readonly stanceQ: number[], readonly maxShiftM = 0.08) {
    this.km = buildKinematicModel(desc);
    const fk = forwardKinematics(this.km, stanceQ, identity());
    this.feet = fk.tips.map((T) => [T[12]!, T[13]!, T[14]!] as Vec3);
    const support = this.feet.filter((_, i) => i !== limbIndex);
    if (support.length < 3) throw new RangeError('Foot lift needs at least three supporting legs');
    const cx = support.reduce((s, p) => s + p[0], 0) / support.length;
    const cy = support.reduce((s, p) => s + p[1], 0) / support.length;
    const n = Math.hypot(cx, cy) || 1;
    this.shiftDir = [cx / n, cy / n, 0];
    this.targets = stanceQ.slice();
  }
  targetsFor(shiftFraction: number, liftM: number): number[] {
    const s = Math.min(1, Math.max(0, shiftFraction)) * this.maxShiftM;
    const lift = Math.min(0.15, Math.max(0, liftM));
    this.km.limbs.forEach((_, li) => {
      const nom = this.feet[li]!;
      const target: Vec3 = [nom[0] - s * this.shiftDir[0], nom[1] - s * this.shiftDir[1], nom[2] + (li === this.limbIndex ? lift : 0)];
      const res = solveIK(this.km, li, target, this.targets, { T_W_B: identity(), maxIterations: 30, tolerance: 1e-5 });
      if (res.status !== 'invalid-input') this.targets = res.q;
    });
    return this.targets.slice();
  }
  reset(): void {
    this.targets = this.stanceQ.slice();
  }
}
