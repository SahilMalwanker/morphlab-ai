import { axisAngle, fromTranslation, fromXyzRpy, identity, isFiniteMat4, multiply, type Mat4 } from '../math/mat4';
import { normalize, scale, type Vec3 } from '../math/vec3';
import type { JointType, RobotDescription } from '../model/types';
import { assertValidRobot } from '../model/validate';

export interface KinematicJoint {
  id: string;
  limb: number;
  type: JointType;
  /** Fixed origin transform from the parent frame to the joint frame. */
  T_parent_joint: Mat4;
  axis: Vec3;
  lower: number;
  upper: number;
}

export interface KinematicLimb {
  id: string;
  /** Index of the first joint of this limb in the global joint vector. */
  start: number;
  count: number;
  T_last_tip: Mat4;
  /** Upper bound on the distance from the first joint origin to the tip. */
  reachBound: number;
}

/** Precomputed, validated kinematic structure derived from a robot description. */
export interface KinematicModel {
  description: RobotDescription;
  dof: number;
  joints: KinematicJoint[];
  limbs: KinematicLimb[];
}

export function buildKinematicModel(desc: RobotDescription): KinematicModel {
  assertValidRobot(desc);
  const joints: KinematicJoint[] = [];
  const limbs: KinematicLimb[] = [];
  desc.limbs.forEach((limb, li) => {
    const start = joints.length;
    let reach = Math.hypot(...limb.tip.xyz);
    limb.joints.forEach((j, ji) => {
      if (ji > 0) reach += Math.hypot(...j.origin.xyz);
      if (j.type === 'prismatic') reach += Math.max(Math.abs(j.limit.lower), Math.abs(j.limit.upper));
      joints.push({
        id: j.id,
        limb: li,
        type: j.type,
        T_parent_joint: fromXyzRpy(j.origin.xyz, j.origin.rpy),
        axis: normalize(j.axis),
        lower: j.type === 'continuous' ? -Infinity : j.limit.lower,
        upper: j.type === 'continuous' ? Infinity : j.limit.upper,
      });
    });
    limbs.push({ id: limb.id, start, count: limb.joints.length, T_last_tip: fromTranslation(limb.tip.xyz), reachBound: reach });
  });
  return { description: desc, dof: joints.length, joints, limbs };
}

/** Joint motion transform M(q): rotation about the axis or translation along it. */
export function jointMotion(j: KinematicJoint, q: number): Mat4 {
  return j.type === 'prismatic' ? fromTranslation(scale(j.axis, q)) : axisAngle(j.axis, q);
}

export interface FKResult {
  /** T_W_Li for every joint i (link frame after joint motion), in global joint order. */
  links: Mat4[];
  /** T_W_E for every limb tip. */
  tips: Mat4[];
  base: Mat4;
}

export function assertJointVector(model: KinematicModel, q: ArrayLike<number>): void {
  if (q.length !== model.dof) throw new RangeError(`Expected ${model.dof} joint values, got ${q.length}`);
  for (let i = 0; i < q.length; i++) {
    if (!Number.isFinite(q[i])) throw new RangeError(`Joint value ${i} is not finite`);
  }
}

/** Forward kinematics: T_W_Li = T_W_B * prod_k (T_parent_Jk * M_k(q_k)). */
export function forwardKinematics(model: KinematicModel, q: ArrayLike<number>, T_W_B: Mat4 = baseTransform(model)): FKResult {
  assertJointVector(model, q);
  if (!isFiniteMat4(T_W_B)) throw new RangeError('Base transform is not finite');
  const links: Mat4[] = new Array<Mat4>(model.dof);
  const tips: Mat4[] = [];
  for (const limb of model.limbs) {
    let T = T_W_B;
    for (let i = limb.start; i < limb.start + limb.count; i++) {
      const j = model.joints[i]!;
      T = multiply(multiply(T, j.T_parent_joint), jointMotion(j, q[i]!));
      links[i] = T;
    }
    tips.push(multiply(T, limb.T_last_tip));
  }
  return { links, tips, base: T_W_B };
}

export function baseTransform(model: KinematicModel): Mat4 {
  const p = model.description.base.pose;
  return fromXyzRpy(p.xyz, p.rpy);
}

export function tipPosition(model: KinematicModel, limb: number, q: ArrayLike<number>, T_W_B?: Mat4): Vec3 {
  const T = forwardKinematics(model, q, T_W_B).tips[limb]!;
  return [T[12]!, T[13]!, T[14]!];
}

export function clampToLimits(model: KinematicModel, q: ArrayLike<number>): number[] {
  return Array.from(q, (v, i) => Math.min(model.joints[i]!.upper, Math.max(model.joints[i]!.lower, v)));
}

export function zeroConfiguration(model: KinematicModel): number[] {
  return clampToLimits(model, new Array<number>(model.dof).fill(0));
}

/** Uniform random configuration within limits (continuous joints in [-pi, pi]). */
export function randomConfiguration(model: KinematicModel, rng: () => number): number[] {
  return model.joints.map((j) => {
    const lo = Number.isFinite(j.lower) ? j.lower : -Math.PI;
    const hi = Number.isFinite(j.upper) ? j.upper : Math.PI;
    return lo + (hi - lo) * rng();
  });
}

export { identity };
