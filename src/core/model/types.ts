import type { Vec3 } from '../math/vec3';

/**
 * Mass model shared by the MJCF generator and core dynamics: each joint's
 * link is a capsule of `link.mass` from the joint to the next joint (or tip),
 * a sphere when that length is zero, a wheel cylinder centred on the joint
 * for wheel limbs, plus a foot sphere of FOOT_MASS at the tip of non-wheel
 * limbs whose tip radius is positive.
 */
export const FOOT_MASS = 0.02;

export type JointType = 'revolute' | 'prismatic' | 'continuous';
export type LimbRole = 'arm' | 'leg' | 'wheel';

export interface Pose {
  xyz: Vec3;
  /** Roll, pitch, yaw in radians: R = Rz(yaw) * Ry(pitch) * Rx(roll). */
  rpy: Vec3;
}

export interface JointLimit {
  /** rad or m. Ignored for continuous joints. */
  lower: number;
  upper: number;
  /** Maximum actuator effort, N*m (revolute) or N (prismatic). */
  effort: number;
  /** Maximum speed, rad/s or m/s. */
  velocity: number;
}

export interface JointDescription {
  id: string;
  name: string;
  type: JointType;
  /** Fixed transform from the parent link frame (or base frame) to this joint frame. */
  origin: Pose;
  /** Unit axis expressed in the joint frame. */
  axis: Vec3;
  limit: JointLimit;
  /** Capsule from this joint to the next joint origin (or the tip). */
  link: { radius: number; mass: number };
  /** Passive viscous damping, N*m*s/rad. */
  damping: number;
  /** Reflected rotor inertia, kg*m^2. */
  armature: number;
}

export interface LimbDescription {
  id: string;
  name: string;
  role: LimbRole;
  joints: JointDescription[];
  /** Tip point in the last link frame, with contact sphere radius (m). */
  tip: { xyz: Vec3; radius: number };
  /** Present only for wheel limbs: wheel cylinder about the joint axis. */
  wheel?: { radius: number; width: number };
}

export interface BaseDescription {
  kind: 'fixed' | 'floating';
  /** Box full extents, m. */
  size: Vec3;
  mass: number;
  /** Initial T_W_B. */
  pose: Pose;
}

export interface RobotDescription {
  schemaVersion: 1;
  id: string;
  name: string;
  base: BaseDescription;
  limbs: LimbDescription[];
  /**
   * Optional home joint positions keyed by joint id (rad or m). Joints not
   * listed use a heuristic. Not part of the morphology hash.
   */
  stance?: Record<string, number>;
}

export interface ValidationIssue {
  path: string;
  message: string;
}
