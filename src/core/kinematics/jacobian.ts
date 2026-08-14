import { symmetricEigen3 } from '../math/linalg';
import type { Mat4 } from '../math/mat4';
import { cross, type Vec3 } from '../math/vec3';
import { forwardKinematics, type FKResult, type KinematicModel } from './fk';

/**
 * Geometric position Jacobian of a limb tip in world coordinates.
 * Returns a 3 x n row-major matrix over the limb's own joints.
 */
export function positionJacobian(model: KinematicModel, limbIndex: number, fk: FKResult): Float64Array {
  const limb = model.limbs[limbIndex];
  if (!limb) throw new RangeError(`No limb ${limbIndex}`);
  const n = limb.count;
  const J = new Float64Array(3 * n);
  const tip = fk.tips[limbIndex]!;
  const p: Vec3 = [tip[12]!, tip[13]!, tip[14]!];
  for (let c = 0; c < n; c++) {
    const gi = limb.start + c;
    const j = model.joints[gi]!;
    const T: Mat4 = fk.links[gi]!;
    // The axis is invariant under its own motion, so the link frame gives the world axis.
    const a: Vec3 = [
      T[0]! * j.axis[0] + T[4]! * j.axis[1] + T[8]! * j.axis[2],
      T[1]! * j.axis[0] + T[5]! * j.axis[1] + T[9]! * j.axis[2],
      T[2]! * j.axis[0] + T[6]! * j.axis[1] + T[10]! * j.axis[2],
    ];
    let col: Vec3;
    if (j.type === 'prismatic') {
      col = a;
    } else {
      col = cross(a, [p[0] - T[12]!, p[1] - T[13]!, p[2] - T[14]!]);
    }
    J[c] = col[0];
    J[n + c] = col[1];
    J[2 * n + c] = col[2];
  }
  return J;
}

export function positionJacobianAt(model: KinematicModel, limbIndex: number, q: ArrayLike<number>, T_W_B?: Mat4): Float64Array {
  return positionJacobian(model, limbIndex, forwardKinematics(model, q, T_W_B));
}

export interface Manipulability {
  /** Singular values of J, descending. */
  singularValues: [number, number, number];
  /** Principal directions (world frame) matching the singular values. */
  directions: [Vec3, Vec3, Vec3];
  /** Yoshikawa measure sqrt(det(J J^T)). */
  measure: number;
  /** sigma_min / sigma_max, 0 at a singularity. */
  inverseCondition: number;
}

export function manipulability(J: Float64Array, n: number): Manipulability {
  const A = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += J[r * n + k]! * J[c * n + k]!;
      A[r * 3 + c] = s;
    }
  }
  const { values, vectors } = symmetricEigen3(A);
  const sv = values.map((v) => Math.sqrt(Math.max(0, v))) as [number, number, number];
  return {
    singularValues: sv,
    directions: vectors,
    measure: sv[0] * sv[1] * sv[2],
    inverseCondition: sv[0] > 0 ? sv[2] / sv[0] : 0,
  };
}
