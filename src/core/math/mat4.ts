import type { Vec3 } from './vec3';

/**
 * 4x4 homogeneous transform, column-vector convention, stored column-major
 * (element (r, c) at index c * 4 + r), identical to THREE.Matrix4.elements.
 */
export type Mat4 = Float64Array;

/** Quaternion in MuJoCo order [w, x, y, z]. */
export type Quat = [number, number, number, number];

export function identity(): Mat4 {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function clone(a: Mat4): Mat4 {
  return new Float64Array(a);
}

/** out = a * b. Safe when out aliases a or b. */
export function multiply(a: Mat4, b: Mat4, out: Mat4 = new Float64Array(16)): Mat4 {
  const r = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row]! * b[c * 4 + k]!;
      r[c * 4 + row] = s;
    }
  }
  out.set(r);
  return out;
}

/** Builds a transform from a row-major 3x3 rotation and a translation. */
export function fromRotationTranslation(R: ArrayLike<number>, t: Readonly<Vec3>): Mat4 {
  const m = identity();
  m[0] = R[0]!; m[4] = R[1]!; m[8] = R[2]!;
  m[1] = R[3]!; m[5] = R[4]!; m[9] = R[5]!;
  m[2] = R[6]!; m[6] = R[7]!; m[10] = R[8]!;
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2];
  return m;
}

export function fromTranslation(t: Readonly<Vec3>): Mat4 {
  const m = identity();
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2];
  return m;
}

/** URDF roll-pitch-yaw: R = Rz(yaw) * Ry(pitch) * Rx(roll). */
export function rpyToRotation(rpy: Readonly<Vec3>): number[] {
  const [r, p, y] = rpy;
  const cr = Math.cos(r), sr = Math.sin(r);
  const cp = Math.cos(p), sp = Math.sin(p);
  const cy = Math.cos(y), sy = Math.sin(y);
  return [
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
    -sp, cp * sr, cp * cr,
  ];
}

export function fromXyzRpy(xyz: Readonly<Vec3>, rpy: Readonly<Vec3>): Mat4 {
  return fromRotationTranslation(rpyToRotation(rpy), xyz);
}

/** Rotation by angle about a unit axis (Rodrigues formula). */
export function axisAngle(axis: Readonly<Vec3>, angle: number): Mat4 {
  const [x, y, z] = axis;
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return fromRotationTranslation(
    [
      c + x * x * t, x * y * t - z * s, x * z * t + y * s,
      y * x * t + z * s, c + y * y * t, y * z * t - x * s,
      z * x * t - y * s, z * y * t + x * s, c + z * z * t,
    ],
    [0, 0, 0],
  );
}

export function transformPoint(T: Mat4, p: Readonly<Vec3>): Vec3 {
  return [
    T[0]! * p[0] + T[4]! * p[1] + T[8]! * p[2] + T[12]!,
    T[1]! * p[0] + T[5]! * p[1] + T[9]! * p[2] + T[13]!,
    T[2]! * p[0] + T[6]! * p[1] + T[10]! * p[2] + T[14]!,
  ];
}

export function transformDirection(T: Mat4, v: Readonly<Vec3>): Vec3 {
  return [
    T[0]! * v[0] + T[4]! * v[1] + T[8]! * v[2],
    T[1]! * v[0] + T[5]! * v[1] + T[9]! * v[2],
    T[2]! * v[0] + T[6]! * v[1] + T[10]! * v[2],
  ];
}

export function getTranslation(T: Mat4): Vec3 {
  return [T[12]!, T[13]!, T[14]!];
}

/** Inverse of a rigid transform (rotation + translation). */
export function invertRigid(T: Mat4): Mat4 {
  const m = identity();
  // R^T
  m[0] = T[0]!; m[4] = T[1]!; m[8] = T[2]!;
  m[1] = T[4]!; m[5] = T[5]!; m[9] = T[6]!;
  m[2] = T[8]!; m[6] = T[9]!; m[10] = T[10]!;
  const t: Vec3 = [T[12]!, T[13]!, T[14]!];
  m[12] = -(m[0]! * t[0] + m[4]! * t[1] + m[8]! * t[2]);
  m[13] = -(m[1]! * t[0] + m[5]! * t[1] + m[9]! * t[2]);
  m[14] = -(m[2]! * t[0] + m[6]! * t[1] + m[10]! * t[2]);
  return m;
}

export function isFiniteMat4(T: ArrayLike<number>): boolean {
  if (T.length !== 16) return false;
  for (let i = 0; i < 16; i++) if (!Number.isFinite(T[i])) return false;
  return true;
}

export function quatToRotation(q: Readonly<Quat>): number[] {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(n > 1e-12)) throw new RangeError('Zero-norm quaternion');
  const w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

export function fromQuatPos(q: Readonly<Quat>, p: Readonly<Vec3>): Mat4 {
  return fromRotationTranslation(quatToRotation(q), p);
}

/** Rotation part of T as a unit quaternion [w, x, y, z] with w >= 0. */
export function quatFromMat4(T: Mat4): Quat {
  const m00 = T[0]!, m11 = T[5]!, m22 = T[10]!;
  const m01 = T[4]!, m02 = T[8]!, m10 = T[1]!, m12 = T[9]!, m20 = T[2]!, m21 = T[6]!;
  const tr = m00 + m11 + m22;
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z);
  const sign = w < 0 ? -1 : 1;
  return [(sign * w) / n, (sign * x) / n, (sign * y) / n, (sign * z) / n];
}

export function rpyToQuat(rpy: Readonly<Vec3>): Quat {
  return quatFromMat4(fromXyzRpy([0, 0, 0], rpy));
}
