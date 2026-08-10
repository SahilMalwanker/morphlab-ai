import type { Vec3 } from './vec3';

/**
 * Solves A x = b for a symmetric positive definite n x n matrix (row-major)
 * using a Cholesky factorization. Returns null if A is not positive definite.
 */
export function choleskySolve(A: ArrayLike<number>, b: ArrayLike<number>, n: number): Float64Array | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j]!;
      for (let k = 0; k < j; k++) s -= L[i * n + k]! * L[j * n + k]!;
      if (i === j) {
        if (!(s > 0)) return null;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j]!;
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i]!;
    for (let k = 0; k < i; k++) s -= L[i * n + k]! * y[k]!;
    y[i] = s / L[i * n + i]!;
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]!;
    for (let k = i + 1; k < n; k++) s -= L[k * n + i]! * x[k]!;
    x[i] = s / L[i * n + i]!;
  }
  return x;
}

export interface SymmetricEigen3 {
  /** Eigenvalues sorted in descending order. */
  values: [number, number, number];
  /** Unit eigenvectors matching `values`. */
  vectors: [Vec3, Vec3, Vec3];
}

/** Eigen-decomposition of a symmetric 3x3 matrix (row-major) by cyclic Jacobi rotations. */
export function symmetricEigen3(Ain: ArrayLike<number>): SymmetricEigen3 {
  const a = [
    [Ain[0]!, Ain[1]!, Ain[2]!],
    [Ain[3]!, Ain[4]!, Ain[5]!],
    [Ain[6]!, Ain[7]!, Ain[8]!],
  ];
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = Math.abs(a[0]![1]!) + Math.abs(a[0]![2]!) + Math.abs(a[1]![2]!);
    if (off < 1e-15) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k]![p]!, akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p]![k]!, aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k]![p]!, vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j]![j]! - a[i]![i]!);
  const values = order.map((i) => a[i]![i]!) as [number, number, number];
  const vectors = order.map((i) => [v[0]![i]!, v[1]![i]!, v[2]![i]!] as Vec3) as [Vec3, Vec3, Vec3];
  return { values, vectors };
}
