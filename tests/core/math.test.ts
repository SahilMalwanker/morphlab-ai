import { describe, expect, it } from 'vitest';
import { choleskySolve, symmetricEigen3 } from '../../src/core/math/linalg';
import { axisAngle, fromQuatPos, fromXyzRpy, invertRigid, multiply, quatFromMat4, rpyToRotation, transformPoint } from '../../src/core/math/mat4';
import { mulberry32 } from '../../src/core/math/rng';

describe('mat4', () => {
  it('stores translation at indices 12..14 (column-major)', () => {
    const T = fromXyzRpy([1, 2, 3], [0, 0, 0]);
    expect([T[12], T[13], T[14], T[15]]).toEqual([1, 2, 3, 1]);
  });

  it('rpy yaw-only equals the closed-form Rz', () => {
    const t = 0.7;
    const R = rpyToRotation([0, 0, t]);
    const expected = [Math.cos(t), -Math.sin(t), 0, Math.sin(t), Math.cos(t), 0, 0, 0, 1];
    R.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 14));
  });

  it('rpy equals Rz(yaw) * Ry(pitch) * Rx(roll) composed about fixed axes', () => {
    const [r, p, y] = [0.3, -0.5, 1.1];
    const composed = multiply(multiply(axisAngle([0, 0, 1], y), axisAngle([0, 1, 0], p)), axisAngle([1, 0, 0], r));
    const T = fromXyzRpy([0, 0, 0], [r, p, y]);
    for (let i = 0; i < 16; i++) expect(T[i]).toBeCloseTo(composed[i]!, 14);
  });

  it('rotation about +Y by +90 degrees maps +Z to +X', () => {
    const p = transformPoint(axisAngle([0, 1, 0], Math.PI / 2), [0, 0, 1]);
    expect(p[0]).toBeCloseTo(1, 14);
    expect(p[2]).toBeCloseTo(0, 14);
  });

  it('quaternion round-trips through a rotation matrix', () => {
    const rng = mulberry32(3);
    for (let k = 0; k < 50; k++) {
      const T = fromXyzRpy([rng(), rng(), rng()], [6 * rng() - 3, 3 * rng() - 1.5, 6 * rng() - 3]);
      const q = quatFromMat4(T);
      const T2 = fromQuatPos(q, [T[12]!, T[13]!, T[14]!]);
      for (let i = 0; i < 16; i++) expect(T2[i]).toBeCloseTo(T[i]!, 12);
    }
  });

  it('invertRigid gives the identity when composed', () => {
    const T = fromXyzRpy([0.3, -1, 2], [0.4, 0.2, -1.3]);
    const I = multiply(T, invertRigid(T));
    for (let i = 0; i < 16; i++) expect(I[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 14);
  });
});

describe('linalg', () => {
  it('Cholesky solves an SPD system', () => {
    const A = [4, 1, 0.5, 1, 3, 0.2, 0.5, 0.2, 2];
    const x = choleskySolve(A, [1, 2, 3], 3)!;
    const b = [0, 1, 2].map((r) => A[r * 3]! * x[0]! + A[r * 3 + 1]! * x[1]! + A[r * 3 + 2]! * x[2]!);
    b.forEach((v, i) => expect(v).toBeCloseTo([1, 2, 3][i]!, 12));
  });

  it('Cholesky rejects a matrix that is not positive definite', () => {
    expect(choleskySolve([1, 2, 2, 1], [1, 1], 2)).toBeNull();
  });

  it('symmetric eigen-decomposition reconstructs the matrix', () => {
    const A = [2, -1, 0.3, -1, 3, 0.5, 0.3, 0.5, 1];
    const { values, vectors } = symmetricEigen3(A);
    expect(values[0]).toBeGreaterThanOrEqual(values[1]);
    expect(values[1]).toBeGreaterThanOrEqual(values[2]);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += values[k]! * vectors[k]![r]! * vectors[k]![c]!;
        expect(s).toBeCloseTo(A[r * 3 + c]!, 12);
      }
    }
  });
});
