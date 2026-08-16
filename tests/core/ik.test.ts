import { describe, expect, it } from 'vitest';
import { buildKinematicModel, forwardKinematics, randomConfiguration, tipPosition, zeroConfiguration } from '../../src/core/kinematics/fk';
import { solveIK } from '../../src/core/kinematics/ik';
import { mulberry32 } from '../../src/core/math/rng';
import { arm3, quadruped, quadrupedStance, serialChain } from '../../src/core/model/presets';

function withinLimits(model: ReturnType<typeof buildKinematicModel>, q: number[]): boolean {
  return q.every((v, i) => v >= model.joints[i]!.lower && v <= model.joints[i]!.upper);
}

describe('damped least squares IK', () => {
  it('reaches >= 98 % of 200 FK-generated targets with residual < 1e-4 m (arm3, restarts allowed)', () => {
    const model = buildKinematicModel(arm3());
    const rng = mulberry32(7);
    let ok = 0;
    for (let k = 0; k < 200; k++) {
      const target = tipPosition(model, 0, randomConfiguration(model, rng));
      const res = solveIK(model, 0, target, [0, 0.5, 1.0], { restarts: 4, seed: k });
      if (res.status === 'converged' && res.residual < 1e-4) ok++;
      expect(withinLimits(model, res.q)).toBe(true);
      // The reported residual is the true distance at the returned configuration.
      const p = tipPosition(model, 0, res.q);
      expect(Math.abs(Math.hypot(p[0] - target[0], p[1] - target[1], p[2] - target[2]) - res.residual)).toBeLessThan(1e-12);
    }
    expect(ok / 200).toBeGreaterThanOrEqual(0.98);
  });

  it('reaches >= 98 % of targets on a 6-joint chain', () => {
    const model = buildKinematicModel(serialChain({ axes: ['z', 'y', 'y', 'x', 'y', 'x'], lengths: [0.2, 0.3, 0.25, 0.1, 0.08, 0.06] }));
    const rng = mulberry32(21);
    let ok = 0;
    for (let k = 0; k < 200; k++) {
      const target = tipPosition(model, 0, randomConfiguration(model, rng));
      const res = solveIK(model, 0, target, zeroConfiguration(model).map(() => 0.3), { restarts: 4, seed: k });
      if (res.status === 'converged' && res.residual < 1e-4) ok++;
      expect(withinLimits(model, res.q)).toBe(true);
    }
    expect(ok / 200).toBeGreaterThanOrEqual(0.98);
  });

  it('solves a quadruped foot target without moving the other legs', () => {
    const model = buildKinematicModel(quadruped());
    const q0 = quadrupedStance();
    const foot = tipPosition(model, 0, q0);
    const res = solveIK(model, 0, [foot[0] + 0.03, foot[1], foot[2] + 0.05], q0);
    expect(res.status).toBe('converged');
    expect(res.q.slice(3)).toEqual(q0.slice(3));
    expect(withinLimits(model, res.q)).toBe(true);
  });

  it('reports a straight-line unreachable target with the gap to the reach bound', () => {
    const model = buildKinematicModel(arm3());
    // Shoulder at z = 0.18 m; reach from the shoulder is 0.65 m. Target straight up at 2x distance.
    const target: [number, number, number] = [0, 0, 0.18 + 1.3];
    const res = solveIK(model, 0, target, [0, 0.4, 0.8]);
    expect(res.status).toBe('unreachable');
    expect(res.beyondReach).toBe(true);
    expect(Math.abs(res.residual - 0.65)).toBeLessThan(1e-3);
    expect(res.q.every(Number.isFinite)).toBe(true);
    expect(withinLimits(model, res.q)).toBe(true);
  });

  it('reports targets inside the reach sphere but blocked by limits as stalled, not proven unreachable', () => {
    const model = buildKinematicModel(arm3());
    // Directly below the base: needs the shoulder beyond its 1.75 rad limit.
    const res = solveIK(model, 0, [0, 0, -0.2], [0, 0.5, 1.0], { restarts: 4 });
    expect(res.status).toBe('stalled');
    expect(res.beyondReach).toBe(false);
    expect(res.residual).toBeGreaterThan(0.01);
    expect(res.message).toContain('does not prove the target is out of reach');
    expect(withinLimits(model, res.q)).toBe(true);
  });

  it('claims unreachable only with the reach-bound proof', () => {
    const model = buildKinematicModel(arm3());
    const rng = mulberry32(11);
    let unreachable = 0;
    for (let i = 0; i < 200; i++) {
      const target: [number, number, number] = [(rng() - 0.5) * 2, (rng() - 0.5) * 2, rng() * 1.2 - 0.3];
      const res = solveIK(model, 0, target, [0, 0.4, 0.8], { restarts: 1, seed: i });
      if (res.status === 'unreachable') {
        unreachable++;
        expect(res.beyondReach).toBe(true);
      }
      if (res.beyondReach) expect(res.status).toBe('unreachable');
    }
    expect(unreachable).toBeGreaterThan(20);
  });

  it('stays finite with bounded steps near the stretched singularity', () => {
    const model = buildKinematicModel(arm3());
    // Start exactly at the singular stretched pose; target just beyond and just inside reach.
    for (const z of [0.18 + 0.6499, 0.18 + 0.66, 0.18 + 0.64]) {
      const res = solveIK(model, 0, [0.001, 0, z], [0, 0, 0], { maxStep: 0.3 });
      expect(res.q.every(Number.isFinite)).toBe(true);
      expect(Number.isFinite(res.residual)).toBe(true);
      expect(withinLimits(model, res.q)).toBe(true);
      if (z < 0.18 + 0.65) expect(res.residual).toBeLessThan(1e-4);
    }
  });

  it('rejects non-finite targets and malformed joint vectors without throwing', () => {
    const model = buildKinematicModel(arm3());
    const a = solveIK(model, 0, [Number.NaN, 0, 0.5], [0, 0.5, 1]);
    expect(a.status).toBe('invalid-input');
    expect(a.q.every(Number.isFinite)).toBe(true);
    const b = solveIK(model, 0, [0.2, 0, 0.5], [0, 0.5]);
    expect(b.status).toBe('invalid-input');
    const c = solveIK(model, 0, [0.2, 0, 0.5], [0, Number.POSITIVE_INFINITY, 1]);
    expect(c.status).toBe('invalid-input');
  });

  it('respects a joint held at its limit (active set) and still converges when possible', () => {
    const model = buildKinematicModel(arm3());
    const q0 = [0, 1.75, 0.2]; // shoulder at upper limit
    const target = tipPosition(model, 0, [0.3, 1.2, 0.9]);
    const res = solveIK(model, 0, target, q0);
    expect(res.status).toBe('converged');
    expect(withinLimits(model, res.q)).toBe(true);
    expect(forwardKinematics(model, res.q).tips[0]!.every(Number.isFinite)).toBe(true);
  });
});
