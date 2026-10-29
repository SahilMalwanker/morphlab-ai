// Spike S3: damped least-squares IK on generated limbs with joint limits.
import { expect, it } from 'vitest';
import { buildKinematicModel, randomConfiguration, tipPosition, type KinematicModel } from '../../src/core/kinematics/fk';
import { solveIK } from '../../src/core/kinematics/ik';
import { mulberry32 } from '../../src/core/math/rng';
import { quadruped, quadrupedStance, radialLegged, serialChain } from '../../src/core/model/presets';

function run(label: string, km: KinematicModel, limb: number, q0: number[], restarts: number) {
  const rng = mulberry32(99);
  const N = 200;
  let ok = 0, iters = 0, limitViolations = 0;
  const times: number[] = [];
  for (let k = 0; k < N; k++) {
    const target = tipPosition(km, limb, randomConfiguration(km, rng));
    const t0 = performance.now();
    const r = solveIK(km, limb, target, q0, { restarts, seed: k });
    times.push(performance.now() - t0);
    if (r.status === 'converged' && r.residual < 1e-4) ok++;
    iters += r.iterations;
    if (r.q.some((v, i) => v < km.joints[i]!.lower || v > km.joints[i]!.upper)) limitViolations++;
  }
  times.sort((a, b) => a - b);
  return { label, restarts, successRate: ok / N, meanIterations: +(iters / N).toFixed(1), medianMs: +times[N / 2]!.toFixed(3), p95Ms: +times[Math.floor(0.95 * N)]!.toFixed(3), limitViolations };
}

it('S3: DLS IK success, limits, unreachable and near-singular behavior on generated limbs', () => {
  const chain6 = buildKinematicModel(serialChain({ axes: ['z', 'y', 'y', 'x', 'y', 'x'], lengths: [0.2, 0.3, 0.25, 0.1, 0.08, 0.06] }));
  const quad = buildKinematicModel(quadruped());
  const leg4 = buildKinematicModel(radialLegged(4, 4));
  const rows = [
    run('6-joint chain, interactive (no restarts)', chain6, 0, new Array(6).fill(0.3), 0),
    run('6-joint chain, 4 restarts', chain6, 0, new Array(6).fill(0.3), 4),
    run('quadruped front-left leg, interactive', quad, 0, quadrupedStance(), 0),
    run('quadruped front-left leg, 4 restarts', quad, 0, quadrupedStance(), 4),
    run('generated 4-joint leg, 4 restarts', leg4, 0, new Array(16).fill(0.2), 4),
  ];
  console.table(rows);

  // Unreachable: 2x beyond the reach bound of the 6-joint chain.
  const far = solveIK(chain6, 0, [0, 0, 2.5], new Array(6).fill(0.3));
  // Near-singular: start fully stretched (all zeros), target a hair inside the reach along the stretch direction.
  const reachTip = tipPosition(chain6, 0, new Array(6).fill(0));
  const near = solveIK(chain6, 0, [reachTip[0] + 0.002, reachTip[1], reachTip[2] - 0.001], new Array(6).fill(0));
  const special = [
    { case: 'far beyond reach', status: far.status, beyondReach: far.beyondReach, residualM: +far.residual.toFixed(4), finite: far.q.every(Number.isFinite) },
    { case: 'near singular start', status: near.status, beyondReach: near.beyondReach, residualM: +near.residual.toExponential(2), finite: near.q.every(Number.isFinite) },
  ];
  console.table(special);

  for (const r of rows) expect(r.limitViolations).toBe(0);
  expect(rows[1]!.successRate).toBeGreaterThanOrEqual(0.98);
  expect(rows[3]!.successRate).toBeGreaterThanOrEqual(0.98);
  expect(rows[4]!.successRate).toBeGreaterThanOrEqual(0.98);
  expect(far.status).toBe('unreachable');
  expect(far.q.every(Number.isFinite)).toBe(true);
  expect(near.q.every(Number.isFinite)).toBe(true);
  expect(near.residual).toBeLessThan(1e-4);
});
