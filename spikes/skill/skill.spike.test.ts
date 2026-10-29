// Spike S5: hand-designed state-machine single-foot lift, 20 randomized trials.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { evaluateFootLift } from '../../src/skills/evaluation';
import { DEFAULT_FOOT_LIFT } from '../../src/skills/footLift';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('S5: state-machine foot lift succeeds in >= 80 % of 20 randomized trials', () => {
  const t0 = performance.now();
  const summary = evaluateFootLift(mj, quadruped(), 0, quadrupedStance(), DEFAULT_FOOT_LIFT, {
    trials: 20,
    seed: 1,
    frictionRange: [0.6, 1.2],
    baseMassRange: [3.5, 4.5],
    jointNoiseRad: 0.02,
  });
  const seconds = (performance.now() - t0) / 1000;
  console.table(summary.results.map((r) => ({ trial: r.trial, friction: r.conditions.friction, massKg: r.conditions.baseMassKg, success: r.success, failures: r.failures.join(',') || '-', holdS: r.longestClearHoldS, maxClearM: r.maxClearanceM, maxTilt: r.maxTiltRad, minZ: r.minBaseHeightM, support: r.supportContactFraction })));
  console.log('S5 summary', JSON.stringify({ successes: summary.successes, trials: summary.trials, successRate: summary.successRate, wilson95: summary.wilson95.map((v) => +v.toFixed(3)), wallSeconds: +seconds.toFixed(1) }));
  expect(summary.successRate).toBeGreaterThanOrEqual(0.8);
});
