// Spike S4 (headless part): train a tiny reaching policy, save, reload and re-evaluate.
import { expect, it } from 'vitest';
import { mulberry32 } from '../../src/core/math/rng';
import { Mlp } from '../../src/learning/mlp';
import { evaluateReach, mlpPolicy, REACH_SIZES, reachTasks, trainReach } from '../../src/learning/reach';

const HELD_OUT_SEED = 424242;

it('S4: trained reaching policies beat the untrained baseline and reload reproducibly', () => {
  const held = reachTasks(HELD_OUT_SEED, 100);
  // Baseline: the exact initial weights both trainers start from (seed 5).
  const untrained = evaluateReach(mlpPolicy(REACH_SIZES, Mlp.initialized(REACH_SIZES, mulberry32(5)).params), held);
  const rows: Record<string, unknown>[] = [{ method: 'untrained', ...untrained, trainSeconds: 0 }];
  for (const method of ['imitation', 'es'] as const) {
    const t0 = performance.now();
    const { net } = trainReach({ method, seed: 5 });
    const trainSeconds = (performance.now() - t0) / 1000;
    const ev = evaluateReach(mlpPolicy(REACH_SIZES, net.params), held);
    const saved = JSON.stringify(net.toJSON());
    const reloaded = Mlp.fromJSON(JSON.parse(saved));
    const ev2 = evaluateReach(mlpPolicy(REACH_SIZES, reloaded.params), held);
    rows.push({ method, ...ev, trainSeconds: +trainSeconds.toFixed(2), reloadIdentical: JSON.stringify(ev) === JSON.stringify(ev2), savedBytes: saved.length });
    expect(JSON.stringify(ev2)).toBe(JSON.stringify(ev));
    if (method === 'imitation') {
      // Reward-based fine-tuning of the imitation policy (the progression planned for M5).
      const t1 = performance.now();
      const { net: tuned } = trainReach({ method: 'es', seed: 5, initialParams: net.params, es: { iterations: 100, sigma: 0.01, learningRate: 0.002 } });
      const evTuned = evaluateReach(mlpPolicy(REACH_SIZES, tuned.params), held);
      rows.push({ method: 'imitation + es fine-tune', ...evTuned, trainSeconds: +((performance.now() - t1) / 1000).toFixed(2) });
    }
  }
  console.table(rows);
  expect(untrained.successRate).toBeLessThan(0.1);
  const best = Math.max(...rows.slice(1).map((r) => r.successRate as number));
  expect(best).toBeGreaterThanOrEqual(0.8);
});
