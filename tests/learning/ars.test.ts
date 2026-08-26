import { describe, expect, it } from 'vitest';
import { gaussian, mulberry32 } from '../../src/core/math/rng';
import { arsStep, augmentedRandomSearch, augmentedRandomSearchBatched, RunningStats } from '../../src/learning/ars';

// Double integrator (position, velocity) at dt 0.1 with force input; cost x'Qx + u'Ru over 100 steps, Q = I, R = 0.1.
const DT = 0.1;
const R_U = 0.1;
const HORIZON = 100;

function cost(K: ArrayLike<number>, starts: readonly (readonly [number, number])[]): number {
  let total = 0;
  for (const [p0, v0] of starts) {
    let p = p0, v = v0;
    for (let t = 0; t < HORIZON; t++) {
      const u = K[0]! * p + K[1]! * v;
      total += p * p + v * v + R_U * u * u;
      p += DT * v;
      v += DT * u;
    }
  }
  return total / starts.length;
}

/** Infinite-horizon optimal gain u = K x from the discrete Riccati recursion (closed form reference). */
function riccatiGain(): [number, number] {
  // P is symmetric 2x2: [a b; b c]. A = [1 dt; 0 1], B = [0; dt].
  let a = 1, b = 0, c = 1;
  for (let i = 0; i < 10_000; i++) {
    // A'PA
    const pa = a, pb = a * DT + b, pc = a * DT * DT + 2 * b * DT + c;
    // B'PB = dt^2 c, B'PA = dt [b, b dt + c]
    const s = R_U + DT * DT * c;
    const g0 = DT * b, g1 = DT * (b * DT + c);
    const na = 1 + pa - (g0 * g0) / s, nb = pb - (g0 * g1) / s, nc = 1 + pc - (g1 * g1) / s;
    if (Math.abs(na - a) + Math.abs(nb - b) + Math.abs(nc - c) < 1e-13) break;
    a = na; b = nb; c = nc;
  }
  const s = R_U + DT * DT * c;
  return [-(DT * b) / s, -(DT * (b * DT + c)) / s];
}

function starts(seed: number, n: number): [number, number][] {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => [gaussian(rng), gaussian(rng)]);
}

describe('augmented random search (ARS V2-t)', () => {
  it('steps along the best directions, scaled by the spread of the returns it used', () => {
    const theta = new Float64Array([0, 0]);
    arsStep(theta, [Float64Array.of(1, 0), Float64Array.of(0, 1)], [[3, 1], [0, 0.5]], 1, 0.1);
    // Best direction 0 (max return 3): returns used 3 and 1 (sd 1), step 0.1 * (3 - 1) along [1, 0].
    expect(Array.from(theta)).toEqual([0.2, 0]);
    const flat = new Float64Array([1, 1]);
    arsStep(flat, [Float64Array.of(1, 0)], [[2, 2]], 1, 0.1);
    expect(Array.from(flat)).toEqual([1, 1]);
  });

  it('finds a regulator gain within 10 % of the Riccati optimum on held-out starts', () => {
    const Kstar = riccatiGain();
    const heldOut = starts(999, 64);
    const run = () => augmentedRandomSearch([0, 0], (K, it) => -cost(K, starts(1000 + it, 4)), { directions: 8, top: 4, stepSize: 0.05, noise: 0.05, iterations: 300, seed: 7 });
    const { theta, history } = run();
    const learned = cost(theta, heldOut), optimal = cost(Kstar, heldOut), open = cost([0, 0], heldOut);
    const seen = `K ${Array.from(theta).map((v) => v.toFixed(3))}, K* ${Kstar.map((v) => v.toFixed(3))}, cost ${learned.toFixed(2)} vs optimal ${optimal.toFixed(2)} vs open loop ${open.toFixed(2)}`;
    expect(learned, seen).toBeLessThan(1.1 * optimal);
    expect(learned, seen).toBeLessThan(0.5 * open);
    expect(history).toHaveLength(300);
    expect(history.at(-1)!.rollouts).toBe(300 * 16);
    // The same seed gives the same result.
    expect(Array.from(run().theta)).toEqual(Array.from(theta));
  });

  it('takes the same steps when every rollout of an iteration is evaluated in one batch (worker pool)', async () => {
    const cfg = { directions: 8, top: 4, stepSize: 0.05, noise: 0.05, iterations: 40, seed: 7 };
    const one = augmentedRandomSearch([0, 0], (K, it) => -cost(K, starts(1000 + it, 4)), cfg);
    const batches: number[] = [];
    const batched = await augmentedRandomSearchBatched([0, 0], async (cands, it) => {
      batches.push(cands.length);
      return cands.map((K) => -cost(K, starts(1000 + it, 4)));
    }, cfg);
    expect(Array.from(batched.theta)).toEqual(Array.from(one.theta));
    expect(batched.history).toEqual(one.history);
    expect(batches).toEqual(new Array(40).fill(16));
    await expect(augmentedRandomSearchBatched([0, 0], async () => [1], cfg)).rejects.toThrow('Expected 16 returns, got 1');
  });
});

describe('running observation statistics', () => {
  it('match the batch mean and variance, and merging two halves equals the whole', () => {
    const rng = mulberry32(3);
    const rows = Array.from({ length: 200 }, () => [gaussian(rng) * 2 + 1, gaussian(rng) * 0.1 - 3, 5]);
    const all = new RunningStats(3), first = new RunningStats(3), second = new RunningStats(3);
    rows.forEach((r, i) => {
      all.push(r);
      (i < 120 ? first : second).push(r);
    });
    first.merge(second);
    for (let j = 0; j < 3; j++) {
      const mean = rows.reduce((s, r) => s + r[j]!, 0) / rows.length;
      const sd = Math.sqrt(rows.reduce((s, r) => s + (r[j]! - mean) ** 2, 0) / rows.length);
      expect(all.mean[j]).toBeCloseTo(mean, 10);
      expect(first.mean[j]).toBeCloseTo(mean, 10);
      expect(Math.max(0.01, sd)).toBeCloseTo(all.std()[j]!, 10);
      expect(first.std()[j]).toBeCloseTo(all.std()[j]!, 10);
    }
    // A constant input keeps the floor instead of a zero standard deviation.
    expect(all.std()[2]).toBe(0.01);
    expect(() => all.push([1, Number.NaN, 2])).toThrow('Observation 1 is not finite');
    expect(new RunningStats(2).std()[0]).toBe(1);
    // Statistics sent between threads come back unchanged; malformed parts are refused.
    const back = RunningStats.fromParts(JSON.parse(JSON.stringify(all.toParts())));
    expect(back.n).toBe(200);
    expect(Array.from(back.std())).toEqual(Array.from(all.std()));
    expect(() => RunningStats.fromParts({ n: 1.5, mean: [0], m2: [0] })).toThrow('Malformed running statistics');
    expect(() => RunningStats.fromParts({ n: 2, mean: [0, 1], m2: [Number.NaN, 0] })).toThrow('Malformed running statistics');
  });
});
