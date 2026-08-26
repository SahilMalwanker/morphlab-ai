import { describe, expect, it } from 'vitest';
import { mulberry32 } from '../../src/core/math/rng';
import { centredRanks, evolutionStrategies } from '../../src/learning/es';
import { Adam, Mlp } from '../../src/learning/mlp';

describe('MLP', () => {
  it('backpropagation matches central finite differences', () => {
    const rng = mulberry32(4);
    const net = Mlp.initialized([4, 7, 5, 3], rng, 1);
    const x = [0.3, -0.7, 0.1, 0.9];
    const target = [0.2, -0.1, 0.4];
    const loss = () => {
      const y = net.forward(x);
      return y.reduce((s, v, i) => s + (v - target[i]!) ** 2, 0);
    };
    const y = net.forward(x);
    const grad = new Float64Array(net.params.length);
    net.backward(x, Array.from(y, (v, i) => 2 * (v - target[i]!)), grad);
    const h = 1e-6;
    for (let i = 0; i < net.params.length; i++) {
      const old = net.params[i]!;
      net.params[i] = old + h; const lp = loss();
      net.params[i] = old - h; const lm = loss();
      net.params[i] = old;
      expect(Math.abs(grad[i]! - (lp - lm) / (2 * h))).toBeLessThan(1e-7);
    }
  });

  it('serializes and restores identical outputs', () => {
    const net = Mlp.initialized([6, 16, 2], mulberry32(1));
    const back = Mlp.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
    const x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    expect(Array.from(back.forward(x))).toEqual(Array.from(net.forward(x)));
  });

  it('rejects non-finite weights and wrong parameter counts', () => {
    expect(() => Mlp.fromJSON({ sizes: [2, 1], activation: 'tanh', params: [1, Number.NaN, 0] })).toThrow(/non-finite/);
    expect(() => new Mlp([2, 1], new Float64Array(2))).toThrow(/Expected 3/);
  });

  it('Adam minimizes a convex quadratic', () => {
    const p = Float64Array.from([3, -2]);
    const adam = new Adam(2, 0.1);
    for (let i = 0; i < 500; i++) adam.step(p, [2 * (p[0]! - 1), 2 * (p[1]! + 0.5)]);
    expect(p[0]).toBeCloseTo(1, 3);
    expect(p[1]).toBeCloseTo(-0.5, 3);
  });
});

describe('evolution strategies', () => {
  it('centred ranks are scale invariant and centred', () => {
    expect(centredRanks([10, -5, 3])).toEqual([0.5, -0.5, 0]);
    expect(centredRanks([1000, -5000, 30])).toEqual([0.5, -0.5, 0]);
  });

  it('maximizes a concave function from a distant start', () => {
    const opt = [0.5, -1.5, 2];
    const { theta } = evolutionStrategies(Float64Array.from([0, 0, 0]), (t) => -t.reduce((s, v, i) => s + (v - opt[i]!) ** 2, 0), { pairs: 16, sigma: 0.1, learningRate: 0.05, iterations: 400, seed: 3 });
    theta.forEach((v, i) => expect(Math.abs(v - opt[i]!)).toBeLessThan(0.05));
  });

  it('is reproducible for a fixed seed', () => {
    const f = (t: Float64Array) => -((t[0]! - 1) ** 2);
    const a = evolutionStrategies(Float64Array.from([0]), f, { pairs: 4, sigma: 0.1, learningRate: 0.05, iterations: 20, seed: 9 });
    const b = evolutionStrategies(Float64Array.from([0]), f, { pairs: 4, sigma: 0.1, learningRate: 0.05, iterations: 20, seed: 9 });
    expect(Array.from(a.theta)).toEqual(Array.from(b.theta));
  });
});
