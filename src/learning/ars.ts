import { gaussian, mulberry32 } from '../core/math/rng';

/**
 * Augmented Random Search, version V2-t (Mania, Guy and Recht 2018, arXiv
 * 1803.07055), reimplemented from the paper and its reference code: each
 * iteration evaluates antithetic perturbations theta +- noise * delta, keeps
 * the best `top` directions by max(r+, r-), and steps along
 * sum (r+ - r-) delta, divided by `top` and by the standard deviation of the
 * returns it used. Observation normalization (the V2 part) is the caller's,
 * with `RunningStats`.
 */

export interface ArsConfig {
  /** Directions sampled per iteration (2 rollouts each). */
  directions: number;
  /** Best directions used for the step (at most `directions`). */
  top: number;
  stepSize: number;
  /** Perturbation scale. */
  noise: number;
  iterations: number;
  seed: number;
}

export interface ArsProgress {
  iteration: number;
  meanReturn: number;
  maxReturn: number;
  rollouts: number;
}

/** Running mean and variance of observation vectors (Welford), mergeable. */
export class RunningStats {
  n = 0;
  readonly mean: Float64Array;
  readonly m2: Float64Array;
  constructor(readonly size: number) {
    this.mean = new Float64Array(size);
    this.m2 = new Float64Array(size);
  }
  push(x: ArrayLike<number>): void {
    if (x.length !== this.size) throw new RangeError(`Expected ${this.size} values, got ${x.length}`);
    for (let i = 0; i < this.size; i++) if (!Number.isFinite(x[i])) throw new RangeError(`Observation ${i} is not finite`);
    this.n++;
    for (let i = 0; i < this.size; i++) {
      const v = x[i]!;
      const d = v - this.mean[i]!;
      this.mean[i]! += d / this.n;
      this.m2[i]! += d * (v - this.mean[i]!);
    }
  }
  /** Adds another set of samples (parallel form of Welford's update). */
  merge(o: RunningStats): void {
    if (o.size !== this.size) throw new RangeError('Statistics of different sizes');
    if (o.n === 0) return;
    const n = this.n + o.n;
    for (let i = 0; i < this.size; i++) {
      const d = o.mean[i]! - this.mean[i]!;
      this.mean[i]! += (d * o.n) / n;
      this.m2[i]! += o.m2[i]! + (d * d * this.n * o.n) / n;
    }
    this.n = n;
  }
  /** Population standard deviation, at least `floor` (a near-constant input is not blown up). */
  std(floor = 1e-2): Float64Array {
    return this.mean.map((_, i) => Math.max(floor, this.n > 0 ? Math.sqrt(this.m2[i]! / this.n) : 1));
  }
  toParts(): RunningStatsParts {
    return { n: this.n, mean: Array.from(this.mean), m2: Array.from(this.m2) };
  }
  /** Statistics sent from another thread. */
  static fromParts(p: RunningStatsParts): RunningStats {
    if (!Number.isInteger(p.n) || p.n < 0 || p.mean.length !== p.m2.length || ![...p.mean, ...p.m2].every(Number.isFinite)) throw new RangeError('Malformed running statistics');
    const s = new RunningStats(p.mean.length);
    s.n = p.n;
    s.mean.set(p.mean);
    s.m2.set(p.m2);
    return s;
  }
}

export interface RunningStatsParts {
  n: number;
  mean: number[];
  m2: number[];
}

/**
 * One ARS update in place. `returns[k]` are the returns of theta + noise *
 * deltas[k] and theta - noise * deltas[k].
 */
export function arsStep(theta: Float64Array, deltas: readonly Float64Array[], returns: readonly (readonly [number, number])[], top: number, stepSize: number): void {
  const k = Math.max(1, Math.min(top, deltas.length));
  const order = returns.map((r, i) => ({ i, best: Math.max(r[0], r[1]) })).sort((a, b) => b.best - a.best || a.i - b.i).slice(0, k);
  const used = order.flatMap(({ i }) => [returns[i]![0], returns[i]![1]]);
  const mean = used.reduce((s, v) => s + v, 0) / used.length;
  const sd = Math.sqrt(used.reduce((s, v) => s + (v - mean) ** 2, 0) / used.length);
  // All returns equal: no information about the direction, so no step.
  if (!(sd > 1e-12)) return;
  const scale = stepSize / (k * sd);
  for (const { i } of order) {
    const w = (returns[i]![0] - returns[i]![1]) * scale;
    const d = deltas[i]!;
    for (let j = 0; j < theta.length; j++) theta[j]! += w * d[j]!;
  }
}

/**
 * Runs ARS. `rollout(theta, iteration)` returns the return of one episode;
 * every rollout of an iteration should use the same conditions (common random
 * numbers). Returning false from `onIteration` stops early.
 */
export function augmentedRandomSearch(
  theta0: ArrayLike<number>,
  rollout: (theta: Float64Array, iteration: number) => number,
  cfg: ArsConfig,
  onIteration?: (p: ArsProgress) => boolean | void,
): { theta: Float64Array; history: ArsProgress[] } {
  const n = theta0.length;
  const theta = Float64Array.from(theta0);
  const rng = mulberry32(cfg.seed);
  const deltas = Array.from({ length: cfg.directions }, () => new Float64Array(n));
  const candidate = new Float64Array(n);
  const history: ArsProgress[] = [];
  let rollouts = 0;
  for (let it = 0; it < cfg.iterations; it++) {
    const returns: [number, number][] = [];
    for (const d of deltas) {
      for (let j = 0; j < n; j++) d[j] = gaussian(rng);
      const pair: [number, number] = [0, 0];
      for (const [s, sign] of [[0, 1], [1, -1]] as const) {
        for (let j = 0; j < n; j++) candidate[j] = theta[j]! + sign * cfg.noise * d[j]!;
        const r = rollout(candidate, it);
        pair[s] = Number.isFinite(r) ? r : -1e9;
        rollouts++;
      }
      returns.push(pair);
    }
    arsStep(theta, deltas, returns, cfg.top, cfg.stepSize);
    const all = returns.flat();
    const p = { iteration: it, meanReturn: all.reduce((s, v) => s + v, 0) / all.length, maxReturn: Math.max(...all), rollouts };
    history.push(p);
    if (onIteration?.(p) === false) break;
  }
  return { theta, history };
}

/**
 * The same search with all rollouts of an iteration in one batch, for a pool
 * of workers: `evaluate` gets theta + noise * delta_k and theta - noise *
 * delta_k for k = 0, 1, ... in that order and returns their returns. The
 * draws and updates equal `augmentedRandomSearch` with the same seed.
 */
export async function augmentedRandomSearchBatched(
  theta0: ArrayLike<number>,
  evaluate: (candidates: Float64Array[], iteration: number) => Promise<number[]>,
  cfg: ArsConfig,
  onIteration?: (p: ArsProgress) => boolean | void,
): Promise<{ theta: Float64Array; history: ArsProgress[] }> {
  const n = theta0.length;
  const theta = Float64Array.from(theta0);
  const rng = mulberry32(cfg.seed);
  const deltas = Array.from({ length: cfg.directions }, () => new Float64Array(n));
  const history: ArsProgress[] = [];
  let rollouts = 0;
  for (let it = 0; it < cfg.iterations; it++) {
    const candidates: Float64Array[] = [];
    for (const d of deltas) {
      for (let j = 0; j < n; j++) d[j] = gaussian(rng);
      for (const sign of [1, -1]) candidates.push(theta.map((v, j) => v + sign * cfg.noise * d[j]!));
    }
    const r = await evaluate(candidates, it);
    if (r.length !== candidates.length) throw new RangeError(`Expected ${candidates.length} returns, got ${r.length}`);
    const safe = r.map((v) => (Number.isFinite(v) ? v : -1e9));
    const returns = deltas.map((_, k) => [safe[2 * k]!, safe[2 * k + 1]!] as const);
    rollouts += candidates.length;
    arsStep(theta, deltas, returns, cfg.top, cfg.stepSize);
    const p = { iteration: it, meanReturn: safe.reduce((s, v) => s + v, 0) / safe.length, maxReturn: Math.max(...safe), rollouts };
    history.push(p);
    if (onIteration?.(p) === false) break;
  }
  return { theta, history };
}
