import { gaussian, mulberry32 } from '../core/math/rng';
import { Adam } from './mlp';

export interface EsConfig {
  /** Antithetic pairs per iteration (population = 2 * pairs). */
  pairs: number;
  /** Perturbation standard deviation. */
  sigma: number;
  learningRate: number;
  iterations: number;
  seed: number;
  /** L2 penalty coefficient applied to the parameters. */
  weightDecay?: number;
}

export interface EsProgress {
  iteration: number;
  meanFitness: number;
  bestFitness: number;
}

/** Centred rank transform: utilities in [-0.5, 0.5], robust to fitness scale. */
export function centredRanks(values: number[]): number[] {
  const idx = values.map((_, i) => i).sort((a, b) => values[a]! - values[b]!);
  const out = new Array<number>(values.length);
  const n = values.length;
  idx.forEach((orig, rank) => (out[orig] = n > 1 ? rank / (n - 1) - 0.5 : 0));
  return out;
}

/**
 * OpenAI-style evolution strategies with antithetic sampling, centred-rank
 * fitness shaping and Adam ascent. `fitness` is maximized; it receives the
 * iteration index so callers can use common random numbers per iteration.
 */
export function evolutionStrategies(
  theta0: Float64Array,
  fitness: (theta: Float64Array, iteration: number) => number,
  cfg: EsConfig,
  onProgress?: (p: EsProgress) => boolean | void,
): { theta: Float64Array; history: EsProgress[] } {
  const n = theta0.length;
  const theta = new Float64Array(theta0);
  const rng = mulberry32(cfg.seed);
  const adam = new Adam(n, cfg.learningRate);
  const history: EsProgress[] = [];
  const eps = Array.from({ length: cfg.pairs }, () => new Float64Array(n));
  const candidate = new Float64Array(n);
  for (let it = 0; it < cfg.iterations; it++) {
    const f: number[] = [];
    for (let p = 0; p < cfg.pairs; p++) {
      const e = eps[p]!;
      for (let i = 0; i < n; i++) e[i] = gaussian(rng);
      for (const sign of [1, -1]) {
        for (let i = 0; i < n; i++) candidate[i] = theta[i]! + sign * cfg.sigma * e[i]!;
        const v = fitness(candidate, it);
        f.push(Number.isFinite(v) ? v : -1e9);
      }
    }
    const u = centredRanks(f);
    const grad = new Float64Array(n);
    for (let p = 0; p < cfg.pairs; p++) {
      const w = u[2 * p]! - u[2 * p + 1]!;
      const e = eps[p]!;
      for (let i = 0; i < n; i++) grad[i]! += w * e[i]!;
    }
    const scale = 1 / (2 * cfg.pairs * cfg.sigma);
    const wd = cfg.weightDecay ?? 0;
    // Adam minimizes, so pass the negated ascent direction.
    for (let i = 0; i < n; i++) grad[i] = -grad[i]! * scale + wd * theta[i]!;
    adam.step(theta, grad);
    const prog = { iteration: it, meanFitness: f.reduce((a, b) => a + b, 0) / f.length, bestFitness: Math.max(...f) };
    history.push(prog);
    if (onProgress?.(prog) === false) break;
  }
  return { theta, history };
}
