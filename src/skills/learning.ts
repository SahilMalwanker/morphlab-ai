import type { MainModule } from '@mujoco/mujoco';
import type { PdGains } from '../core/control/pd';
import { mulberry32 } from '../core/math/rng';
import type { RobotDescription } from '../core/model/types';
import { evolutionStrategies, type EsProgress } from '../learning/es';
import { Adam, Mlp } from '../learning/mlp';
import { skillDuration, type PolicyJson } from './controllers';
import { DEFAULT_CRITERIA, runSkillTrial, trialConditions, type TrialRecording, type TrialResult } from './evaluation';
import type { FootLiftParams } from './footLift';

export interface ConditionRanges {
  frictionRange: [number, number];
  baseMassRange: [number, number];
  jointNoiseRad: number;
  /** Joint PD gains for every trial (the evaluation default when absent). */
  gains?: PdGains;
}

export interface ImitationConfig extends ConditionRanges {
  /** Expert episodes recorded under randomized conditions. */
  episodes: number;
  seed: number;
  hidden: number[];
  iterations: number;
  batchSize: number;
  learningRate: number;
}

export const DEFAULT_IMITATION: ImitationConfig = {
  episodes: 12, seed: 1, hidden: [64, 64], iterations: 2000, batchSize: 64, learningRate: 3e-3,
  frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02,
};

export interface Demonstrations {
  /** Policy features per control step. */
  X: number[][];
  /** Expert joint targets minus the stance, per control step. */
  Y: number[][];
  horizonS: number;
}

/** Records the state-machine expert under randomized conditions (behavior-cloning data). */
export function collectDemonstrations(
  mj: MainModule,
  desc: RobotDescription,
  limbIndex: number,
  stanceQ: number[],
  expert: FootLiftParams,
  cfg: ImitationConfig,
  onEpisode?: (episode: number) => void,
): Demonstrations {
  const spec = { kind: 'state-machine', params: expert } as const;
  const horizonS = skillDuration(spec);
  const X: number[][] = [], Y: number[][] = [];
  for (let e = 0; e < cfg.episodes; e++) {
    const rec: TrialRecording = { features: [], qCmd: [] };
    runSkillTrial(mj, {
      desc, limbIndex, stanceQ, spec, criteria: DEFAULT_CRITERIA, conditions: trialConditions({ ...cfg, seed: cfg.seed + 5000 }, e), trial: e,
      noiseSeed: cfg.seed * 31 + e, featureHorizonS: horizonS, gains: cfg.gains,
    }, rec);
    rec.features.forEach((f, i) => {
      X.push(f);
      Y.push(rec.qCmd[i]!.map((v, k) => v - stanceQ[k]!));
    });
    onEpisode?.(e);
  }
  return { X, Y, horizonS };
}

function meanStd(rows: number[][]): { mean: number[]; std: number[] } {
  const n = rows.length, d = rows[0]!.length;
  const mean = new Array<number>(d).fill(0), std = new Array<number>(d).fill(0);
  for (const r of rows) for (let i = 0; i < d; i++) mean[i]! += r[i]! / n;
  for (const r of rows) for (let i = 0; i < d; i++) std[i]! += (r[i]! - mean[i]!) ** 2 / n;
  // Constant features (for example a foot that never leaves the ground) keep unit scale.
  return { mean, std: std.map((v) => (Math.sqrt(v) > 1e-6 ? Math.sqrt(v) : 1)) };
}

export interface ImitationResult {
  policy: PolicyJson;
  initialLoss: number;
  finalLoss: number;
  samples: number;
}

/**
 * Behavior cloning: mean squared error between the network output and the
 * normalized expert targets, minimized with Adam on minibatches. The output
 * normalization is folded into the last layer so the stored policy maps
 * normalized features directly to joint offsets in radians.
 */
export function trainImitation(data: Demonstrations, cfg: ImitationConfig, onProgress?: (iteration: number, loss: number) => void): ImitationResult {
  if (data.X.length === 0) throw new RangeError('No demonstration samples');
  const nIn = data.X[0]!.length, nOut = data.Y[0]!.length;
  const xs = meanStd(data.X), ys = meanStd(data.Y);
  const X = data.X.map((r) => r.map((v, i) => (v - xs.mean[i]!) / xs.std[i]!));
  const Y = data.Y.map((r) => r.map((v, i) => (v - ys.mean[i]!) / ys.std[i]!));
  const sizes = [nIn, ...cfg.hidden, nOut];
  const rng = mulberry32(cfg.seed);
  const net = Mlp.initialized(sizes, rng, 0.5);
  const adam = new Adam(net.params.length, cfg.learningRate);
  const grad = new Float64Array(net.params.length);
  const loss = (idx: number[]) => {
    let s = 0;
    for (const i of idx) {
      const p = net.forward(X[i]!);
      for (let k = 0; k < nOut; k++) s += (p[k]! - Y[i]![k]!) ** 2;
    }
    return s / (idx.length * nOut);
  };
  const probe = Array.from({ length: Math.min(512, X.length) }, (_, k) => Math.floor((k * X.length) / Math.min(512, X.length)));
  const initialLoss = loss(probe);
  for (let it = 0; it < cfg.iterations; it++) {
    grad.fill(0);
    for (let b = 0; b < cfg.batchSize; b++) {
      const i = Math.floor(rng() * X.length);
      const p = net.forward(X[i]!);
      const d = new Float64Array(nOut);
      for (let k = 0; k < nOut; k++) d[k] = (2 * (p[k]! - Y[i]![k]!)) / (cfg.batchSize * nOut);
      net.backward(X[i]!, d, grad);
    }
    adam.step(net.params, grad);
    if (onProgress && (it % 100 === 0 || it === cfg.iterations - 1)) onProgress(it, loss(probe));
  }
  const finalLoss = loss(probe);
  // Fold the output de-normalization into the last layer: y = std * (W h + b) + mean.
  const params = Float64Array.from(net.params);
  const nH = sizes[sizes.length - 2]!;
  const off = params.length - (nOut * nH + nOut);
  for (let r = 0; r < nOut; r++) {
    for (let c = 0; c < nH; c++) params[off + r * nH + c]! *= ys.std[r]!;
    params[off + nOut * nH + r] = ys.std[r]! * params[off + nOut * nH + r]! + ys.mean[r]!;
  }
  return {
    policy: { layers: sizes, activation: 'tanh', weights: Array.from(params), inputMean: xs.mean, inputStd: xs.std },
    initialLoss,
    finalLoss,
    samples: X.length,
  };
}

/** State-machine parameters that reward-based improvement may change, with bounds and step scale. */
export const TUNABLE: { key: keyof FootLiftParams; label: string; min: number; max: number; scale: number }[] = [
  { key: 'shiftM', label: 'Body shift (m)', min: 0, max: 0.1, scale: 0.02 },
  { key: 'liftHeightM', label: 'Lift height (m)', min: 0.02, max: 0.12, scale: 0.02 },
  { key: 'shiftS', label: 'Shift time (s)', min: 0.3, max: 2, scale: 0.3 },
  { key: 'liftS', label: 'Lift time (s)', min: 0.2, max: 1.5, scale: 0.2 },
  { key: 'lowerS', label: 'Lower time (s)', min: 0.2, max: 1.5, scale: 0.2 },
];

export function paramsFromTheta(base: FootLiftParams, theta: ArrayLike<number>): FootLiftParams {
  const p = { ...base };
  TUNABLE.forEach((t, i) => {
    p[t.key] = Math.min(t.max, Math.max(t.min, base[t.key] + t.scale * theta[i]!));
  });
  return p;
}

/**
 * Scalar reward of one trial, computed by the referee from the trial result:
 * foot clearance relative to the required clearance (dense, so small
 * improvements are visible), the fraction of the required clear hold, a
 * success bonus, and penalties for tilt beyond 0.1 rad and for falling or
 * aborting.
 */
export function trialReward(r: TrialResult, holdS: number, minClearanceM = DEFAULT_CRITERIA.minClearanceM): number {
  let v = 0.5 * Math.min(r.maxClearanceM / minClearanceM, 1.5) + Math.min(r.longestClearHoldS, holdS) / holdS + (r.success ? 1 : 0) - 2 * Math.max(0, r.maxTiltRad - 0.1);
  if (r.failures.includes('fell') || r.failures.includes('aborted') || r.failures.includes('numerical-instability')) v -= 1;
  return v;
}

export interface ImproveConfig extends ConditionRanges {
  iterations: number;
  pairs: number;
  sigma: number;
  learningRate: number;
  trialsPerEvaluation: number;
  /** Held-out trials used to compare baseline and result. */
  heldOutTrials: number;
  seed: number;
}

export const DEFAULT_IMPROVE: ImproveConfig = {
  iterations: 8, pairs: 3, sigma: 0.5, learningRate: 0.3, trialsPerEvaluation: 1, heldOutTrials: 6, seed: 3,
  frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02,
};

export interface ImproveResult {
  params: FootLiftParams;
  history: EsProgress[];
  baselineReward: number;
  improvedReward: number;
  trialsRun: number;
}

/**
 * Reward-based improvement of the state-machine parameters with evolution
 * strategies, starting from `base` (decision D3: never from scratch). Each
 * iteration uses common random conditions for all candidates. The result is
 * compared with the baseline on held-out conditions.
 */
export function improveParameters(
  mj: MainModule,
  desc: RobotDescription,
  limbIndex: number,
  stanceQ: number[],
  base: FootLiftParams,
  cfg: ImproveConfig,
  onProgress?: (p: EsProgress) => boolean | void,
): ImproveResult {
  let trialsRun = 0;
  const meanReward = (params: FootLiftParams, seed: number, n: number) => {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const r = runSkillTrial(mj, {
        desc, limbIndex, stanceQ, spec: { kind: 'es-improved', params }, criteria: DEFAULT_CRITERIA,
        conditions: trialConditions({ ...cfg, seed }, i), trial: i, noiseSeed: seed * 104729 + i, gains: cfg.gains,
      });
      trialsRun++;
      s += trialReward(r, params.holdS);
    }
    return s / n;
  };
  const { theta, history } = evolutionStrategies(
    new Float64Array(TUNABLE.length),
    (th, it) => meanReward(paramsFromTheta(base, th), cfg.seed * 1000 + it, cfg.trialsPerEvaluation),
    { pairs: cfg.pairs, sigma: cfg.sigma, learningRate: cfg.learningRate, iterations: cfg.iterations, seed: cfg.seed },
    onProgress,
  );
  const params = paramsFromTheta(base, theta);
  const heldOutSeed = cfg.seed * 1000 + 999;
  return {
    params,
    history,
    baselineReward: meanReward(base, heldOutSeed, cfg.heldOutTrials),
    improvedReward: meanReward(params, heldOutSeed, cfg.heldOutTrials),
    trialsRun,
  };
}
