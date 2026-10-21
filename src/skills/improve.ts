import type { MainModule } from '@mujoco/mujoco';
import { buildKinematicModel, clampToLimits } from '../core/kinematics/fk';
import type { RobotDescription } from '../core/model/types';
import { evolutionStrategies, type EsProgress } from '../learning/es';
import type { SkillSpec } from './controllers';
import { runSkillTrial, trialConditions } from './evaluation';
import { GAIT_BOUNDS, legIndices, type GaitParams } from './gait';
import type { ConditionRanges } from './learning';
import { goalReward, type KeyframeMotion, type SkillGoal } from './taught';

/**
 * Reward-based improvement of a taught skill (D25): evolution strategies
 * (antithetic sampling, centred ranks, common random conditions per
 * iteration, after Salimans et al. 2017) adjust a keyframe motion's poses
 * and timing, or a gait's parameters and feedback gains, to raise the goal
 * reward measured in randomized trials. It starts from the skill as composed
 * or posed (decision D3: never from scratch) and keeps the result only if it
 * scores better than the original on held-out conditions.
 */

export interface TaughtImproveConfig extends ConditionRanges {
  iterations: number;
  pairs: number;
  sigma: number;
  learningRate: number;
  trialsPerEvaluation: number;
  heldOutTrials: number;
  seed: number;
}

export const DEFAULT_TAUGHT_IMPROVE: TaughtImproveConfig = {
  iterations: 10, pairs: 4, sigma: 0.5, learningRate: 0.25, trialsPerEvaluation: 1, heldOutTrials: 8, seed: 11,
  frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02,
};

/** Joint change per unit of a keyframe parameter, rad (or m for sliding joints). */
const POSE_SCALE = 0.08;
/** Log change of a keyframe's duration per unit. */
const TIME_SCALE = 0.2;

/** Gait parameters improvement may change, with the change per unit. Stride, side step and turn keep their direction. */
const GAIT_TUNABLE: { key: keyof typeof GAIT_BOUNDS; scale: number; keepSign?: true }[] = [
  { key: 'strideM', scale: 0.02, keepSign: true },
  { key: 'lateralM', scale: 0.015, keepSign: true },
  { key: 'turnRad', scale: 0.06, keepSign: true },
  { key: 'stepHeightM', scale: 0.01 },
  { key: 'cycleS', scale: 0.08 },
  { key: 'dutyFactor', scale: 0.05 },
  { key: 'bodyHeightM', scale: 0.01 },
  { key: 'feedbackKp', scale: 0.04 },
  { key: 'feedbackKd', scale: 0.004 },
];

/** Number of parameters improvement searches for a skill. */
export function parameterCount(spec: SkillSpec): number {
  if (spec.kind === 'keyframes') return spec.motion.keyframes.length * (spec.motion.jointIds.length + 1);
  if (spec.kind === 'gait') return GAIT_TUNABLE.length;
  throw new RangeError('Only keyframe and gait skills are improved this way');
}

/** The skill with parameters theta applied (theta = 0 is the original). */
export function specFromTheta(desc: RobotDescription, base: SkillSpec, theta: ArrayLike<number>): SkillSpec {
  if (base.kind === 'keyframes') {
    const km = buildKinematicModel(desc);
    const n = base.motion.jointIds.length;
    let t = 0;
    let prevT = 0;
    const keyframes = base.motion.keyframes.map((k, i) => {
      const off = i * (n + 1);
      const q = clampToLimits(km, k.q.map((v, j) => v + POSE_SCALE * theta[off + j]!));
      const d0 = k.t - prevT;
      prevT = k.t;
      const d = Math.min(10, Math.max(0.1, d0 * Math.exp(TIME_SCALE * theta[off + n]!)));
      t += d;
      return { t: Number(t.toFixed(4)), q };
    });
    const motion: KeyframeMotion = { ...base.motion, keyframes };
    return { kind: 'keyframes', motion };
  }
  if (base.kind === 'gait') {
    const g: GaitParams = { ...base.gait };
    // A walk or wave lifts one leg at a time and needs its feet down at least this share of the cycle (gaitProblems).
    const legs = legIndices(desc).length;
    const minDuty = (g.pattern === 'walk' || g.pattern === 'wave') && g.phases === undefined && legs >= 3 ? 1 - 1 / legs : 0;
    GAIT_TUNABLE.forEach((p, i) => {
      const [lo0, hi] = GAIT_BOUNDS[p.key];
      // Rounded up, so the rounding below never takes it under the minimum.
      const lo = p.key === 'dutyFactor' ? Math.max(lo0, Math.ceil(minDuty * 1e4) / 1e4) : lo0;
      const v0 = base.gait[p.key];
      if (p.keepSign && v0 === 0) return;
      let v = v0 + p.scale * theta[i]!;
      if (p.keepSign) v = v0 > 0 ? Math.max(0.005, v) : Math.min(-0.005, v);
      g[p.key] = Number(Math.min(hi, Math.max(lo, v)).toFixed(4));
    });
    return { kind: 'gait', gait: g };
  }
  throw new RangeError('Only keyframe and gait skills are improved this way');
}

export interface TaughtImproveResult {
  spec: SkillSpec;
  /** False when the search found nothing better on held-out conditions; `spec` is then the original. */
  improved: boolean;
  history: EsProgress[];
  baseline: { reward: number; successes: number };
  result: { reward: number; successes: number };
  heldOutTrials: number;
  rollouts: number;
}

/** Improves a keyframe or gait skill against its goals with evolution strategies in randomized trials. */
export function improveTaughtSkill(
  mj: MainModule,
  desc: RobotDescription,
  stanceQ: number[],
  base: SkillSpec,
  goals: SkillGoal[],
  cfg: TaughtImproveConfig,
  onProgress?: (p: EsProgress) => boolean | void,
): TaughtImproveResult {
  let rollouts = 0;
  const score = (spec: SkillSpec, seed: number, n: number) => {
    let reward = 0;
    let successes = 0;
    for (let i = 0; i < n; i++) {
      const r = runSkillTrial(mj, {
        desc, limbIndex: -1, stanceQ, spec, criteria: { goals }, conditions: trialConditions({ ...cfg, seed }, i), trial: i, noiseSeed: seed * 104729 + i, gains: cfg.gains,
      });
      rollouts++;
      reward += goalReward({ success: r.success, goals: r.goals ?? [], ...(r.checks ? { checks: r.checks } : {}) }, r.failures.includes('numerical-instability'));
      if (r.success) successes++;
    }
    return { reward: reward / n, successes };
  };
  const { theta, history } = evolutionStrategies(
    new Float64Array(parameterCount(base)),
    (th, it) => score(specFromTheta(desc, base, th), cfg.seed * 1000 + it, cfg.trialsPerEvaluation).reward,
    { pairs: cfg.pairs, sigma: cfg.sigma, learningRate: cfg.learningRate, iterations: cfg.iterations, seed: cfg.seed },
    onProgress,
  );
  const candidate = specFromTheta(desc, base, theta);
  const heldOut = cfg.seed * 1000 + 999;
  const baseline = score(base, heldOut, cfg.heldOutTrials);
  const result = score(candidate, heldOut, cfg.heldOutTrials);
  const improved = result.reward > baseline.reward + 1e-9;
  return { spec: improved ? candidate : base, improved, history, baseline, result: improved ? result : baseline, heldOutTrials: cfg.heldOutTrials, rollouts };
}
