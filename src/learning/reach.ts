import { choleskySolve } from '../core/math/linalg';
import { mulberry32 } from '../core/math/rng';
import { Adam, Mlp } from './mlp';
import { evolutionStrategies, type EsConfig, type EsProgress } from './es';

/**
 * Kinematic two-link planar reaching task used by the learning spike.
 * The policy commands joint velocities; there is no dynamics. Observation:
 * [sin q1, cos q1, sin q2, cos q2, 10 * (target - tip)].
 */
export const REACH = { l1: 0.3, l2: 0.25, dt: 0.05, steps: 60, vmax: 2.0, q2Limit: 2.6, successRadius: 0.02 } as const;

export interface ReachTask {
  q0: [number, number];
  target: [number, number];
}

export function reachTip(q: readonly number[]): [number, number] {
  const { l1, l2 } = REACH;
  return [l1 * Math.cos(q[0]!) + l2 * Math.cos(q[0]! + q[1]!), l1 * Math.sin(q[0]!) + l2 * Math.sin(q[0]! + q[1]!)];
}

export function sampleReachTask(rng: () => number): ReachTask {
  const r = 0.12 + 0.38 * Math.sqrt(rng());
  const a = 2 * Math.PI * rng();
  return { q0: [2 * Math.PI * rng() - Math.PI, (2 * rng() - 1) * 2.2], target: [r * Math.cos(a), r * Math.sin(a)] };
}

export function reachTasks(seed: number, n: number): ReachTask[] {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => sampleReachTask(rng));
}

export function reachObservation(q: readonly number[], target: readonly number[]): number[] {
  const p = reachTip(q);
  return [Math.sin(q[0]!), Math.cos(q[0]!), Math.sin(q[1]!), Math.cos(q[1]!), 10 * (target[0]! - p[0]), 10 * (target[1]! - p[1])];
}

export type ReachPolicy = (obs: number[]) => [number, number];

export function rollout(policy: ReachPolicy, task: ReachTask): { finalError: number; meanError: number } {
  const q = [task.q0[0], task.q0[1]];
  let sum = 0;
  for (let t = 0; t < REACH.steps; t++) {
    const a = policy(reachObservation(q, task.target));
    q[0] = q[0]! + REACH.dt * Math.max(-REACH.vmax, Math.min(REACH.vmax, a[0]));
    q[1] = Math.max(-REACH.q2Limit, Math.min(REACH.q2Limit, q[1]! + REACH.dt * Math.max(-REACH.vmax, Math.min(REACH.vmax, a[1]))));
    const p = reachTip(q);
    sum += Math.hypot(task.target[0] - p[0], task.target[1] - p[1]);
  }
  const p = reachTip(q);
  return { finalError: Math.hypot(task.target[0] - p[0], task.target[1] - p[1]), meanError: sum / REACH.steps };
}

/** Network output passes through tanh and is scaled to the velocity limit. */
export function mlpPolicy(sizes: number[], params: ArrayLike<number>): ReachPolicy {
  return (obs) => {
    const y = Mlp.forwardWith(sizes, params, obs);
    return [REACH.vmax * Math.tanh(y[0]!), REACH.vmax * Math.tanh(y[1]!)];
  };
}

/** Hand-written expert used only for imitation data: damped least-squares velocity toward the target. */
export function expertPolicy(obs: number[]): [number, number] {
  const q1 = Math.atan2(obs[0]!, obs[1]!), q2 = Math.atan2(obs[2]!, obs[3]!);
  const ex = obs[4]! / 10, ey = obs[5]! / 10;
  const { l1, l2 } = REACH;
  const s1 = Math.sin(q1), c1 = Math.cos(q1), s12 = Math.sin(q1 + q2), c12 = Math.cos(q1 + q2);
  const J = [-l1 * s1 - l2 * s12, -l2 * s12, l1 * c1 + l2 * c12, l2 * c12];
  const lam2 = 0.01 * 0.01;
  const A = [J[0]! * J[0]! + J[1]! * J[1]! + lam2, J[0]! * J[2]! + J[1]! * J[3]!, J[0]! * J[2]! + J[1]! * J[3]!, J[2]! * J[2]! + J[3]! * J[3]! + lam2];
  const k = 4;
  const y = choleskySolve(A, [k * ex, k * ey], 2) ?? new Float64Array(2);
  const v = [J[0]! * y[0]! + J[2]! * y[1]!, J[1]! * y[0]! + J[3]! * y[1]!];
  const m = Math.max(Math.abs(v[0]!), Math.abs(v[1]!));
  const s = m > REACH.vmax * 0.95 ? (REACH.vmax * 0.95) / m : 1;
  return [v[0]! * s, v[1]! * s];
}

export interface ReachEvaluation {
  tasks: number;
  successes: number;
  successRate: number;
  meanFinalErrorM: number;
}

export function evaluateReach(policy: ReachPolicy, tasks: ReachTask[]): ReachEvaluation {
  let ok = 0, sum = 0;
  for (const t of tasks) {
    const r = rollout(policy, t);
    if (r.finalError < REACH.successRadius) ok++;
    sum += r.finalError;
  }
  return { tasks: tasks.length, successes: ok, successRate: ok / tasks.length, meanFinalErrorM: sum / tasks.length };
}

export const REACH_SIZES = [6, 32, 2];

export interface TrainReachConfig {
  method: 'es' | 'imitation';
  seed: number;
  /** Start from these parameters instead of a fresh initialization. */
  initialParams?: ArrayLike<number>;
  es?: Partial<EsConfig> & { tasksPerIteration?: number };
  imitation?: { episodes: number; epochs: number; batch: number; learningRate: number };
}

/** Trains a reaching policy; returns the network and a training log. */
export function trainReach(cfg: TrainReachConfig, onProgress?: (p: EsProgress | { epoch: number; loss: number }) => void): { net: Mlp; log: unknown[] } {
  const net = Mlp.initialized(REACH_SIZES, mulberry32(cfg.seed));
  if (cfg.initialParams) net.params.set(cfg.initialParams);
  const log: unknown[] = [];
  if (cfg.method === 'es') {
    const tasksPer = cfg.es?.tasksPerIteration ?? 8;
    const es: EsConfig = { pairs: 24, sigma: 0.05, learningRate: 0.03, iterations: 300, seed: cfg.seed, weightDecay: 0.001, ...cfg.es };
    const { theta } = evolutionStrategies(
      net.params,
      (theta, it) => {
        const tasks = reachTasks(cfg.seed * 1000 + it, tasksPer);
        const pol = mlpPolicy(REACH_SIZES, theta);
        let f = 0;
        for (const t of tasks) {
          const r = rollout(pol, t);
          f -= r.meanError + 2 * r.finalError;
        }
        return f / tasksPer;
      },
      es,
      (p) => {
        if (p.iteration % 10 === 0) log.push(p);
        onProgress?.(p);
      },
    );
    net.params.set(theta);
  } else {
    const c = { episodes: 400, epochs: 40, batch: 64, learningRate: 0.003, ...cfg.imitation };
    const rng = mulberry32(cfg.seed + 7);
    const X: number[][] = [], Y: [number, number][] = [];
    for (let e = 0; e < c.episodes; e++) {
      const task = sampleReachTask(rng);
      const q = [task.q0[0], task.q0[1]];
      for (let t = 0; t < REACH.steps; t++) {
        const obs = reachObservation(q, task.target);
        const a = expertPolicy(obs);
        X.push(obs); Y.push(a);
        q[0] = q[0]! + REACH.dt * a[0];
        q[1] = Math.max(-REACH.q2Limit, Math.min(REACH.q2Limit, q[1]! + REACH.dt * a[1]));
      }
    }
    const adam = new Adam(net.params.length, c.learningRate);
    const order = X.map((_, i) => i);
    for (let epoch = 0; epoch < c.epochs; epoch++) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [order[i], order[j]] = [order[j]!, order[i]!];
      }
      let lossSum = 0;
      for (let b = 0; b < order.length; b += c.batch) {
        const grad = new Float64Array(net.params.length);
        const idx = order.slice(b, b + c.batch);
        for (const i of idx) {
          const y = net.forward(X[i]!);
          const dOut = [0, 0];
          for (let k = 0; k < 2; k++) {
            const th = Math.tanh(y[k]!);
            const err = REACH.vmax * th - Y[i]![k]!;
            lossSum += err * err;
            dOut[k] = (2 * err * REACH.vmax * (1 - th * th)) / idx.length;
          }
          net.backward(X[i]!, dOut, grad);
        }
        adam.step(net.params, grad);
      }
      const entry = { epoch, loss: lossSum / order.length };
      log.push(entry);
      onProgress?.(entry);
    }
  }
  return { net, log };
}
