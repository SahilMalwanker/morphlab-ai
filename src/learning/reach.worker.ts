/// <reference lib="webworker" />
import { evaluateReach, mlpPolicy, REACH_SIZES, reachTasks, trainReach, type TrainReachConfig } from './reach';

export type ReachWorkerRequest = { type: 'train'; config: TrainReachConfig; heldOutSeed: number };
export type ReachWorkerMessage =
  | { type: 'progress'; detail: unknown }
  | { type: 'done'; params: number[]; sizes: number[]; evaluation: ReturnType<typeof evaluateReach>; wallTimeS: number }
  | { type: 'error'; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<ReachWorkerRequest>) => {
  const req = e.data;
  if (req.type !== 'train') return;
  try {
    const t0 = performance.now();
    let last = 0;
    const { net } = trainReach(req.config, (p) => {
      const now = performance.now();
      if (now - last > 200) {
        last = now;
        ctx.postMessage({ type: 'progress', detail: p } satisfies ReachWorkerMessage);
      }
    });
    const evaluation = evaluateReach(mlpPolicy(REACH_SIZES, net.params), reachTasks(req.heldOutSeed, 100));
    ctx.postMessage({ type: 'done', params: Array.from(net.params), sizes: REACH_SIZES, evaluation, wallTimeS: (performance.now() - t0) / 1000 } satisfies ReachWorkerMessage);
  } catch (err) {
    ctx.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) } satisfies ReachWorkerMessage);
  }
};
