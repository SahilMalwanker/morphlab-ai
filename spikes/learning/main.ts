import { openDB } from 'idb';
import { evaluateReach, mlpPolicy, reachTasks } from '../../src/learning/reach';
import type { ReachWorkerMessage } from '../../src/learning/reach.worker';

const HELD_OUT_SEED = 424242;
const out = document.getElementById('out')!;
const trainBtn = document.getElementById('train') as HTMLButtonElement;
const db = await openDB('morphlab-spikes', 1, { upgrade: (d) => d.createObjectStore('policies') });

interface SavedPolicy {
  sizes: number[];
  params: number[];
  evaluation: ReturnType<typeof evaluateReach>;
  wallTimeS: number;
  savedAt: string;
}

const state: Record<string, unknown> = {};
(window as unknown as { __spike: unknown }).__spike = state;

function show(): void {
  out.textContent = JSON.stringify(state, null, 1);
}

const saved = (await db.get('policies', 'reach')) as SavedPolicy | undefined;
if (saved) {
  const reEval = evaluateReach(mlpPolicy(saved.sizes, saved.params), reachTasks(HELD_OUT_SEED, 100));
  state.reloaded = { savedAt: saved.savedAt, savedEvaluation: saved.evaluation, evaluationAfterReload: reEval, identical: JSON.stringify(reEval) === JSON.stringify(saved.evaluation) };
} else {
  state.reloaded = null;
}
show();

trainBtn.addEventListener('click', () => {
  trainBtn.disabled = true;
  const worker = new Worker(new URL('../../src/learning/reach.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = async (e: MessageEvent<ReachWorkerMessage>) => {
    const m = e.data;
    if (m.type === 'progress') {
      state.progress = m.detail;
    } else if (m.type === 'done') {
      const record: SavedPolicy = { sizes: m.sizes, params: m.params, evaluation: m.evaluation, wallTimeS: +m.wallTimeS.toFixed(2), savedAt: new Date().toISOString() };
      await db.put('policies', record, 'reach');
      state.trained = { evaluation: m.evaluation, trainingWallTimeS: record.wallTimeS, saved: true };
      worker.terminate();
      trainBtn.disabled = false;
    } else {
      state.error = m.message;
      worker.terminate();
      trainBtn.disabled = false;
    }
    show();
  };
  worker.postMessage({ type: 'train', config: { method: 'imitation', seed: 5 }, heldOutSeed: HELD_OUT_SEED });
  state.progress = 'started';
  show();
});

document.getElementById('clear')!.addEventListener('click', async () => {
  await db.delete('policies', 'reach');
  state.reloaded = null;
  show();
});
