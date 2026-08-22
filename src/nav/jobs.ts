import type { MainModule } from '@mujoco/mujoco';
import type { RobotDescription } from '../core/model/types';
import { runNavigation, type NavRunResult } from '../physics/navigation';
import type { LocomotionController } from '../skills/locomotion';
import type { JobMessage } from '../skills/workerClient';
import { TEST_MAPS } from './maps';

/** Batch evaluation of the test maps in one fixed controller mode, chosen when the job starts. */
export interface NavWorkerRequest {
  id: number;
  type: 'evaluate';
  mode: 'sensor-only' | 'ground-truth';
  mapIds: string[];
  /** The body selected when the job started; the rover when absent. */
  robot?: RobotDescription;
  /** A legged body's walking controller at that moment, if it had one. */
  walking?: LocomotionController;
}

/**
 * Runs one navigation job with a loaded MuJoCo module, reporting through
 * `post`. Progress messages carry the maps finished so far. Shared by the
 * worker and tests.
 */
export async function handleNavJob(loadModule: () => Promise<MainModule>, req: NavWorkerRequest, post: (m: JobMessage) => void): Promise<void> {
  try {
    const t0 = performance.now();
    post({ id: req.id, type: 'progress', fraction: 0, detail: 'Loading the physics engine in the worker', partial: [] });
    const maps = req.mapIds.map((id) => {
      const m = TEST_MAPS.find((x) => x.id === id);
      if (!m) throw new Error(`Unknown test map ${id}.`);
      return m;
    });
    const mj = await loadModule();
    const results: NavRunResult[] = [];
    for (let i = 0; i < maps.length; i++) {
      const map = maps[i]!;
      const run = runNavigation(mj, map, { mode: req.mode, ...(req.robot ? { robot: req.robot } : {}), ...(req.walking ? { walking: req.walking } : {}) });
      let r = run.next();
      let last = performance.now();
      while (!r.done) {
        const now = performance.now();
        if (now - last > 150) {
          last = now;
          post({ id: req.id, type: 'progress', fraction: (i + r.value) / maps.length, detail: `Map ${i + 1} of ${maps.length}: ${map.name}`, partial: results.slice() });
        }
        r = run.next();
      }
      results.push(r.value);
      post({ id: req.id, type: 'progress', fraction: (i + 1) / maps.length, detail: `Map ${i + 1} of ${maps.length}: ${map.name} ${r.value.reached ? 'reached' : 'failed'}`, partial: results.slice() });
    }
    post({ id: req.id, type: 'result', result: results, wallTimeS: (performance.now() - t0) / 1000 });
  } catch (err) {
    post({ id: req.id, type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
}
