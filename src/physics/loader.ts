import type { MainModule } from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';

let modulePromise: Promise<MainModule> | null = null;

/**
 * Loads the single-threaded MuJoCo build once per thread, with a
 * bundler-resolved WASM URL. The JavaScript glue is imported on first use so
 * it stays out of the first bundle (kinematics needs no physics engine).
 */
export function loadMujocoBrowser(): Promise<MainModule> {
  modulePromise ??= import('@mujoco/mujoco')
    .then(({ default: loadMujoco }) => loadMujoco({ locateFile: (path: string) => (path.endsWith('.wasm') ? wasmUrl : path) }))
    .catch((e: unknown) => {
      // A failed download must not be cached: the next call tries again.
      modulePromise = null;
      throw e;
    });
  return modulePromise;
}
