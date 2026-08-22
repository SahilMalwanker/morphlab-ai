import { JobWorkerClient } from '../skills/workerClient';
import type { NavWorkerRequest } from './jobs';

/** Navigation batch evaluations in their own worker, so the live view keeps running. */
export class NavWorkerClient extends JobWorkerClient<NavWorkerRequest> {
  constructor(create: () => Worker = () => new Worker(new URL('./nav.worker.ts', import.meta.url), { type: 'module' })) {
    super(create, 'Navigation worker');
  }
}
