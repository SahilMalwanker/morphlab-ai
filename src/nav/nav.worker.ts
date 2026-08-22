/// <reference lib="webworker" />
import { loadMujocoBrowser } from '../physics/loader';
import type { JobMessage } from '../skills/workerClient';
import { handleNavJob, type NavWorkerRequest } from './jobs';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<NavWorkerRequest>) => {
  void handleNavJob(loadMujocoBrowser, e.data, (m: JobMessage) => ctx.postMessage(m));
};
