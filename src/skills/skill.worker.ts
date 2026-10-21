/// <reference lib="webworker" />
import { loadMujocoBrowser } from '../physics/loader';
import { handleSkillJob, type SkillWorkerMessage, type SkillWorkerRequest } from './jobs';

export type { SkillWorkerMessage, SkillWorkerRequest } from './jobs';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<SkillWorkerRequest>) => {
  void handleSkillJob(loadMujocoBrowser, e.data, (m: SkillWorkerMessage) => ctx.postMessage(m));
};
