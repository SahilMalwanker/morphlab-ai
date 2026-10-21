import { describe, expect, it } from 'vitest';
import type { SkillWorkerMessage, SkillWorkerRequest } from '../../src/skills/jobs';
import { SkillWorkerClient } from '../../src/skills/workerClient';

type Behavior = 'fail-to-load' | 'crash' | 'answer';

/** Minimal stand-in for a Worker; behavior is decided per created instance. */
class FakeWorker {
  onmessage: ((e: MessageEvent<SkillWorkerMessage>) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  terminated = false;
  readonly received: SkillWorkerRequest[] = [];
  constructor(private readonly behavior: Behavior) {}
  postMessage(req: SkillWorkerRequest): void {
    this.received.push(req);
    queueMicrotask(() => {
      if (this.terminated) return;
      if (this.behavior === 'fail-to-load') this.onerror?.({ type: 'error' } as Event);
      else if (this.behavior === 'crash') {
        this.onmessage?.({ data: { id: req.id, type: 'progress', fraction: 0.1, detail: 'started' } } as MessageEvent<SkillWorkerMessage>);
        this.onerror?.({ type: 'error', message: 'Uncaught RangeError: boom' } as ErrorEvent);
      } else this.onmessage?.({ data: { id: req.id, type: 'result', result: 42, wallTimeS: 0.01 } } as MessageEvent<SkillWorkerMessage>);
    });
  }
  terminate(): void {
    this.terminated = true;
  }
}

function clientWith(behaviors: Behavior[]) {
  const made: FakeWorker[] = [];
  const client = new SkillWorkerClient(() => {
    const w = new FakeWorker(behaviors[Math.min(made.length, behaviors.length - 1)]!);
    made.push(w);
    return w as unknown as Worker;
  });
  return { client, made };
}

const JOB = { type: 'demonstrate' } as unknown as Parameters<SkillWorkerClient['run']>[0];

describe('skill worker client', () => {
  it('resends the job once to a fresh worker when the first one never started', async () => {
    const { client, made } = clientWith(['fail-to-load', 'answer']);
    await expect(client.run<number>(JOB)).resolves.toEqual({ result: 42, wallTimeS: 0.01 });
    expect(made).toHaveLength(2);
    expect(made[0]!.terminated).toBe(true);
    expect(made[1]!.received[0]).toEqual(made[0]!.received[0]);
  });

  it('reports a start failure after one retry instead of retrying forever', async () => {
    const { client, made } = clientWith(['fail-to-load']);
    await expect(client.run(JOB)).rejects.toThrow('Skill worker failed: the worker could not be started');
    expect(made).toHaveLength(2);
  });

  it('does not retry a worker that started and then crashed; the error message is kept', async () => {
    const { client, made } = clientWith(['crash', 'answer']);
    await expect(client.run(JOB)).rejects.toThrow('Skill worker failed: Uncaught RangeError: boom');
    expect(made).toHaveLength(1);
    // The next job gets a fresh worker.
    await expect(client.run<number>(JOB)).resolves.toMatchObject({ result: 42 });
    expect(made).toHaveLength(2);
  });
});
