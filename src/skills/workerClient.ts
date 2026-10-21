import type { SkillWorkerRequest } from './jobs';

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
export type SkillJob = DistributiveOmit<SkillWorkerRequest, 'id'>;

/** Messages every job worker sends back; `partial` carries results finished so far. */
export type JobMessage =
  | { id: number; type: 'progress'; fraction: number; detail: string; partial?: unknown }
  | { id: number; type: 'result'; result: unknown; wallTimeS: number }
  | { id: number; type: 'error'; message: string };

export type ProgressCallback = (fraction: number, detail: string, partial?: unknown) => void;

interface Pending<Req> {
  request: Req;
  resolve: (v: { result: unknown; wallTimeS: number }) => void;
  reject: (e: Error) => void;
  onProgress?: ProgressCallback;
}

/**
 * Runs batch jobs in a dedicated Web Worker with its own MuJoCo instance.
 * Cancelling terminates the worker; the next job starts a fresh one. If a
 * worker fails before it sends anything (its script did not load), the
 * pending jobs are sent once more to a fresh worker before the failure is
 * reported.
 */
export class JobWorkerClient<Req extends { id: number }> {
  private worker: Worker | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, Pending<Req>>();
  private startRetried = false;

  constructor(private readonly create: () => Worker, private readonly label: string) {}

  private ensure(): Worker {
    if (this.worker) return this.worker;
    const w = this.create();
    let heard = false;
    w.onmessage = (e: MessageEvent<JobMessage>) => {
      heard = true;
      this.startRetried = false;
      const m = e.data;
      const p = this.pending.get(m.id);
      if (!p) return;
      if (m.type === 'progress') p.onProgress?.(m.fraction, m.detail, m.partial);
      else {
        this.pending.delete(m.id);
        if (m.type === 'result') p.resolve({ result: m.result, wallTimeS: m.wallTimeS });
        else p.reject(new Error(m.message));
      }
    };
    w.onerror = (e: Event) => {
      // Uncaught errors arrive as ErrorEvent with a message; a script that failed to load gives a plain Event.
      const message = typeof (e as ErrorEvent).message === 'string' ? (e as ErrorEvent).message : '';
      w.terminate();
      if (this.worker === w) this.worker = null;
      if (!heard && !message && !this.startRetried && this.pending.size) {
        this.startRetried = true;
        const fresh = this.ensure();
        for (const p of this.pending.values()) fresh.postMessage(p.request);
        return;
      }
      const err = new Error(`${this.label} failed: ${message || (heard ? 'unknown error' : 'the worker could not be started')}`);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
    this.worker = w;
    return w;
  }

  run<T>(job: DistributiveOmit<Req, 'id'>, onProgress?: ProgressCallback): Promise<{ result: T; wallTimeS: number }> {
    const id = this.nextId++;
    const w = this.ensure();
    const request = { ...job, id } as unknown as Req;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { request, resolve: resolve as Pending<Req>['resolve'], reject, onProgress });
      w.postMessage(request);
    });
  }

  /** Whether a worker is running (a terminated or never-started worker is not). */
  get alive(): boolean {
    return this.worker !== null;
  }

  /** Stops the running job; pending promises reject with a cancellation error. */
  cancel(): void {
    this.worker?.terminate();
    this.worker = null;
    for (const p of this.pending.values()) p.reject(new Error('Cancelled.'));
    this.pending.clear();
  }

  dispose(): void {
    this.cancel();
  }
}

/** Skill jobs (evaluation, demonstration recording, training) in their own worker. */
export class SkillWorkerClient extends JobWorkerClient<SkillWorkerRequest> {
  constructor(create: () => Worker = () => new Worker(new URL('./skill.worker.ts', import.meta.url), { type: 'module' })) {
    super(create, 'Skill worker');
  }
}
