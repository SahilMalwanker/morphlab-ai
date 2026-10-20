import type { BodyKind } from './capabilities.ts';
import { rejected, type TaskStatus, type ToolResult } from './result.ts';

export type StopBehavior = 'hold' | 'brake' | 'stance';

export function stopBehaviorFor(body: BodyKind): StopBehavior {
  return body === 'arm' ? 'hold' : body === 'wheeled' ? 'brake' : 'stance';
}

const STOP_TEXT: Record<StopBehavior, string> = {
  hold: 'The arm holds its current pose.',
  brake: 'The rover brakes to a stop.',
  stance: 'The robot returns to a stable stance.',
};

export interface TaskChange {
  change: 'slower' | 'faster' | 'higher' | 'lower' | 'closer' | 'farther';
  amount: 'little' | 'normal';
}

export type FinalResult = Omit<ToolResult, 'taskId'>;

export interface TaskExecutor {
  /** Runs the task to its end and reports measured outcomes; must stop promptly when the signal aborts. */
  run(signal: AbortSignal): Promise<FinalResult>;
  /** Applies a validated live change, if the task supports one. */
  modify?(change: TaskChange): { ok: true; description: string } | { ok: false; reason: string };
  /** What stopping this task does, where the body's stop text does not describe it (an experiment in a separate simulation). */
  stopText?: string;
}

export interface TaskRecord {
  id: string;
  tool: string;
  status: TaskStatus;
  stateVersion: number;
  startedAt: number;
  endedAt: number | null;
  result: ToolResult | null;
}

export interface TaskManagerOptions {
  /** Applies the body's safe stop immediately; never depends on an AI service. */
  stop: (behavior: StopBehavior) => void;
  /** The body in view; a task keeps the body it started on. */
  body: () => BodyKind;
  now?: () => number;
  onChange?: (task: TaskRecord) => void;
}

interface Active {
  record: TaskRecord;
  executor: TaskExecutor;
  abort: AbortController;
  body: BodyKind;
  settle: (result: ToolResult) => void;
}

const HISTORY_LIMIT = 50;

function errorText(e: unknown): string {
  return e instanceof Error ? e.message.slice(0, 200) : 'unknown error';
}

/** One active task at a time; stale commands are rejected by task ID and robot state version. */
export class TaskManager {
  readonly #opts: TaskManagerOptions;
  readonly #history: TaskRecord[] = [];
  #active: Active | null = null;
  #version = 0;
  #seq = 0;

  constructor(opts: TaskManagerOptions) {
    this.#opts = opts;
  }

  get stateVersion(): number {
    return this.#version;
  }

  get active(): TaskRecord | null {
    return this.#active ? { ...this.#active.record } : null;
  }

  get history(): readonly TaskRecord[] {
    return this.#history;
  }

  /** The robot, scene or morphology changed: earlier plans become stale and a running task is cancelled. */
  invalidate(reason: string): void {
    this.#version++;
    const a = this.#active;
    if (!a) return;
    a.abort.abort();
    this.#finish(a, { status: 'cancelled', reason: `Cancelled: ${reason}.`, measurements: {}, fallback: this.#stopTask(a) });
  }

  start(tool: string, plannedVersion: number, executor: TaskExecutor): { taskId: string | null; done: Promise<ToolResult> } {
    if (plannedVersion !== this.#version) {
      return { taskId: null, done: Promise.resolve(rejected('The robot or scene changed after this command was planned; ask again.')) };
    }
    const previous = this.#active;
    if (previous) {
      previous.abort.abort();
      this.#finish(previous, { status: 'cancelled', reason: `Replaced by ${tool}.`, measurements: {}, fallback: null });
    }
    const record: TaskRecord = {
      id: `task-${++this.#seq}`,
      tool,
      status: 'running',
      stateVersion: this.#version,
      startedAt: this.#now(),
      endedAt: null,
      result: null,
    };
    let settle: (result: ToolResult) => void = () => {};
    const done = new Promise<ToolResult>((resolve) => {
      settle = resolve;
    });
    const active: Active = { record, executor, abort: new AbortController(), body: this.#opts.body(), settle };
    this.#active = active;
    this.#emit(record);
    let started: Promise<FinalResult>;
    try {
      started = executor.run(active.abort.signal);
    } catch (e) {
      started = Promise.reject(e);
    }
    started.then(
      (r) => {
        this.#finish(active, r);
      },
      (e: unknown) => {
        if (record.result) return;
        this.#finish(active, { status: 'failed', reason: `The task stopped with an error: ${errorText(e)}`, measurements: {}, fallback: this.#stopTask(active) });
      },
    );
    return { taskId: record.id, done };
  }

  modify(taskId: string, plannedVersion: number, change: TaskChange): ToolResult {
    if (plannedVersion !== this.#version) return rejected('The robot or scene changed after this command was planned.', null, taskId);
    const a = this.#active;
    if (!a || a.record.id !== taskId) return rejected(`Task ${taskId} is not running.`, null, taskId);
    if (!a.executor.modify) return rejected(`The running ${a.record.tool} task cannot be changed while it runs.`, null, taskId);
    const r = a.executor.modify(change);
    if (!r.ok) return rejected(r.reason, null, taskId);
    return { taskId, status: 'running', reason: r.description, measurements: {}, fallback: null };
  }

  cancel(taskId: string, plannedVersion: number): ToolResult {
    if (plannedVersion !== this.#version) return rejected('The robot or scene changed after this command was planned.', null, taskId);
    const a = this.#active;
    if (!a || a.record.id !== taskId) return rejected(`Task ${taskId} is not running.`, null, taskId);
    a.abort.abort();
    return this.#finish(a, { status: 'cancelled', reason: 'Cancelled on request.', measurements: {}, fallback: this.#stopTask(a) });
  }

  /** The local Stop control: applies the safe stop first, with or without a running task. */
  stopAll(): ToolResult | null {
    const a = this.#active;
    const fallback = this.#stopNow(a ? a.body : this.#opts.body());
    if (!a) return null;
    a.abort.abort();
    return this.#finish(a, { status: 'cancelled', reason: 'Stopped with the Stop control.', measurements: {}, fallback: a.executor.stopText ?? fallback });
  }

  #finish(active: Active, partial: FinalResult): ToolResult {
    const rec = active.record;
    if (rec.result) return rec.result;
    const status: TaskStatus = partial.status === 'succeeded' || partial.status === 'cancelled' ? partial.status : 'failed';
    const result: ToolResult = { ...partial, status, taskId: rec.id };
    rec.status = status;
    rec.result = result;
    rec.endedAt = this.#now();
    if (this.#active === active) this.#active = null;
    this.#history.push({ ...rec });
    if (this.#history.length > HISTORY_LIMIT) this.#history.shift();
    this.#emit(rec);
    active.settle(result);
    return result;
  }

  #stopNow(body: BodyKind): string {
    const behavior = stopBehaviorFor(body);
    this.#opts.stop(behavior);
    return STOP_TEXT[behavior];
  }

  /** Applies the body's safe stop, then says what stopping did to this task. */
  #stopTask(a: Active): string {
    const text = this.#stopNow(a.body);
    return a.executor.stopText ?? text;
  }

  #now(): number {
    return this.#opts.now ? this.#opts.now() : Date.now();
  }

  #emit(rec: TaskRecord): void {
    this.#opts.onChange?.({ ...rec });
  }
}
