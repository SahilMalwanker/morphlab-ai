import { describe, expect, it } from 'vitest';
import type { BodyKind } from '../../src/ai/capabilities.ts';
import { stopBehaviorFor, TaskManager, type FinalResult, type StopBehavior, type TaskExecutor, type TaskRecord } from '../../src/ai/tasks.ts';

function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void } {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function setup(body: BodyKind = 'arm') {
  const stops: StopBehavior[] = [];
  const changes: TaskRecord[] = [];
  let t = 1000;
  const tm = new TaskManager({ stop: (b) => stops.push(b), body: () => body, now: () => t++, onChange: (r) => changes.push(r) });
  return { tm, stops, changes };
}

function controllable(modify?: TaskExecutor['modify']) {
  const d = deferred<FinalResult>();
  let signal: AbortSignal | null = null;
  const executor: TaskExecutor = {
    run(s) {
      signal = s;
      return d.promise;
    },
    ...(modify ? { modify } : {}),
  };
  return { executor, finish: d.resolve, fail: d.reject, signal: () => signal };
}

const ok: FinalResult = { status: 'succeeded', reason: 'Reached the goal.', measurements: { distance: { value: 0.03, unit: 'm' } }, fallback: null };

describe('stopBehaviorFor', () => {
  it('holds an arm, brakes a rover and returns a legged robot to stance', () => {
    expect(stopBehaviorFor('arm')).toBe('hold');
    expect(stopBehaviorFor('wheeled')).toBe('brake');
    expect(stopBehaviorFor('legged')).toBe('stance');
  });
});

describe('TaskManager', () => {
  it('runs one task to success and reports measured results with its task ID', async () => {
    const { tm, changes } = setup();
    const c = controllable();
    const { taskId, done } = tm.start('navigateTo', tm.stateVersion, c.executor);
    expect(taskId).toBe('task-1');
    expect(tm.active?.status).toBe('running');
    c.finish(ok);
    const r = await done;
    expect(r).toEqual({ ...ok, taskId: 'task-1' });
    expect(tm.active).toBeNull();
    expect(changes.map((x) => x.status)).toEqual(['running', 'succeeded']);
    expect(tm.history[0]).toMatchObject({ id: 'task-1', status: 'succeeded', startedAt: 1000, endedAt: 1001 });
  });

  it('rejects a command planned against an older state version without running it', async () => {
    const { tm } = setup();
    const planned = tm.stateVersion;
    tm.invalidate('the scene was reset');
    let ran = false;
    const { taskId, done } = tm.start('pointAt', planned, {
      run: async () => {
        ran = true;
        return ok;
      },
    });
    expect(taskId).toBeNull();
    expect((await done).status).toBe('rejected');
    expect(ran).toBe(false);
  });

  it('cancels the running task when the state is invalidated, applying the body stop', async () => {
    const { tm, stops } = setup('wheeled');
    const c = controllable();
    const { done } = tm.start('navigateTo', tm.stateVersion, c.executor);
    tm.invalidate('the robot was rebuilt');
    const r = await done;
    expect(r).toMatchObject({ status: 'cancelled', reason: 'Cancelled: the robot was rebuilt.', fallback: 'The rover brakes to a stop.' });
    expect(c.signal()?.aborted).toBe(true);
    expect(stops).toEqual(['brake']);
  });

  it('replaces the active task and ignores the late result of the replaced one', async () => {
    const { tm } = setup();
    const first = controllable();
    const a = tm.start('pointAt', tm.stateVersion, first.executor);
    const second = controllable();
    const b = tm.start('executeSkill', tm.stateVersion, second.executor);
    expect(first.signal()?.aborted).toBe(true);
    first.finish(ok);
    expect(await a.done).toMatchObject({ taskId: 'task-1', status: 'cancelled', reason: 'Replaced by executeSkill.' });
    expect(tm.active?.id).toBe('task-2');
    second.finish(ok);
    expect((await b.done).status).toBe('succeeded');
  });

  it('cancels only the matching current task', async () => {
    const { tm, stops } = setup('legged');
    const c = controllable();
    const { taskId, done } = tm.start('executeSkill', tm.stateVersion, c.executor);
    expect(tm.cancel('task-99', tm.stateVersion)).toMatchObject({ status: 'rejected', reason: 'Task task-99 is not running.' });
    expect(tm.cancel(taskId ?? '', tm.stateVersion - 1).status).toBe('rejected');
    const r = tm.cancel(taskId ?? '', tm.stateVersion);
    expect(r).toMatchObject({ taskId: 'task-1', status: 'cancelled', fallback: 'The robot returns to a stable stance.' });
    expect(await done).toEqual(r);
    expect(stops).toEqual(['stance']);
  });

  it('Stop applies the safe stop even with no task running', () => {
    const { tm, stops } = setup('arm');
    expect(tm.stopAll()).toBeNull();
    expect(stops).toEqual(['hold']);
    const c = controllable();
    tm.start('pointAt', tm.stateVersion, c.executor);
    expect(tm.stopAll()).toMatchObject({ status: 'cancelled', reason: 'Stopped with the Stop control.', fallback: 'The arm holds its current pose.' });
    expect(stops).toEqual(['hold', 'hold']);
    expect(tm.active).toBeNull();
  });

  it('a task can say what stopping it does; the body stop still applies', async () => {
    const { tm, stops } = setup('arm');
    const c = controllable();
    const stopText = 'The experiment stopped; its unfinished results were discarded.';
    const first = tm.start('runControllerExperiment', tm.stateVersion, { ...c.executor, stopText });
    expect(tm.stopAll()).toMatchObject({ status: 'cancelled', fallback: stopText });
    expect((await first.done).fallback).toBe(stopText);
    const second = tm.start('runControllerExperiment', tm.stateVersion, { ...controllable().executor, stopText });
    expect(tm.cancel(second.taskId ?? '', tm.stateVersion).fallback).toBe(stopText);
    const third = tm.start('runControllerExperiment', tm.stateVersion, { ...controllable().executor, stopText });
    tm.invalidate('the view changed');
    expect((await third.done).fallback).toBe(stopText);
    expect(stops).toEqual(['hold', 'hold', 'hold']);
  });

  it('reports executor errors as failed and applies the safe stop', async () => {
    const { tm, stops } = setup('wheeled');
    const thrown = tm.start('navigateTo', tm.stateVersion, {
      run: () => {
        throw new Error('solver diverged');
      },
    });
    expect(await thrown.done).toMatchObject({ status: 'failed', reason: 'The task stopped with an error: solver diverged' });
    const c = controllable();
    const rejectedRun = tm.start('navigateTo', tm.stateVersion, c.executor);
    c.fail(new Error('collision'));
    expect(await rejectedRun.done).toMatchObject({ status: 'failed', reason: 'The task stopped with an error: collision' });
    expect(stops).toEqual(['brake', 'brake']);
  });

  it('applies live modifications only to a running task that supports them', () => {
    const { tm } = setup();
    const plain = controllable();
    const a = tm.start('pointAt', tm.stateVersion, plain.executor);
    expect(tm.modify(a.taskId ?? '', tm.stateVersion, { change: 'slower', amount: 'little' })).toMatchObject({ status: 'rejected' });
    const adjustable = controllable((change) => ({ ok: true, description: `Speed reduced (${change.amount}).` }));
    const b = tm.start('navigateTo', tm.stateVersion, adjustable.executor);
    expect(tm.modify(a.taskId ?? '', tm.stateVersion, { change: 'slower', amount: 'little' }).status).toBe('rejected');
    expect(tm.modify(b.taskId ?? '', tm.stateVersion, { change: 'slower', amount: 'little' })).toEqual({
      taskId: 'task-2',
      status: 'running',
      reason: 'Speed reduced (little).',
      measurements: {},
      fallback: null,
    });
  });

  it('stops the body a task started on, even after the view changed', async () => {
    const stops: StopBehavior[] = [];
    let body: BodyKind = 'wheeled';
    const tm = new TaskManager({ stop: (b) => stops.push(b), body: () => body });
    const c = controllable();
    const { done } = tm.start('navigateTo', tm.stateVersion, c.executor);
    body = 'legged';
    tm.invalidate('the view changed');
    expect(await done).toMatchObject({ status: 'cancelled', reason: 'Cancelled: the view changed.', fallback: 'The rover brakes to a stop.' });
    expect(stops).toEqual(['brake']);
    expect(tm.stopAll()).toBeNull();
    expect(stops).toEqual(['brake', 'stance']);
  });

  it('keeps a bounded history', async () => {
    const { tm } = setup();
    for (let i = 0; i < 60; i++) await tm.start('getRobotState', tm.stateVersion, { run: async () => ok }).done;
    expect(tm.history).toHaveLength(50);
    expect(tm.history[0]?.id).toBe('task-11');
  });
});
