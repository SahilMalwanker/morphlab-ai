import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { LongTermMemory } from '../../src/ai/memory';
import type { ToolResult } from '../../src/ai/result';
import { TaskManager } from '../../src/ai/tasks';
import { groundTruthPose } from '../../src/physics/navigation';
import { workspaceBridge } from '../../src/ui/ai/bridge';
import { applyLiveCommand } from '../../src/ui/ai/live';
import { NavSession } from '../../src/ui/nav/session';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

function run(s: NavSession, seconds: number): void {
  const n = Math.round(seconds / s.runtime!.timestep);
  for (let i = 0; i < n; i++) s.stepOnce();
}

/** Mean ground-truth body speed over the next `seconds` of simulation, m/s. */
function speedOver(s: NavSession, seconds: number): number {
  const rt = s.runtime!;
  const a = groundTruthPose(rt);
  const t0 = rt.time;
  run(s, seconds);
  const b = groundTruthPose(rt);
  return Math.hypot(b.x - a.x, b.y - a.y) / (rt.time - t0);
}

async function ready(): Promise<NavSession> {
  const s = new NavSession(() => Promise.resolve(mj));
  await s.init();
  s.setMap('open');
  return s;
}

describe('live changes to a running navigation task', () => {
  it('drives under the requested speed limit, and slower during the run lowers the measured speed', async () => {
    const s = await ready();
    const bridge = workspaceBridge({ mode: () => 'navigate', nav: () => s, skills: () => { throw new Error('not used'); }, memory: () => new LongTermMemory(), control: () => null });
    const snap = await bridge.snapshot();
    if ('unsupported' in snap) throw new Error(snap.unsupported);
    const tasks = new TaskManager({ body: () => 'wheeled', stop: () => s.stop('Stopped with the Stop control.') });
    const exec = snap.action('navigateTo', { x: 3, y: 0.6, maxSpeed: 0.3 });
    if (!('run' in exec)) throw new Error(exec.reason);
    const { taskId, done } = tasks.start('navigateTo', tasks.stateVersion, exec);
    const box: { result: ToolResult | null } = { result: null };
    void done.then((r) => (box.result = r));
    expect(s.state.speedLimit).toBe(0.3);

    run(s, 2);
    const before = speedOver(s, 1);
    expect(before).toBeGreaterThan(0.2);
    expect(before).toBeLessThanOrEqual(0.32);

    const live = { tasks, stop: () => 'stopped', nav: s };
    expect(applyLiveCommand('slower', live)).toBe(`navigateTo (${taskId}): Speed limit 0.20 m/s (was 0.30 m/s); the path is unchanged.`);
    run(s, 0.8);
    const after = speedOver(s, 1);
    expect(after).toBeLessThanOrEqual(0.22);
    expect(after).toBeGreaterThan(0.12);
    expect(applyLiveCommand('higher', live)).toBe('Not changed: A navigation run can go slower or faster, not higher.');

    for (let i = 0; i < 400 && !box.result; i++) {
      run(s, 0.1);
      await new Promise((r) => setTimeout(r, 0));
    }
    expect(box.result?.status).toBe('succeeded');
    expect(box.result?.measurements['commandedSpeedLimit']).toEqual({ value: 0.2, unit: 'm/s' });
    // The task's own limits end with it.
    expect(s.state.speedLimit).toBe(0.4);
    s.dispose();
  });

  it('changes a goal run started in the Navigate tab, refuses past the limits, and ignores live words when nothing runs', async () => {
    const s = await ready();
    const tasks = new TaskManager({ body: () => 'wheeled', stop: () => {} });
    const live = { tasks, stop: () => 'stopped', nav: s };
    expect(applyLiveCommand('slower', live)).toBeNull();
    expect(applyLiveCommand('stop', live)).toBe('stopped');
    expect(s.setGoal(2.5, 0)).toBe(true);
    s.startAuto();
    expect(applyLiveCommand('a little slower', live)).toBe('Speed limit 0.35 m/s (was 0.40 m/s); the path is unchanged.');
    expect(applyLiveCommand('faster', live)).toBe('Speed limit 0.40 m/s (was 0.35 m/s); the path is unchanged.');
    expect(applyLiveCommand('faster', live)).toBe('Not changed: The speed limit is already at its maximum, 0.40 m/s.');
    for (let i = 0; i < 3; i++) applyLiveCommand('slower', live);
    expect(applyLiveCommand('slower', live)).toBe('Not changed: The speed limit is already at its minimum, 0.10 m/s.');
    expect(s.setSpeedLimit(0.05)).toBe(false);
    expect(s.setSpeedLimit(Number.NaN)).toBe(false);
    expect(s.state.speedLimit).toBe(0.1);
    s.dispose();
  });
});
