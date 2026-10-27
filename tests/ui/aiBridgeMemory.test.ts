import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import type { BridgeSnapshot } from '../../src/ai/controller';
import { LongTermMemory } from '../../src/ai/memory';
import type { ToolResult } from '../../src/ai/result';
import type { FinalResult, TaskExecutor } from '../../src/ai/tasks';
import { workspaceBridge } from '../../src/ui/ai/bridge';
import { NavSession } from '../../src/ui/nav/session';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const NOW = '2026-10-04T12:00:00.000Z';

function run(s: NavSession, seconds: number): void {
  const n = Math.round(seconds / s.runtime!.timestep);
  for (let i = 0; i < n; i++) s.stepOnce();
}

async function snapshotOf(s: NavSession, memory: LongTermMemory): Promise<BridgeSnapshot> {
  const bridge = workspaceBridge({
    mode: () => 'navigate',
    nav: () => s,
    skills: () => {
      throw new Error('The Skills session is not used here.');
    },
    memory: () => memory,
    control: () => null,
  });
  const snap = await bridge.snapshot();
  if ('unsupported' in snap) throw new Error(snap.unsupported);
  return snap;
}

function immediate(x: TaskExecutor | ToolResult): ToolResult {
  if ('run' in x) throw new Error('Expected an immediate result, got a task.');
  return x;
}

describe('world memory through the AI bridge', () => {
  it('remembers places where the controller believes the rover is, or at given free coordinates, per map', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    s.setMap('block');
    const memory = new LongTermMemory([], () => NOW);
    let snap = await snapshotOf(s, memory);
    expect(snap.caps.places).toEqual([]);
    expect(snap.summary).toContain('No places are remembered on this map yet.');

    // Sensor-only mode: the stored position is the controller's estimate, and is labeled so.
    const here = immediate(snap.action('rememberPlace', { name: 'Start' }));
    expect(here).toEqual({
      taskId: null,
      status: 'succeeded',
      reason: 'Remembered "Start" (0.00, 0.00) m, from the sensor-only position estimate on the map "Single block".',
      measurements: { estimatedX: { value: 0, unit: 'm' }, estimatedY: { value: 0, unit: 'm' } },
      fallback: null,
    });
    const given = immediate(snap.action('rememberPlace', { name: 'Dock', x: 3, y: 1 }));
    expect(given.measurements).toEqual({ givenX: { value: 3, unit: 'm' }, givenY: { value: 1, unit: 'm' } });
    // Inside the block: refused, nothing stored.
    expect(immediate(snap.action('rememberPlace', { name: 'Wall', x: 1.5, y: 0 }))).toMatchObject({ status: 'rejected', reason: 'That position is outside the arena or too close to an obstacle.' });

    // Ground-truth mode stores simulator ground truth, labeled as such.
    s.setMode('ground-truth');
    run(s, 0.2);
    snap = await snapshotOf(s, memory);
    const truth = immediate(snap.action('rememberPlace', { name: 'Start' }));
    expect(truth.reason).toMatch(/^Updated "Start" \(-?0\.0\d, -?0\.0\d\) m, from simulator ground truth on the map "Single block"\.$/);
    expect(Object.keys(truth.measurements)).toEqual(['groundTruthX', 'groundTruthY']);

    snap = await snapshotOf(s, memory);
    expect(snap.caps.places).toEqual(['Dock', 'Start']);
    expect(snap.summary).toContain('Remembered places on this map (world memory): "Dock" (3.00, 1.00) m, from coordinates given in a request; "Start" (');
    expect(immediate(snap.query('getRobotCapabilities', {})).reason).toContain('Tools here: getRobotState, navigateTo, rememberPlace, navigateToPlace, forgetPlace, selectRobot, cancelTask.');

    // Places belong to their map.
    s.setMap('open');
    expect((await snapshotOf(s, memory)).caps.places).toEqual([]);
    s.dispose();
  });

  it('drives to a remembered place by name and reports the measured outcome with the place, then forgets it', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    s.setMap('open');
    const memory = new LongTermMemory([], () => NOW);
    expect(immediate((await snapshotOf(s, memory)).action('rememberPlace', { name: 'dock', x: 1.5, y: 0.5 })).status).toBe('succeeded');
    const snap = await snapshotOf(s, memory);
    const task = snap.action('navigateToPlace', { place: 'DOCK' });
    if (!('run' in task)) throw new Error(task.reason);
    const box: { result: FinalResult | null } = { result: null };
    void task.run(new AbortController().signal).then((r) => (box.result = r));
    for (let i = 0; i < 400 && !box.result; i++) {
      run(s, 0.1);
      await new Promise((r) => setTimeout(r, 0));
    }
    const out = box.result;
    expect(out).not.toBeNull();
    expect(out!.status).toBe('succeeded');
    expect(out!.reason.startsWith('"dock" (1.50, 0.50) m, from coordinates given in a request: ')).toBe(true);
    expect(out!.measurements['groundTruthDistanceToGoal']!.value).toBeLessThan(0.3);
    expect(s.state.goal).toEqual([1.5, 0.5]);

    expect(immediate(snap.action('navigateToPlace', { place: 'kitchen' })).status).toBe('rejected');
    expect(immediate(snap.action('forgetPlace', { place: 'Dock' }))).toMatchObject({ status: 'succeeded', reason: 'Forgot "dock" on the map "Open floor".' });
    expect(memory.places('open')).toEqual([]);
    s.dispose();
  });
});
