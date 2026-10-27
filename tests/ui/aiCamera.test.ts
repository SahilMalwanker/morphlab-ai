import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import type { BridgeSnapshot, RobotBridge } from '../../src/ai/controller';
import { LongTermMemory } from '../../src/ai/memory';
import { toolsForModel } from '../../src/ai/tools';
import { RAY_ELEVATION, RAY_ORIGIN } from '../../src/nav/stack';
import { workspaceBridge } from '../../src/ui/ai/bridge';
import { NavSession } from '../../src/ui/nav/session';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

async function snapshotOf(bridge: RobotBridge): Promise<BridgeSnapshot> {
  const snap = await bridge.snapshot();
  if ('unsupported' in snap) throw new Error(snap.unsupported);
  return snap;
}

describe('the rover camera and the AI', () => {
  it('never offers the AI a camera image (images cost too much per request, D31)', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    const bridge = workspaceBridge({ mode: () => 'navigate', nav: () => s, skills: () => { throw new Error('not used'); }, memory: () => new LongTermMemory(), control: () => null });
    const snap = await snapshotOf(bridge);
    expect(toolsForModel(snap.caps).map((t) => t.name)).not.toContain('getCameraSnapshot');
    expect(snap.summary).not.toMatch(/getCameraSnapshot|camera image/);
    expect(snap.query('getCameraSnapshot', { width: 256 })).toMatchObject({ status: 'rejected', reason: 'getCameraSnapshot is not available for the rover.' });
    s.dispose();
  });
});

describe('the rover range sensor for the AI', () => {
  it('reports the nine rays as measured distances, with the ray ahead and the nearest hit', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    s.setMap('block');
    const bridge = workspaceBridge({ mode: () => 'navigate', nav: () => s, skills: () => { throw new Error('not used'); }, memory: () => new LongTermMemory(), control: () => null });
    let snap = await snapshotOf(bridge);
    expect(snap.summary).toContain("getRobotState also reports the range sensor: nine rays from -80 to 80 deg (positive to the left), measured from 0.2 m ahead of the rover's center up to 2 m");
    let state = snap.query('getRobotState', {});
    expect(state.reason).toContain('Range sensor: no reading yet.');
    expect(state.measurements['rangeAhead']).toBeUndefined();

    for (let i = 0; i < 200; i++) s.stepOnce();
    // The session publishes its state at most every 100 ms of wall time.
    await new Promise((r) => setTimeout(r, 120));
    s.advance(0);
    snap = await snapshotOf(bridge);
    state = snap.query('getRobotState', {});
    // Independent reference: the block's near face is the plane x = 1.3 m; closed-form range along the ray ahead,
    // which starts 0.2 m ahead of the center and tilts up by RAY_ELEVATION.
    const p = s.state.truth!;
    const ox = p.x + Math.cos(p.yaw) * RAY_ORIGIN[0];
    const expected = (1.3 - ox) / Math.cos(p.yaw) / Math.cos(RAY_ELEVATION);
    expect(state.measurements['rangeAhead']!.value).toBeCloseTo(expected, 2);
    expect(state.measurements['nearestRange']).toEqual(state.measurements['rangeAhead']);
    expect(state.measurements['nearestRangeBearing']).toEqual({ value: 0, unit: 'rad' });
    expect(state.reason).toMatch(/Range sensor, measured from 0\.2 m ahead of the rover's center \(bearing in deg, positive to the left, then range in m; none means no hit within 2 m\): -80 none, -60 none, -40 none, -20 1\.1\d, 0 1\.10, 20 1\.1\d, 40 none, 60 none, 80 none\.$/);

    // A reset drops the old readings until the sensor reports again.
    s.reset();
    expect((await snapshotOf(bridge)).query('getRobotState', {}).reason).toContain('Range sensor: no reading yet.');
    s.dispose();
  });
});
