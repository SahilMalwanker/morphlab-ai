import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { PRESETS } from '../../src/core/model/catalog';
import { ControlSession } from '../../src/ui/control/session';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

async function ready(): Promise<ControlSession> {
  const s = new ControlSession(() => Promise.resolve(mj));
  await s.init();
  expect(s.state.status).toBe('ready');
  return s;
}

describe('control lab session', () => {
  it('compares two controllers on the same step and reports stability notes', async () => {
    const s = await ready();
    expect(s.state.stability!.a.ok).toBe(true);
    await s.runStepTest();
    const { stepA, stepB } = s.state;
    expect(stepA).not.toBeNull();
    expect(stepB).not.toBeNull();
    expect(stepA!.t).toHaveLength(stepB!.t.length);
    expect(Math.abs(stepB!.metrics.steadyStateError)).toBeLessThan(Math.abs(stepA!.metrics.steadyStateError));
    expect(s.state.running).toBeNull();
  });

  it('runs a challenge for both gain sets and keeps the verdicts', async () => {
    const s = await ready();
    await s.runChallenge('hold');
    const r = s.state.challenges.hold!;
    expect(r.a.passed).toBe(false);
    expect(r.b!.passed).toBe(true);
  });

  it('switching robots clears results and marks floating robots as tested on a stand', async () => {
    const s = await ready();
    await s.runStepTest();
    s.setPreset('quadruped');
    expect(s.state.stepA).toBeNull();
    expect(s.state.onStand).toBe(true);
    await s.runChallenge('reach');
    expect(s.state.challenges.reach!.a.applicable).toBe(false);
    expect(s.state.challenges.reach!.b).toBeNull();
  });

  it('rejects invalid gains and invalid robots without changing state', async () => {
    const s = await ready();
    const before = s.state.a;
    expect(s.setController('a', { kp: Number.NaN })).toBe(false);
    expect(s.setController('a', { kd: -1 })).toBe(false);
    expect(s.state.a).toBe(before);
    const bad = PRESETS[0]!.make();
    bad.base.mass = 0;
    expect(s.setRobot(bad)).toBe(false);
    expect(s.state.message).toBe('Refused robot: base.mass: Base mass must be a positive number of kilograms.');
    expect(s.setController('b', { kp: 5000 })).toBe(true);
    expect(s.state.stability!.b.ok).toBe(false);
  });

  it('a cancelled run leaves no partial result', async () => {
    const s = await ready();
    const p = s.runStepTest();
    s.cancel();
    await p;
    expect(s.state.stepA).toBeNull();
    expect(s.state.running).toBeNull();
  });

  it('changing the setup cancels a running experiment; results keep the setup they ran with', async () => {
    const s = await ready();
    const p = s.runStepTest();
    expect(s.setController('a', { kp: 80 })).toBe(true);
    await p;
    expect(s.state.stepA).toBeNull();
    expect(s.state.running).toBeNull();
    await s.runStepTest();
    expect(s.state.stepSetup).toMatchObject({ joint: 1, step: 20, durationS: 2, command: 'rate-limited', a: { gains: { kp: 80, kd: 3 } }, b: { gravityCompensation: true } });
    expect(s.state.stepA!.command).toBe('rate-limited');
    const q = s.runStepTest();
    s.setCommand('ideal-step');
    await q;
    // The earlier results stay, labeled with the setup they were run with.
    expect(s.state.stepSetup!.command).toBe('rate-limited');
    expect(s.state.command).toBe('ideal-step');
    await s.runStepTest();
    expect(s.state.stepA!.command).toBe('ideal-step');
    expect(s.state.stepSetup!.command).toBe('ideal-step');
  });
});
