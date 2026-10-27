import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { PhysicsSession, PUSH_DURATION_S } from '../../src/ui/physics/session';
import type { PresetId } from '../../src/core/model/catalog';
import { DEFAULT_LEGGED, generateLegged } from '../../src/core/model/builder';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

async function ready(preset: PresetId): Promise<PhysicsSession> {
  const s = new PhysicsSession(() => Promise.resolve(mj), preset);
  await s.init();
  expect(s.state.status).toBe('ready');
  return s;
}

function run(s: PhysicsSession, seconds: number): void {
  const n = Math.round(seconds / s.runtime!.timestep);
  for (let i = 0; i < n; i++) s.stepOnce();
}

describe('physics session', () => {
  it('holds the quadruped stance under the default PD gains with torques inside the effort limits', async () => {
    const s = await ready('quadruped');
    run(s, 2);
    const b = s.runtime!.groundTruthBaseState();
    expect(b.pos[2]).toBeGreaterThan(0.24);
    expect(b.pos[2]).toBeLessThan(0.29);
    expect(b.tilt).toBeLessThan(0.05);
    const tel = s.telemetry!;
    expect(tel.length).toBeGreaterThan(100);
    const effort = s.runtime!.layout.effort;
    for (let j = 0; j < effort.length; j++) {
      const applied = tel.channelIndex(`tauApplied${j}`);
      for (let k = 0; k < tel.length; k++) expect(Math.abs(tel.value(k, applied))).toBeLessThanOrEqual(effort[j]! + 1e-12);
    }
    s.dispose();
  });

  it('applies new gains without rebuilding the model', async () => {
    const s = await ready('quadruped');
    const rt = s.runtime;
    run(s, 0.2);
    const t = s.runtime!.time;
    expect(s.setGains({ kp: 60, kd: 1.5 })).toBe(true);
    expect(s.runtime).toBe(rt);
    expect(s.runtime!.time).toBe(t);
    expect(s.state.gains).toEqual({ kp: 60, kd: 1.5 });
    s.dispose();
  });

  it('refuses gains beyond the explicit stability bound and keeps the previous gains', async () => {
    const s = await ready('quadruped');
    const before = s.state.gains;
    expect(s.setGains({ kp: 5000, kd: 1 })).toBe(false);
    expect(s.state.gainError).toMatch(/^Refused/);
    expect(s.state.gains).toEqual(before);
    expect(s.setGains({ kp: Number.NaN, kd: 1 })).toBe(false);
    expect(s.setGains({ kp: 10, kd: -1 })).toBe(false);
    expect(s.state.gains).toEqual(before);
    s.dispose();
  });

  it('reset restores the initial state exactly and clears telemetry', async () => {
    const s = await ready('quadruped');
    const rt = s.runtime!;
    const q0 = Array.from(rt.readSensors().q);
    const base0 = rt.groundTruthBaseState().pos.slice();
    s.setTarget(2, -2.0);
    s.push([1, 0, 0], 3);
    run(s, 0.5);
    expect(Array.from(rt.readSensors().q)).not.toEqual(q0);
    s.reset();
    expect(rt.time).toBe(0);
    expect(s.state.simTime).toBe(0);
    expect(Array.from(rt.readSensors().q)).toEqual(q0);
    expect(rt.groundTruthBaseState().pos).toEqual(base0);
    expect(s.telemetry!.length).toBe(0);
    expect(s.state.qCmd[2]).toBeCloseTo(-1.4, 12);
    s.dispose();
  });

  it('a push changes the base velocity, is logged, and the stance recovers', async () => {
    const s = await ready('quadruped');
    run(s, 1);
    const v0 = s.runtime!.groundTruthBaseState().linVel[1];
    s.push([0, 1, 0], 2);
    expect(s.state.events[0]!.text).toMatch(/^Push 2\.0 N\*s along \(0, 1, 0\) at the base/);
    run(s, PUSH_DURATION_S);
    const v1 = s.runtime!.groundTruthBaseState().linVel[1];
    expect(v1 - v0).toBeGreaterThan(0.1);
    run(s, 2);
    const b = s.runtime!.groundTruthBaseState();
    expect(b.tilt).toBeLessThan(0.1);
    expect(b.pos[2]).toBeGreaterThan(0.2);
    s.dispose();
  });

  it('clamps commanded targets to joint limits and ignores non-finite targets', async () => {
    const s = await ready('quadruped');
    s.setTarget(2, -10);
    expect(s.state.qCmd[2]).toBe(-2.7);
    s.setTarget(2, Number.POSITIVE_INFINITY);
    expect(s.state.qCmd[2]).toBe(-2.7);
    s.dispose();
  });

  it('switching presets disposes the previous runtime', async () => {
    const s = await ready('quadruped');
    const old = s.runtime!;
    s.setPreset('arm3');
    expect(old.disposed).toBe(true);
    expect(s.runtime).not.toBe(old);
    expect(s.state.qCmd).toHaveLength(3);
    expect(s.state.description!.base.kind).toBe('fixed');
    s.dispose();
    expect(s.runtime).toBeNull();
  });

  it('caps substeps per frame so a long frame cannot stall the page', async () => {
    const s = await ready('arm3');
    expect(s.advance(1.0)).toBe(25);
    expect(s.advance(0.004)).toBe(2);
    s.setRunning(false);
    expect(s.advance(0.1)).toBe(0);
    s.dispose();
  });

  it('simulates a builder robot, including one sent before the engine finished loading', async () => {
    const desc = generateLegged({ ...DEFAULT_LEGGED, pairs: 3, dofPerLeg: 3 });
    const s = new PhysicsSession(() => Promise.resolve(mj));
    expect(s.loadRobot(desc)).toBe(true);
    await Promise.all([s.init(), s.init()]);
    expect(s.state.presetId).toBe('custom');
    expect(s.state.description).toBe(desc);
    expect(s.state.qCmd).toHaveLength(18);
    run(s, 1.5);
    const b = s.runtime!.groundTruthBaseState();
    expect(b.tilt).toBeLessThan(0.1);
    expect(b.pos[2]).toBeGreaterThan(0.5 * desc.base.pose.xyz[2]);
    const bad = structuredClone(desc);
    bad.limbs[0]!.joints[0]!.link.mass = -1;
    const rt = s.runtime;
    expect(s.loadRobot(bad)).toBe(false);
    expect(s.runtime).toBe(rt);
    expect(s.state.events[0]!.text).toBe('Refused robot: limbs[0].joints[0].link.mass: Link mass must be positive.');
    s.dispose();
  });
});
