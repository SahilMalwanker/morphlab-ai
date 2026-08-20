import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { joint } from '../../src/core/model/presets';
import { PRESETS } from '../../src/core/model/catalog';
import type { RobotDescription } from '../../src/core/model/types';
import { runChallenge, runToEnd, stepTest, type ControllerSpec } from '../../src/physics/experiments';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const preset = (id: string) => PRESETS.find((p) => p.id === id)!.make();
const pd = (kp: number, kd: number): ControllerSpec => ({ gains: { kp, kd }, gravityCompensation: false });
const pdg = (kp: number, kd: number): ControllerSpec => ({ gains: { kp, kd }, gravityCompensation: true });

function turntable(velocity = 6): RobotDescription {
  const j = joint({ id: 'yaw', name: 'Yaw', xyz: [0, 0, 0.05], rpy: [Math.PI / 2, 0, 0], axis: [0, 1, 0], lower: -3, upper: 3, effort: 50, velocity, radius: 0.03, mass: 1 });
  j.damping = 0;
  return {
    schemaVersion: 1,
    id: 'turntable',
    name: 'Turntable',
    base: { kind: 'fixed', size: [0.1, 0.1, 0.05], mass: 1, pose: { xyz: [0, 0, 0.025], rpy: [0, 0, 0] } },
    limbs: [{ id: 'arm', name: 'Arm', role: 'arm', joints: [j], tip: { xyz: [0, 0, 0.3], radius: 0.02 } }],
  };
}

/** A horizontal slider: one prismatic joint along x, no gravity load on the axis. */
function slider(): RobotDescription {
  const j = joint({ id: 'slide', name: 'Slide', type: 'prismatic', xyz: [0, 0, 0.05], axis: [1, 0, 0], lower: -0.2, upper: 0.2, effort: 40, velocity: 0.5, radius: 0.03, mass: 1 });
  return {
    schemaVersion: 1,
    id: 'slider',
    name: 'Slider',
    base: { kind: 'fixed', size: [0.1, 0.1, 0.05], mass: 1, pose: { xyz: [0, 0, 0.025], rpy: [0, 0, 0] } },
    limbs: [{ id: 'arm', name: 'Arm', role: 'arm', joints: [j], tip: { xyz: [0, 0, 0.1], radius: 0.02 } }],
  };
}

describe('step test', () => {
  it('on a gravity-free hinge the measured response matches the linear prediction (M2 tolerances)', () => {
    const r = runToEnd(stepTest(mj, turntable(), { joint: 0, step: 0.5, settleS: 0.2, durationS: 4, controller: pd(8, 0.3), command: 'ideal-step' }));
    expect(r.command).toBe('ideal-step');
    expect(r.unstable).toBe(false);
    expect(Math.abs(r.metrics.overshootPct - r.linear!.overshootPct)).toBeLessThan(2);
    expect(Math.abs(r.metrics.settlingTimeS! - r.linear!.settlingTimeS!) / r.linear!.settlingTimeS!).toBeLessThan(0.1);
    expect(Math.abs(r.metrics.riseTimeS! - r.linear!.riseTimeS!) / r.linear!.riseTimeS!).toBeLessThan(0.05);
    expect(r.t[0]).toBe(0);
    expect(r.y0).toBe(0);
  });

  it('by default the command ramps at the velocity limit and the linear model gets the same ramp', () => {
    const r = runToEnd(stepTest(mj, turntable(1), { joint: 0, step: 0.5, settleS: 0.2, durationS: 4, controller: pd(8, 0.3) }));
    expect(r.command).toBe('rate-limited');
    expect(r.velocityLimit).toBe(1);
    const h = r.t[1]! - r.t[0]!;
    for (let i = 1; i < r.qCmd.length; i++) expect(Math.abs(r.qCmd[i]! - r.qCmd[i - 1]!) / h).toBeLessThanOrEqual(1 + 1e-9);
    expect(r.qCmd.at(-1)).toBeCloseTo(0.5, 12);
    // 0.5 rad at 1 rad/s: held back for 0.5 s.
    expect(r.rateLimitedS).toBeGreaterThan(0.49);
    expect(r.rateLimitedS).toBeLessThan(0.51);
    // Same M2 tolerances against the linear model driven by the same command.
    expect(Math.abs(r.metrics.overshootPct - r.linear!.overshootPct)).toBeLessThan(2);
    expect(Math.abs(r.metrics.settlingTimeS! - r.linear!.settlingTimeS!) / r.linear!.settlingTimeS!).toBeLessThan(0.1);
    expect(Math.abs(r.metrics.riseTimeS! - r.linear!.riseTimeS!) / r.linear!.riseTimeS!).toBeLessThan(0.05);
    expect(r.linearQ).toHaveLength(r.t.length);
    // The ramp is slower than the ideal jump.
    const ideal = runToEnd(stepTest(mj, turntable(1), { joint: 0, step: 0.5, settleS: 0.2, durationS: 4, controller: pd(8, 0.3), command: 'ideal-step' }));
    expect(r.metrics.riseTimeS!).toBeGreaterThan(ideal.metrics.riseTimeS!);
    expect(ideal.rateLimitedS).toBe(0);
  });

  it('a prismatic joint steps in metres with its own velocity limit', () => {
    const r = runToEnd(stepTest(mj, slider(), { joint: 0, step: 0.1, settleS: 0.2, durationS: 2, controller: pd(400, 40) }));
    expect(r.unstable).toBe(false);
    expect(r.velocityLimit).toBe(0.5);
    expect(r.rateLimitedS).toBeGreaterThan(0.19);
    expect(r.rateLimitedS).toBeLessThan(0.21);
    expect(r.target).toBeCloseTo(0.1, 12);
    expect(Math.abs(r.metrics.steadyStateError)).toBeLessThan(1e-3);
    expect(Math.abs(r.metrics.overshootPct - r.linear!.overshootPct)).toBeLessThan(2);
  });

  it('gravity compensation removes the steady-state sag that PD alone leaves', () => {
    const cfg = { joint: 1, step: 0.3, settleS: 0.5, durationS: 2 };
    const plain = runToEnd(stepTest(mj, preset('arm3'), { ...cfg, controller: pd(60, 3) }));
    const comp = runToEnd(stepTest(mj, preset('arm3'), { ...cfg, controller: pdg(60, 3) }));
    expect(Math.abs(plain.metrics.steadyStateError)).toBeGreaterThan(0.01);
    expect(Math.abs(comp.metrics.steadyStateError)).toBeLessThan(1e-3);
  });

  it('reports saturation and never applies more than the effort limit', () => {
    // An instant 1.5 rad jump demands more torque than the elbow has.
    const r = runToEnd(stepTest(mj, preset('arm3'), { joint: 2, step: 1.5, settleS: 0.2, durationS: 1, controller: pd(150, 4), command: 'ideal-step' }));
    expect(r.saturationFraction).toBeGreaterThan(0);
    expect(Math.max(...r.tauCmd.map(Math.abs))).toBeGreaterThan(r.effort);
    expect(Math.max(...r.tauApplied.map(Math.abs))).toBeLessThanOrEqual(r.effort);
    // The same step ramped at the 6 rad/s velocity limit stays within the effort limit.
    const ramp = runToEnd(stepTest(mj, preset('arm3'), { joint: 2, step: 1.5, settleS: 0.2, durationS: 1, controller: pd(150, 4) }));
    expect(ramp.saturationFraction).toBeLessThan(r.saturationFraction);
    expect(Math.max(...ramp.tauApplied.map(Math.abs))).toBeLessThanOrEqual(ramp.effort);
  });

  it('clamps a step beyond the joint limit and says so', () => {
    const r = runToEnd(stepTest(mj, preset('arm3'), { joint: 1, step: 5, settleS: 0.1, durationS: 0.5, controller: pdg(60, 3) }));
    expect(r.clamped).toBe(true);
    expect(r.target).toBe(1.75);
  });

  it('floating robots are tested on a stand where the limbs cannot reach the ground', () => {
    const r = runToEnd(stepTest(mj, preset('hexapod'), { joint: 1, step: 0.3, settleS: 0.5, durationS: 2, controller: pdg(40, 1) }));
    expect(r.onStand).toBe(true);
    expect(r.saturationFraction).toBeLessThan(0.05);
    expect(Math.abs(r.metrics.steadyStateError)).toBeLessThan(1e-3);
  });

  it('rejects invalid configurations', () => {
    expect(() => runToEnd(stepTest(mj, preset('arm3'), { joint: 9, step: 0.3, settleS: 0.1, durationS: 0.5, controller: pd(60, 3) }))).toThrow('Joint index out of range');
    expect(() => runToEnd(stepTest(mj, preset('arm3'), { joint: 0, step: Number.NaN, settleS: 0.1, durationS: 0.5, controller: pd(60, 3) }))).toThrow('Step size must be a non-zero finite number');
  });
});

describe('challenges are scored from measured data with objective thresholds', () => {
  it('reach: PD alone misses because of gravity sag; gravity compensation passes', () => {
    const plain = runToEnd(runChallenge(mj, preset('arm3'), 'reach', pd(60, 3)));
    const comp = runToEnd(runChallenge(mj, preset('arm3'), 'reach', pdg(60, 3)));
    expect(plain.passed).toBe(false);
    expect(plain.criteria[0]!.measured).toBeGreaterThan(10);
    expect(comp.passed).toBe(true);
    expect(comp.criteria.every((c) => c.source === 'encoders')).toBe(true);
  });

  it('hold: stiffer gains or gravity compensation pass, soft gains fail', () => {
    expect(runToEnd(runChallenge(mj, preset('arm3'), 'hold', pd(60, 3))).passed).toBe(false);
    expect(runToEnd(runChallenge(mj, preset('arm3'), 'hold', pd(150, 4))).passed).toBe(true);
    expect(runToEnd(runChallenge(mj, preset('arm3'), 'hold', pdg(60, 3))).passed).toBe(true);
  });

  it('hold: prismatic joints are judged in metres, revolute joints in radians', () => {
    const arm = runToEnd(runChallenge(mj, preset('arm3'), 'hold', pdg(60, 3)));
    expect(arm.criteria.map((c) => c.unit)).toEqual(['rad', 'rad/s']);
    const r = runToEnd(runChallenge(mj, slider(), 'hold', pd(400, 40)));
    expect(r.criteria.map((c) => [c.label, c.unit, c.threshold])).toEqual([
      ['Largest prismatic joint error in the last second', 'm', 0.005],
      ['Largest prismatic joint speed in the last second', 'm/s', 0.0125],
    ]);
    expect(r.passed).toBe(true);
  });

  it('push: the quadruped recovers with the default gains and falls behind with soft gains', () => {
    const ok = runToEnd(runChallenge(mj, preset('quadruped'), 'push', pd(40, 1)));
    const soft = runToEnd(runChallenge(mj, preset('quadruped'), 'push', pd(10, 0.3)));
    expect(ok.passed).toBe(true);
    expect(soft.passed).toBe(false);
    expect(ok.criteria.map((c) => c.source)).toEqual(['IMU', 'ground truth', 'IMU']);
  });

  it('challenges that do not fit the robot say why', () => {
    expect(runToEnd(runChallenge(mj, preset('quadruped'), 'reach', pd(40, 1)))).toMatchObject({ applicable: false, reason: 'Reaching needs a fixed-base arm; this robot has a floating base.' });
    expect(runToEnd(runChallenge(mj, preset('arm3'), 'push', pd(60, 3)))).toMatchObject({ applicable: false, reason: 'The push test needs a free-standing robot; this robot is fixed to the ground.' });
  });
});
