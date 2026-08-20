import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { pdControl, secondOrderParameters } from '../../src/core/control/pd';
import { analyticStepMetrics, stepMetrics } from '../../src/core/control/stepMetrics';
import { joint } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { MujocoRuntime } from '../../src/physics/runtime';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

/** One hinge about the vertical axis: gravity exerts no torque, so the joint is a pure inertia. */
function turntable(): RobotDescription {
  const j = joint({ id: 'yaw', name: 'Yaw', xyz: [0, 0, 0.05], rpy: [Math.PI / 2, 0, 0], axis: [0, 1, 0], lower: -3, upper: 3, effort: 50, radius: 0.03, mass: 1 });
  j.damping = 0;
  j.armature = 0.01;
  return {
    schemaVersion: 1,
    id: 'turntable',
    name: 'Turntable',
    base: { kind: 'fixed', size: [0.1, 0.1, 0.05], mass: 1, pose: { xyz: [0, 0, 0.025], rpy: [0, 0, 0] } },
    limbs: [{ id: 'arm', name: 'Arm', role: 'arm', joints: [j], tip: { xyz: [0, 0, 0.3], radius: 0.02 } }],
  };
}

function simulateStep(kp: number, kd: number, yRef: number, duration: number) {
  const rt = MujocoRuntime.create(mj, turntable(), { initialQ: [0] });
  try {
    const I = rt.jointInertias()[0]!;
    const t: number[] = [], y: number[] = [];
    let maxApplied = 0;
    for (let k = 0; k < duration / rt.timestep; k++) {
      t.push(rt.time);
      rt.controlStep((s) => {
        y.push(s.q[0]!);
        const r = pdControl([{ kp, kd }], rt.layout.effort, { qCmd: [yRef], qMeas: s.q, qdMeas: s.qd });
        maxApplied = Math.max(maxApplied, Math.abs(r.tauApplied[0]!));
        return r.tauApplied;
      });
    }
    return { I, t, y, maxApplied };
  } finally {
    rt.dispose();
  }
}

describe('PD step response in MuJoCo vs the analytical second-order system', () => {
  it('the hinge axis is vertical, so gravity produces no torque', () => {
    const sim = simulateStep(0, 0, 0, 1);
    expect(Math.max(...sim.y.map(Math.abs))).toBeLessThan(1e-9);
  });

  for (const [kp, kd] of [[8, 0.12], [8, 0.3], [20, 0.4]] as const) {
    it(`kp=${kp}, kd=${kd}: overshoot within 2 points, settling within 10 %`, () => {
      const yRef = 0.5;
      const sim = simulateStep(kp, kd, yRef, 4);
      const { wn, zeta } = secondOrderParameters({ kp, kd }, sim.I);
      const measured = stepMetrics(sim.t, sim.y, 0, yRef);
      const ideal = analyticStepMetrics(wn, zeta, 4);
      console.log(JSON.stringify({ kp, kd, inertia: +sim.I.toFixed(5), wn: +wn.toFixed(2), zeta: +zeta.toFixed(3), overshoot: [+measured.overshootPct.toFixed(2), +ideal.overshootPct.toFixed(2)], settling: [measured.settlingTimeS, +ideal.settlingTimeS!.toFixed(3)] }));
      expect(Math.abs(measured.overshootPct - ideal.overshootPct)).toBeLessThan(2);
      expect(Math.abs(measured.settlingTimeS! - ideal.settlingTimeS!) / ideal.settlingTimeS!).toBeLessThan(0.1);
      expect(sim.maxApplied).toBeLessThanOrEqual(50);
    });
  }
});
