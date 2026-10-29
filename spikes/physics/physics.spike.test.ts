// Spike S1: 12-joint quadruped in MuJoCo WASM under TypeScript PD control, stance and lateral push.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { pdControl, type PdGains } from '../../src/core/control/pd';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { MujocoRuntime } from '../../src/physics/runtime';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

export const STANCE_GAINS: PdGains = { kp: 40, kd: 1.0 };

function percentile(xs: number[], p: number): number {
  const s = xs.slice().sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))]!;
}

it('S1: quadruped holds a PD stance and survives a lateral push', () => {
  const desc = quadruped();
  const qCmd = quadrupedStance();
  const rt = MujocoRuntime.create(mj, desc, { initialQ: qCmd });
  try {
    const gains = qCmd.map(() => STANCE_GAINS);
    const effort = rt.layout.effort;
    const dt = rt.timestep;
    let saturatedSteps = 0;
    const tick = () => {
      rt.controlStep((s) => {
        const r = pdControl(gains, effort, { qCmd, qMeas: s.q, qdMeas: s.qd });
        if (r.saturated.some((v) => v === 1)) saturatedSteps++;
        return r.tauApplied;
      });
    };
    // Settle 1 s after the 1.3 cm drop onto the ground.
    for (let i = 0; i < 1 / dt; i++) tick();
    const stance0 = rt.groundTruthBaseState();
    // Stand 5 s.
    let zMin = Infinity, zMax = -Infinity, tiltMax = 0;
    for (let i = 0; i < 5 / dt; i++) {
      tick();
      const b = rt.groundTruthBaseState();
      zMin = Math.min(zMin, b.pos[2]); zMax = Math.max(zMax, b.pos[2]); tiltMax = Math.max(tiltMax, b.tilt);
    }
    const stanceHeight = rt.groundTruthBaseState().pos[2];
    // Lateral push: 40 N for 0.05 s = 2 N*s at the base centre of mass.
    const pushSteps = Math.round(0.05 / dt);
    for (let i = 0; i < pushSteps; i++) {
      rt.setBodyWrench(rt.layout.baseBodyId, [0, 40, 0]);
      tick();
    }
    rt.clearWrenches();
    let pzMin = Infinity, ptiltMax = 0, lastUnsettled = 0, lateralDisp = 0;
    const y0 = stance0.pos[1];
    for (let i = 0; i < 3 / dt; i++) {
      tick();
      const b = rt.groundTruthBaseState();
      pzMin = Math.min(pzMin, b.pos[2]);
      ptiltMax = Math.max(ptiltMax, b.tilt);
      lateralDisp = Math.max(lateralDisp, Math.abs(b.pos[1] - y0));
      const speed = Math.hypot(...b.linVel);
      if (b.tilt > 0.05 || speed > 0.05) lastUnsettled = (i + 1) * dt;
    }
    // Timing: controller + mj_step per step, batches of 50 steps.
    const batch: number[] = [];
    for (let k = 0; k < 60; k++) {
      const t0 = performance.now();
      for (let i = 0; i < 50; i++) tick();
      batch.push((performance.now() - t0) / 50);
    }
    const warm = batch.slice(10);
    const summary = {
      stanceHeightM: +stanceHeight.toFixed(4),
      standHeightDriftM: +(zMax - zMin).toFixed(4),
      standTiltMaxRad: +tiltMax.toFixed(4),
      pushImpulseNs: 40 * pushSteps * dt,
      pushMinHeightM: +pzMin.toFixed(4),
      pushMinHeightFraction: +(pzMin / stanceHeight).toFixed(3),
      pushTiltMaxRad: +ptiltMax.toFixed(4),
      pushLateralDisplacementM: +lateralDisp.toFixed(4),
      pushSettleTimeS: +lastUnsettled.toFixed(3),
      saturatedStepFraction: +(saturatedSteps / ((1 + 5 + 0.05 + 3) / dt + 3000)).toFixed(4),
      stepMsMedian: +percentile(warm, 0.5).toFixed(4),
      stepMsP95: +percentile(warm, 0.95).toFixed(4),
      warnings: rt.warningCounts(),
      nonFinite: rt.hasNonFiniteState(),
    };
    console.log('S1 summary', JSON.stringify(summary, null, 1));
    expect(summary.nonFinite).toBe(false);
    expect(summary.standHeightDriftM).toBeLessThan(0.02);
    expect(summary.standTiltMaxRad).toBeLessThan(0.1);
    expect(summary.pushMinHeightFraction).toBeGreaterThan(0.6);
    expect(summary.pushTiltMaxRad).toBeLessThan(0.5);
    expect(summary.pushSettleTimeS).toBeLessThan(2);
    expect(summary.stepMsMedian).toBeLessThan(0.25);
  } finally {
    rt.dispose();
  }
});
