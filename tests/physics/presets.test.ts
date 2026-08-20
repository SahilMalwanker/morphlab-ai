import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { pdControl, pdStabilityMargin, suggestGains } from '../../src/core/control/pd';
import { DEFAULT_LEGGED, generateArm, generateLegged } from '../../src/core/model/builder';
import { homeConfiguration, PRESETS } from '../../src/core/model/catalog';
import type { RobotDescription } from '../../src/core/model/types';
import { MujocoRuntime } from '../../src/physics/runtime';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

function simulate(desc: RobotDescription, seconds: number) {
  const q0 = homeConfiguration(desc);
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q0 });
  try {
    const minI = Math.min(...rt.jointInertias());
    const { gains } = suggestGains(desc.base.kind === 'floating', minI, rt.timestep);
    expect(pdStabilityMargin(gains, minI, rt.timestep).ok).toBe(true);
    const g = q0.map(() => gains);
    const z0 = rt.groundTruthBaseState().pos[2];
    let finite = true;
    for (let k = 0; k < seconds / rt.timestep; k++) {
      rt.controlStep((s) => pdControl(g, rt.layout.effort, { qCmd: q0, qMeas: s.q, qdMeas: s.qd }).tauApplied);
      if (rt.hasNonFiniteState()) {
        finite = false;
        break;
      }
    }
    const b = rt.groundTruthBaseState();
    const warnings = Object.fromEntries(Object.entries(rt.warningCounts()).filter(([, n]) => n > 0));
    return { finite, z0, z: b.pos[2], tilt: b.tilt, warnings };
  } finally {
    rt.dispose();
  }
}

describe('every preset and generated robot builds and simulates', () => {
  for (const p of PRESETS) {
    it(`preset ${p.label}`, () => {
      const r = simulate(p.make(), 2);
      expect(r.finite).toBe(true);
      expect(r.warnings).toEqual({});
      // Floating robots stay up (no collapse, no tip-over); fixed bases cannot move.
      expect(r.z).toBeGreaterThan(0.5 * r.z0);
      expect(r.tilt).toBeLessThan(0.2);
    });
  }

  for (let pairs = 1; pairs <= 4; pairs++) {
    for (let dof = 1; dof <= 4; dof++) {
      it(`generated legged robot, ${pairs} pair(s), ${dof} joint(s) per leg`, () => {
        const r = simulate(generateLegged({ ...DEFAULT_LEGGED, pairs, dofPerLeg: dof }), 2);
        expect(r.finite).toBe(true);
        expect(r.warnings).toEqual({});
        if (pairs >= 2 && dof >= 2) {
          expect(r.z).toBeGreaterThan(0.5 * r.z0);
          expect(r.tilt).toBeLessThan(0.2);
        }
      });
    }
  }

  it('generated arms with 1 to 7 joints', () => {
    for (let dof = 1; dof <= 7; dof++) {
      const r = simulate(generateArm(dof, 0.8), 1);
      expect(r.finite).toBe(true);
      expect(r.warnings).toEqual({});
    }
  });
});
