// Spike S2: physics models generated from JSON descriptions with N limbs x M joints, rebuilt repeatedly.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { pdControl } from '../../src/core/control/pd';
import { buildKinematicModel, zeroConfiguration } from '../../src/core/kinematics/fk';
import { arm3, hexapod, quadruped, quadrupedStance, radialLegged, rover, serialChain } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { MujocoRuntime } from '../../src/physics/runtime';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

interface Config {
  label: string;
  desc: RobotDescription;
  q0?: number[];
}

const configs: Config[] = [
  { label: 'quadruped 4x3', desc: quadruped(), q0: quadrupedStance() },
  { label: 'hexapod 6x3', desc: hexapod() },
  { label: 'tripod 3x2 (odd)', desc: radialLegged(3, 2) },
  { label: 'pentapod 5x1 (odd)', desc: radialLegged(5, 1) },
  { label: 'biped 2x4', desc: radialLegged(2, 4) },
  { label: 'octopod 8x3', desc: radialLegged(8, 3) },
  { label: 'monopod 1x8 (odd)', desc: radialLegged(1, 8) },
  { label: 'fixed arm 1x3', desc: arm3() },
  { label: 'fixed chain 1x7', desc: serialChain({ axes: ['z', 'y', 'y', 'x', 'y', 'x', 'z'], lengths: [0.2, 0.25, 0.2, 0.08, 0.08, 0.06, 0.05] }) },
  { label: 'rover 4 wheels', desc: rover() },
];

it('S2: generated models compile with the expected sizes and simulate 2 s without NaN', () => {
  const rows: Record<string, unknown>[] = [];
  for (const c of configs) {
    const km = buildKinematicModel(c.desc);
    const q0 = c.q0 ?? zeroConfiguration(km);
    const t0 = performance.now();
    const rt = MujocoRuntime.create(mj, c.desc, { initialQ: q0 });
    const buildMs = performance.now() - t0;
    try {
      const floating = c.desc.base.kind === 'floating';
      const expected = { nq: km.dof + (floating ? 7 : 0), nv: km.dof + (floating ? 6 : 0), nu: km.dof };
      expect({ nq: rt.layout.nq, nv: rt.layout.nv, nu: rt.layout.nu }).toEqual(expected);
      const gains = q0.map(() => ({ kp: 20, kd: 0.5 }));
      let nonFinite = false;
      for (let i = 0; i < 2 / rt.timestep; i++) {
        rt.controlStep((s) => pdControl(gains, rt.layout.effort, { qCmd: q0, qMeas: s.q, qdMeas: s.qd }).tauApplied);
        if (rt.hasNonFiniteState()) { nonFinite = true; break; }
      }
      const w = rt.warningCounts();
      rows.push({ config: c.label, ...expected, buildMs: +buildMs.toFixed(1), simSeconds: +rt.time.toFixed(2), nonFinite, badqacc: w.badqacc, baseZ: +rt.groundTruthBaseState().pos[2].toFixed(3) });
      expect(nonFinite).toBe(false);
      expect(w.badqacc).toBe(0);
    } finally {
      rt.dispose();
    }
  }
  console.table(rows);
  expect(rows.length).toBeGreaterThanOrEqual(6);
});

it('S2: 50 live rebuilds do not grow WASM memory beyond 5 %', () => {
  const cycle = (n: number) => {
    let bytes = 0;
    for (let k = 0; k < n; k++) {
      const c = configs[k % configs.length]!;
      const km = buildKinematicModel(c.desc);
      const rt = MujocoRuntime.create(mj, c.desc, { initialQ: c.q0 ?? zeroConfiguration(km) });
      for (let i = 0; i < 50; i++) rt.step();
      rt.warningCounts();
      bytes = rt.wasmMemoryBytes();
      rt.dispose();
    }
    return bytes;
  };
  const warm = cycle(configs.length * 2);
  const after = cycle(50);
  console.log('S2 memory', JSON.stringify({ warmBytes: warm, after50RebuildsBytes: after, growth: +(after / warm - 1).toFixed(4) }));
  expect(after).toBeLessThanOrEqual(warm * 1.05);
});
