import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { gravityTorques } from '../../src/core/dynamics/gravity';
import { buildKinematicModel, randomConfiguration } from '../../src/core/kinematics/fk';
import { mulberry32 } from '../../src/core/math/rng';
import { DEFAULT_LEGGED, generateArm, generateLegged, placeOnGround } from '../../src/core/model/builder';
import { PRESETS } from '../../src/core/model/catalog';
import { joint } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { MujocoRuntime } from '../../src/physics/runtime';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

function onStand(desc: RobotDescription): RobotDescription {
  return placeOnGround({ ...desc, base: { ...desc.base, kind: 'fixed', pose: { xyz: [0.1, -0.2, 0], rpy: [0.1, -0.05, 0.7] } } });
}

const skewed: RobotDescription = {
  schemaVersion: 1,
  id: 'skewed',
  name: 'Skewed chain with a prismatic joint',
  base: { kind: 'fixed', size: [0.1, 0.1, 0.1], mass: 1, pose: { xyz: [0, 0, 0.05], rpy: [0, 0, 0] } },
  limbs: [
    {
      id: 'arm',
      name: 'Arm',
      role: 'arm',
      joints: [
        joint({ id: 'a1', name: 'A1', xyz: [0.02, 0.01, 0.05], rpy: [0.3, -0.4, 0.7], axis: [0.2, 0.9, -0.3], lower: -2, upper: 2, effort: 50, radius: 0.02, mass: 0.7 }),
        joint({ id: 'a2', name: 'A2', type: 'prismatic', xyz: [0.05, -0.1, 0.2], rpy: [-0.5, 0.2, 0.1], axis: [0.3, -0.5, 0.8], lower: -0.1, upper: 0.2, effort: 50, radius: 0.02, mass: 0.4 }),
        joint({ id: 'a3', name: 'A3', xyz: [0, 0.07, 0.15], rpy: [0.1, 0.9, -1.2], axis: [1, 0, 0], lower: -2, upper: 2, effort: 50, radius: 0.02, mass: 0.3 }),
      ],
      tip: { xyz: [0.02, -0.03, 0.1], radius: 0.015 },
    },
  ],
};

describe('gravity compensation torques vs MuJoCo bias forces at rest', () => {
  const robots: RobotDescription[] = [
    ...PRESETS.filter((p) => p.make().base.kind === 'fixed').map((p) => p.make()),
    generateArm(7, 0.9),
    skewed,
    onStand(PRESETS.find((p) => p.id === 'quadruped')!.make()),
    onStand(generateLegged({ ...DEFAULT_LEGGED, pairs: 3, dofPerLeg: 4 })),
    onStand(PRESETS.find((p) => p.id === 'rover')!.make()),
  ];
  for (const desc of robots) {
    it(`${desc.name}: agrees to 1e-9 N*m at 20 random configurations`, () => {
      const km = buildKinematicModel(desc);
      const rng = mulberry32(11);
      const rt = MujocoRuntime.create(mj, desc);
      try {
        for (let t = 0; t < 20; t++) {
          const q = randomConfiguration(km, rng).map((v) => (Number.isFinite(v) ? v : 0));
          rt.setState(q);
          const ref = rt.biasForces();
          const ours = gravityTorques(km, q);
          ours.forEach((v, i) => expect(Math.abs(v - ref[i]!)).toBeLessThan(1e-9));
        }
      } finally {
        rt.dispose();
      }
    });
  }
});
