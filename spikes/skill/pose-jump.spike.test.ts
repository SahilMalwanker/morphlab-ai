// A jump that lands on its feet (D30): crouch, extend within 0.1 s so every foot leaves the ground, land and stand.
// Each run is pushed once at a random moment; the feedback policy, trained by reinforcement learning under pushes,
// has to keep the body upright through the flight and the landing. A measurement.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { quadruped } from '../../src/core/model/presets';
import { measurePose } from './poseMeasure';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('quadruped: a jump that lands on its feet, open loop against a learned feedback policy', async () => {
  const row = await measurePose(
    mj,
    quadruped(),
    'jump and land',
    [{ seconds: 0.5, body: { z: -0.08 } }, { seconds: 0.1, body: { z: 0.06 } }, { seconds: 0.15 }, { seconds: 0.6 }],
    1,
    [
      { kind: 'airborne', minS: 0.05 },
      { kind: 'upright', maxTiltRad: 0.6 },
      { kind: 'standsAtEnd', maxTiltRad: 0.35, minBaseHeightM: 0.15 },
    ],
  );
  expect(row.heldOut.trials).toBeGreaterThan(0);
}, 1_500_000);
