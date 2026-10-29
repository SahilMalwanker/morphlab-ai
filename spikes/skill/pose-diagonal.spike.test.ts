// Standing on two diagonal legs (D30): the quadruped lifts its front left and rear right feet and balances on the other
// two. The center of mass sits over the line between the two feet that stay down, an unstable balance that open-loop
// keyframes cannot hold; a feedback policy trained by reinforcement learning under pushes has to. A measurement.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { quadruped } from '../../src/core/model/presets';
import { measurePose } from './poseMeasure';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('quadruped: balancing on two diagonal legs, open loop against a learned feedback policy', async () => {
  const lift = [{ limb: 'front_left', z: 0.06 }, { limb: 'rear_right', z: 0.06 }];
  const row = await measurePose(
    mj,
    quadruped(),
    'two diagonal legs',
    [{ seconds: 0.6, tips: lift }, { seconds: 3, tips: lift }, { seconds: 0.6 }],
    0.5,
    [
      { kind: 'airborne', limbIds: ['front_left', 'rear_right'], minS: 1.5 },
      { kind: 'upright', maxTiltRad: 0.5 },
      { kind: 'noFall', minBaseHeightM: 0.15 },
    ],
  );
  expect(row.heldOut.trials).toBeGreaterThan(0);
}, 1_500_000);
