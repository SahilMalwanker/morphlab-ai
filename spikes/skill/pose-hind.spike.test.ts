// Standing on the hind legs (D30): a rough sketch in round numbers (pitch nose up, shift back, front feet up, hold,
// come down) that does not balance by itself; the feedback policy, trained by reinforcement learning under pushes,
// has to find the balance on the two rear feet and bring the body back down onto four feet. A measurement.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { quadruped } from '../../src/core/model/presets';
import { measurePose } from './poseMeasure';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('quadruped: standing on the hind legs, open loop against a learned feedback policy', async () => {
  const up = { x: -0.15, pitch: -0.8 };
  const tuck = [{ limb: 'front_left', x: -0.15, z: 0.2 }, { limb: 'front_right', x: -0.15, z: 0.2 }];
  const row = await measurePose(
    mj,
    quadruped(),
    'hind legs',
    [{ seconds: 1, body: up, tips: tuck }, { seconds: 2.5, body: up, tips: tuck }, { seconds: 1 }],
    0.5,
    [
      { kind: 'bodyAngle', axis: 'pitch', minRad: -1.1, maxRad: -0.5, holdS: 1.5 },
      { kind: 'airborne', limbIds: ['front_left', 'front_right'], minS: 1.5 },
      { kind: 'noFall', minBaseHeightM: 0.1 },
      { kind: 'standsAtEnd', maxTiltRad: 0.35, minBaseHeightM: 0.15 },
    ],
  );
  expect(row.heldOut.trials).toBeGreaterThan(0);
}, 1_500_000);
