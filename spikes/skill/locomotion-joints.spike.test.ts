// Comparison for D29: the quadruped's walking controller when the policy also corrects every joint target, besides
// changing the generator's command (the default changes the command only). Default settings otherwise, one thread. A measurement.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { quadruped } from '../../src/core/model/presets';
import { DEFAULT_LOCOMOTION } from '../../src/skills/locomotion';
import { measureLocomotion } from './locomotionMeasure';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('quadruped: a walking controller whose policy also corrects the joints', async () => {
  const row = await measureLocomotion(mj, quadruped(), true);
  expect(row.evaluation.trials).toBe(DEFAULT_LOCOMOTION.evaluationTrials);
}, 1_500_000);
