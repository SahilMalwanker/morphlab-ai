// Walking controller of the hexapod (D28, D29; docs/physical-ai-plan.md Phase 2) with the default settings, one thread:
// the calibration, the held-out comparison of the feedback policy, and the evaluation over randomized commands. A measurement.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { hexapod } from '../../src/core/model/presets';
import { DEFAULT_LOCOMOTION } from '../../src/skills/locomotion';
import { measureLocomotion } from './locomotionMeasure';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('hexapod: a walking controller learned from simulated runs', async () => {
  const row = await measureLocomotion(mj, hexapod());
  expect(row.evaluation.trials).toBe(DEFAULT_LOCOMOTION.evaluationTrials);
}, 1_500_000);
