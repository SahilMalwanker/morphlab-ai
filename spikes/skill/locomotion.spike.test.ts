// Walking controller of the quadruped (D28, D29; docs/physical-ai-plan.md Phase 2) with the default settings, one thread:
// the calibration, the held-out comparison of the feedback policy, and the evaluation over randomized commands. A measurement.
// The hexapod runs in locomotion-hexapod.spike.test.ts, so the two can run in parallel.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { quadruped } from '../../src/core/model/presets';
import { DEFAULT_LOCOMOTION } from '../../src/skills/locomotion';
import { measureLocomotion } from './locomotionMeasure';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('quadruped: a walking controller learned from simulated runs', async () => {
  const row = await measureLocomotion(mj, quadruped());
  expect(row.evaluation.trials).toBe(DEFAULT_LOCOMOTION.evaluationTrials);
}, 1_500_000);
