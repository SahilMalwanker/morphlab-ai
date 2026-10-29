// Spike S6: can a skill taught as keyframes make the quadruped hop? A measurement, not a pass/fail claim.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { DEFAULT_EVALUATION, evaluateSkill } from '../../src/skills/evaluation';
import type { KeyframeMotion, SkillGoal } from '../../src/skills/taught';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

it('S6: keyframe hop on the quadruped over 20 randomized trials (measured)', () => {
  const desc = quadruped();
  const legs = (hip: number, knee: number) => desc.limbs.flatMap(() => [0, hip, knee]);
  // Crouch slowly, extend as fast as the joints allow (feet stay under the hips), then return to the stance for landing.
  const motion: KeyframeMotion = {
    jointIds: desc.limbs.flatMap((l) => l.joints.map((j) => j.id)),
    keyframes: [
      { t: 0.8, q: legs(1.1, -2.2) },
      { t: 0.9, q: legs(0.35, -0.7) },
      { t: 1.4, q: legs(0.7, -1.4) },
    ],
    endHoldS: 1.5,
  };
  const goals: SkillGoal[] = [
    { kind: 'airborne', minS: 0.02 },
    { kind: 'baseRise', minRiseM: 0.01 },
    { kind: 'standsAtEnd', maxTiltRad: 0.5, minBaseHeightM: 0.15 },
  ];
  const t0 = performance.now();
  const e = evaluateSkill(mj, desc, -1, quadrupedStance(), { kind: 'keyframes', motion }, { ...DEFAULT_EVALUATION, seed: 7 }, { goals });
  const seconds = (performance.now() - t0) / 1000;
  console.table(e.results.map((r) => ({ trial: r.trial, friction: r.conditions.friction, massKg: r.conditions.baseMassKg, success: r.success, airborneS: r.goals![0]!.measured, riseM: r.goals![1]!.measured, endTilt: r.goals![2]!.measured, maxTilt: r.maxTiltRad, minZ: r.minBaseHeightM })));
  console.log('S6 summary', JSON.stringify({ successes: e.successes, trials: e.trials, wilson95: e.wilson95.map((v) => +v.toFixed(3)), goalsMet: goals.map((_, i) => e.results.filter((r) => r.goals![i]!.met).length), wallSeconds: +seconds.toFixed(1) }));
  expect(e.trials).toBe(20);
  expect(e.results.every((r) => r.goals!.length === 3)).toBe(true);
});
