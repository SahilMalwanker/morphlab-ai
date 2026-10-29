// Gait sweep: how far and how straight the contact-pattern gaits move the presets. A measurement, not a pass/fail claim.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { homeConfiguration } from '../../src/core/model/catalog';
import { hexapod, quadruped } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { DEFAULT_EVALUATION, evaluateSkill } from '../../src/skills/evaluation';
import { DEFAULT_GAIT, type GaitParams } from '../../src/skills/gait';
import type { SkillGoal } from '../../src/skills/taught';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const goals: SkillGoal[] = [
  { kind: 'travel', direction: 'forward', minDistanceM: 0.01 },
  { kind: 'travel', direction: 'left', minDistanceM: 0.01 },
  { kind: 'turn', left: true, minAngleRad: 0.05 },
  { kind: 'upright', maxTiltRad: 0.5 },
];

function measure(robot: RobotDescription, label: string, g: Partial<GaitParams>) {
  const gait: GaitParams = { ...DEFAULT_GAIT, durationS: 4, ...g };
  const e = evaluateSkill(mj, robot, -1, homeConfiguration(robot), { kind: 'gait', gait }, { ...DEFAULT_EVALUATION, trials: 5, seed: 3, baseMassRange: [robot.base.mass * 0.875, robot.base.mass * 1.125] }, { goals });
  const mean = (i: number) => +(e.results.reduce((s, r) => s + r.goals![i]!.measured, 0) / e.results.length).toFixed(3);
  return {
    robot: robot.name, gait: label, forwardM: mean(0), leftM: mean(1), turnRad: mean(2),
    maxTiltRad: +Math.max(...e.results.map((r) => r.maxTiltRad)).toFixed(3), upright: e.results.filter((r) => r.goals![3]!.met).length, trials: e.trials,
  };
}

it('gait sweep on the quadruped and the hexapod, 4 s each over 5 randomized trials (measured)', () => {
  const dog = quadruped();
  const hex = hexapod();
  const rows = [
    measure(dog, 'trot stride 0.06 cycle 0.5 (default)', {}),
    measure(dog, 'trot stride 0.06, no feedback', { feedbackKp: 0, feedbackKd: 0 }),
    measure(dog, 'trot stride 0.10 cycle 0.5', { strideM: 0.1 }),
    measure(dog, 'trot stride 0.06 cycle 0.4', { cycleS: 0.4 }),
    measure(dog, 'trot stride 0.06 step 0.06', { stepHeightM: 0.06 }),
    measure(dog, 'trot stride -0.06 (backward)', { strideM: -0.06 }),
    measure(dog, 'trot lateral 0.04 (left)', { strideM: 0, lateralM: 0.04 }),
    measure(dog, 'trot turn 0.3 rad per cycle (left)', { strideM: 0, turnRad: 0.3 }),
    measure(dog, 'walk duty 0.75 stride 0.06 cycle 0.8', { pattern: 'walk', dutyFactor: 0.75, cycleS: 0.8 }),
    measure(hex, 'tripod (trot) stride 0.06 cycle 0.5', {}),
    measure(hex, 'tripod turn 0.3 rad per cycle (left)', { strideM: 0, turnRad: 0.3 }),
  ];
  console.table(rows);
  console.log('Gait sweep', JSON.stringify(rows));
  expect(rows.every((r) => r.trials === 5)).toBe(true);
});
