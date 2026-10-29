// Learned feedback policy in the joints of a gait (D27, docs/physical-ai-plan.md Phase 1): the gait alone versus
// the gait with a policy trained by augmented random search with the default settings, compared on held-out
// randomized trials and in 20-trial evaluations. A measurement, not a pass/fail claim.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { homeConfiguration } from '../../src/core/model/catalog';
import { hexapod, quadruped } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import type { SkillSpec } from '../../src/skills/controllers';
import { DEFAULT_SKILL_GAINS, evaluateSkill, evaluationOptionsFor } from '../../src/skills/evaluation';
import { DEFAULT_GAIT, type GaitParams } from '../../src/skills/gait';
import type { SkillGoal } from '../../src/skills/taught';
import { DEFAULT_GAIT_POLICY, trainGaitPolicy } from '../../src/skills/train';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const upright: SkillGoal = { kind: 'upright', maxTiltRad: 0.5 };

function measure(label: string, robot: RobotDescription, gait: GaitParams, goals: SkillGoal[]) {
  const stance = homeConfiguration(robot);
  const o = evaluationOptionsFor(robot, 2026, DEFAULT_SKILL_GAINS);
  const cfg = { ...DEFAULT_GAIT_POLICY, frictionRange: o.frictionRange, baseMassRange: o.baseMassRange, jointNoiseRad: o.jointNoiseRad, gains: o.gains };
  const t0 = performance.now();
  const r = trainGaitPolicy(mj, robot, stance, gait, goals, cfg);
  const wallS = (performance.now() - t0) / 1000;
  const evaluate = (spec: SkillSpec) => {
    const e = evaluateSkill(mj, robot, -1, stance, spec, o, { goals });
    const mean = (i: number) => +(e.results.reduce((s, t) => s + (t.goals?.[i]?.measured ?? 0), 0) / e.trials).toFixed(3);
    return { successes: e.successes, trials: e.trials, meanGoal1: mean(0) };
  };
  const round = (s: typeof r.baseline) => ({
    return: +s.meanReturn.toFixed(3), goalReward: +s.meanGoalReward.toFixed(3), velErrMps: +s.trackingErrorMps.toFixed(4), yawRateErrRadps: +s.yawRateErrorRadps.toFixed(4),
    forwardM: +s.meanForwardM.toFixed(3), leftM: +s.meanLeftM.toFixed(3), turnRad: +s.meanTurnRad.toFixed(3), falls: s.falls, goalsMet: s.successes,
  });
  const row = {
    case: label, robot: robot.name, improved: r.improved, rollouts: r.rollouts, parameters: r.parameters, trainWallS: +wallS.toFixed(1),
    target: r.target.map((v) => +v.toFixed(4)),
    firstMeanReturn: +r.history[0]!.meanReturn.toFixed(3), lastMeanReturn: +r.history[r.history.length - 1]!.meanReturn.toFixed(3),
    heldOutGait: round(r.baseline), heldOutLearned: round(r.result),
    evalGait: evaluate({ kind: 'gait', gait }), evalLearned: evaluate({ kind: 'gait-learned', gait, policy: r.policy }),
  };
  console.log('LearnedGait', JSON.stringify(row));
  return row;
}

it('quadruped side step right: the gait alone versus with a learned feedback policy', () => {
  const row = measure('side step right', quadruped(), { ...DEFAULT_GAIT, strideM: 0, lateralM: -0.04, durationS: 4 }, [{ kind: 'travel', direction: 'right', minDistanceM: 0.2 }, upright]);
  expect(row.rollouts).toBe(1 + 30 * 16 + 16);
});

it('quadruped trot forward: the gait alone versus with a learned feedback policy', () => {
  const row = measure('trot forward', quadruped(), { ...DEFAULT_GAIT, strideM: 0.06, durationS: 4 }, [{ kind: 'travel', direction: 'forward', minDistanceM: 0.3 }, upright]);
  expect(row.rollouts).toBe(1 + 30 * 16 + 16);
});

it('hexapod tripod turn left: the gait alone versus with a learned feedback policy', () => {
  const row = measure('turn left', hexapod(), { ...DEFAULT_GAIT, strideM: 0, turnRad: 0.3, durationS: 4 }, [{ kind: 'turn', left: true, minAngleRad: 1.0 }, upright]);
  expect(row.rollouts).toBe(1 + 30 * 16 + 16);
});

// The browser check of 8f898d1 kept a policy that followed the nominal side-step speed and halved the travel of a gait tuned to a 0.15 to 0.25 m goal.
it('hexapod side step right about 20 cm (a goal with a maximum): the gait alone versus with a learned feedback policy', () => {
  const row = measure('side step right about 20 cm', hexapod(), { ...DEFAULT_GAIT, strideM: 0, lateralM: -0.03, durationS: 2.5 }, [{ kind: 'travel', direction: 'right', minDistanceM: 0.15, maxDistanceM: 0.25 }, { kind: 'upright', maxTiltRad: 0.2 }]);
  expect(row.rollouts).toBe(1 + 30 * 16 + 16);
});
