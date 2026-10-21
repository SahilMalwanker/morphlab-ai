import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import type { SkillSpec } from '../../src/skills/controllers';
import * as evaluation from '../../src/skills/evaluation';
import { DEFAULT_GAIT } from '../../src/skills/gait';
import { DEFAULT_TAUGHT_IMPROVE, improveTaughtSkill, parameterCount, specFromTheta } from '../../src/skills/improve';
import { goalReward, type SkillGoal } from '../../src/skills/taught';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const dog = quadruped();
const stance = quadrupedStance();
const ids = dog.limbs.flatMap((l) => l.joints.map((j) => j.id));
const joints = dog.limbs.flatMap((l) => l.joints);
const crouch = (hip: number, knee: number): SkillSpec => ({ kind: 'keyframes', motion: { jointIds: ids, keyframes: [{ t: 1, q: dog.limbs.flatMap(() => [0, hip, knee]) }], endHoldS: 1.5 } });

describe('improvement by trial and error (evolution strategies)', () => {
  it('includes failed body checks in training and held-out fitness even when all requested goals pass', () => {
    const base = crouch(0.7, -1.4);
    const goals: SkillGoal[] = [{ kind: 'noFall', minBaseHeightM: 0.1 }];
    const trial = evaluation.runSkillTrial(mj, {
      desc: dog, limbIndex: -1, stanceQ: stance, spec: base, criteria: { goals },
      conditions: evaluation.trialConditions({ ...DEFAULT_TAUGHT_IMPROVE, seed: 11 }, 0), trial: 0, noiseSeed: 11,
    });
    expect(trial.success).toBe(true);
    const checked = {
      ...trial, success: false,
      checks: [{ kind: 'feetOnly' as const, met: false, text: 'a knee touched the ground' }, { kind: 'noTipOver' as const, met: true, text: 'never tipped over' }],
    };
    const run = vi.spyOn(evaluation, 'runSkillTrial').mockReturnValue(checked);
    try {
      const cfg = { ...DEFAULT_TAUGHT_IMPROVE, iterations: 1, pairs: 1, heldOutTrials: 1 };
      const result = improveTaughtSkill(mj, dog, stance, base, goals, cfg);
      const expected = goalReward({ success: false, goals: trial.goals ?? [], checks: checked.checks }, false);
      expect(expected).toBeLessThan(goalReward({ success: false, goals: trial.goals ?? [] }, false));
      expect(result.baseline).toEqual({ reward: expected, successes: 0 });
      expect(result.result).toEqual(result.baseline);
      expect(result.improved).toBe(false);
      expect(run).toHaveBeenCalledTimes(result.rollouts);
    } finally {
      run.mockRestore();
    }
  });

  it('maps zero parameters to the skill as given, and stays within ranges and directions', () => {
    const keyframes = crouch(1.1, -2.2);
    expect(parameterCount(keyframes)).toBe(13);
    expect(specFromTheta(dog, keyframes, new Float64Array(13))).toEqual(keyframes);
    const gait: SkillSpec = { kind: 'gait', gait: { ...DEFAULT_GAIT, strideM: 0.06, turnRad: 0 } };
    expect(parameterCount(gait)).toBe(9);
    expect(specFromTheta(dog, gait, new Float64Array(9))).toEqual(gait);
    // Large steps stay in range, never reverse the stride and never start a turn that was not asked for.
    const far = specFromTheta(dog, gait, new Float64Array(9).fill(-100));
    if (far.kind !== 'gait') throw new Error('expected a gait');
    expect(far.gait.strideM).toBe(0.005);
    expect(far.gait.turnRad).toBe(0);
    expect(far.gait.lateralM).toBe(0);
    expect(far.gait.cycleS).toBe(0.25);
    expect(far.gait.stepHeightM).toBe(0);
    expect(far.gait.feedbackKp).toBe(0);
    expect(far.gait.durationS).toBe(DEFAULT_GAIT.durationS);
    // Keyframe durations stay from 0.1 to 10 s, and poses within the joint limits.
    const fast = specFromTheta(dog, keyframes, new Float64Array(13).fill(-100));
    if (fast.kind !== 'keyframes') throw new Error('expected keyframes');
    expect(fast.motion.keyframes[0]!.t).toBe(0.1);
    fast.motion.keyframes[0]!.q.forEach((v, j) => expect(v).toBe(joints[j]!.limit.lower));
    expect(() => parameterCount({ kind: 'replay', trajectory: { dt: 0.01, jointIds: ids, frames: [stance] } })).toThrow('Only keyframe and gait skills');
  });

  it('keeps a change only when it scores better on held-out randomized trials (MuJoCo)', () => {
    const goals: SkillGoal[] = [{ kind: 'baseHeight', below: true, heightM: 0.2, holdS: 1 }, { kind: 'upright', maxTiltRad: 0.35 }];
    // A shallow crouch that does not get low enough.
    const shallow = crouch(0.85, -1.7);
    const cfg = { ...DEFAULT_TAUGHT_IMPROVE, iterations: 3, pairs: 2, heldOutTrials: 3 };
    const r = improveTaughtSkill(mj, dog, stance, shallow, goals, cfg);
    expect(r.rollouts).toBe(3 * 2 * 2 + 2 * 3);
    expect(r.history).toHaveLength(3);
    expect(r.heldOutTrials).toBe(3);
    if (r.improved) {
      expect(r.result.reward).toBeGreaterThan(r.baseline.reward);
      expect(r.spec).not.toEqual(shallow);
    } else {
      expect(r.spec).toEqual(shallow);
      expect(r.result).toEqual(r.baseline);
    }
    // Seeded: the same search gives the same result.
    expect(improveTaughtSkill(mj, dog, stance, shallow, goals, cfg).baseline).toEqual(r.baseline);
  });
});
