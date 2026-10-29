// Shared measurement of learned feedback policies for keyframe skills (D30): the keyframes alone (open loop) against
// the keyframes with a policy trained by reinforcement learning under pushes (closed loop), on the same randomized runs.
// Default training settings, one thread. A measurement: it prints what it finds and asserts only that it ran.
import type { MainModule } from '@mujoco/mujoco';
import { homeConfiguration } from '../../src/core/model/catalog';
import type { RobotDescription } from '../../src/core/model/types';
import { composeKeyframes, type ComposedFrame } from '../../src/skills/compose';
import type { PolicyJson } from '../../src/skills/controllers';
import { DEFAULT_SKILL_GAINS, evaluateSkill, evaluationOptionsFor } from '../../src/skills/evaluation';
import { inProcessRunner } from '../../src/skills/rollouts';
import { motionDuration, type KeyframeMotion, type SkillGoal } from '../../src/skills/taught';
import { DEFAULT_POSE_POLICY, posePolicyRollout, poseTask, trainPosePolicy, type PosePolicyConfig, type PoseRolloutJob } from '../../src/skills/trainPose';

export function composedMotion(robot: RobotDescription, frames: readonly ComposedFrame[], endHoldS: number): { motion: KeyframeMotion; notes: string[] } {
  const c = composeKeyframes(robot, frames);
  if (!c.ok) throw new Error(`The poses were refused: ${c.problems.join(' ')}`);
  let t = 0;
  const keyframes = c.keyframes.map((k) => ({ t: Number((t += k.durationS).toFixed(4)), q: k.q.slice() }));
  return { motion: { jointIds: robot.limbs.flatMap((l) => l.joints.map((j) => j.id)), keyframes, endHoldS }, notes: c.notes };
}

export async function measurePose(mj: MainModule, robot: RobotDescription, task: string, frames: readonly ComposedFrame[], endHoldS: number, goals: SkillGoal[], overrides: Partial<PosePolicyConfig> = {}) {
  const stanceQ = homeConfiguration(robot);
  const { motion, notes } = composedMotion(robot, frames, endHoldS);
  const o = evaluationOptionsFor(robot, 2026, DEFAULT_SKILL_GAINS);
  const cfg: PosePolicyConfig = { ...DEFAULT_POSE_POLICY, frictionRange: o.frictionRange, baseMassRange: o.baseMassRange, jointNoiseRad: o.jointNoiseRad, gains: o.gains, ...overrides };
  const durationS = motionDuration(motion);
  const setup = { desc: robot, stanceQ, gains: o.gains };

  // The standard evaluation (20 randomized trials, no push) of the keyframes alone, with the goals of the first trial.
  const evaluate = (spec: Parameters<typeof evaluateSkill>[4]) => evaluateSkill(mj, robot, -1, stanceQ, spec, { ...o, trials: 20 }, { goals });
  const openEval = evaluate({ kind: 'keyframes', motion });
  const firstGoals = (openEval.results[0]?.goals ?? []).map((g) => `${g.met ? 'met' : 'missed'}: ${g.text}`);

  const t0 = performance.now();
  const r = await trainPosePolicy(inProcessRunner(mj), robot, stanceQ, motion, goals, cfg);
  const wallS = (performance.now() - t0) / 1000;

  // Fresh seeds: 40 pushed runs and 20 without a push, the same runs for both.
  const compare = (seed: number, n: number, pushed: boolean, policy: PolicyJson | null) => {
    let successes = 0, falls = 0, instant = 0;
    for (let k = 0; k < n; k++) {
      const t = poseTask(cfg, durationS, seed, k);
      const job: PoseRolloutJob = { motion, policy, goals, conditions: t.conditions, noiseSeed: t.noiseSeed, ...(pushed && t.push ? { push: t.push } : {}) };
      const x = posePolicyRollout(mj, setup, job).rollout;
      if (x.success) successes++;
      if (x.fell || x.unstable) falls++;
      instant += x.instantMean / n;
    }
    return { trials: n, successes, falls, shareHolding: +instant.toFixed(3) };
  };
  const learnedEval = evaluate({ kind: 'keyframes-learned', motion, policy: r.policy });
  const score = (s: typeof r.baseline) => ({ successes: s.successes, falls: s.falls, goalReward: +s.meanGoalReward.toFixed(3), shareHolding: +s.meanInstant.toFixed(3), ret: +s.meanReturn.toFixed(3) });
  const row = {
    task, robot: robot.name, durationS: +durationS.toFixed(2), poseNotes: notes, rollouts: r.rollouts, wallS: +wallS.toFixed(1), parameters: r.parameters, kept: r.improved,
    pushImpulseNs: cfg.pushImpulseNs,
    heldOut: { trials: r.heldOutTrials, keyframesAlone: score(r.baseline), learned: score(r.result) },
    returns: r.history.filter((_, i) => i % 10 === 0 || i === r.history.length - 1).map((x) => +x.meanReturn.toFixed(3)),
    pushed40: { keyframesAlone: compare(4242, 40, true, null), learned: compare(4242, 40, true, r.policy) },
    unpushed20: { keyframesAlone: compare(4343, 20, false, null), learned: compare(4343, 20, false, r.policy) },
    evaluation20: { keyframesAlone: openEval.successes, learned: learnedEval.successes, firstTrialGoalsAlone: firstGoals, firstTrialGoalsLearned: (learnedEval.results[0]?.goals ?? []).map((g) => `${g.met ? 'met' : 'missed'}: ${g.text}`) },
  };
  console.log('Pose', JSON.stringify(row));
  return row;
}
