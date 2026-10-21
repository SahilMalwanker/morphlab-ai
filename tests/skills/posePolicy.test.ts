import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel, clampToLimits } from '../../src/core/kinematics/fk';
import type { Quat } from '../../src/core/math/mat4';
import { mulberry32 } from '../../src/core/math/rng';
import { morphologyHash } from '../../src/core/model/hash';
import { hexapod, quadruped, quadrupedStance } from '../../src/core/model/presets';
import { Mlp } from '../../src/learning/mlp';
import { checkSkillCompatibility, createSkillArtifact, parseSkillArtifact, skillLabel, specFromArtifact } from '../../src/skills/artifact';
import { composeKeyframes } from '../../src/skills/compose';
import { makeController, METHOD_LABELS, skillDuration, type SkillSpec } from '../../src/skills/controllers';
import { DEFAULT_EVALUATION, evaluateSkill, runSkillTrial } from '../../src/skills/evaluation';
import { LearnedPoseController, POSE_RESIDUAL_RAD, policyFeet, posePolicyInputs, posePolicyObservation, posePolicyProblem, posePolicyWithoutFeedback } from '../../src/skills/posePolicy';
import { inProcessRunner } from '../../src/skills/rollouts';
import { KeyframeController, motionDuration, type KeyframeMotion, type SkillGoal } from '../../src/skills/taught';
import { DEFAULT_POSE_POLICY, linearPosePolicy, posePolicyRollout, poseTask, randomPush, trainPosePolicy, type PoseRunner } from '../../src/skills/trainPose';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const dog = quadruped();
const stance = quadrupedStance();
const km = buildKinematicModel(dog);
const J = stance.length;
const N = posePolicyInputs(dog);
const zeros = (n: number) => new Array<number>(n).fill(0);
const ones = (n: number) => new Array<number>(n).fill(1);
const obs = (imuQuat: Quat = [1, 0, 0, 0], gyro: [number, number, number] = [0, 0, 0], touch = [5, 5, 5, 5]) => ({ q: stance, qd: zeros(J), touch, imuQuat, gyro });
/** A policy whose outputs are all `bias`, whatever it observes. */
const constant = (bias: number) => {
  const theta = new Float64Array(J * (N + 1));
  theta.fill(bias, J * N);
  return linearPosePolicy(theta, dog, zeros(N), ones(N));
};

/** Lift the front left foot after shifting the body away from it: 1.5 s, a stable three-leg stand. */
function liftMotion(): KeyframeMotion {
  const shift = { x: -0.03, y: -0.04 };
  const c = composeKeyframes(dog, [{ seconds: 0.5, body: shift }, { seconds: 0.5, body: shift, tips: [{ limb: 'front_left', z: 0.06 }] }]);
  if (!c.ok) throw new Error(c.problems.join(' '));
  let t = 0;
  return { jointIds: km.joints.map((j) => j.id), keyframes: c.keyframes.map((k) => ({ t: (t += k.durationS), q: k.q })), endHoldS: 0.5 };
}
const motion = liftMotion();
const goals: SkillGoal[] = [{ kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.03, holdS: 0.3 }, { kind: 'noFall', minBaseHeightM: 0.15 }];

describe('learned feedback policy in the joints of a keyframe skill (D30)', () => {
  it.each([{ layers: [N, J] }, { layers: [N, 16, J] }])('ablates sensor feedback while preserving phase for layers $layers', ({ layers }) => {
    const network = Mlp.initialized(layers, mulberry32(7));
    const policy = { layers, activation: 'tanh' as const, weights: Array.from(network.params), inputMean: zeros(N), inputStd: ones(N) };
    const original = structuredClone(policy);
    const ablated = posePolicyWithoutFeedback(policy, dog);
    const timeOnly = zeros(N);
    timeOnly[9] = 0.5;
    timeOnly[10] = 1;
    const changedSensors = timeOnly.map((value, index) => index === 9 || index === 10 ? value : 5);
    const expected = network.forward(timeOnly);
    const actual = Mlp.forwardWith(ablated.layers, ablated.weights, changedSensors);
    actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, 12));
    expect(Mlp.forwardWith(ablated.layers, ablated.weights, zeros(N))).not.toEqual(actual);
    expect(policy).toEqual(original);
  });

  it('observes the gravity direction, angular rate, foot contacts, motion progress and each joint against the sketch (23 inputs on four feet)', () => {
    expect(policyFeet(dog)).toEqual([0, 1, 2, 3]);
    expect(N).toBe(23);
    expect(posePolicyInputs(hexapod())).toBe(31);
    const pitch = 0.2;
    const sketch = stance.map((v, j) => v + 0.01 * j);
    const x = posePolicyObservation(0.75, obs([Math.cos(pitch / 2), 0, Math.sin(pitch / 2), 0], [0.1, -0.2, 0.3], [5, 0.5, 0, 12]), 3, policyFeet(dog), sketch, new Float64Array(N));
    // Front down by 0.2 rad: the body x axis points down by sin(0.2).
    expect(x[0]).toBeCloseTo(-Math.sin(pitch), 12);
    expect(x[1]).toBeCloseTo(0, 12);
    expect(Array.from(x.slice(2, 5))).toEqual([0.1, -0.2, 0.3]);
    // A foot counts as down above 1 N.
    expect(Array.from(x.slice(5, 9))).toEqual([1, 0, 0, 1]);
    expect(x[9]).toBeCloseTo(0.25, 12);
    expect(x[10]).toBeCloseTo(Math.sin(Math.PI / 4), 12);
    // The encoders read the stance; the sketch asks for 0.01 rad more per joint index.
    Array.from(x.slice(11)).forEach((v, j) => expect(v).toBeCloseTo(-0.01 * j, 12));
  });

  it('adds nothing when the policy is zero: the keyframes alone', () => {
    const alone = new KeyframeController(km, stance, motion);
    const learned = new LearnedPoseController(km, stance, motion, constant(0));
    const tilted: Quat = [Math.cos(0.03), 0.02, Math.sin(0.03), 0];
    for (const t of [0, 0.3, 0.7, 1.2, 1.49, 1.6]) {
      const o = obs(tilted, [0.2, -0.1, 0.05]);
      expect(learned.update(t, o)).toEqual(alone.update(t));
    }
    expect(learned.finished).toBe(true);
    expect(learned.durationS).toBe(motionDuration(motion));
    expect(Array.from(learned.action)).toEqual(zeros(J));
  });

  it('bounds every change to 0.5 rad around the keyframes, within the joint limits', () => {
    const alone = new KeyframeController(km, stance, motion);
    const learned = new LearnedPoseController(km, stance, motion, constant(10));
    for (const t of [0, 0.4, 1, 1.4]) {
      const want = clampToLimits(km, alone.update(t).map((v) => v + POSE_RESIDUAL_RAD));
      learned.update(t, obs()).forEach((v, i) => expect(v).toBeCloseTo(want[i]!, 12));
    }
    // The output of 10 is clipped to 1.
    expect(Array.from(learned.action)).toEqual(ones(J));
    expect(learned.phase).toBe('holding the last pose with learned feedback');
    // After the motion it no longer corrects: the last pose, as the keyframes hold it.
    expect(learned.update(1.6, obs())).toEqual(alone.update(1.6));
    expect(learned.phase).toBe('done');
  });

  it('falls back to the keyframes for the rest of the run when the policy output is not finite', () => {
    const alone = new KeyframeController(km, stance, motion);
    const learned = new LearnedPoseController(km, stance, motion, constant(0.5));
    const broken = obs([1, 0, 0, 0], [Number.NaN, 0, 0]);
    expect(learned.update(0.5, broken)).toEqual(alone.update(0.5));
    expect(learned.phase).toBe('keyframes only (the policy output was not finite)');
    expect(learned.update(0.6, obs())).toEqual(alone.update(0.6));
    expect(Array.from(learned.action)).toEqual(zeros(J));
  });

  it('refuses a policy that does not fit the robot', () => {
    const six = hexapod();
    const sixJoints = six.limbs.reduce((n, l) => n + l.joints.length, 0);
    const big = linearPosePolicy(new Float64Array(sixJoints * (posePolicyInputs(six) + 1)), six, zeros(31), ones(31));
    expect(posePolicyProblem(big, dog)).toBe('the policy maps 31 inputs to 18 outputs; a keyframe policy on Quadruped needs 23 inputs and 12 outputs.');
    expect(() => new LearnedPoseController(km, stance, motion, big)).toThrow('The policy maps 31 inputs to 18 outputs');
    expect(posePolicyProblem({ ...constant(0), inputStd: [...ones(N - 1), 0] }, dog)).toBe('the policy normalization does not match its inputs.');
    expect(posePolicyProblem(constant(0), dog)).toBeNull();
    expect(() => linearPosePolicy(new Float64Array(5), dog, zeros(N), ones(N))).toThrow('Expected 288 parameters, got 5');
  });

  it('pushes each run once, seeded, sideways or lengthwise within the size and time ranges', () => {
    const cfg = { pushImpulseNs: [0.2, 0.8] as [number, number], pushWindow: [0.2, 0.9] as [number, number] };
    const pushes = Array.from({ length: 50 }, (_, k) => randomPush(cfg, 3, 11, k));
    expect(randomPush(cfg, 3, 11, 7)).toEqual(pushes[7]);
    expect(randomPush(cfg, 3, 12, 7)).not.toEqual(pushes[7]);
    for (const p of pushes) {
      const impulse = Math.hypot(p.forceN[0], p.forceN[1]) * p.durationS;
      expect(impulse).toBeGreaterThanOrEqual(0.2 - 1e-3);
      expect(impulse).toBeLessThanOrEqual(0.8 + 1e-3);
      expect(p.forceN[2]).toBe(0);
      expect(p.durationS).toBe(0.1);
      expect(p.tS).toBeGreaterThanOrEqual(0.6 - 1e-3);
      expect(p.tS).toBeLessThanOrEqual(2.7 + 1e-3);
    }
    // Directions spread around the circle.
    expect(new Set(pushes.map((p) => Math.min(3, Math.floor(((Math.atan2(p.forceN[1], p.forceN[0]) + Math.PI) / (2 * Math.PI)) * 4)))).size).toBe(4);
    const t = poseTask(DEFAULT_POSE_POLICY, 3, 11, 7);
    expect(t.push).toEqual(randomPush(DEFAULT_POSE_POLICY, 3, 11, 7));
    expect(t.noiseSeed).toBe(11 * 104729 + 7);
  });
});

describe('training a feedback policy for keyframes under pushes (MuJoCo)', () => {
  const setup = { desc: dog, stanceQ: stance };

  it('does not replace an incumbent with a weaker candidate that only beats the keyframes', async () => {
    const incumbentPolicy = constant(0.1);
    const runner: PoseRunner = {
      workers: 1,
      async runPose(_setup, jobs) {
        return jobs.map((job) => {
          const reward = job.policy === incumbentPolicy ? 2 : job.policy ? 1 : 0;
          return { stats: null, rollout: { ret: reward, goalReward: reward, success: reward > 0, fell: false, unstable: false, maxTiltRad: 0, instantMean: 0 } };
        });
      },
    };
    const cfg = { ...DEFAULT_POSE_POLICY, architecture: 'mlp' as const, incumbentPolicy, iterations: 0, heldOutTrials: 2 };
    const result = await trainPosePolicy(runner, dog, stance, motion, goals, cfg);
    expect(result.result.successes).toBeGreaterThan(result.baseline.successes);
    expect(result.incumbent).toMatchObject({ trials: 2, successes: 2, meanGoalReward: 2 });
    expect(result.result).toMatchObject({ trials: 2, successes: 2, meanGoalReward: 1 });
    expect(result.improved).toBe(false);
    expect(result.rollouts).toBe(12);
  });

  it('initializes nonlinear features reproducibly without changing the reference motion', async () => {
    const cfg = { ...DEFAULT_POSE_POLICY, architecture: 'mlp' as const, iterations: 0, heldOutTrials: 1 };
    const result = await trainPosePolicy(inProcessRunner(mj), dog, stance, motion, goals, cfg);
    expect(result.policy.layers).toEqual([N, 16, J]);
    expect(result.parameters).toBe(588);
    expect(result.policy.weights.slice(0, 16 * (N + 1)).some((weight) => weight !== 0)).toBe(true);
    expect(result.policy.weights.slice(16 * (N + 1)).every((weight) => weight === 0)).toBe(true);
    const reference = new KeyframeController(km, stance, motion);
    const learned = new LearnedPoseController(km, stance, motion, result.policy);
    for (const time of [0, 0.4, 0.9, 1.4, 1.6]) expect(learned.update(time, obs())).toEqual(reference.update(time));
    expect(result.result).toEqual(result.baseline);
    expect(result.feedbackAblation).toEqual(result.baseline);
    expect(result.improved).toBe(false);
    expect(await trainPosePolicy(inProcessRunner(mj), dog, stance, motion, goals, cfg)).toEqual(result);
  });

  it('trains nonlinear feedback under the same held-out acceptance rule without changing the linear default', async () => {
    const cfg = { ...DEFAULT_POSE_POLICY, architecture: 'mlp' as const, iterations: 2, directions: 2, top: 1, trialsPerCandidate: 1, heldOutTrials: 2 };
    const result = await trainPosePolicy(inProcessRunner(mj), dog, stance, motion, goals, cfg);
    expect(result.parameters).toBe(Mlp.paramCount([N, 16, J]));
    expect(result.history).toHaveLength(2);
    expect(result.rollouts).toBe(18);
    expect(posePolicyProblem(result.policy, dog)).toBeNull();
    expect(result.policy.weights.every(Number.isFinite)).toBe(true);
    expect(result.improved).toBe(
      result.result.falls <= result.baseline.falls && result.result.successes >= result.baseline.successes &&
      (result.result.successes > result.baseline.successes || result.result.meanGoalReward > result.baseline.meanGoalReward + 0.05),
    );
    expect(await trainPosePolicy(inProcessRunner(mj), dog, stance, motion, goals, cfg)).toEqual(result);
    expect(DEFAULT_POSE_POLICY.architecture).toBeUndefined();
  });

  it('runs a zero policy exactly like the keyframes alone, and a push moves the body', () => {
    const t = poseTask(DEFAULT_POSE_POLICY, motionDuration(motion), 5, 0);
    const job = { motion, goals, conditions: t.conditions, noiseSeed: t.noiseSeed };
    const alone = posePolicyRollout(mj, setup, { ...job, policy: null });
    expect(posePolicyRollout(mj, setup, { ...job, policy: constant(0) })).toEqual(alone);
    expect(alone.rollout.fell).toBe(false);
    expect(alone.rollout.unstable).toBe(false);
    expect(Number.isFinite(alone.rollout.ret)).toBe(true);
    expect(alone.rollout.instantMean).toBeGreaterThan(0);
    expect(alone.stats).toBeNull();
    // Judged like an evaluation trial of the same skill.
    const trial = runSkillTrial(mj, { desc: dog, limbIndex: -1, stanceQ: stance, spec: { kind: 'keyframes', motion }, criteria: { goals }, conditions: t.conditions, trial: 0, noiseSeed: t.noiseSeed });
    expect(alone.rollout.success).toBe(trial.success);
    // A hard push sideways during the lift tilts the body more than no push.
    const pushed = posePolicyRollout(mj, setup, { ...job, policy: null, push: { tS: 0.9, forceN: [0, 60, 0], durationS: 0.1 } });
    expect(pushed.rollout.maxTiltRad).toBeGreaterThan(alone.rollout.maxTiltRad + 0.05);
  });

  it('trains by augmented random search on pushed runs, seeded, and keeps the policy only when it beats the keyframes on held-out pushed runs', async () => {
    const cfg = { ...DEFAULT_POSE_POLICY, iterations: 2, directions: 2, top: 1, trialsPerCandidate: 1, heldOutTrials: 2 };
    const progress: number[] = [];
    const r = await trainPosePolicy(inProcessRunner(mj), dog, stance, motion, goals, cfg, (f) => progress.push(f));
    // Four runs of the keyframes alone for the input statistics, 2 iterations of 2 direction pairs, 2 held-out runs without and with the policy.
    expect(r.rollouts).toBe(4 + 2 * 2 * 2 + 2 * 2);
    expect(r.history).toHaveLength(2);
    expect(r.parameters).toBe(J * (N + 1));
    expect(r.policy.layers).toEqual([N, J]);
    expect(posePolicyProblem(r.policy, dog)).toBeNull();
    // Foot contacts map to -1 and 1; the other inputs are normalized by what was observed.
    for (let k = 5; k < 9; k++) {
      expect(r.policy.inputMean[k]).toBe(0.5);
      expect(r.policy.inputStd[k]).toBe(0.5);
    }
    expect(r.policy.inputStd[9]).toBeGreaterThan(0.1);
    expect(r.pushImpulseNs).toEqual(cfg.pushImpulseNs);
    expect(r.heldOutTrials).toBe(2);
    expect(r.baseline.trials).toBe(2);
    expect(r.result.trials).toBe(2);
    expect(r.improved).toBe(
      r.result.falls <= r.baseline.falls &&
        r.result.successes >= r.baseline.successes &&
        (r.result.successes > r.baseline.successes || r.result.meanGoalReward > r.baseline.meanGoalReward + 0.05),
    );
    expect(progress[0]).toBe(0);
    expect(progress[progress.length - 1]).toBe(1);
    expect(await trainPosePolicy(inProcessRunner(mj), dog, stance, motion, goals, cfg)).toEqual(r);
  });

  it.each(['linear', 'mlp'] as const)('saves %s learned keyframes bound to the body and evaluates them like any skill', async (architecture) => {
    const policy = constant(0.05);
    if (architecture === 'mlp') {
      policy.layers = [N, 16, J];
      policy.weights = Array.from(new Mlp(policy.layers).params);
      policy.weights.fill(0.05, policy.weights.length - J);
    }
    const spec: SkillSpec = { kind: 'keyframes-learned', motion, policy };
    expect(makeController(km, -1, stance, spec)).toBeInstanceOf(LearnedPoseController);
    expect(skillDuration(spec)).toBe(motionDuration(motion));
    const e = evaluateSkill(mj, dog, -1, stance, spec, { ...DEFAULT_EVALUATION, trials: 2, seed: 4 }, { goals });
    expect(e.trials).toBe(2);
    const hash = await morphologyHash(dog);
    const a = createSkillArtifact({ robot: dog, morphologyHash: hash, limbId: '', spec, name: 'lift (learned keyframes)', evaluation: e, taught: { name: 'lift', goals, author: 'ai' }, now: '2026-10-08T12:00:00.000Z' });
    expect(a.method).toBe('keyframes-learned');
    expect(a.observation.reduce((n, o) => n + o.size, 0)).toBe(N);
    expect(skillLabel(a)).toBe('Keyframes the AI composed, with a feedback policy learned by reinforcement learning under pushes (closed loop)');
    expect(METHOD_LABELS['keyframes-learned']).toBe('Keyframes with a learned feedback policy (reinforcement learning under pushes, closed loop)');
    const parsed = parseSkillArtifact(JSON.parse(JSON.stringify(a)));
    if (!parsed.ok) throw new Error(parsed.errors.join(' '));
    expect(specFromArtifact(parsed.value)).toEqual(spec);
    expect(checkSkillCompatibility(parsed.value, dog, hash)).toEqual({ ok: true });
    // A stored policy for another number of joints is refused on load.
    const tampered = JSON.parse(JSON.stringify(a)) as { policy: { layers: number[] } };
    tampered.policy.layers = [N, 18];
    expect(parseSkillArtifact(tampered).ok).toBe(false);
    // A policy for another body's inputs is refused before it runs.
    const wrong = JSON.parse(JSON.stringify(a)) as { policy: { layers: number[]; weights: number[]; inputMean: number[]; inputStd: number[] }; observation: { size: number }[] };
    wrong.policy = { ...wrong.policy, layers: [N + 2, J], weights: zeros(J * (N + 3)), inputMean: zeros(N + 2), inputStd: ones(N + 2) };
    wrong.observation[2]!.size = 6;
    const w = parseSkillArtifact(wrong);
    if (!w.ok) throw new Error(w.errors.join(' '));
    expect(checkSkillCompatibility(w.value, dog, hash)).toEqual({
      ok: false,
      reason: 'The learned feedback in "lift (learned keyframes)" does not fit this robot: the policy maps 25 inputs to 12 outputs; a keyframe policy on Quadruped needs 23 inputs and 12 outputs. Train it again on this robot.',
    });
  });
});
