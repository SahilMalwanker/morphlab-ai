import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel, clampToLimits } from '../../src/core/kinematics/fk';
import type { Quat } from '../../src/core/math/mat4';
import { morphologyHash } from '../../src/core/model/hash';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { checkSkillCompatibility, createSkillArtifact, parseSkillArtifact, skillLabel, specFromArtifact } from '../../src/skills/artifact';
import { makeController, METHOD_LABELS, type SkillSpec } from '../../src/skills/controllers';
import { DEFAULT_EVALUATION, evaluateSkill, runSkillTrial, trialConditions } from '../../src/skills/evaluation';
import { DEFAULT_GAIT, GaitController, gaitEnvelope, type GaitParams } from '../../src/skills/gait';
import { GAIT_POLICY_INPUTS, gaitCommand, gaitPolicyObservation, gaitPolicyProblem, LearnedGaitController, RESIDUAL_SCALE_RAD } from '../../src/skills/policyGait';
import type { SkillGoal } from '../../src/skills/taught';
import { DEFAULT_GAIT_POLICY, gaitPolicyRollout, linearGaitPolicy, trackingTarget, trainGaitPolicy } from '../../src/skills/train';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const dog = quadruped();
const stance = quadrupedStance();
const km = buildKinematicModel(dog);
const J = stance.length;
/** A side step to the right: no stride, 4 cm to the right per cycle. */
const side: GaitParams = { ...DEFAULT_GAIT, strideM: 0, lateralM: -0.04, durationS: 2 };
const zeros = (n: number) => new Array<number>(n).fill(0);
const ones = (n: number) => new Array<number>(n).fill(1);
const obs = (imuQuat: Quat = [1, 0, 0, 0], gyro: [number, number, number] = [0, 0, 0]) => ({ q: stance, qd: zeros(J), touch: [1, 1, 1, 1], imuQuat, gyro });
/** A policy whose outputs are all `bias`, whatever it observes. */
const constant = (bias: number) => {
  const theta = new Float64Array(J * (GAIT_POLICY_INPUTS + 1));
  theta.fill(bias, J * GAIT_POLICY_INPUTS);
  return linearGaitPolicy(theta, J, zeros(GAIT_POLICY_INPUTS), ones(GAIT_POLICY_INPUTS));
};

describe('learned feedback policy in the joints of a gait', () => {
  it('observes the gravity direction, angular rate, gait phase and command (12 inputs)', () => {
    const g: GaitParams = { ...side, turnRad: 0.2 };
    expect(gaitCommand(g)).toEqual([0, -0.08, 0.4]);
    const roll = 0.1;
    const x = gaitPolicyObservation(g.cycleS / 4, obs([Math.cos(roll / 2), Math.sin(roll / 2), 0, 0], [0.1, -0.2, 0.3]), g);
    expect(x).toHaveLength(GAIT_POLICY_INPUTS);
    // Left side up by 0.1 rad: the body y axis points up by sin(0.1).
    expect(x[0]).toBeCloseTo(0, 12);
    expect(x[1]).toBeCloseTo(Math.sin(roll), 12);
    expect(Array.from(x.slice(2, 5))).toEqual([0.1, -0.2, 0.3]);
    // A quarter of the cycle: phase pi/2.
    expect(x[5]).toBeCloseTo(1, 12);
    expect(x[6]).toBeCloseTo(0, 12);
    expect(x[7]).toBeCloseTo(0, 12);
    expect(x[8]).toBeCloseTo(-1, 12);
    expect(Array.from(x.slice(9))).toEqual(gaitCommand(g));
  });

  it('adds nothing when the policy is zero: the gait generator alone', () => {
    const tg = new GaitController(km, stance, side);
    const learned = new LearnedGaitController(km, stance, side, constant(0));
    const tilted: Quat = [Math.cos(0.03), 0.02, Math.sin(0.03), 0];
    for (const t of [0, 0.25, 0.8, 1.3, 1.9, 2.05]) {
      const o = obs(tilted, [0.2, -0.1, 0.05]);
      expect(learned.update(t, o)).toEqual(tg.update(t, o));
    }
    expect(learned.finished).toBe(true);
    expect(Array.from(learned.action)).toEqual(zeros(J));
  });

  it('bounds every correction to 0.15 rad, faded in and out with the steps', () => {
    const tg = new GaitController(km, stance, side);
    const learned = new LearnedGaitController(km, stance, side, constant(10));
    for (const t of [0, 0.25, 1, 1.75, 2]) {
      const base = tg.update(t, obs());
      const want = clampToLimits(km, base.map((v) => v + gaitEnvelope(side, t) * RESIDUAL_SCALE_RAD));
      learned.update(t, obs()).forEach((v, i) => expect(v).toBeCloseTo(want[i]!, 12));
    }
    expect(gaitEnvelope(side, 0)).toBe(0);
    expect(gaitEnvelope(side, 0.25)).toBeCloseTo(0.5, 12);
    expect(gaitEnvelope(side, 1)).toBe(1);
    expect(gaitEnvelope(side, 2)).toBe(0);
    // The output of 10 is clipped to 1.
    expect(Array.from(learned.action)).toEqual(ones(J));
    expect(learned.phase).toBe('coming to a stop with learned feedback');
  });

  it('falls back to the gait generator for the rest of the run when the policy output is not finite', () => {
    const tg = new GaitController(km, stance, side);
    const learned = new LearnedGaitController(km, stance, side, constant(0.5));
    const broken = obs([1, 0, 0, 0], [Number.NaN, 0, 0]);
    expect(learned.update(1, broken)).toEqual(tg.update(1, broken));
    expect(learned.phase).toBe('gait generator only (the policy output was not finite)');
    expect(learned.update(1.1, obs())).toEqual(tg.update(1.1, obs()));
    expect(Array.from(learned.action)).toEqual(zeros(J));
  });

  it('refuses a policy that does not fit the robot', () => {
    const six = linearGaitPolicy(new Float64Array(18 * (GAIT_POLICY_INPUTS + 1)), 18, zeros(GAIT_POLICY_INPUTS), ones(GAIT_POLICY_INPUTS));
    expect(gaitPolicyProblem(six, J)).toBe('the policy maps 12 inputs to 18 outputs; a learned gait on this robot needs 12 inputs and 12 outputs.');
    expect(() => new LearnedGaitController(km, stance, side, six)).toThrow('The policy maps 12 inputs to 18 outputs');
    expect(gaitPolicyProblem({ ...constant(0), inputStd: [...ones(11), 0] }, J)).toBe('the policy normalization does not match its inputs.');
    expect(gaitPolicyProblem(constant(0), J)).toBeNull();
    expect(() => linearGaitPolicy(new Float64Array(5), J, zeros(12), ones(12))).toThrow('Expected 156 parameters, got 5');
  });

  it('follows the gait\'s own measured speed along the directions it commands, and no motion along the others', () => {
    // A side step that went faster than commanded, drifted forward and turned: keep its sideways speed, stop the drift and the turn.
    expect(trackingTarget([0, -0.08, 0], [0.05, -0.11, 0.07])).toEqual([0, -0.11, 0]);
    // A gait that went the wrong way is asked for its command.
    expect(trackingTarget([0.12, 0, 0.6], [-0.01, 0.02, 0.9])).toEqual([0.12, 0, 0.9]);
  });
});

describe('training a learned gait policy (MuJoCo)', () => {
  const goals: SkillGoal[] = [{ kind: 'travel', direction: 'right', minDistanceM: 0.05 }, { kind: 'upright', maxTiltRad: 0.5 }];

  it('runs a zero policy exactly like the gait alone, judged like an evaluation trial', () => {
    const conditions = trialConditions({ ...DEFAULT_GAIT_POLICY, seed: 3 }, 0);
    const input = { desc: dog, stanceQ: stance, gait: side, conditions, noiseSeed: 17, goals };
    const alone = gaitPolicyRollout(mj, { ...input, policy: null });
    expect(gaitPolicyRollout(mj, { ...input, policy: constant(0) })).toEqual(alone);
    expect(alone.fell).toBe(false);
    expect(alone.unstable).toBe(false);
    // It steps to the right.
    expect(alone.leftM).toBeLessThan(-0.05);
    expect(alone.meanVelocity[1]).toBeLessThan(0);
    expect(Number.isFinite(alone.ret)).toBe(true);
    expect(alone.trackingErrorMps).toBeGreaterThan(0);
    const trial = runSkillTrial(mj, { desc: dog, limbIndex: -1, stanceQ: stance, spec: { kind: 'gait', gait: side }, criteria: { goals }, conditions, trial: 0, noiseSeed: 17 });
    expect(alone.goalSuccess).toBe(trial.success);
  });

  it('trains by random search, seeded, and keeps the policy only when it beats the gait on held-out trials', () => {
    const cfg = { ...DEFAULT_GAIT_POLICY, iterations: 2, directions: 2, top: 1, heldOutTrials: 2 };
    const r = trainGaitPolicy(mj, dog, stance, side, goals, cfg);
    // One run of the gait alone for the input statistics, 2 iterations of 2 direction pairs, 2 held-out trials without and with the policy.
    expect(r.rollouts).toBe(1 + 2 * 2 * 2 + 2 * 2);
    expect(r.history).toHaveLength(2);
    expect(r.parameters).toBe(J * (GAIT_POLICY_INPUTS + 1));
    expect(r.policy.layers).toEqual([GAIT_POLICY_INPUTS, J]);
    expect(gaitPolicyProblem(r.policy, J)).toBeNull();
    // Inputs are normalized by what was observed: the command is constant, the phase is not.
    expect(r.policy.inputStd[9]).toBe(0.01);
    expect(r.policy.inputMean[10]).toBeCloseTo(-0.08, 12);
    expect(r.policy.inputStd[5]).toBeGreaterThan(0.5);
    // The side step is followed at the speed the gait reached, with no forward motion or turn.
    expect(r.target[0]).toBe(0);
    expect(r.target[1]).toBeLessThan(0);
    expect(r.target[2]).toBe(0);
    expect(r.improved).toBe(
      r.result.meanReturn > r.baseline.meanReturn + 1e-9 && r.result.falls <= r.baseline.falls && r.result.successes >= r.baseline.successes && r.result.meanGoalReward >= r.baseline.meanGoalReward - 0.01,
    );
    expect(trainGaitPolicy(mj, dog, stance, side, goals, cfg)).toEqual(r);
  });

  it('saves a learned gait as a skill bound to the body, and evaluates it like any skill', async () => {
    const spec: SkillSpec = { kind: 'gait-learned', gait: side, policy: constant(0.1) };
    expect(makeController(km, -1, stance, spec)).toBeInstanceOf(LearnedGaitController);
    const e = evaluateSkill(mj, dog, -1, stance, spec, { ...DEFAULT_EVALUATION, trials: 2, seed: 4 }, { goals });
    expect(e.trials).toBe(2);
    const hash = await morphologyHash(dog);
    const a = createSkillArtifact({ robot: dog, morphologyHash: hash, limbId: '', spec, name: 'step right (learned gait)', evaluation: e, taught: { name: 'step right', goals }, now: '2026-10-07T12:00:00.000Z' });
    expect(a.method).toBe('gait-learned');
    expect(a.observation.reduce((n, o) => n + o.size, 0)).toBe(GAIT_POLICY_INPUTS);
    expect(skillLabel(a)).toBe('Gait you set, with a feedback policy learned by random search (closed loop)');
    expect(METHOD_LABELS['gait-learned']).toBe('Gait with a learned feedback policy (random search, closed loop)');
    const parsed = parseSkillArtifact(JSON.parse(JSON.stringify(a)));
    if (!parsed.ok) throw new Error(parsed.errors.join(' '));
    expect(specFromArtifact(parsed.value)).toEqual(spec);
    expect(checkSkillCompatibility(parsed.value, dog, hash)).toEqual({ ok: true });
    // A stored policy for another number of joints is refused on load.
    const tampered = JSON.parse(JSON.stringify(a)) as { policy: { layers: number[] } };
    tampered.policy.layers = [GAIT_POLICY_INPUTS, 18];
    expect(parseSkillArtifact(tampered).ok).toBe(false);
  });
});
