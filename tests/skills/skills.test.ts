import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel } from '../../src/core/kinematics/fk';
import { editLimb, patchJoint } from '../../src/core/model/builder';
import { morphologyHash } from '../../src/core/model/hash';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { checkSkillCompatibility, createSkillArtifact, parseSkillArtifact, specFromArtifact } from '../../src/skills/artifact';
import { MlpController, ReplayController, METHOD_LABELS } from '../../src/skills/controllers';
import { DEFAULT_EVALUATION, evaluateSkill, recordExpertDemonstration } from '../../src/skills/evaluation';
import { DEFAULT_FOOT_LIFT } from '../../src/skills/footLift';
import { collectDemonstrations, DEFAULT_IMITATION, DEFAULT_IMPROVE, improveParameters, trainImitation } from '../../src/skills/learning';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const desc = quadruped();
const stance = quadrupedStance();
const opts = { ...DEFAULT_EVALUATION, seed: 42 };

describe('skill controllers', () => {
  it('replay interpolates linearly and finishes after the last frame', () => {
    const r = new ReplayController({ dt: 0.1, jointIds: ['a', 'b'], frames: [[0, 1], [1, 3], [2, 5]] });
    expect(r.update(0.05)).toEqual([0.5, 2]);
    expect(r.update(0.15)[1]).toBeCloseTo(4, 12);
    expect(r.finished).toBe(false);
    expect(r.update(0.5)).toEqual([2, 5]);
    expect(r.finished).toBe(true);
  });

  it('a neural policy must match the robot it drives', () => {
    const km = buildKinematicModel(desc);
    const policy = { layers: [19, 4, 3], activation: 'tanh' as const, weights: new Array(19 * 4 + 4 + 4 * 3 + 3).fill(0), inputMean: new Array(19).fill(0), inputStd: new Array(19).fill(1) };
    expect(() => new MlpController(km, stance, policy, 7)).toThrow('Policy output size does not match the robot');
  });

  it('every method has a plain-language label', () => {
    expect(Object.keys(METHOD_LABELS).sort()).toEqual(['es-improved', 'gait', 'gait-learned', 'imitation-mlp', 'keyframes', 'keyframes-learned', 'replay', 'state-machine', 'tuned-parameters']);
  });
});

describe('methods over 20 randomized trials (quadruped, front-left foot)', () => {
  it('replay of a recorded expert demonstration (open loop)', () => {
    const traj = recordExpertDemonstration(mj, desc, 0, stance, DEFAULT_FOOT_LIFT);
    expect(traj.jointIds).toHaveLength(12);
    const e = evaluateSkill(mj, desc, 0, stance, { kind: 'replay', trajectory: traj }, opts);
    expect(e.trials).toBe(20);
    expect(e.successes).toBeGreaterThanOrEqual(18);
  });

  it('imitation-learned neural policy trained on expert demonstrations', () => {
    const demos = collectDemonstrations(mj, desc, 0, stance, DEFAULT_FOOT_LIFT, DEFAULT_IMITATION);
    const tr = trainImitation(demos, DEFAULT_IMITATION);
    expect(tr.finalLoss).toBeLessThan(0.01 * tr.initialLoss);
    const e = evaluateSkill(mj, desc, 0, stance, { kind: 'imitation-mlp', policy: tr.policy, durationS: demos.horizonS }, opts);
    expect(e.successes).toBeGreaterThanOrEqual(16);
  });

  it('reward-based improvement raises held-out reward and success from a poor baseline', () => {
    const poor = { ...DEFAULT_FOOT_LIFT, liftHeightM: 0.025, shiftM: 0.02 };
    const imp = improveParameters(mj, desc, 0, stance, poor, DEFAULT_IMPROVE);
    expect(imp.improvedReward).toBeGreaterThan(imp.baselineReward + 1);
    const before = evaluateSkill(mj, desc, 0, stance, { kind: 'tuned-parameters', params: poor }, { ...opts, trials: 10 });
    const after = evaluateSkill(mj, desc, 0, stance, { kind: 'es-improved', params: imp.params }, { ...opts, trials: 10 });
    expect(before.successes).toBeLessThanOrEqual(2);
    expect(after.successes).toBeGreaterThanOrEqual(8);
  });
});

describe('skill artifacts and morphology binding', () => {
  it('stores method, hash, evaluation and conditions, and rebuilds the controller', async () => {
    const e = evaluateSkill(mj, desc, 0, stance, { kind: 'state-machine', params: DEFAULT_FOOT_LIFT }, { ...opts, trials: 3 });
    const hash = await morphologyHash(desc);
    const a = createSkillArtifact({ robot: desc, morphologyHash: hash, limbId: 'front_left', spec: { kind: 'state-machine', params: DEFAULT_FOOT_LIFT }, name: 'Lift front left', evaluation: e, now: '2026-10-03T12:00:00.000Z' });
    expect(a.method).toBe('state-machine');
    expect(a.robot.morphologyHash).toBe(hash);
    expect(a.evaluation!.trials).toBe(3);
    expect(a.evaluation!.conditions.friction).toEqual([0.6, 1.2]);
    expect(specFromArtifact(a)).toEqual({ kind: 'state-machine', params: DEFAULT_FOOT_LIFT });
    expect(parseSkillArtifact(JSON.parse(JSON.stringify(a))).ok).toBe(true);
    expect(checkSkillCompatibility(a, desc, hash)).toEqual({ ok: true });
  });

  it('refuses a different morphology and a limb it was not evaluated on', async () => {
    const e = evaluateSkill(mj, desc, 0, stance, { kind: 'state-machine', params: DEFAULT_FOOT_LIFT }, { ...opts, trials: 1 });
    const hash = await morphologyHash(desc);
    const a = createSkillArtifact({ robot: desc, morphologyHash: hash, limbId: 'front_left', spec: { kind: 'state-machine', params: DEFAULT_FOOT_LIFT }, name: 'Lift front left', evaluation: e });
    const longer = editLimb(desc, 'front_left', (l) => ({ ...l, joints: l.joints.map((j, i) => (i === 2 ? patchJoint(j, { xyz: [0, 0, -0.18] }) : j)) }), true);
    const other = await morphologyHash(longer);
    const refused = checkSkillCompatibility(a, longer, other);
    expect(refused.ok).toBe(false);
    expect(!refused.ok && refused.reason).toContain('A skill is never reused on a different body');
    const wrongLimb = checkSkillCompatibility(a, desc, hash, 'rear_right');
    expect(wrongLimb).toEqual({ ok: false, reason: '"Lift front left" was evaluated on front_left, not on rear_right.' });
  });

  it('rejects malformed skill records with specific reasons', () => {
    const res = parseSkillArtifact({ kind: 'morphlab.skill', schemaVersion: 1, id: 'x', name: 'n', createdAt: 'a', updatedAt: 'b', skillType: 'foot_lift', method: 'imitation-mlp', robot: { robotId: 'q', morphologyHash: 'abc' }, limbs: ['front_left'], parameters: { durationS: 7 }, policy: { layers: [1, 1], activation: 'relu', weights: [], inputMean: [], inputStd: [] } });
    expect(res.ok).toBe(false);
    expect(!res.ok && res.errors).toEqual([
      'skill.robot must contain robotId and a 64-character hex morphologyHash.',
      'skill.policy must contain integer layer sizes, tanh activation, one finite weight per connection and bias, and input normalization with positive scales.',
    ]);
  });
});
