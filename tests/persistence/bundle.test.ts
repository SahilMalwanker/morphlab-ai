import { describe, expect, it } from 'vitest';
import { DEFAULT_LEGGED, generateArm, generateLegged } from '../../src/core/model/builder';
import { PRESETS } from '../../src/core/model/catalog';
import { canonicalJson, morphologyHash } from '../../src/core/model/hash';
import { quadruped } from '../../src/core/model/presets';
import { exportBundle, importBundle, MAX_BUNDLE_CHARS } from '../../src/persistence/bundle';
import { createSkillArtifact, policyInputSize, type SkillArtifact } from '../../src/skills/artifact';
import type { SkillSpec } from '../../src/skills/controllers';
import { wilson95, type EvaluationSummary } from '../../src/skills/evaluation';
import { DEFAULT_FOOT_LIFT } from '../../src/skills/footLift';

const AT = '2026-10-03T12:00:00.000Z';

const EVALUATION: EvaluationSummary = {
  trials: 20, successes: 19, successRate: 0.95, wilson95: wilson95(19, 20), conditions: { friction: [0.6, 1.2], baseMassKg: [3.5, 4.5], jointNoiseRad: [-0.02, 0.02] },
  results: [], simulator: 'MuJoCo 3.14.0', evaluatedAt: AT,
};

async function sampleSkills(): Promise<SkillArtifact[]> {
  const robot = quadruped();
  const hash = await morphologyHash(robot);
  const joints = robot.limbs.flatMap((l) => l.joints.map((j) => j.id));
  // A linear policy with the quadruped's dimensions: 12 joints + 2 IMU + 4 contacts + 1 time inputs, 12 outputs.
  const inputs = policyInputSize(robot), outputs = joints.length;
  const weights = Array.from({ length: outputs * (inputs + 1) }, (_, i) => ((i % 7) - 3) / 10);
  const policy = { layers: [inputs, outputs], activation: 'tanh' as const, weights, inputMean: new Array<number>(inputs).fill(0), inputStd: new Array<number>(inputs).fill(1) };
  const specs: [string, SkillSpec][] = [
    ['skill_sm', { kind: 'state-machine', params: DEFAULT_FOOT_LIFT }],
    ['skill_replay', { kind: 'replay', trajectory: { dt: 0.01, jointIds: joints, frames: [joints.map(() => 0.1), joints.map(() => 0.2)] } }],
    ['skill_mlp', { kind: 'imitation-mlp', durationS: 6, policy }],
  ];
  return specs.map(([id, spec]) => createSkillArtifact({ robot, morphologyHash: hash, limbId: 'front_left', spec, name: id, evaluation: EVALUATION, now: AT, id }));
}

describe('bundle export and import', () => {
  const robots = [...PRESETS.map((p) => p.make()), generateLegged({ ...DEFAULT_LEGGED, pairs: 3, dofPerLeg: 4 }), generateArm(5, 0.9)];

  for (const r of robots) {
    it(`${r.name}: export then import round-trips byte-identical canonical JSON`, async () => {
      const text = exportBundle([r], AT);
      const res = importBundle(text);
      expect(res.errors).toEqual([]);
      expect(res.rejected).toEqual([]);
      expect(res.robots).toHaveLength(1);
      expect(canonicalJson(res.robots[0])).toBe(canonicalJson(r));
      expect(exportBundle(res.robots, res.exportedAt!)).toBe(text);
      expect(await morphologyHash(res.robots[0]!)).toBe(await morphologyHash(r));
    });
  }

  it('refuses files that are not bundles, with a specific reason', () => {
    expect(importBundle('{"kind": "morphlab.bundle", ').errors[0]).toMatch(/^The file is not valid JSON/);
    expect(importBundle('[]').errors).toEqual(['The file does not contain a MorphLab bundle object.']);
    expect(importBundle('{"kind": "other"}').errors).toEqual(['The file is not a MorphLab bundle (expected kind "morphlab.bundle").']);
    expect(importBundle('{"kind": "morphlab.bundle", "schemaVersion": 2, "robots": []}').errors).toEqual([
      'Unsupported bundle schema version 2; this version reads 1.',
    ]);
    expect(importBundle(' '.repeat(MAX_BUNDLE_CHARS + 1)).errors).toEqual(['The file is larger than 5 MB.']);
  });

  it('imports valid robots and reports each rejected robot by reason', () => {
    const good = quadruped();
    const missingAxis = structuredClone(good) as unknown as { limbs: { joints: Record<string, unknown>[] }[] };
    delete missingAxis.limbs[0]!.joints[1]!.axis;
    const badLimits = structuredClone(good);
    badLimits.limbs[0]!.joints[2]!.limit.lower = 0.5;
    const text = exportBundle([good], AT).replace('"robots": [', `"robots": [${JSON.stringify(missingAxis)}, ${JSON.stringify(badLimits)}, `);
    const res = importBundle(text);
    expect(res.errors).toEqual([]);
    expect(res.robots).toHaveLength(1);
    expect(res.rejected).toEqual([
      { kind: 'robot', index: 0, reasons: ['robot.limbs[0].joints[1].axis must be a list of three numbers.'] },
      { kind: 'robot', index: 1, reasons: ['Front left leg, Knee: Lower limit must be below upper limit.', 'Front left leg, Knee: Stance position must lie within the joint limits.'] },
    ]);
  });

  it('rejects non-finite numbers and suspicious stance keys, and drops unknown fields', () => {
    const text = exportBundle([quadruped()], AT);
    const inf = importBundle(text.replace('"mass": 4,', '"mass": 1e400,'));
    expect(inf.rejected[0]!.reasons).toEqual(['Base: Base mass must be a positive number of kilograms.']);
    const proto = importBundle(text.replace('"stance": {', '"stance": {"__proto__": {"polluted": 1}, '));
    expect(proto.rejected[0]!.reasons).toEqual(['robot.stance has an invalid joint id "__proto__".']);
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    const extra = importBundle(text.replace('"name": "Quadruped",', '"name": "Quadruped", "script": "alert(1)",'));
    expect(extra.robots).toHaveLength(1);
    expect('script' in extra.robots[0]!).toBe(false);
    expect(canonicalJson(extra.robots[0])).toBe(canonicalJson(quadruped()));
  });

  it('exports skills and re-imports them unchanged, keeping only known fields', async () => {
    const skills = await sampleSkills();
    const text = exportBundle([quadruped()], AT, skills);
    const res = importBundle(text);
    expect(res.errors).toEqual([]);
    expect(res.rejected).toEqual([]);
    expect(canonicalJson(res.skills)).toBe(canonicalJson(skills));
    expect(exportBundle(res.robots, AT, res.skills)).toBe(text);
    const extra = importBundle(text.replace('"id": "skill_sm",', '"id": "skill_sm", "script": "alert(1)",'));
    expect(extra.skills).toHaveLength(3);
    expect(extra.skills.some((s) => 'script' in s)).toBe(false);
  });

  it('rejects malformed skills with specific reasons and keeps the valid ones', async () => {
    const [sm, replay, mlp] = await sampleSkills();
    const ev = sm!.evaluation!;
    const bad = [
      { ...mlp!, id: 'w', policy: { ...mlp!.policy!, weights: [0.5, -0.25] } },
      { ...mlp!, id: 's', policy: { ...mlp!.policy!, inputStd: [1, 0] } },
      { ...replay!, id: 'f', trajectory: { ...replay!.trajectory!, frames: [[0.1], [0.2]] } },
      { ...sm!, id: 'e', evaluation: { ...ev, successes: 21 } },
      { ...sm!, id: 'm', method: 'teleport' },
      { ...sm!, id: 'rate', evaluation: { ...ev, successRate: 1 } },
      { ...sm!, id: 'ci', evaluation: { ...ev, wilson95: [0.99, 1] } },
      { ...sm!, id: 'cond', evaluation: { ...ev, conditions: { ...ev.conditions, friction: [1.2, 0.6] } } },
      { ...sm!, id: 'p', parameters: { ...sm!.parameters, shiftM: 5 } },
      { ...mlp!, id: 'd', parameters: {} },
      { ...mlp!, id: 'dim', observation: mlp!.observation.slice(1) },
      { ...replay!, id: 'dup', trajectory: { ...replay!.trajectory!, jointIds: replay!.trajectory!.jointIds.map(() => 'same') } },
    ];
    const res = importBundle(exportBundle([], AT, [sm!, ...(bad as SkillArtifact[])]));
    expect(res.skills.map((s) => s.id)).toEqual(['skill_sm']);
    const policy = 'skill.policy must contain integer layer sizes, tanh activation, one finite weight per connection and bias, and input normalization with positive scales.';
    expect(res.rejected).toEqual([
      { kind: 'skill', index: 1, reasons: [policy] },
      { kind: 'skill', index: 2, reasons: [policy] },
      { kind: 'skill', index: 3, reasons: ['skill.trajectory must contain a dt between 0 and 1 s, joint ids and at least two finite frames with one value per joint.'] },
      { kind: 'skill', index: 4, reasons: ['skill.evaluation.successes must be a whole number from 0 to trials (20).'] },
      { kind: 'skill', index: 5, reasons: ['skill.method must be one of state-machine, tuned-parameters, replay, imitation-mlp, es-improved.'] },
      { kind: 'skill', index: 6, reasons: ['skill.evaluation.successRate 1 does not equal successes / trials = 0.9500.'] },
      { kind: 'skill', index: 7, reasons: ['skill.evaluation.wilson95 must be the Wilson 95 % interval of 19 of 20 (0.764 to 0.991).'] },
      { kind: 'skill', index: 8, reasons: ['skill.evaluation.conditions must map names to [low, high] ranges with low <= high.'] },
      { kind: 'skill', index: 9, reasons: ['skill.parameters for a state machine: shiftM = 5 is outside 0 to 0.1.'] },
      { kind: 'skill', index: 10, reasons: ['skill.parameters.durationS must be between 0.5 and 30 s for an imitation policy.'] },
      { kind: 'skill', index: 11, reasons: ['skill.policy takes 19 inputs but skill.observation lists 7.'] },
      { kind: 'skill', index: 12, reasons: ['skill.trajectory.jointIds must not repeat.'] },
    ]);
    const notList = importBundle(exportBundle([], AT).replace('"skills": []', '"skills": {}'));
    expect(notList.errors).toEqual(['The bundle skills entry must be a list.']);
  });
});
