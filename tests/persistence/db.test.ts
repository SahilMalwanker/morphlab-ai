import 'fake-indexeddb/auto';
import { openDB } from 'idb';
import { describe, expect, it } from 'vitest';
import type { MemoryRecord } from '../../src/ai/memory';
import { canonicalJson, morphologyHash } from '../../src/core/model/hash';
import { hexapod, quadruped } from '../../src/core/model/presets';
import { MorphStore } from '../../src/persistence/db';
import { createSkillArtifact } from '../../src/skills/artifact';
import { wilson95, type EvaluationSummary } from '../../src/skills/evaluation';
import { DEFAULT_FOOT_LIFT } from '../../src/skills/footLift';
import { DEFAULT_GAIT } from '../../src/skills/gait';
import { DEFAULT_WALK_RANGES, type LocomotionController, type WalkScore } from '../../src/skills/locomotion';

const walkScore: WalkScore = { trials: 4, successes: 3, falls: 0, meanPositionErrorM: 0.02, meanHeadingErrorRad: 0.05 };
const controller = (morphologyHash: string): LocomotionController => ({
  kind: 'morphlab.locomotion', schemaVersion: 1, robot: { robotId: 'quadruped', morphologyHash }, createdAt: '2026-10-07T12:00:00.000Z',
  base: { ...DEFAULT_GAIT, strideM: 0, lateralM: 0, turnRad: 0, durationS: 3 }, ranges: DEFAULT_WALK_RANGES,
  calibration: [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1], policy: null,
  training: {
    algorithm: 'test', rollouts: 30, workers: 2, iterations: 0, parameters: 156, calibrationRuns: 18, wallTimeS: 1, seed: 1,
    heldOut: {
      trials: 0,
      generator: { meanReturn: 0, trackingErrorMps: 0, yawRateErrorRadps: 0, falls: 0, meanPositionErrorM: 0, meanHeadingErrorRad: 0 },
      learned: { meanReturn: 0, trackingErrorMps: 0, yawRateErrorRadps: 0, falls: 0, meanPositionErrorM: 0, meanHeadingErrorRad: 0 },
    },
  },
  evaluation: {
    seed: 1, trials: 4, durationS: 3, final: { ...walkScore, wilson95: wilson95(3, 4) }, calibrated: walkScore, uncalibrated: { ...walkScore, successes: 1 },
    conditions: { frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02 },
  },
});

let n = 0;
const dbName = () => `morphlab-test-${n++}`;

const evaluation: EvaluationSummary = {
  trials: 20, successes: 19, successRate: 0.95, wilson95: wilson95(19, 20),
  conditions: { friction: [0.6, 1.2], baseMassKg: [3.5, 4.5], jointNoiseRad: [-0.02, 0.02] },
  results: [], simulator: 'MuJoCo 3.14.0 (WASM)', evaluatedAt: '2026-10-03T12:00:00.000Z',
};

async function skillFor(robot = quadruped()) {
  return createSkillArtifact({ robot, morphologyHash: await morphologyHash(robot), limbId: 'front_left', spec: { kind: 'state-machine', params: DEFAULT_FOOT_LIFT }, name: 'Lift front left', evaluation, now: '2026-10-03T12:00:00.000Z', id: 'skill_a' });
}

describe('IndexedDB storage', () => {
  it('keeps one walking controller per morphology, refuses invalid ones and reports a malformed record', async () => {
    const s = await MorphStore.open(dbName());
    const hash = await morphologyHash(quadruped());
    await s.saveLocomotion(controller(hash));
    expect(await s.loadLocomotion()).toEqual({ records: [controller(hash)], corrupted: [] });
    await expect(s.saveLocomotion({ ...controller(hash), calibration: [1] })).rejects.toThrow('Refusing to store an invalid walking controller');
    const odd = `locomotion.${'c'.repeat(64)}`;
    await s.putRaw('settings', { key: odd, value: { kind: 'morphlab.locomotion' } });
    const loaded = await s.loadLocomotion();
    expect(loaded.records).toHaveLength(1);
    expect(loaded.corrupted.map((c) => c.key)).toEqual([odd]);
    await s.deleteLocomotion(hash);
    expect((await s.loadLocomotion()).records).toEqual([]);
    s.close();
  });
  it('robots and skills survive closing and reopening the database', async () => {
    const name = dbName();
    const a = await MorphStore.open(name);
    const q = quadruped();
    await a.saveRobot(q);
    await a.saveRobot(hexapod());
    await a.saveSkill(await skillFor());
    await a.setSetting('workspace.builderRobot', ['quadruped', await morphologyHash(q)]);
    a.close();

    const b = await MorphStore.open(name);
    const robots = await b.loadRobots();
    expect(robots.corrupted).toEqual([]);
    expect(robots.records.map((r) => r.id).sort()).toEqual(['hexapod', 'quadruped']);
    expect(canonicalJson(robots.records.find((r) => r.id === 'quadruped')!.robot)).toBe(canonicalJson(q));
    const skills = await b.loadSkills();
    expect(skills.corrupted).toEqual([]);
    expect(skills.records.map((s) => s.id)).toEqual(['skill_a']);
    expect(await b.getSetting('workspace.builderRobot')).toEqual(['quadruped', await morphologyHash(q)]);
    expect(canonicalJson(await b.getRobot('quadruped', await morphologyHash(q)))).toBe(canonicalJson(q));
    b.close();
  });

  it('keeps each morphology version as its own record', async () => {
    const s = await MorphStore.open(dbName());
    const q = quadruped();
    const heavier = { ...q, base: { ...q.base, mass: 5 } };
    await s.saveRobot(q);
    await s.saveRobot(heavier);
    await s.saveRobot(q);
    const { records } = await s.loadRobots();
    expect(records).toHaveLength(2);
    expect(new Set(records.map((r) => r.morphologyHash)).size).toBe(2);
    s.close();
  });

  it('reports corrupted records with reasons instead of dropping them, until removed', async () => {
    const s = await MorphStore.open(dbName());
    const q = quadruped();
    await s.saveRobot(q);
    const hash = await morphologyHash(q);
    await s.putRaw('robots', { id: 'quadruped', morphologyHash: 'f'.repeat(64), name: 'Tampered', savedAt: 'x', robot: q });
    await s.putRaw('robots', { id: 'broken', morphologyHash: 'abc', robot: { schemaVersion: 1, id: 'broken' } });
    await s.putRaw('skills', { id: 'skill_bad', kind: 'morphlab.skill', method: 'teleport' });
    const robots = await s.loadRobots();
    expect(robots.records.map((r) => r.morphologyHash)).toEqual([hash]);
    expect(robots.corrupted.map((c) => c.key).sort()).toEqual(['broken / abc', `quadruped / ${'f'.repeat(64)}`]);
    expect(robots.corrupted.find((c) => c.key.startsWith('quadruped'))!.reasons).toEqual(['The stored morphology hash or id does not match the robot it contains.']);
    expect(robots.corrupted.find((c) => c.key.startsWith('broken'))!.reasons).toEqual(expect.arrayContaining(['robot.base must be an object.', 'robot.name must be a string.']));
    const skills = await s.loadSkills();
    expect(skills.records).toEqual([]);
    expect(skills.corrupted[0]!.key).toBe('skill_bad');
    expect(skills.corrupted[0]!.reasons).toContain('skill.method must be one of state-machine, tuned-parameters, replay, imitation-mlp, es-improved.');
    // Still present on the next read; removal is explicit.
    expect((await s.loadSkills()).corrupted).toHaveLength(1);
    await s.removeRecord(skills.corrupted[0]!);
    expect((await s.loadSkills()).corrupted).toHaveLength(0);
    s.close();
  });

  it('refuses invalid robots and credential-like settings', async () => {
    const s = await MorphStore.open(dbName());
    const bad = quadruped();
    bad.base.mass = -1;
    await expect(s.saveRobot(bad)).rejects.toThrow('Refusing to store an invalid robot: Base: Base mass must be a positive number of kilograms.');
    await expect(s.setSetting('cloud.apiKey', 'sk-test')).rejects.toThrow('Credentials are never stored.');
    await expect(s.setSetting('openai_token', 'x')).rejects.toThrow('Credentials are never stored.');
    expect(await s.dumpText()).not.toContain('sk-test');
    s.close();
  });

  it('upgrades a version 1 database without losing records, then stores and validates memory records', async () => {
    const name = dbName();
    // The store layout of version 1: robots, skills and settings.
    const v1 = await openDB(name, 1, {
      upgrade(d) {
        d.createObjectStore('robots', { keyPath: ['id', 'morphologyHash'] });
        d.createObjectStore('skills', { keyPath: 'id' }).createIndex('morphologyHash', 'robot.morphologyHash');
        d.createObjectStore('settings', { keyPath: 'key' });
      },
    });
    await v1.put('skills', await skillFor());
    await v1.put('settings', { key: 'ui.preferences', value: { guideHidden: true, learnMode: false } });
    v1.close();

    const s = await MorphStore.open(name);
    expect((await s.loadSkills()).records.map((x) => x.id)).toEqual(['skill_a']);
    expect(await s.getSetting('ui.preferences')).toEqual({ guideHidden: true, learnMode: false });
    expect(await s.loadMemories()).toEqual({ records: [], corrupted: [] });

    const place: MemoryRecord = { id: 'place:open:dock', kind: 'place', mapId: 'open', name: 'dock', x: 1, y: 1.5, source: 'given', savedAt: '2026-10-04T12:00:00.000Z' };
    await s.putMemory(place);
    await expect(s.putMemory({ ...place, x: Number.NaN })).rejects.toThrow('Refusing to store an invalid memory record: x: ');
    await s.putRaw('memories', { ...place, id: 'place:open:other' });
    const loaded = await s.loadMemories();
    expect(loaded.records).toEqual([place]);
    expect(loaded.corrupted.map((c) => [c.store, c.key, c.reasons])).toEqual([['memories', 'place:open:other', ['record: The record id does not match its map and name.']]]);
    expect(await s.dumpText()).toContain('"place:open:dock"');
    await s.deleteMemory('place:open:dock');
    await s.removeRecord(loaded.corrupted[0]!);
    expect(await s.loadMemories()).toEqual({ records: [], corrupted: [] });
    s.close();
  });
});
