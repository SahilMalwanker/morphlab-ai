import { describe, expect, it } from 'vitest';
import { buildKinematicModel, forwardKinematics } from '../../src/core/kinematics/fk';
import { homeConfiguration } from '../../src/core/model/catalog';
import { canonicalJson, morphologyHash } from '../../src/core/model/hash';
import { quadruped } from '../../src/core/model/presets';
import { BuilderSession } from '../../src/ui/builder/session';

const AT = '2026-10-03T12:00:00.000Z';
/** crypto.subtle.digest resolves on another thread; wait until the current version's hash is set. */
async function hashed(s: BuilderSession) {
  for (let i = 0; i < 2000 && s.state.hash === null; i++) await new Promise((r) => setTimeout(r, 1));
}

function knee(s: BuilderSession) {
  s.selectLimb('front_left');
  s.selectJoint(2);
}

describe('builder session', () => {
  it('refuses an invalid edit with a specific message and leaves the robot unchanged', () => {
    const s = new BuilderSession(quadruped());
    const before = s.state.robot;
    knee(s);
    expect(s.updateJoint({ lower: 0.5 })).toBe(false);
    expect(s.state.issues).toContain('Front left leg, Knee: Lower limit must be below upper limit.');
    expect(s.state.robot).toBe(before);
    expect(s.state.version).toBe(1);
    expect(s.state.attempt).toBe(1);
    expect(s.setBase({ mass: -1 })).toBe(false);
    expect(s.state.issues).toEqual(['Base: Base mass must be a positive number of kilograms.']);
    expect(s.rename('   ')).toBe(false);
    expect(s.state.issues).toEqual(['Robot name is required.']);
    expect(s.state.robot).toBe(before);
  });

  it('every accepted edit is a new version with a new morphology hash', async () => {
    const s = new BuilderSession(quadruped());
    await hashed(s);
    const h1 = s.state.hash!;
    expect(h1).toBe(await morphologyHash(quadruped()));
    knee(s);
    expect(s.updateJoint({ mass: 0.2 })).toBe(true);
    expect(s.state.version).toBe(2);
    await hashed(s);
    expect(s.state.hash).not.toBe(h1);
    expect(s.state.history.map((h) => h.version)).toEqual([2, 1]);
    expect(s.state.history[0]!.hash).toBe(s.state.hash);
    expect(s.state.history[1]!.hash).toBe(h1);
    expect(s.state.history[0]!.change).toBe('Front left leg, Knee: mass (mirrored)');
    // Renaming is a new version but not a new morphology.
    const h2 = s.state.hash;
    s.rename('My quadruped');
    await hashed(s);
    expect(s.state.version).toBe(3);
    expect(s.state.hash).toBe(h2);
  });

  it('submitting a value that is already there is not a new version', () => {
    const s = new BuilderSession(quadruped());
    const before = s.state.robot;
    expect(s.setBase({ mass: before.base.mass })).toBe(true);
    knee(s);
    // Mirroring off: a mirrored edit regenerates the partner limb, which is a real change for this preset.
    s.setMirror(false);
    expect(s.updateJoint({ upper: before.limbs[0]!.joints[2]!.limit.upper })).toBe(true);
    expect(s.state.version).toBe(1);
    expect(s.state.history).toHaveLength(1);
    expect(s.state.robot).toBe(before);
  });

  it('mirrors edits to the partner limb only when mirroring is on', () => {
    const s = new BuilderSession(quadruped());
    knee(s);
    s.updateJoint({ upper: -0.3 });
    const fr = () => s.state.robot.limbs.find((l) => l.id === 'front_right')!.joints[2]!;
    expect(fr().limit.upper).toBe(-0.3);
    s.setMirror(false);
    s.updateJoint({ upper: -0.4 });
    expect(fr().limit.upper).toBe(-0.3);
  });

  it('clamps the stance into narrowed limits instead of refusing the edit', () => {
    const s = new BuilderSession(quadruped());
    knee(s);
    expect(s.updateJoint({ upper: -1.6 })).toBe(true);
    expect(s.state.robot.stance!.front_left_knee).toBe(-1.6);
    expect(s.state.robot.stance!.front_right_knee).toBe(-1.6);
    expect(s.state.history[0]!.change).toBe('Front left leg, Knee: upper (mirrored); stance clamped to the new limits');
  });

  it('re-seats the robot on the ground after a geometric edit', () => {
    const s = new BuilderSession(quadruped());
    knee(s);
    s.updateJoint({ xyz: [0, 0, -0.22] });
    const r = s.state.robot;
    const fk = forwardKinematics(buildKinematicModel(r), homeConfiguration(r));
    const lowest = Math.min(...fk.tips.map((T, i) => T[14]! - r.limbs[i]!.tip.radius));
    expect(lowest).toBeCloseTo(0.002, 9);
  });

  it('export then import restores byte-identical canonical JSON', () => {
    const s = new BuilderSession(quadruped());
    knee(s);
    s.updateJoint({ effort: 10 });
    const text = s.exportText(AT);
    const original = s.state.robot;
    const other = new BuilderSession(quadruped());
    expect(other.importText(text)).toBe(true);
    expect(canonicalJson(other.state.robot)).toBe(canonicalJson(original));
    expect(other.exportText(AT)).toBe(text);
    expect(other.state.notice).toBe('Imported Quadruped.');
  });

  it('reports a broken import and keeps the current robot', () => {
    const s = new BuilderSession(quadruped());
    const before = s.state.robot;
    expect(s.importText('{"kind": "morphlab.bundle", "schemaVersion": 1, "robots": [{"id": "x"}]}')).toBe(false);
    expect(s.state.issues[0]).toBe('Robot 1: robot.schemaVersion must be 1.');
    expect(s.importText('not json')).toBe(false);
    expect(s.state.issues[0]).toMatch(/^The file is not valid JSON/);
    expect(s.state.robot).toBe(before);
  });

  it('refuses structural edits that would break the robot', () => {
    const s = new BuilderSession(quadruped());
    expect(s.generateLegged({ ...s.state.legged, pairs: 5 })).toBe(false);
    expect(s.state.issues).toEqual(['Leg pairs must be a whole number from 1 to 4.']);
    s.startFromPreset('arm3');
    expect(s.removeSelectedLimb()).toBe(false);
    expect(s.state.issues).toEqual(['A robot needs at least one limb.']);
    s.startFromPreset('rover');
    expect(s.setJointCount(2)).toBe(false);
    expect(s.state.issues).toEqual(['A wheel has exactly one joint.']);
  });
});
