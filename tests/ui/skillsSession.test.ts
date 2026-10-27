import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it, vi } from 'vitest';
import { LongTermMemory } from '../../src/ai/memory';
import { homeConfiguration, PRESETS } from '../../src/core/model/catalog';
import { methodLabel, skillLabel } from '../../src/skills/artifact';
import { handleSkillJob, type SkillWorkerMessage, type SkillWorkerRequest } from '../../src/skills/jobs';
import { RolloutPool } from '../../src/skills/rollouts';
import { SkillWorkerClient } from '../../src/skills/workerClient';
import { workspaceBridge } from '../../src/ui/ai/bridge';
import { PhysicsSession } from '../../src/ui/physics/session';
import { ATTEMPT_DT, describePlay, SkillsSession, walkOutcomeText, type Attempt, type CommandTarget, type SkillsSessionOptions } from '../../src/ui/skills/session';
import { homeTipClearance } from '../../src/skills/taught';
import { DEFAULT_GAIT } from '../../src/skills/gait';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

/** Runs worker jobs in-process with the same handler the real worker uses. */
class InProcessWorker {
  onmessage: ((e: MessageEvent<SkillWorkerMessage>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  terminated = false;
  postMessage(req: SkillWorkerRequest): void {
    void handleSkillJob(() => Promise.resolve(mj), req, (m) => {
      if (!this.terminated) this.onmessage?.({ data: m } as MessageEvent<SkillWorkerMessage>);
    });
  }
  terminate(): void {
    this.terminated = true;
  }
}

/** A worker that accepts jobs and never answers, so a job stays in progress. */
class SilentWorker {
  onmessage = null;
  onerror = null;
  postMessage(): void {}
  terminate(): void {}
}

async function ready(makeWorker: () => object = () => new InProcessWorker(), options: SkillsSessionOptions = {}): Promise<SkillsSession> {
  const physics = new PhysicsSession(() => Promise.resolve(mj));
  const s = new SkillsSession(physics, new SkillWorkerClient(() => makeWorker() as unknown as Worker), undefined, undefined, options);
  await physics.init();
  // crypto.subtle.digest resolves on another thread; wait for the morphology hash.
  for (let i = 0; i < 2000 && s.state.hash === null; i++) await new Promise((r) => setTimeout(r, 1));
  expect(s.state.hash).toMatch(/^[0-9a-f]{64}$/);
  return s;
}

function runPhysics(s: SkillsSession, seconds: number): void {
  const n = Math.round(seconds / s.physics.runtime!.timestep);
  for (let i = 0; i < n; i++) s.physics.stepOnce();
}

/** Steps until the playback settles (at most 12 s of simulated time). */
function settle(s: SkillsSession): void {
  for (let i = 0; i < 6000 && s.state.playing; i++) s.physics.stepOnce();
  expect(s.state.playing).toBeNull();
}

describe('skills session', () => {
  it('stops a fixed-arm driver at the measured pose instead of restoring its home target', async () => {
    const session = await ready();
    try {
      session.setRobot(PRESETS.find((preset) => preset.id === 'arm3')!.make());
      const home = homeConfiguration(session.state.robot);
      session.physics.setDriver(() => home.map((value, index) => index === 1 ? value + 0.3 : value));
      runPhysics(session, 0.5);
      const measured = Array.from(session.physics.runtime!.readSensors().q);
      expect(Math.abs(measured[1]! - home[1]!)).toBeGreaterThan(0.1);
      session.stopActivity('the Stop control was pressed');
      expect(session.physics.hasDriver).toBe(false);
      expect(session.physics.state.qCmd).toEqual(measured);
    } finally {
      session.dispose();
    }
  });

  it('evaluates a method over 20 trials and stores a labeled skill card bound to the hash', async () => {
    const s = await ready();
    const a = await s.evaluate('state-machine');
    expect(a).not.toBeNull();
    expect(s.state.library).toHaveLength(1);
    expect(a!.method).toBe('state-machine');
    expect(a!.robot.morphologyHash).toBe(s.state.hash);
    expect(a!.evaluation!.trials).toBe(20);
    expect(s.state.message!.text).toMatch(/^State machine \(hand-designed\): \d+ of 20 trials succeeded/);
    expect(s.state.job).toBeNull();
  });

  it('explains which step is missing before a method can be evaluated', async () => {
    const s = await ready();
    expect(await s.evaluate('replay')).toBeNull();
    expect(s.state.message).toEqual({ tone: 'error', text: 'Record a demonstration first.' });
    expect(await s.evaluate('imitation-mlp')).toBeNull();
    expect(s.state.message!.text).toBe('Train the imitation policy first.');
    expect(s.state.library).toHaveLength(0);
  });

  it('records a teleoperated demonstration at 100 Hz from the live simulation', async () => {
    const s = await ready();
    s.startTeleop();
    expect(s.physics.hasDriver).toBe(true);
    s.startRecording();
    runPhysics(s, 0.3);
    s.setTeleop(1, 0.06);
    runPhysics(s, 0.7);
    s.stopRecording();
    const d = s.state.demonstration!;
    expect(d.source).toBe('you');
    expect(d.trajectory.dt).toBe(0.01);
    expect(d.trajectory.frames.length).toBeGreaterThanOrEqual(99);
    expect(d.trajectory.frames.length).toBeLessThanOrEqual(101);
    // Starts at the stance (sliders at zero) and ends at the shifted, lifted pose.
    const stance = homeConfiguration(s.state.robot);
    d.trajectory.frames[0]!.forEach((v, i) => expect(v).toBeCloseTo(stance[i]!, 4));
    expect(Math.max(...d.trajectory.frames.at(-1)!.map((v, i) => Math.abs(v - stance[i]!)))).toBeGreaterThan(0.1);
    s.stopActivity();
    expect(s.physics.hasDriver).toBe(false);
  });

  it('plays a compatible skill and refuses it on a different morphology', async () => {
    const s = await ready();
    const a = await s.evaluate('state-machine');
    expect(await s.play(a!.id)).toBe(true);
    expect(s.state.playing).toBe(a!.id);
    expect(s.physics.hasDriver).toBe(true);
    s.setPreset('hexapod');
    await new Promise((r) => setTimeout(r, 0));
    expect(await s.play(a!.id)).toBe(false);
    expect(s.state.message!.tone).toBe('error');
    expect(s.state.message!.text).toContain('A skill is never reused on a different body');
    expect(s.physics.hasDriver).toBe(false);
  });

  it('robots without four feet are told why the skill is unavailable', async () => {
    const s = await ready();
    s.setRobot(PRESETS.find((p) => p.id === 'arm3')!.make(), 'arm3');
    expect(s.state.unsupported).toBe('The foot-lift skill needs at least four legs with feet: three to stand on and one to lift.');
    expect(await s.evaluate('state-machine')).toBeNull();
    expect(s.state.message!.text).toBe(s.state.unsupported);
  });

  it('the flagship command runs a stored skill and reports the measured hold', async () => {
    const s = await ready();
    await s.evaluate('state-machine');
    const p = s.runCommand('lift the front left leg and hold for three seconds');
    let done = false;
    void p.then(() => (done = true));
    for (let i = 0; i < 20000 && !done; i++) {
      s.physics.stepOnce();
      if (i % 50 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    const out = await p;
    expect(out.kind).toBe('done');
    expect(out.source).toBe('local rules');
    expect(out.measuredHoldS).toBeGreaterThan(3);
    expect(out.measuredHoldS).toBeLessThan(3.6);
    expect(out.text).toMatch(/^Lifted the front left leg\. Measured hold at least 3 cm above the ground: \d\.\d\d s \(requested 3 s; simulator ground truth\)\.$/);
    expect(out.note).toContain('this 3 s hold was not part of that evaluation');
    expect(s.state.commandLog[0]).toBe(out);
  });

  it('unsupported and unrunnable commands are refused with a suggestion', async () => {
    const s = await ready();
    const walk = await s.runCommand('walk forward');
    expect(walk.kind).toBe('refusal');
    expect(walk.suggestion).toBeTruthy();
    const none = await s.runCommand('lift the rear right leg');
    expect(none.kind).toBe('refusal');
    expect(none.text).toContain('There is no evaluated foot-lift skill for the rear right leg');
  });

  it('exports the library and imports it elsewhere, labeled as imported, saved once and playable', async () => {
    const a = await ready();
    const skill = (await a.evaluate('state-machine'))!;
    const text = a.exportText('2026-10-03T12:00:00.000Z');
    const b = await ready();
    const saved: string[] = [];
    const robots: string[] = [];
    b.attachPersistence({ saveSkill: async (x) => void saved.push(x.id), deleteSkill: async () => {}, saveRobot: async () => {}, storeRobot: async (r) => void robots.push(r.name) });
    expect(await b.importText(text, '2026-10-04T08:00:00.000Z')).toBe(1);
    const imported = b.state.library[0]!;
    expect(imported.id).toBe(skill.id);
    expect(imported.importedAt).toBe('2026-10-04T08:00:00.000Z');
    expect(imported.evaluation).toEqual(skill.evaluation);
    expect(b.state.message).toEqual({ tone: 'info', text: 'Imported 1 skill. Their evaluation results come from the file and were not re-run here; text commands will not use them until you re-evaluate them here. Saved 1 robot the skills were evaluated on.' });
    await new Promise((r) => setTimeout(r, 0));
    expect(saved).toEqual([skill.id]);
    expect(robots).toEqual(['Quadruped']);
    expect(await b.importText(text)).toBe(0);
    expect(b.state.library).toHaveLength(1);
    expect(b.state.message!.text).toContain('1 was already in the library.');
    expect(await b.play(imported.id)).toBe(true);
    b.stopActivity();
    expect(await b.importText('{"kind": "something else"}')).toBe(0);
    expect(b.state.message).toEqual({ tone: 'error', text: 'Import refused: The file is not a MorphLab bundle (expected kind "morphlab.bundle").' });
  });

  it('exports every robot the library was evaluated on and says which are missing', async () => {
    const s = await ready();
    const quad = s.state.robot;
    await s.evaluate('state-machine');
    s.setPreset('hexapod');
    for (let i = 0; i < 2000 && s.state.hash === null; i++) await new Promise((r) => setTimeout(r, 1));
    await s.evaluate('state-machine');
    const bundle = JSON.parse(s.exportText('2026-10-03T12:00:00.000Z')) as { robots: { name: string }[]; skills: unknown[] };
    expect(bundle.skills).toHaveLength(2);
    expect(bundle.robots.map((r) => r.name)).toEqual(['Hexapod', 'Quadruped']);
    expect(s.exportPlan().missing).toEqual([]);
    // A skill whose robot is stored nowhere is exported without it, and the export says so.
    const orphan = { ...s.state.library[0]!, id: 'skill_orphan', robot: { robotId: 'lost', morphologyHash: 'c'.repeat(64) } };
    s.addSkills([orphan]);
    expect(s.exportPlan().missing).toEqual([`lost (morphology ${'c'.repeat(12)})`]);
    s.exportText();
    expect(s.state.message!.text).toBe(`Exported 3 skills and 2 robots. Not stored here, so not in the file: lost (morphology ${'c'.repeat(12)}).`);
    expect(quad.name).toBe('Quadruped');
  });

  it('records every playback as an attempt of simulator states and opens two side by side', async () => {
    const s = await ready();
    const a = (await s.evaluate('state-machine'))!;
    for (let k = 0; k < 2; k++) {
      expect(await s.play(a.id)).toBe(true);
      for (let i = 0; i < 6000 && s.state.playing; i++) s.physics.stepOnce();
      expect(s.state.playing).toBeNull();
    }
    expect(s.state.attempts.map((x) => x.label)).toEqual([`Attempt 2: ${a.name}`, `Attempt 1: ${a.name}`]);
    const [second, first] = s.state.attempts as [Attempt, Attempt];
    const links = s.state.robot.limbs.reduce((n, l) => n + l.joints.length, 0);
    expect(first.frames[0]!.length).toBe(16 * (1 + links));
    // One frame every 40 ms of simulated time; each holds the recorded base and link transforms.
    const span = first.t.at(-1)! - first.t[0]!;
    expect(first.frames.length).toBeGreaterThanOrEqual(Math.floor(span / ATTEMPT_DT));
    expect(first.frames.length).toBeLessThanOrEqual(Math.ceil(span / ATTEMPT_DT) + 1);
    expect(first.holdS).toBe(s.state.lastPlay!.holdS);
    expect(Math.max(...first.clearanceM)).toBeCloseTo(s.state.lastPlay!.maxClearanceM, 3);
    // The same skill from the same reset state is deterministic.
    expect(second.frames.at(-1)).toEqual(first.frames.at(-1));
    expect(s.openComparison(first.id, second.id)).toBe(true);
    expect(s.state.comparing).toEqual([first.id, second.id]);
    expect(s.openComparison(first.id, 'attempt_99')).toBe(false);
    s.closeComparison();
    expect(s.state.comparing).toBeNull();
    // An interrupted playback is kept and labeled as such.
    await s.play(a.id);
    for (let i = 0; i < 300; i++) s.physics.stepOnce();
    s.stopActivity();
    expect(s.state.attempts[0]!.label).toBe(`Attempt 3: ${a.name}`);
    expect(s.state.attempts[0]!.outcome).toBe('interrupted');
    expect(s.state.attempts[0]!.completed).toBe(false);
    expect(s.state.attempts[1]!.outcome).toBe('succeeded');
  });

  it('judges live playback by the evaluation referee with the requested hold', async () => {
    const s = await ready();
    const a = (await s.evaluate('state-machine'))!;
    // The stored controller holds for 2 s; demanding 5 s must be reported as a failure, not a lift.
    expect(await s.play(a.id, undefined, { requiredHoldS: 5 })).toBe(true);
    settle(s);
    const r = s.state.lastPlay!;
    expect(r.outcome).toBe('failed');
    expect(r.success).toBe(false);
    expect(r.completed).toBe(true);
    expect(r.requiredHoldS).toBe(5);
    expect(r.failures).toContain('hold-too-short');
    expect(r.holdS).toBeGreaterThan(1.8);
    expect(r.holdS).toBeLessThan(5);
    const text = describePlay(r, 'front left leg', 5);
    expect(text.kind).toBe('failed');
    expect(text.text).not.toMatch(/^Lifted/);
    expect(text.text).toContain('the foot was not held clear long enough');
    // A taught skill whose goals pass while the body ends on its back is reported as tipped over.
    const air = { goal: { kind: 'airborne', minS: 0.01 }, met: true, measured: 0.348, unit: 's', text: 'every foot off the ground with the body upright for at most 0.348 s at once (needed 0.01 s)', score: 1 } as const;
    const flipped = describePlay({ ...r, skillName: 'jump', outcome: 'succeeded', success: true, failures: [], goals: [air], maxTiltRad: Math.PI }, '', 0);
    expect(flipped.text).toBe(`Did "jump": every goal was met, but the body tipped over (largest tilt 180 deg), which none of its goals checks. Measured (simulator ground truth): met, ${air.text}.`);
    expect(flipped.suggestion).toContain('Add an upright or standsAtEnd goal');
    expect(describePlay({ ...r, skillName: 'jump', outcome: 'succeeded', success: true, failures: [], goals: [air], maxTiltRad: 0.3 }, '', 0).text).toMatch(/^Did "jump": every goal was met\. Measured/);
    // The default demand is the evaluation's 2 s, which the same controller meets.
    expect(await s.play(a.id)).toBe(true);
    settle(s);
    expect(s.state.lastPlay!.outcome).toBe('succeeded');
    expect(s.state.lastPlay!.failures).toEqual([]);
    expect(s.state.lastPlay!.setupNotes).toEqual([]);
  });

  it('records every finished playback in skill memory, shows it to the AI, and forgets it with the skill', async () => {
    const s = await ready();
    const memory = new LongTermMemory([], () => '2026-10-04T12:00:00.000Z');
    s.attachMemory(memory);
    const a = (await s.evaluate('state-machine'))!;
    expect(await s.play(a.id, undefined, { requiredHoldS: 5 })).toBe(true);
    settle(s);
    expect(await s.play(a.id)).toBe(true);
    settle(s);
    expect(await s.play(a.id)).toBe(true);
    for (let i = 0; i < 300; i++) s.physics.stepOnce();
    s.stopActivity();
    const runs = memory.skillRuns(a.id)!;
    expect(runs).toMatchObject({ skillId: a.id, morphologyHash: s.state.hash, completed: 2, succeeded: 1, interrupted: 1 });
    expect(runs.last.outcome).toBe('interrupted');
    expect(runs.last.holdS).toBe(s.state.lastPlay!.holdS);

    const bridge = workspaceBridge({ mode: () => 'skills', nav: () => null, skills: () => s, memory: () => memory, control: () => null });
    const snap = await bridge.snapshot();
    if ('unsupported' in snap) throw new Error(snap.unsupported);
    const e = a.evaluation!;
    expect(snap.summary).toContain(`"${a.name}" (${methodLabel(a.method)}, evaluation ${e.successes}/${e.trials}; live runs: 2 completed, 1 succeeded, 1 stopped early; the last one interrupted)`);
    expect(snap.summary).toContain('they are a record, not an evaluation');
    const listed = snap.query('listSkills', {});
    expect(listed.measurements[`${a.name} live runs completed`]).toEqual({ value: 2, unit: 'count' });
    expect(listed.measurements[`${a.name} live runs succeeded`]).toEqual({ value: 1, unit: 'count' });

    s.deleteSkill(a.id);
    expect(memory.skillRuns(a.id)).toBeNull();
  });

  it('a reset or a controller error during playback settles the run and the command', async () => {
    const s = await ready();
    await s.evaluate('state-machine');
    const p = s.runCommand('lift the front left leg');
    for (let i = 0; i < 200 && !s.state.playing; i++) await new Promise((r) => setTimeout(r, 0));
    for (let i = 0; i < 400; i++) s.physics.stepOnce();
    s.physics.reset();
    const out = await p;
    expect(out.kind).toBe('failed');
    expect(out.text).toMatch(/^The run was interrupted before it finished \(the simulation was reset\)/);
    expect(s.state.playing).toBeNull();
    expect(s.physics.hasDriver).toBe(false);
    expect(s.state.lastPlay!.outcome).toBe('interrupted');

    // A controller that throws ends the run as a fault instead of leaving it pending.
    const attach = s.physics.setDriver.bind(s.physics);
    s.physics.setDriver = (d, label) => attach(d && (() => { throw new Error('policy exploded'); }), label);
    const q = s.runCommand('lift the front left leg');
    for (let i = 0; i < 200 && !s.state.playing; i++) await new Promise((r) => setTimeout(r, 0));
    for (let i = 0; i < 20 && s.state.playing; i++) s.physics.stepOnce();
    const fault = await q;
    expect(fault.kind).toBe('failed');
    expect(fault.text).toContain('the controller failed: policy exploded');
    expect(s.state.lastPlay!.outcome).toBe('faulted');
    expect(s.state.playing).toBeNull();
    expect(s.physics.hasDriver).toBe(false);
  });

  it('imported skills are refused by commands until re-evaluated here', async () => {
    const a = await ready();
    const skill = (await a.evaluate('state-machine'))!;
    const b = await ready();
    expect(await b.importText(a.exportText())).toBe(1);
    expect(b.state.library[0]!.evaluationOrigin).toBe('imported');
    const refused = await b.runCommand('lift the front left leg');
    expect(refused.kind).toBe('refusal');
    expect(refused.text).toContain('was imported and its evaluation (');
    expect(refused.text).toContain('has not been reproduced here.');
    expect(refused.suggestion).toContain('Re-evaluate here');
    const updated = (await b.reevaluate(skill.id))!;
    expect(updated.evaluationOrigin).toBe('local');
    expect(updated.importedEvaluation).toEqual(skill.evaluation);
    expect(updated.evaluation!.trials).toBe(20);
    expect(b.state.message!.text).toMatch(/^Re-evaluated .*: \d+ of 20 trials succeeded \(the file claimed \d+ of 20\)\.$/);
    const p = b.runCommand('lift the front left leg');
    let done = false;
    void p.then(() => (done = true));
    for (let i = 0; i < 20000 && !done; i++) {
      b.physics.stepOnce();
      if (i % 50 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    expect((await p).kind).toBe('done');
  });

  it('discards work for a leg that is no longer selected', async () => {
    const s = await ready();
    const other = s.state.legs[1]!.id;
    // The leg changes before the worker job starts: the result arrives and is discarded.
    const pending = s.evaluate('state-machine');
    s.setLimb(other);
    expect(await pending).toBeNull();
    expect(s.state.library).toHaveLength(0);
    expect(s.state.job).toBeNull();
    expect(s.state.message!.text).toBe('Evaluating: State machine (hand-designed): the result was discarded because the robot or leg changed while it ran.');
    // The leg changes while the job runs: the job is cancelled.
    const slow = await ready(() => new SilentWorker());
    const running = slow.evaluate('state-machine');
    for (let i = 0; i < 200 && !slow.state.job; i++) await new Promise((r) => setTimeout(r, 0));
    expect(slow.state.job).not.toBeNull();
    slow.setLimb(other);
    expect(await running).toBeNull();
    expect(slow.state.library).toHaveLength(0);
    expect(slow.state.job).toBeNull();
    expect(slow.state.message!.text).toBe('Evaluating: State machine (hand-designed) was cancelled.');
  });

  it('commands typed in another tab act on that tab or are refused, never on a hidden robot', async () => {
    const s = await ready();
    await s.evaluate('state-machine');
    const hexapod = PRESETS.find((p) => p.id === 'hexapod')!.make();
    let shown = 0;
    const calls: string[] = [];
    const physicsTab: CommandTarget = {
      tab: 'Physics',
      robot: hexapod,
      simulation: { stand: () => (calls.push('stand'), 'Physics: stood.'), reset: () => (calls.push('reset'), 'Physics: reset.') },
      showSkills: () => shown++,
    };
    const lift = await s.runCommand('lift the front left leg', physicsTab);
    expect(lift.kind).toBe('refusal');
    expect(lift.text).toMatch(/^The Physics tab shows Hexapod \(morphology [0-9a-f]{12}\), but foot-lift skills run on the Skills robot, Quadruped \(morphology [0-9a-f]{12}\)\. A skill is never reused on a different body\.$/);
    expect(shown).toBe(0);
    expect(s.state.playing).toBeNull();

    // Stand and reset act on the tab's own simulation; the Skills simulation is untouched.
    const t0 = s.physics.runtime!.time;
    s.physics.stepOnce();
    expect((await s.runCommand('stand', physicsTab)).text).toBe('Physics: stood.');
    expect((await s.runCommand('reset', physicsTab)).text).toBe('Physics: reset.');
    expect(calls).toEqual(['stand', 'reset']);
    expect(s.physics.runtime!.time).toBeGreaterThan(t0);

    const buildTab: CommandTarget = { tab: 'Build', robot: s.state.robot, simulation: null, showSkills: () => shown++ };
    const stand = await s.runCommand('stand', buildTab);
    expect(stand.kind).toBe('refusal');
    expect(stand.text).toBe('The Build tab has no live simulation to stand.');

    // The same morphology shown elsewhere: the lift runs in the Skills view.
    const p = s.runCommand('lift the front left leg', buildTab);
    let done = false;
    void p.then(() => (done = true));
    for (let i = 0; i < 20000 && !done; i++) {
      s.physics.stepOnce();
      if (i % 50 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    expect((await p).kind).toBe('done');
    expect(shown).toBe(1);
  });
});

/** Steps the simulation until the promise settles, yielding so its microtasks run. */
async function stepUntil<T>(s: SkillsSession, p: Promise<T>): Promise<T> {
  let done = false;
  void p.then(() => (done = true));
  for (let i = 0; i < 20000 && !done; i++) {
    s.physics.stepOnce();
    if (i % 50 === 0) await new Promise((r) => setTimeout(r, 0));
  }
  return p;
}

describe('skills the user teaches', () => {
  it('poses keyframes on the live robot, tries them, evaluates them, and runs the skill by its name', async () => {
    const s = await ready();
    const memory = new LongTermMemory([], () => '2026-10-05T12:00:00.000Z');
    s.attachMemory(memory);
    expect(s.state.panel).toBe('teach');
    s.newDraft();
    expect(s.state.draft!.goals.map((g) => g.kind)).toEqual(['upright', 'noFall']);
    expect(await s.evaluateDraft()).toBeNull();
    expect(s.state.message!.text).toMatch(/^Name the skill first/);
    s.setDraftName('crouch');
    expect(await s.previewDraft()).toBe(false);
    expect(s.state.message!.text).toBe('Add at least one keyframe: pose the robot with the joint sliders, then press Add keyframe.');

    // Pose every leg with the joint sliders; the live robot follows under its joint controller.
    s.state.robot.limbs.forEach((_, li) => {
      s.setPoseJoint(3 * li + 1, 1.1);
      s.setPoseJoint(3 * li + 2, -2.2);
    });
    runPhysics(s, 1);
    expect(s.physics.runtime!.groundTruthBaseState().pos[2]).toBeLessThan(0.2);
    expect(s.addKeyframe()).toBe(true);
    expect(s.state.draft!.keyframes[0]!.q.slice(0, 3)).toEqual([0, 1.1, -2.2]);
    expect(s.setEndHold(1.5)).toBe(true);
    expect(s.addGoal('baseHeight')).toBe(true);
    expect(s.setGoal(2, { kind: 'baseHeight', below: true, heightM: 0.2, holdS: 1 })).toBe(true);
    expect(s.setGoal(0, { kind: 'upright', maxTiltRad: 3 })).toBe(false);
    expect(s.state.message!.text).toBe('Goal 1.maxTiltRad must be a number from 0.05 to 1.6.');

    // One try in the live simulation: judged by the goals, kept as an attempt, not saved.
    expect(await s.previewDraft()).toBe(true);
    expect(s.state.playing).toBe('preview');
    settle(s);
    const preview = s.state.lastPlay!;
    expect(preview.skillName).toBe('crouch (preview)');
    expect(preview.outcome).toBe('succeeded');
    expect(preview.goals!.map((g) => g.met)).toEqual([true, true, true]);
    const attempt = s.state.attempts[0]!;
    expect(attempt.label).toBe('Attempt 1: crouch (preview)');
    expect(attempt.resultText).toMatch(/^3 of 3 goals met, max tilt \d+\.\d deg, succeeded$/);
    expect(attempt.plot).toMatchObject({ title: 'Body height', unit: 'm', threshold: 0.2 });
    expect(Math.min(...attempt.plot.values)).toBeLessThan(0.2);
    expect(attempt.clearanceM).toEqual([]);
    expect(s.state.library).toHaveLength(0);

    const a = (await s.evaluateDraft())!;
    expect(a.skillType).toBe('taught');
    expect(a.name).toBe('crouch (keyframes)');
    expect(a.taught).toEqual({ name: 'crouch', goals: s.state.draft!.goals });
    expect(a.evaluation!.trials).toBe(20);
    expect(a.evaluation!.successes).toBeGreaterThanOrEqual(19);
    expect(s.state.message!.text).toMatch(/^"crouch": (19|20) of 20 trials met every goal \(\d+\.\d s\)\. Added to the library as "crouch \(keyframes\)"; say "crouch" to run it\.$/);

    // A command with the skill's name runs the evaluated version and reports every goal as measured.
    const out = await stepUntil(s, s.runCommand('please crouch'));
    expect(out.kind).toBe('done');
    expect(out.source).toBe('local rules');
    expect(out.text).toMatch(
      /^Did "crouch": every goal was met\. Measured \(simulator ground truth\): met, largest tilt \d+ deg \(limit 29 deg\); met, lowest body height 0\.\d{3} m \(minimum 0\.13 m\); met, body below 0\.2 m for \d\.\d\d s of 1 s required \(lowest 0\.\d{3} m\)\.$/,
    );
    expect(out.note).toMatch(/^Using "crouch \(keyframes\)" \(Keyframe motion you posed \(open loop\), (19|20)\/20 in evaluation\)\.$/);
    expect(memory.skillRuns(a.id)).toMatchObject({ completed: 1, succeeded: 1 });
    expect(memory.skillRuns(a.id)!.last.holdS).toBeNull();

    // Other requests point to teaching and list what this robot was taught.
    const jump = await s.runCommand('jump');
    expect(jump.kind).toBe('refusal');
    expect(jump.text).toBe('No jumping skill has been taught to this robot yet.');
    expect(jump.suggestion).toContain('Teach it in the Skills tab');
    expect(jump.suggestion).toContain('Skills taught on this robot run by their name: "crouch".');
    // More than the name: explained locally (the app sends such requests to the AI when it is available).
    const more = await s.runCommand('crouch twice and tell me how low it went');
    expect(more.kind).toBe('refusal');
    expect(more.text).toBe('"crouch" runs when the command is its name alone, such as "crouch" or "please crouch"; the local rules cannot follow the rest of this request.');
    expect(s.taughtSkillMentioned('crouch twice and tell me how low it went')).toBe('crouch');
    expect(s.taughtSkillMentioned('crouching low')).toBeNull();
    // A tab showing another body never runs it.
    const hexapod = PRESETS.find((p) => p.id === 'hexapod')!.make();
    const elsewhere = await s.runCommand('crouch', { tab: 'Physics', robot: hexapod, simulation: null, showSkills: () => {} });
    expect(elsewhere.kind).toBe('refusal');
    expect(elsewhere.text).toMatch(/^"crouch" is a skill of the Skills robot, Quadruped \(morphology [0-9a-f]{12}\); the Physics tab shows Hexapod \(morphology [0-9a-f]{12}\)\. A skill is never reused on a different body\.$/);

    // The AI sees the skill with its goals and runs it through the same referee.
    const bridge = workspaceBridge({ mode: () => 'skills', nav: () => null, skills: () => s, memory: () => memory, control: () => null });
    const snap = await bridge.snapshot();
    if ('unsupported' in snap) throw new Error(snap.unsupported);
    expect(snap.caps.skills).toEqual(['crouch (keyframes)']);
    expect(snap.summary).toContain(
      `"crouch (keyframes)" (Keyframe motion you posed (open loop), evaluation ${a.evaluation!.successes}/20; live runs: 1 completed, 1 succeeded; the last one succeeded; succeeds when the body stays upright (tilt at most 29 deg), and the body stays above 0.13 m, and the body is below 0.2 m for 1 s)`,
    );
    const exec = snap.action('executeSkill', { skill: 'crouch (keyframes)' });
    if (!('run' in exec)) throw new Error(exec.reason);
    const res = await stepUntil(s, exec.run(new AbortController().signal));
    expect(res.status).toBe('succeeded');
    expect(res.reason).toMatch(/^Did "crouch \(keyframes\)": every goal was met\./);
    expect(res.measurements['goalsMet']).toEqual({ value: 3, unit: 'count' });
    expect(res.measurements['goal 3 (Body height reached)']!.unit).toBe('s');
    const held = snap.action('executeSkill', { skill: 'crouch (keyframes)', holdSeconds: 2 });
    if (!('run' in held)) throw new Error(held.reason);
    expect(await held.run(new AbortController().signal)).toMatchObject({ status: 'failed', reason: '"crouch (keyframes)" is a taught skill; holdSeconds applies to foot lifts only.' });

    // The draft belongs to its robot.
    s.setPreset('hexapod');
    expect(s.state.draft).toBeNull();
  });

  it('offers only limb tip goals on a fixed-base arm', async () => {
    const s = await ready();
    const arm = PRESETS.find((p) => p.id === 'arm3')!.make();
    s.setRobot(arm, 'arm3');
    s.newDraft();
    // The tip rests far above the ground, so the default asks for 5 cm above where it rests.
    const [goal] = s.state.draft!.goals;
    expect(goal).toMatchObject({ kind: 'tipHeight', limbId: 'arm', holdS: 1 });
    expect(Math.abs((goal!.kind === 'tipHeight' ? goal!.minHeightM : 0) - (homeTipClearance(arm, 0) + 0.05))).toBeLessThanOrEqual(0.005);
    expect(s.state.draft!.goals).toHaveLength(1);
    expect(s.addGoal('upright')).toBe(false);
    expect(s.state.message!.text).toBe(`"Stays upright" measures the body, but ${arm.name} has a fixed base; use limb tip goals.`);
    expect(s.state.draft!.goals).toHaveLength(1);
  });
});

describe('skills the AI composes', () => {
  const crouch = {
    name: 'crouch',
    keyframes: [{ seconds: 1, body: { z: -0.08 } }],
    finalHoldSeconds: 1.5,
    goals: [{ kind: 'baseHeight', below: true, heightM: 0.21, holdS: 1 }, { kind: 'upright', maxTiltRad: 0.35 }, { kind: 'noFall', minBaseHeightM: 0.13 }],
  };

  async function aiView(s: SkillsSession) {
    const memory = new LongTermMemory([], () => '2026-10-05T12:00:00.000Z');
    s.attachMemory(memory);
    const bridge = workspaceBridge({ mode: () => 'skills', nav: () => null, skills: () => s, memory: () => memory, control: () => null });
    const snap = await bridge.snapshot();
    if ('unsupported' in snap) throw new Error(snap.unsupported);
    return { bridge, snap };
  }

  it.each(['evaluateSkillDraft', 'improveSkillDraft', 'trainGaitPolicy', 'trainLocomotion'] as const)('reports job Cancel during %s as cancellation, not an invalid draft', async (tool) => {
    const makeWorker = () => new SilentWorker();
    const session = await ready(makeWorker, { rolloutPool: () => new RolloutPool(() => new SkillWorkerClient(() => makeWorker() as unknown as Worker), 1) });
    try {
      const { snap } = await aiView(session);
      const composed = snap.action('draftSkill', crouch);
      if (!('run' in composed)) throw new Error(composed.reason);
      const draft = session.state.draft;
      const executor = snap.action(tool, {});
      if (!('run' in executor)) throw new Error(executor.reason);
      const result = executor.run(new AbortController().signal);
      await vi.waitFor(() => expect(session.state.job).not.toBeNull());
      session.cancelJob();
      expect(await result).toMatchObject({ status: 'cancelled', fallback: null });
      expect(session.state.draft).toBe(draft);
      expect(session.state.library).toHaveLength(0);
      expect(session.state.job).toBeNull();
    } finally {
      session.dispose();
    }
  });

  it('composes a draft from a description of the body, tries it, evaluates and saves it, and runs it by name', async () => {
    const s = await ready();
    const { bridge, snap } = await aiView(s);
    expect(snap.caps.features.skillAuthoring).toBe(true);
    expect(snap.caps.skills).toEqual([]);
    expect(snap.summary).toContain('compose it instead of refusing');
    expect(snap.summary).toContain('front_left (Front left leg, leg with a foot): tip at home (0.170, 0.130, -0.245) m');

    // A draft the body cannot take runs nothing and changes nothing.
    const refused = snap.action('draftSkill', { ...crouch, keyframes: [{ seconds: 1, tips: [{ limb: 'tail', z: 0.1 }] }] });
    if ('run' in refused) throw new Error('a refused draft must not run');
    expect(refused.status).toBe('rejected');
    expect(refused.reason).toContain('Quadruped has no limb "tail"');
    expect(s.state.draft).toBeNull();

    const draft = snap.action('draftSkill', crouch);
    if (!('run' in draft)) throw new Error(draft.reason);
    expect(s.state.draft).toMatchObject({ name: 'crouch', author: 'ai', endHoldS: 1.5, notes: [] });
    const tried = await stepUntil(s, draft.run(new AbortController().signal));
    expect(tried.status).toBe('succeeded');
    expect(tried.reason).toMatch(/^Tried the draft "crouch" once in the live simulation \(one run, not evaluated, not saved\): 3 of 3 goals met\. goal 1 met: body below 0\.21 m for /);
    expect(tried.measurements['goalsMet']).toEqual({ value: 3, unit: 'count' });
    expect(tried.measurements['groundTruthMinBaseHeight']!.value).toBeLessThan(0.21);
    expect(s.state.library).toHaveLength(0);

    const evaluation = snap.action('evaluateSkillDraft', {});
    if (!('run' in evaluation)) throw new Error(evaluation.reason);
    const evaluated = await evaluation.run(new AbortController().signal);
    expect(evaluated.status).toBe('succeeded');
    expect(evaluated.reason).toMatch(
      /^Evaluated "crouch": (19|20) of 20 randomized trials met every goal \(95 % interval \d+ % to \d+ %; friction, body mass and joint offsets varied\)\. Saved to the library as "crouch \(keyframes\)": executeSkill runs it under that name, and the command "crouch" runs it\. Trials per goal met: Body height reached \d+\/20, Stays upright \d+\/20, Does not fall \d+\/20, only the feet touched the ground \d+\/20, never tipped over \d+\/20\.$/,
    );
    const a = s.state.library[0]!;
    expect(a.taught).toMatchObject({ name: 'crouch', author: 'ai' });
    expect(skillLabel(a)).toBe('Keyframe motion the AI composed (open loop)');

    // A fresh view offers the saved skill, so the same command can run it; the command bar runs it by name.
    const after = await bridge.snapshot();
    if ('unsupported' in after) throw new Error(after.unsupported);
    expect(after.caps.skills).toEqual(['crouch (keyframes)']);
    expect(after.summary).toContain('"crouch (keyframes)" (Keyframe motion the AI composed (open loop), evaluation');
    const out = await stepUntil(s, s.runCommand('crouch'));
    expect(out.kind).toBe('done');
    expect(out.note).toMatch(/^Using "crouch \(keyframes\)" \(Keyframe motion the AI composed \(open loop\), (19|20)\/20 in evaluation\)\.$/);
  });

  it('reports an opposing body-angle target as a sketch error before suggesting optimization', async () => {
    const session = await ready();
    const { snap } = await aiView(session);
    try {
      const draft = snap.action('draftSkill', {
        name: 'bow', keyframes: [{ seconds: 0.5, body: { pitch: -0.3 } }, { seconds: 0.5 }], finalHoldSeconds: 0.5,
        goals: [{ kind: 'bodyAngle', axis: 'pitch', minRad: 0.35, maxRad: 0.6, holdS: 1.5 }],
      });
      if (!('run' in draft)) throw new Error(draft.reason);
      const result = await stepUntil(session, draft.run(new AbortController().signal));
      expect(result.status).toBe('failed');
      expect(result.reason).toContain('body.pitch sign conflict');
      expect(result.fallback).toContain('Revise the sketch with draftSkill before optimizing or training it');
      expect(result.fallback).toContain('positive pitch lowers the front (nose down)');
      expect(session.state.library).toHaveLength(0);
    } finally {
      session.dispose();
    }
  });

  it('suggests learned feedback after measured balance failures even without a balance keyword', async () => {
    const session = await ready();
    const { snap } = await aiView(session);
    const draft = snap.action('draftSkill', crouch);
    if (!('run' in draft)) throw new Error(draft.reason);
    const trial = await stepUntil(session, session.tryDraft(new AbortController().signal));
    if (!trial.ok) throw new Error(trial.reason);
    const run = vi.spyOn(session, 'tryDraft');
    try {
      for (const kind of ['upright', 'noFall', 'standsAtEnd'] as const) {
        run.mockResolvedValue({ ok: true, report: {
          ...trial.report, outcome: 'failed',
          goals: [{ goal: kind === 'upright' ? { kind, maxTiltRad: 0.35 } : kind === 'noFall' ? { kind, minBaseHeightM: 0.15 } : { kind, maxTiltRad: 0.35, minBaseHeightM: 0.15 }, met: false, measured: 0, unit: 'test', text: 'balance failed', score: 0 }],
        } });
        const result = await draft.run(new AbortController().signal);
        expect(result.status).toBe('failed');
        expect(result.fallback).toContain('call trainGaitPolicy');
      }
      run.mockResolvedValue({ ok: true, report: {
        ...trial.report, outcome: 'failed',
        checks: [{ kind: 'feetOnly', met: false, text: 'a knee touched the ground' }],
      } });
      expect((await draft.run(new AbortController().signal)).fallback).toContain('call trainGaitPolicy');
      run.mockResolvedValue(trial);
      expect((await draft.run(new AbortController().signal)).fallback).toContain('Call evaluateSkillDraft now');
    } finally {
      run.mockRestore();
      session.dispose();
    }
  });

  it("never replaces the user's unsaved draft; an edit makes the AI's draft the user's too", async () => {
    const s = await ready();
    const { snap } = await aiView(s);
    s.newDraft();
    s.setDraftName('mine');
    const kept = snap.action('draftSkill', crouch);
    if ('run' in kept) throw new Error('the user draft must not be replaced');
    expect(kept.reason).toContain('The Teach panel holds an unsaved draft "mine" with 0 keyframes; the AI does not replace it.');
    expect(s.state.draft).toMatchObject({ name: 'mine', author: 'user' });
    s.discardDraft();
    expect('run' in snap.action('draftSkill', crouch)).toBe(true);
    expect(s.state.draft!.author).toBe('ai');
    expect(s.setEndHold(2)).toBe(true);
    expect(s.state.draft!.author).toBe('ai-edited');
    expect('run' in snap.action('draftSkill', crouch)).toBe(false);
  });

  it('saves nothing when no trial meets every goal, and says so', async () => {
    const s = await ready();
    const { snap } = await aiView(s);
    const none = snap.action('evaluateSkillDraft', {});
    if (!('run' in none)) throw new Error(none.reason);
    expect(await none.run(new AbortController().signal)).toMatchObject({ status: 'failed', reason: 'There is no skill draft for this robot in the Teach panel; compose one first.' });
    // A crouch cannot keep every foot off the ground for a second.
    const draft = snap.action('draftSkill', { ...crouch, name: 'hover', goals: [{ kind: 'airborne', minS: 1 }] });
    if (!('run' in draft)) throw new Error(draft.reason);
    const tried = await stepUntil(s, draft.run(new AbortController().signal));
    expect(tried.status).toBe('failed');
    expect(tried.reason).toContain('0 of 1 goals met');
    const evaluation = snap.action('evaluateSkillDraft', {});
    if (!('run' in evaluation)) throw new Error(evaluation.reason);
    const r = await evaluation.run(new AbortController().signal);
    expect(r.status).toBe('failed');
    expect(r.reason).toBe('Evaluated "hover": none of 20 randomized trials met every goal, so it was not saved. Trials per goal met: Feet leave the ground 0/20, only the feet touched the ground 20/20, never tipped over 20/20.');
    expect(s.state.library).toHaveLength(0);
    expect(s.state.draft!.name).toBe('hover');
  });

  it('refuses a draft whose goals do not measure what was asked, and reports the check with the result', async () => {
    const s = await ready();
    const { snap } = await aiView(s);
    // A motion request answered without acting gets a reminder; questions and other text do not.
    expect(snap.followUp?.([], 'look up at the sky')).toMatch(/^The user asked the robot for a motion\. Act now/);
    expect(snap.followUp?.([], 'what does sit mean?')).toBeNull();
    expect(snap.followUp?.([], 'hello there')).toBeNull();
    const level = snap.action('draftSkill', { ...crouch, name: 'sit' }, { request: 'sit down like a dog' });
    if ('run' in level) throw new Error('a sit without a nose-up pitch goal must be refused');
    expect(level.reason).toContain('The goals do not measure what was asked: "sit" means a sit lowers the rear below the front, so the body pitches nose up; add a bodyAngle goal on pitch');
    expect(s.state.draft).toBeNull();
    const pitched = snap.action('draftSkill', { ...crouch, name: 'sit', goals: [...crouch.goals, { kind: 'bodyAngle', axis: 'pitch', minRad: -1, maxRad: -0.1, holdS: 0.5 }] }, { request: 'sit down like a dog' });
    if (!('run' in pitched)) throw new Error(pitched.reason);
    expect(s.state.draft).toMatchObject({ name: 'sit', request: 'sit down like a dog' });
    const tried = await stepUntil(s, pitched.run(new AbortController().signal));
    expect(tried.reason).toContain('Request check: "sit" (a sit lowers the rear below the front, so the body pitches nose up) is measured by goal 4.');
    // These poses lower the body evenly, so the pitch goal is measured and not met.
    expect(tried.status).toBe('failed');
    expect(tried.reason).toContain('goal 4 not met: body pitch between -57 deg and -6 deg');
    // Checkpoints tell the AI where the body was at each keyframe and at the end.
    expect(tried.reason).toMatch(/Checkpoints \(simulator ground truth; x forward, y left, z up from the start\): keyframe 1 at 1 s: body -?\d+(\.\d+)? m forward, -?\d+(\.\d+)? m left, 0\.\d+ m high, pitch -?\d+(\.\d)?, roll -?\d+(\.\d)?, turned -?\d+(\.\d)? deg \| end of the hold at 2\.5 s: .* \| end of the run at /);
  });

  it('tells the AI where the center of mass is over the feet still on the ground', async () => {
    const s = await ready();
    const { snap } = await aiView(s);
    // Seen in the browser: the model shifted the body toward the foot it lifted. Here it shifts away, back and left, so it stays up.
    const lift = { seconds: 1, body: { x: -0.04, y: 0.04 }, tips: [{ limb: 'front_right', z: 0.06 }] };
    const draft = snap.action('draftSkill', { ...crouch, name: 'lean', keyframes: [lift], goals: [{ kind: 'tipHeight', limbId: 'front_right', minHeightM: 0.03, holdS: 0.5 }] });
    if (!('run' in draft)) throw new Error(draft.reason);
    const tried = await stepUntil(s, draft.run(new AbortController().signal));
    expect(tried.status).toBe('succeeded');
    expect(tried.reason).toMatch(/keyframe 1 at 1 s: body [^|]*; center of mass 0\.0\d+ m inside the feet on the ground \(front_left, rear_left, rear_right\) \|/);
  });

  it('composes a gait from a contact pattern and tries it in the live simulation', async () => {
    const s = await ready();
    const { bridge, snap } = await aiView(s);
    expect(snap.summary).toContain('Walking, turning, side steps and stepping in place use a gait');
    const walk = {
      name: 'walk',
      gait: { ...DEFAULT_GAIT, strideM: 0.06, durationS: 3 },
      finalHoldSeconds: 1,
      goals: [{ kind: 'travel', direction: 'forward', minDistanceM: 0.05 }, { kind: 'upright', maxTiltRad: 0.6 }],
    };
    const aimless = snap.action('draftSkill', { ...walk, goals: [walk.goals[1]] }, { request: 'walk forward' });
    if ('run' in aimless) throw new Error('a walk without a travel goal must be refused');
    expect(aimless.reason).toContain('"walk" means walking moves the body over the ground');
    const draft = snap.action('draftSkill', walk, { request: 'walk forward' });
    if (!('run' in draft)) throw new Error(draft.reason);
    expect(s.state.draft).toMatchObject({ name: 'walk', author: 'ai', request: 'walk forward', keyframes: [] });
    expect(s.state.draft!.gait).toMatchObject({ pattern: 'trot', strideM: 0.06 });
    const tried = await stepUntil(s, draft.run(new AbortController().signal));
    expect(tried.reason).toContain('Request check: "walk" (walking moves the body over the ground) is measured by goal 1.');
    expect(tried.measurements['goal 1 (Body moves)']!.unit).toBe('m');
    const view = await bridge.snapshot();
    if ('unsupported' in view) throw new Error(view.unsupported);
    expect(view.summary).toContain('The Teach panel holds a draft "walk" (a gait: trot pattern, 0.5 s cycle');
  });

  it('improves a draft by trial and error, and changes it only when held-out trials score better', async () => {
    const s = await ready();
    const { snap } = await aiView(s);
    const none = snap.action('improveSkillDraft', {});
    if (!('run' in none)) throw new Error(none.reason);
    expect(await none.run(new AbortController().signal)).toMatchObject({ status: 'failed', reason: 'There is no skill draft for this robot in the Teach panel; compose one first.' });
    // A crouch too shallow for its height goal.
    const draft = snap.action('draftSkill', { ...crouch, keyframes: [{ seconds: 1, body: { z: -0.03 } }] });
    if (!('run' in draft)) throw new Error(draft.reason);
    const before = s.state.draft!;
    const improving = snap.action('improveSkillDraft', {});
    if (!('run' in improving)) throw new Error(improving.reason);
    const r = await improving.run(new AbortController().signal);
    expect(r.measurements['simulatedRuns']).toEqual({ value: 96, unit: 'count' });
    expect(r.measurements['heldOutTrials']).toEqual({ value: 8, unit: 'count' });
    if (r.status === 'succeeded') {
      expect(r.measurements['resultMeanReward']!.value).toBeGreaterThanOrEqual(r.measurements['baselineMeanReward']!.value);
      expect(s.state.draft!.training).toMatchObject({ heldOutTrials: 8, rollouts: 96 });
      expect(s.state.draft!.keyframes).not.toEqual(before.keyframes);
      expect(s.state.draft!.author).toBe('ai');
    } else {
      expect(r.status).toBe('failed');
      expect(r.reason).toContain('The draft is unchanged.');
      expect(s.state.draft).toBe(before);
    }
  });

  it('trains a learned feedback policy for a gait draft, keeps it only when held-out trials score better, and saves it as a learned gait', async () => {
    const s = await ready();
    const { snap } = await aiView(s);
    const none = snap.action('trainGaitPolicy', {});
    if (!('run' in none)) throw new Error(none.reason);
    expect(await none.run(new AbortController().signal)).toMatchObject({ status: 'failed', reason: 'There is no skill draft for this robot in the Teach panel; compose one first.' });
    const walk = {
      name: 'walk',
      gait: { ...DEFAULT_GAIT, strideM: 0.06, durationS: 1.5 },
      finalHoldSeconds: 1,
      goals: [{ kind: 'travel', direction: 'forward', minDistanceM: 0.02 }, { kind: 'upright', maxTiltRad: 0.6 }],
    };
    const draft = snap.action('draftSkill', walk, { request: 'walk forward with a learned feedback policy' });
    if (!('run' in draft)) throw new Error(draft.reason);
    const before = s.state.draft!;
    expect(before.policy).toBeNull();
    const training = snap.action('trainGaitPolicy', {});
    if (!('run' in training)) throw new Error(training.reason);
    const r = await training.run(new AbortController().signal);
    // One run of the gait alone for the input statistics, 30 iterations of 8 direction pairs, then 8 held-out trials each without and with the policy.
    expect(r.measurements['simulatedRuns']).toEqual({ value: 1 + 30 * 16 + 16, unit: 'count' });
    expect(r.measurements['heldOutTrials']).toEqual({ value: 8, unit: 'count' });
    expect(r.reason).toMatch(/on 8 held-out randomized trials, gait alone versus with the policy: training return -?\d+\.\d{3} and -?\d+\.\d{3}, goal reward -?\d+\.\d{3} and -?\d+\.\d{3}/);
    expect(r.reason).toMatch(/The reward asked for 0\.\d{3} m\/s forward, (-?0\.000|0\.000) m\/s left and -?0\.000 rad\/s turning while stepping/);
    if (r.status === 'succeeded') {
      expect(r.measurements['learnedReturn']!.value).toBeGreaterThanOrEqual(r.measurements['gaitOnlyReturn']!.value);
      expect(s.state.draft!.policy).toMatchObject({ heldOutTrials: 8, parameters: 156, iterations: 30 });
      expect(s.state.draft!.gait).toEqual(before.gait);
      const evaluation = snap.action('evaluateSkillDraft', {});
      if (!('run' in evaluation)) throw new Error(evaluation.reason);
      const e = await evaluation.run(new AbortController().signal);
      if (e.status === 'succeeded') {
        const saved = s.state.library[0]!;
        expect(saved.method).toBe('gait-learned');
        expect(saved.name).toBe('walk (learned gait)');
        expect(saved.policy!.layers).toEqual([12, 12]);
        expect(saved.training!.algorithm).toBe('augmented random search on speed tracking, stability and the goal reward (learned feedback policy)');
        expect(skillLabel(saved)).toBe('Gait the AI composed, with a feedback policy learned by random search (closed loop)');
        // The request asked for learning, and the policy was trained: nothing more to train.
        expect(e.fallback ?? '').not.toContain('trainGaitPolicy');
      }
      // Any edit of the gait drops the policy trained for it.
      expect(s.setGaitParam('stepHeightM', 0.05)).toBe(true);
      expect(s.state.draft!.policy).toBeNull();
    } else {
      expect(r.status).toBe('failed');
      expect(r.reason).toContain('so the draft is unchanged');
      expect(s.state.draft).toBe(before);
      // Saved without the policy the request asked for: the result says to train it.
      const evaluation = snap.action('evaluateSkillDraft', {});
      if (!('run' in evaluation)) throw new Error(evaluation.reason);
      const e = await evaluation.run(new AbortController().signal);
      if (e.status === 'succeeded') expect(e.fallback).toContain('call trainGaitPolicy on the draft');
    }
  });

  it.each(['linear', 'mlp'] as const)('trains %s feedback for a keyframe draft under pushes and saves only held-out improvements', async (architecture) => {
    const physics = new PhysicsSession(() => Promise.resolve(mj));
    const inProcess = () => new SkillWorkerClient(() => new InProcessWorker() as unknown as Worker);
    const s = new SkillsSession(physics, inProcess(), undefined, undefined, {
      rolloutPool: () => new RolloutPool(inProcess, 2),
      posePolicyConfig: { iterations: 2, directions: 2, top: 1, trialsPerCandidate: 1, heldOutTrials: 2 },
    });
    await physics.init();
    for (let i = 0; i < 2000 && s.state.hash === null; i++) await new Promise((r) => setTimeout(r, 1));
    const { snap } = await aiView(s);
    expect(snap.summary).toContain('Keyframes play open loop: they cannot hold an unstable pose');
    // Balancing on two diagonal legs: the request asks for balance, so the next step after the try is the learned policy.
    const lift = [{ limb: 'front_left', z: 0.06 }, { limb: 'rear_right', z: 0.06 }];
    const diagonal = {
      name: 'two legs',
      keyframes: [{ seconds: 0.5, tips: lift }, { seconds: 0.6, tips: lift }, { seconds: 0.5 }],
      finalHoldSeconds: 0.3,
      goals: [
        { kind: 'airborne', limbIds: ['front_left', 'rear_right'], minS: 0.3 },
        { kind: 'noFall', minBaseHeightM: 0.15 },
      ],
    };
    // Goals that do not lift the other feet together are refused for a stand on two legs.
    const flat = snap.action('draftSkill', { ...diagonal, goals: [{ kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.03, holdS: 0.3 }, { kind: 'noFall', minBaseHeightM: 0.15 }] }, { request: 'balance on two diagonal legs' });
    if ('run' in flat) throw new Error('goals that do not measure the request must be refused');
    expect(flat.reason).toContain('"two legs" means standing on two legs lifts the other feet off the ground together');
    const draft = snap.action('draftSkill', diagonal, { request: 'balance on two diagonal legs' });
    if (!('run' in draft)) throw new Error(draft.reason);
    const tried = await stepUntil(s, draft.run(new AbortController().signal));
    expect(tried.fallback).toMatch(/[Cc]all trainGaitPolicy/);
    const before = s.state.draft!;
    const training = snap.action('trainGaitPolicy', { architecture });
    if (!('run' in training)) throw new Error(training.reason);
    const r = await training.run(new AbortController().signal);
    // Four runs of the keyframes alone for the input statistics, 2 iterations of 2 direction pairs, then 2 held-out pushed runs each without and with the policy.
    expect(r.measurements['simulatedRuns']).toEqual({ value: 4 + 2 * 4 + 4 + (architecture === 'mlp' ? 2 : 0), unit: 'count' });
    expect(r.measurements['heldOutTrials']).toEqual({ value: 2, unit: 'count' });
    expect(r.measurements['pushImpulseMin']).toEqual({ value: 0.2, unit: 'N*s' });
    expect(r.measurements['pushImpulseMax']).toEqual({ value: 0.8, unit: 'N*s' });
    expect(r.reason).toMatch(/on 2 held-out randomized runs, each pushed once with 0\.2 to 0\.8 N\*s at a random moment and direction, keyframes alone \(open loop\) versus with the policy \(closed loop\): every goal met in \d and \d, falls \d and \d, goal reward -?\d\.\d{3} and -?\d\.\d{3}/);
    if (r.status === 'succeeded') {
      expect(r.reason).toMatch(/^A feedback policy learned by reinforcement learning under pushes now corrects the joints of "two legs" every 10 ms from the IMU, the foot contacts and the joint encoders/);
      expect(s.state.draft!.policy).toMatchObject({ kind: 'pose', heldOutTrials: 2, parameters: architecture === 'mlp' ? 588 : 288, iterations: 2, pushImpulseNs: [0.2, 0.8] });
      expect(s.state.draft!.keyframes).toEqual(before.keyframes);
      const evaluation = snap.action('evaluateSkillDraft', {});
      if (!('run' in evaluation)) throw new Error(evaluation.reason);
      const e = await evaluation.run(new AbortController().signal);
      if (e.status === 'succeeded') {
        const saved = s.state.library[0]!;
        expect(saved.method).toBe('keyframes-learned');
        expect(saved.name).toBe('two legs (learned keyframes)');
        expect(saved.policy!.layers).toEqual(architecture === 'mlp' ? [23, 16, 12] : [23, 12]);
        expect(saved.training!.algorithm).toBe('augmented random search on the goals in randomized, pushed runs (learned feedback policy for keyframes)');
        expect(saved.training!.config).toMatchObject({ pushImpulseMinNs: 0.2, pushImpulseMaxNs: 0.8 });
        expect(skillLabel(saved)).toBe('Keyframes the AI composed, with a feedback policy learned by reinforcement learning under pushes (closed loop)');
      }
      // Any edit of the keyframes drops the policy trained for them.
      expect(s.setEndHold(0.5)).toBe(true);
      expect(s.state.draft!.policy).toBeNull();
    } else {
      expect(r.status).toBe('failed');
      expect(r.reason).toMatch(/^The learned feedback policy did not beat the keyframes of "two legs" alone, so the draft is unchanged/);
      expect(s.state.draft).toBe(before);
    }
  });

  it('trains a walking controller on a pool of workers, then walks by command and reports where the body went', async () => {
    const physics = new PhysicsSession(() => Promise.resolve(mj));
    const inProcess = () => new SkillWorkerClient(() => new InProcessWorker() as unknown as Worker);
    const s = new SkillsSession(physics, inProcess(), undefined, undefined, {
      rolloutPool: () => new RolloutPool(inProcess, 2),
      locomotionConfig: { iterations: 2, directions: 2, top: 1, commandsPerCandidate: 1, heldOutTrials: 2, evaluationTrials: 6, episodeS: 2 },
    });
    await physics.init();
    for (let i = 0; i < 2000 && s.state.hash === null; i++) await new Promise((r) => setTimeout(r, 1));
    const { bridge, snap } = await aiView(s);
    expect(snap.caps.walking).toBeNull();
    expect(snap.summary).toContain('This body has no learned walking controller yet.');
    const early = snap.action('locomote', { forwardMps: 0.1, leftMps: 0, turnRadps: 0, durationS: 2 });
    if ('run' in early) throw new Error('walking needs a trained controller');
    expect(early.reason).toBe('No walking controller is trained for Quadruped yet.');
    const training = snap.action('trainLocomotion', {});
    if (!('run' in training)) throw new Error(training.reason);
    const t = await training.run(new AbortController().signal);
    expect(t.status).toBe('succeeded');
    expect(t.reason).toMatch(/^Trained and saved this body's walking controller \(\d+ simulated runs on 2 workers/);
    expect(t.measurements['evaluationTrials']).toEqual({ value: 6, unit: 'count' });
    expect(s.state.walking!.robot.morphologyHash).toBe(s.state.hash);
    const view = await bridge.snapshot();
    if ('unsupported' in view) throw new Error(view.unsupported);
    expect(view.caps.walking).toEqual(s.state.walking!.ranges);
    expect(view.summary).toContain('This body has a learned walking controller (D28; forward -0.12 to 0.15 m/s');
    expect(view.summary).toContain('Walking, turning and side steps by a distance or an angle are done with locomote');
    expect(snap.summary).not.toContain('are done with locomote');
    const far = view.action('locomote', { forwardMps: 0.4, leftMps: 0, turnRadps: 0, durationS: 2 });
    if ('run' in far) throw new Error('a command beyond the trained range must be refused');
    expect(far.reason).toBe('The walk was refused: forwardMps must be from -0.12 to 0.15 (the range this controller was trained for).');
    const walk = view.action('locomote', { forwardMps: 0.12, leftMps: 0, turnRadps: 0, durationS: 2 });
    if (!('run' in walk)) throw new Error(walk.reason);
    const w = await stepUntil(s, walk.run(new AbortController().signal));
    expect(['succeeded', 'failed']).toContain(w.status);
    // 2 s of stepping with 0.5 s fades move as far as 1.5 s at full speed.
    expect(w.measurements['expectedForward']!.value).toBeCloseTo(0.18, 3);
    expect(w.measurements['measuredForward']!.value).toBeGreaterThan(0.05);
    expect(w.reason).toMatch(/^Walked 0\.12 m\/s forward, 0 m\/s left and 0 rad\/s turning for 2 s with the learned walking controller/);
    expect(s.state.lastWalk!.judgement.expected[0]).toBeCloseTo(0.18, 3);
    // The result names where the walk ended, never "succeeded" next to an error outside the tolerance.
    expect(w.reason).toMatch(new RegExp(`\\): ${walkOutcomeText(s.state.lastWalk!)}\\. It started (where the body stood|from the home stance \\(the simulation was reset first\\))\\. Where the command leads`));
    const judged = (success: boolean, upright: boolean) => ({ ...s.state.lastWalk!.judgement, success, upright });
    expect(walkOutcomeText({ outcome: 'succeeded', judgement: judged(true, true) })).toBe('ended within tolerance');
    expect(walkOutcomeText({ outcome: 'succeeded', judgement: judged(false, true) })).toBe('stayed upright but ended outside tolerance');
    expect(walkOutcomeText({ outcome: 'failed', judgement: judged(false, false) })).toBe('did not stay upright');
    expect(walkOutcomeText({ outcome: 'interrupted', judgement: judged(false, false) })).toBe('interrupted');
    const firstUp = s.state.lastWalk!.judgement.upright;
    // A target instead of speeds: the shortest walk within the allowed share of the ranges, 10 cm at most 0.12 m/s.
    const tooFar = view.action('locomote', { forwardM: 2 });
    if ('run' in tooFar) throw new Error('a target beyond 10 s of walking must be refused');
    expect(tooFar.reason).toBe('The walk was refused: reaching it takes longer than 10 s at the speeds this controller was trained for; walk part of the way and call locomote again from there.');
    const toTarget = view.action('locomote', { forwardM: 0.1 });
    if (!('run' in toTarget)) throw new Error(toTarget.reason);
    const tw = await stepUntil(s, toTarget.run(new AbortController().signal));
    expect(tw.reason).toMatch(/^Target 0\.100 m forward, 0\.000 m left, turned 0 deg: Walked 0\.1111 m\/s forward, 0 m\/s left and 0 rad\/s turning for 1\.4 s /);
    expect(tw.measurements['expectedForward']!.value).toBeCloseTo(0.1, 3);
    // Walks chain into a path: the second started where the first left the body.
    if (firstUp) {
      expect(s.state.lastWalk!.continued).toBe(true);
      expect(tw.reason).toContain('It started where the body stood.');
    }
    s.deleteLocomotion();
    expect(s.state.walking).toBeNull();
  });

  it('runs saved skills one after another, each judged on its own', async () => {
    const s = await ready();
    const { bridge, snap } = await aiView(s);
    const draft = snap.action('draftSkill', crouch);
    if (!('run' in draft)) throw new Error(draft.reason);
    await stepUntil(s, draft.run(new AbortController().signal));
    const evaluation = snap.action('evaluateSkillDraft', {});
    if (!('run' in evaluation)) throw new Error(evaluation.reason);
    expect((await evaluation.run(new AbortController().signal)).status).toBe('succeeded');
    const view = await bridge.snapshot();
    if ('unsupported' in view) throw new Error(view.unsupported);
    expect(view.caps.features.skillSequences).toBe(true);
    const seq = view.action('runSkillSequence', { steps: [{ skill: 'crouch (keyframes)', times: 2 }] });
    if (!('run' in seq)) throw new Error(seq.reason);
    const r = await stepUntil(s, seq.run(new AbortController().signal));
    expect(r.status).toBe('succeeded');
    expect(r.measurements['runsPlanned']).toEqual({ value: 2, unit: 'count' });
    expect(r.measurements['runsSucceeded']).toEqual({ value: 2, unit: 'count' });
    expect(r.measurements['predictedSuccess']!.value).toBeGreaterThan(0.8);
    expect(r.reason).toMatch(/^Ran 2 of 2 planned runs: 1\. "crouch \(keyframes\)" .*; 2\. "crouch \(keyframes\)" .*Each skill started from the home stance\./);
    const unknown = view.action('runSkillSequence', { steps: [{ skill: 'fly', times: 1 }] });
    if (!('run' in unknown)) throw new Error(unknown.reason);
    expect(await unknown.run(new AbortController().signal)).toMatchObject({ status: 'failed', reason: 'No skill named "fly" is in the library.' });

    // A level crouch saved under the name "sit" is not a sit: the list says so, and the AI may not run it as one.
    const a = s.state.library[0]!;
    const sit = { ...a, id: 'sit-level', name: 'sit (keyframes)', taught: { ...a.taught!, name: 'sit' } };
    s.store.update({ library: [sit, ...s.state.library] });
    const listed = await bridge.snapshot();
    if ('unsupported' in listed) throw new Error(listed.unsupported);
    expect(listed.summary).toContain('its goals do not measure "sit" (a sit lowers the rear below the front, so the body pitches nose up), so it is not that motion');
    const refused = listed.action('executeSkill', { skill: 'sit (keyframes)' }, { request: 'sit down like a dog' });
    if ('run' in refused) throw new Error('a level crouch must not run as a sit');
    expect(refused).toMatchObject({ status: 'rejected', reason: '"sit (keyframes)" does not do what was asked: "sit" means a sit lowers the rear below the front, so the body pitches nose up, and none of its goals measures that.' });
    const inSequence = listed.action('runSkillSequence', { steps: [{ skill: 'crouch (keyframes)', times: 1 }, { skill: 'sit (keyframes)', times: 1 }] });
    if ('run' in inSequence) throw new Error('a level crouch must not run as a sit');
    expect(inSequence.status).toBe('rejected');
  });

  it('mirrors a saved skill for the other side; the mirror image is tried like any draft and mirrors back', async () => {
    const s = await ready();
    const { bridge, snap } = await aiView(s);
    const lean = {
      name: 'lean left',
      keyframes: [{ seconds: 1, body: { y: 0.03, roll: -0.2 } }],
      finalHoldSeconds: 1,
      goals: [{ kind: 'bodyAngle', axis: 'roll', minRad: -0.5, maxRad: -0.1, holdS: 0.5 }, { kind: 'upright', maxTiltRad: 0.5 }],
    };
    const draft = snap.action('draftSkill', lean, { request: 'lean left' });
    if (!('run' in draft)) throw new Error(draft.reason);
    expect((await stepUntil(s, draft.run(new AbortController().signal))).status).toBe('succeeded');
    const evaluation = snap.action('evaluateSkillDraft', {});
    if (!('run' in evaluation)) throw new Error(evaluation.reason);
    expect((await evaluation.run(new AbortController().signal)).status).toBe('succeeded');
    const saved = s.state.library[0]!;
    const view = await bridge.snapshot();
    if ('unsupported' in view) throw new Error(view.unsupported);
    expect(view.summary).toContain('call mirrorSkill with that skill instead of composing it again');
    const mirrored = view.action('mirrorSkill', { skill: 'lean left (keyframes)' }, { request: 'now lean right' });
    if (!('run' in mirrored)) throw new Error(mirrored.reason);
    expect(s.state.draft).toMatchObject({ name: 'lean right', author: 'ai', request: 'now lean right', notes: ['Left-right mirror image of "lean left (keyframes)".'] });
    expect(s.state.draft!.goals).toEqual([{ kind: 'bodyAngle', axis: 'roll', minRad: 0.1, maxRad: 0.5, holdS: 0.5 }, { kind: 'upright', maxTiltRad: 0.5 }]);
    const tried = await stepUntil(s, mirrored.run(new AbortController().signal));
    expect(tried.reason).toMatch(/^Tried the draft "lean right" once in the live simulation/);
    expect(tried.reason).toContain('Request check: "lean right" (a lean lowers one side, so the body rolls) is measured by goal 1.');
    expect(tried.status).toBe('succeeded');
    // The Teach panel mirrors it back to the original poses; the edit makes the draft the AI's and the user's.
    expect(s.mirrorDraft()).toBe(true);
    expect(s.state.draft).toMatchObject({ name: 'lean left', author: 'ai-edited' });
    s.state.draft!.keyframes.forEach((k, i) => k.q.forEach((v, j) => expect(v).toBeCloseTo(saved.motion!.keyframes[i]!.q[j]!, 9)));
    expect(s.state.draft!.goals).toEqual(saved.taught!.goals);
    // The draft is now the user's too, so the AI may not replace it with another mirror image.
    const unnamed = view.action('mirrorSkill', { skill: 'lean left (keyframes)', name: 'tilt' }, { request: 'tilt' });
    if ('run' in unnamed) throw new Error('the user-edited draft must not be replaced');
    expect(unnamed.reason).toContain('the AI does not replace it');
  });

  it('switches the robot it works on, and refuses the Build tab robot before one exists', async () => {
    const s = await ready();
    const { snap } = await aiView(s);
    expect(snap.caps.features.robotChoice).toBe(true);
    const none = snap.action('selectRobot', { robot: 'built' });
    if ('run' in none) throw new Error('selectRobot answers at once');
    expect(none).toMatchObject({ status: 'rejected', reason: 'The Build tab has no robot yet.' });
    const same = snap.action('selectRobot', { robot: s.state.source });
    if ('run' in same) throw new Error('selectRobot answers at once');
    expect(same).toMatchObject({ status: 'succeeded', reason: `The Skills tab already shows ${s.state.robot.name}.` });
    const hex = snap.action('selectRobot', { robot: 'hexapod' });
    if ('run' in hex) throw new Error('selectRobot answers at once');
    expect(hex.status).toBe('succeeded');
    expect(hex.reason).toMatch(/^The Skills tab now shows Hexapod \(6 limbs, 18 joints, floating base\)/);
    expect(s.state.robot.name).toBe('Hexapod');
  });
});
