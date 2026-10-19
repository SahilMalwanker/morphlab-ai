import { describe, expect, it } from 'vitest';
import { editLimb, patchJoint } from '../../src/core/model/builder';
import { morphologyHash } from '../../src/core/model/hash';
import { quadruped } from '../../src/core/model/presets';
import { CommandInterpreter } from '../../src/language/interpreter';
import { parseCommand } from '../../src/language/parser';
import { resolveCommand } from '../../src/language/resolver';
import { createSkillArtifact, type SkillArtifact } from '../../src/skills/artifact';
import type { SkillSpec } from '../../src/skills/controllers';
import { wilson95, type EvaluationSummary } from '../../src/skills/evaluation';
import { DEFAULT_FOOT_LIFT } from '../../src/skills/footLift';

const robot = quadruped();
const ctx = { robotName: robot.name, limbs: robot.limbs.map((l) => ({ id: l.id, name: l.name, role: l.role })) };

function evaluation(successes: number): EvaluationSummary {
  return { trials: 20, successes, successRate: successes / 20, wilson95: wilson95(successes, 20), conditions: { friction: [0.6, 1.2], baseMassKg: [3.5, 4.5], jointNoiseRad: [-0.02, 0.02] }, results: [], simulator: 'MuJoCo', evaluatedAt: '2026-10-03T12:00:00.000Z' };
}

async function skill(spec: SkillSpec, successes: number, opts: { robot?: typeof robot; limb?: string; id: string; now?: string }): Promise<SkillArtifact> {
  const r = opts.robot ?? robot;
  return createSkillArtifact({ robot: r, morphologyHash: await morphologyHash(r), limbId: opts.limb ?? 'front_left', spec, name: opts.id, evaluation: evaluation(successes), id: opts.id, now: opts.now ?? '2026-10-03T12:00:00.000Z' });
}

const sm: SkillSpec = { kind: 'state-machine', params: DEFAULT_FOOT_LIFT };
const replay: SkillSpec = { kind: 'replay', trajectory: { dt: 0.01, jointIds: robot.limbs.flatMap((l) => l.joints.map((j) => j.id)), frames: [new Array(12).fill(0)] } };

describe('command resolver', () => {
  it('picks the best evaluated skill for this morphology and leg, with the requested hold', async () => {
    const hash = await morphologyHash(robot);
    const lib = [await skill(sm, 15, { id: 'weaker' }), await skill({ kind: 'tuned-parameters', params: DEFAULT_FOOT_LIFT }, 20, { id: 'best' }), await skill(sm, 20, { id: 'other_leg', limb: 'rear_left' })];
    const r = resolveCommand(parseCommand('lift the front left leg and hold for three seconds', ctx), { robot, hash, library: lib });
    expect(r.kind).toBe('run-skill');
    if (r.kind !== 'run-skill') return;
    expect(r.skill.id).toBe('best');
    expect(r.holdS).toBe(3);
    expect('params' in r.spec && r.spec.params.holdS).toBe(3);
    expect(r.note).toContain('It was evaluated with a 2 s hold; this 3 s hold was not part of that evaluation.');
  });

  it('refuses when no skill exists for this body, and says why other skills do not count', async () => {
    const longer = editLimb(robot, 'front_left', (l) => ({ ...l, joints: l.joints.map((j, i) => (i === 2 ? patchJoint(j, { xyz: [0, 0, -0.18] }) : j)) }), true);
    const lib = [await skill(sm, 20, { id: 'old_body' })];
    const r = resolveCommand(parseCommand('lift the front left leg', ctx), { robot: longer, hash: await morphologyHash(longer), library: lib });
    expect(r.kind).toBe('refusal');
    if (r.kind !== 'refusal') return;
    expect(r.reason).toContain('There is no evaluated foot-lift skill for the front left leg of this robot');
    expect(r.reason).toContain('1 skill for that leg belong to other morphologies and are never reused on a different body.');
    expect(r.suggestion).toContain('Open the Skills tab');
  });

  it('fixed-timing skills cannot change the hold; out-of-range holds are refused', async () => {
    const hash = await morphologyHash(robot);
    const lib = [await skill(replay, 20, { id: 'replay_only' })];
    const r = resolveCommand(parseCommand('lift the front left leg for 3 seconds', ctx), { robot, hash, library: lib });
    expect(r).toMatchObject({ kind: 'refusal', reason: 'The skills for the front left leg replay a fixed recording or a learned timing and cannot hold for 3 s.' });
    expect(resolveCommand(parseCommand('lift the front left leg', ctx), { robot, hash, library: lib }).kind).toBe('run-skill');
    const long = resolveCommand(parseCommand('lift the front left leg for 20 seconds', ctx), { robot, hash, library: [await skill(sm, 20, { id: 's' })] });
    expect(long).toMatchObject({ kind: 'refusal', reason: 'A 20 s hold is outside the supported range of 0.5 to 5 s.' });
  });

  it('passes refusals, clarifications, stand and reset through', async () => {
    const hash = await morphologyHash(robot);
    const walk = resolveCommand(parseCommand('walk forward', ctx), { robot, hash, library: [] });
    expect(walk.kind).toBe('refusal');
    expect(walk.kind === 'refusal' && walk.suggestion.length).toBeGreaterThan(10);
    expect(resolveCommand(parseCommand('lift the left leg', ctx), { robot, hash, library: [] }).kind).toBe('clarify');
    expect(resolveCommand(parseCommand('stand', ctx), { robot, hash, library: [] })).toEqual({ kind: 'stand' });
    expect(resolveCommand(parseCommand('reset', ctx), { robot, hash, library: [] })).toEqual({ kind: 'reset' });
  });
});

describe('local interpreter and AI escalation', () => {
  it('interprets with the local rules only', async () => {
    const r = await new CommandInterpreter().interpret('please raise the back right leg for two and a half seconds', ctx);
    expect(r).toEqual({ result: { kind: 'skill', skillType: 'foot_lift', limbId: 'rear_right', holdS: 2.5 }, source: 'local rules', note: null });
  });

  it('marks only requests the local rules do not cover for the AI path', () => {
    expect(parseCommand('make me a sandwich', ctx)).toMatchObject({ kind: 'refusal', reason: 'This command is not recognized.', escalate: true });
    expect(parseCommand('drive to the red box', ctx)).toMatchObject({ kind: 'refusal', reason: 'The local rules do not handle navigation commands.', escalate: true });
    // Motions nobody taught go to the AI, which can compose them; without it the refusal says how to teach them.
    for (const text of ['walk forward', 'jump', 'wave your front right leg', 'sit down', 'crouch low']) {
      const r = parseCommand(text, ctx);
      expect(r, text).toMatchObject({ kind: 'refusal', escalate: true });
      expect(r.kind === 'refusal' && r.suggestion, text).toContain('Teach it in the Skills tab');
    }
    const arm = { robotName: 'Three-joint arm', limbs: [{ id: 'arm', name: 'Arm', role: 'arm' as const }] };
    expect(parseCommand('raise your arm', arm)).toMatchObject({ kind: 'refusal', reason: 'Three-joint arm has no legs.', escalate: true });
    // Safety and factual refusals stay local.
    for (const text of ['apply 5 N*m to the knee', 'lift all four legs', 'lift the middle left leg']) {
      const r = parseCommand(text, ctx);
      expect(r.kind, text).toBe('refusal');
      expect(r.kind === 'refusal' && r.escalate, text).toBeFalsy();
    }
  });

  it('treats questions as questions: they never run a skill or a reset, and go to the AI path', () => {
    for (const text of ['how have your live runs of the front left lift gone so far?', 'why did the robot reset?', 'did the lift work', 'is the front left leg lifted', 'tell me how you lift the front left leg', 'lift the front left leg?']) {
      expect(parseCommand(text, ctx), text).toMatchObject({ kind: 'refusal', reason: 'This is a question; the local rules only carry out commands.', escalate: true });
    }
    expect(parseCommand('can you lift the front left leg?', ctx)).toMatchObject({ kind: 'skill', skillType: 'foot_lift', limbId: 'front_left' });
    expect(parseCommand('please lift the front left leg', ctx)).toMatchObject({ kind: 'skill', skillType: 'foot_lift' });
    expect(parseCommand('reset', ctx)).toEqual({ kind: 'skill', skillType: 'reset' });
    // Local refusals still come first: a question about walking is answered by the walking rule.
    expect(parseCommand('how do I make it walk?', ctx)).toMatchObject({ kind: 'refusal', reason: 'Walking is not a skill this robot has learned yet.' });
    // The torque rule refuses commands only; a question about torque goes to the AI (the experiment coach).
    expect(parseCommand('why is the torque saturating?', ctx)).toMatchObject({ kind: 'refusal', reason: 'This is a question; the local rules only carry out commands.', escalate: true });
    expect(parseCommand('apply 5 N*m to the knee', ctx)).toMatchObject({ kind: 'refusal', reason: 'Language commands select evaluated skills; they never set joint torques or targets directly.' });
    expect(parseCommand('apply 3 newton meters to the hip', ctx)).toMatchObject({ kind: 'refusal', reason: 'Language commands select evaluated skills; they never set joint torques or targets directly.' });
    // An impulse in newton seconds is a push, which the AI handles in the Physics tab.
    expect(parseCommand('push the robot to the left with 2 newton seconds', ctx)).toMatchObject({ kind: 'refusal', reason: 'This command is not recognized.', escalate: true });
    expect(parseCommand('push it forward with 1.5 N*s', ctx)).toMatchObject({ kind: 'refusal', reason: 'This command is not recognized.', escalate: true });
    // Standing tall is a posture to compose, not the home stance.
    expect(parseCommand('stand up tall', ctx)).toMatchObject({ kind: 'refusal', reason: 'That posture has not been taught to this robot yet.', escalate: true });
    expect(parseCommand('stand up', ctx)).toMatchObject({ kind: 'skill', skillType: 'stand' });
    for (const text of ['stand', 'please stand', 'stand still', 'hold still', 'balance', 'stand now']) expect(parseCommand(text, ctx), text).toEqual({ kind: 'skill', skillType: 'stand' });
    // Seen in the browser: a request to teach standing on the hind legs ran the home stance. Balancing on fewer feet goes to the AI.
    for (const text of ['teach a skill called "rear up": stand on the hind legs and balance there', 'stand on two legs', 'balance on your back legs', 'lift both front legs', 'do a handstand']) {
      expect(parseCommand(text, ctx), text).toMatchObject({ kind: 'refusal', reason: 'Balancing on fewer feet has not been taught to this robot yet.', escalate: true });
    }
    // A longer request that mentions standing is not the stand command.
    expect(parseCommand('stand for a while and then look around', ctx)).toMatchObject({ kind: 'refusal', reason: 'This command is not recognized.', escalate: true });
    // One hind leg is still a foot lift.
    expect(parseCommand('lift the left hind leg and hold it for half a second', ctx)).toMatchObject({ kind: 'skill', skillType: 'foot_lift', limbId: 'rear_left', holdS: 0.5 });
  });

  it('reads "run" as walking only when it means locomotion', () => {
    for (const text of ['run', 'run forward', 'walk forward']) expect(parseCommand(text, ctx), text).toMatchObject({ kind: 'refusal', reason: 'Walking is not a skill this robot has learned yet.' });
    for (const text of ['summarize the live run record of your front left skill', 'run the evaluation again']) {
      expect(parseCommand(text, ctx), text).toMatchObject({ kind: 'refusal', reason: 'This command is not recognized.', escalate: true });
    }
  });
});
