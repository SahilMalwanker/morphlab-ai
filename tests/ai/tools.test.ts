import { describe, expect, it } from 'vitest';
import type { RobotCapabilities } from '../../src/ai/capabilities.ts';
import { DRAFT_GAIT_BOUNDS, DRAFT_GAIT_PATTERNS, DRAFT_GOAL_BOUNDS, DRAFT_NAME_PATTERN, DRAFT_POSE_LIMITS, MAX_DRAFT_KEYFRAMES, TOOL_DEFINITIONS, TOOL_NAMES, toolsForModel, validateToolCall } from '../../src/ai/tools.ts';
import { COMPOSE_LIMITS, MAX_COMPOSED_KEYFRAMES } from '../../src/skills/compose.ts';
import { GAIT_BOUNDS, GAIT_PATTERNS } from '../../src/skills/gait.ts';
import { GOAL_BOUNDS, MAX_GOALS, TAUGHT_NAME_PATTERN } from '../../src/skills/taught.ts';

function caps(over: Partial<RobotCapabilities> = {}, features: Partial<RobotCapabilities['features']> = {}): RobotCapabilities {
  return {
    name: 'Test rover',
    morphologyHash: 'hash-1',
    bodyKind: 'wheeled',
    jointCount: 0,
    skills: [],
    places: [],
    features: {
      pointing: false,
      navigation: true,
      sceneObjects: true,
      demonstrations: false,
      controllerExperiments: false,
      skillEvaluation: false,
      skillAuthoring: false,
      skillSequences: false,
      robotChoice: false,
      physicsTools: false,
      kinematicsTools: false,
      designTools: false,
      ...features,
    },
    ...over,
  };
}

const arm = caps({ name: 'Test arm', bodyKind: 'arm', jointCount: 3, skills: ['Lift front left (state machine)'] }, { pointing: true, navigation: false, controllerExperiments: true, skillEvaluation: true });

describe('tool definitions', () => {
  it('defines the thirty tools once, with provider-safe names and short descriptions, none returning an image (D31)', () => {
    expect(TOOL_DEFINITIONS).toHaveLength(30);
    expect(TOOL_NAMES.size).toBe(30);
    expect(TOOL_NAMES.has('getCameraSnapshot')).toBe(false);
    expect(TOOL_NAMES.has('reviewLastRun')).toBe(false);
    for (const t of TOOL_DEFINITIONS) {
      expect(t.name).toMatch(/^[A-Za-z0-9_-]{1,64}$/);
      expect(t.description.length).toBeGreaterThan(10);
      expect(t.description.length).toBeLessThanOrEqual(1024);
    }
  });

  it('offers a model only the tools this robot supports', () => {
    const roverTools = toolsForModel(caps()).map((t) => t.name);
    expect(roverTools).toContain('navigateTo');
    expect(roverTools).not.toContain('pointAt');
    expect(roverTools).not.toContain('executeSkill');
    const armTools = toolsForModel(arm).map((t) => t.name);
    expect(armTools).toContain('pointAt');
    expect(armTools).toContain('executeSkill');
    expect(armTools).toContain('runControllerExperiment');
    expect(armTools).not.toContain('navigateTo');
    expect(armTools).not.toContain('saveSkillFromDemonstration');
  });

  it('offers place tools only for driving robots, and drive-to and forget only when places are remembered', () => {
    const none = toolsForModel(caps()).map((t) => t.name);
    expect(none).toContain('rememberPlace');
    expect(none).not.toContain('navigateToPlace');
    expect(none).not.toContain('forgetPlace');
    const some = toolsForModel(caps({ places: ['dock'] })).map((t) => t.name);
    expect(some).toEqual(expect.arrayContaining(['rememberPlace', 'navigateToPlace', 'forgetPlace']));
    expect(toolsForModel(arm).map((t) => t.name)).not.toContain('rememberPlace');
    const remember = toolsForModel(caps()).find((t) => t.name === 'rememberPlace')?.parameters as { required: string[]; properties: Record<string, unknown> };
    expect(remember.required).toEqual(['name']);
    expect(Object.keys(remember.properties).sort()).toEqual(['name', 'x', 'y']);
  });

  it('emits closed JSON Schemas with the bounds from the Zod schemas', () => {
    const nav = toolsForModel(caps()).find((t) => t.name === 'navigateTo');
    expect(nav).toBeDefined();
    const p = nav?.parameters as { type: string; additionalProperties: boolean; required: string[]; properties: Record<string, { minimum?: number; maximum?: number }> };
    expect(p).not.toHaveProperty('$schema');
    expect(p.type).toBe('object');
    expect(p.additionalProperties).toBe(false);
    expect([...p.required].sort()).toEqual(['x', 'y']);
    expect(p.properties['x']).toMatchObject({ minimum: -0.4, maximum: 3.4 });
    expect(p.properties['y']).toMatchObject({ minimum: -1.9, maximum: 1.9 });
  });
});

describe('validateToolCall', () => {
  it('accepts valid JSON arguments and applies defaults', () => {
    expect(validateToolCall('navigateTo', '{"x":1.2,"y":-0.5}', caps())).toEqual({ ok: true, name: 'navigateTo', args: { x: 1.2, y: -0.5 } });
    expect(validateToolCall('getSceneObjects', '{}', caps())).toEqual({ ok: true, name: 'getSceneObjects', args: { mode: 'sensor' } });
  });

  it('treats empty arguments as an empty object for parameterless tools', () => {
    expect(validateToolCall('getRobotState', '', caps())).toEqual({ ok: true, name: 'getRobotState', args: {} });
  });

  it('rejects unknown tools without echoing unsafe characters', () => {
    const r = validateToolCall('rm -rf <x>', '{}', caps());
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.result.status).toBe('rejected');
      expect(r.result.taskId).toBeNull();
      expect(r.result.reason).toBe('There is no tool named rm-rfx.');
    }
  });

  it('rejects tools the current robot does not support, with the reason', () => {
    const r = validateToolCall('navigateTo', '{"x":0,"y":0}', arm);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.result.reason).toBe('Cannot do this yet: this robot cannot drive.');
  });

  it('rejects malformed, oversized, out-of-range, non-finite and extra arguments', () => {
    const bad: [string, unknown][] = [
      ['navigateTo', '{"x":1,'],
      ['navigateTo', `{"x":1,"y":1,"pad":"${'a'.repeat(4000)}"}`],
      ['navigateTo', '{"x":5,"y":0}'],
      ['navigateTo', { x: Number.POSITIVE_INFINITY, y: 0 }],
      ['navigateTo', { x: Number.NaN, y: 0 }],
      ['navigateTo', '{"x":0,"y":0,"turbo":true}'],
      ['getSceneObjects', '{"mode":"xray"}'],
    ];
    for (const [name, args] of bad) {
      const r = validateToolCall(name, args, caps());
      expect(r.ok, `${name} ${String(args).slice(0, 40)}`).toBe(false);
      if (!r.ok) expect(r.result.status).toBe('rejected');
    }
    const range = validateToolCall('navigateTo', '{"x":5,"y":0}', caps());
    expect(range.ok).toBe(false);
    if (!range.ok) expect(range.result.reason).toMatch(/^Invalid arguments for navigateTo: x: /);
  });

  it('rejects skills that are not saved for this robot and lists the saved ones', () => {
    const r = validateToolCall('executeSkill', '{"skill":"jump"}', arm);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.result.reason).toBe('No evaluated skill named "jump" is saved for Test arm.');
      expect(r.result.fallback).toBe('Saved skills: Lift front left (state machine).');
    }
    expect(validateToolCall('executeSkill', '{"skill":"Lift front left (state machine)","holdSeconds":2}', arm).ok).toBe(true);
    expect(validateToolCall('executeSkill', '{"skill":"rm -rf /; echo"}', arm).ok).toBe(false);
  });

  it('accepts a saved skill named without its method suffix when only one matches', () => {
    const legged = caps({ name: 'Quadruped', bodyKind: 'legged', jointCount: 12, skills: ['hop (keyframes)', 'sit (keyframes)', 'sit (keyframes 2)'] }, { navigation: false, sceneObjects: false, skillSequences: true });
    expect(validateToolCall('executeSkill', '{"skill":"hop"}', legged)).toEqual({ ok: true, name: 'executeSkill', args: { skill: 'hop (keyframes)' } });
    expect(validateToolCall('runSkillSequence', '{"steps":[{"skill":"HOP","times":2}]}', legged)).toEqual({ ok: true, name: 'runSkillSequence', args: { steps: [{ skill: 'hop (keyframes)', times: 2 }] } });
    // Two versions match "sit", so the name must be exact.
    const sit = validateToolCall('executeSkill', '{"skill":"sit"}', legged);
    expect(sit.ok).toBe(false);
    if (!sit.ok) expect(sit.result.reason).toBe('No evaluated skill named "sit" is saved for Quadruped.');
  });

  it('matches place names regardless of case and spacing, and lists the remembered places when one is unknown', () => {
    const rover = caps({ places: ['Loading dock', 'start'] });
    expect(validateToolCall('navigateToPlace', '{"place":"loading  DOCK"}', rover)).toEqual({ ok: true, name: 'navigateToPlace', args: { place: 'loading  DOCK' } });
    const r = validateToolCall('navigateToPlace', '{"place":"kitchen"}', rover);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.result.reason).toBe('No place named "kitchen" is remembered on this map.');
      expect(r.result.fallback).toBe('Remembered places: Loading dock, start.');
    }
    expect(validateToolCall('forgetPlace', '{"place":"x; rm -rf"}', rover).ok).toBe(false);
    const empty = validateToolCall('forgetPlace', '{"place":"start"}', caps());
    expect(empty.ok).toBe(false);
    if (!empty.ok) expect(empty.result.reason).toBe('Cannot do this yet: no places are remembered on this map.');
  });

  it('bounds the optional speed limit of navigation tools', () => {
    expect(validateToolCall('navigateTo', '{"x":1,"y":0,"maxSpeed":0.2}', caps())).toEqual({ ok: true, name: 'navigateTo', args: { x: 1, y: 0, maxSpeed: 0.2 } });
    for (const v of [0.05, 0.5, '"fast"']) expect(validateToolCall('navigateTo', `{"x":1,"y":0,"maxSpeed":${v}}`, caps()).ok, String(v)).toBe(false);
    expect(validateToolCall('navigateToPlace', '{"place":"dock","maxSpeed":0.1}', caps({ places: ['dock'] })).ok).toBe(true);
  });

  it('remembers a place at the current position or at both given coordinates, never at one of them', () => {
    expect(validateToolCall('rememberPlace', '{"name":"dock"}', caps())).toEqual({ ok: true, name: 'rememberPlace', args: { name: 'dock' } });
    expect(validateToolCall('rememberPlace', '{"name":"dock","x":1,"y":0.5}', caps())).toEqual({ ok: true, name: 'rememberPlace', args: { name: 'dock', x: 1, y: 0.5 } });
    const half = validateToolCall('rememberPlace', '{"name":"dock","x":1}', caps());
    expect(half.ok).toBe(false);
    if (!half.ok) expect(half.result.reason).toBe('Invalid arguments for rememberPlace: give x and y together, or neither.');
    expect(validateToolCall('rememberPlace', '{"name":"dock","x":9,"y":0}', caps()).ok).toBe(false);
    expect(validateToolCall('rememberPlace', '{"name":""}', caps()).ok).toBe(false);
    expect(validateToolCall('rememberPlace', '{"name":"dock"}', arm).ok).toBe(false);
  });

  it('bounds controller experiments', () => {
    expect(validateToolCall('runControllerExperiment', '{"configuration":{"kp":50,"kd":2}}', arm)).toEqual({
      ok: true,
      name: 'runControllerExperiment',
      args: { configuration: { kp: 50, kd: 2, gravityCompensation: false } },
    });
    expect(validateToolCall('runControllerExperiment', '{"configuration":{"kp":5000,"kd":2}}', arm).ok).toBe(false);
    // Without a configuration the lab's current setup runs unchanged.
    expect(validateToolCall('runControllerExperiment', '{}', arm)).toEqual({ ok: true, name: 'runControllerExperiment', args: {} });
    expect(validateToolCall('runControllerExperiment', '{"configuration":{}}', arm).ok).toBe(false);
    // Step tests are deterministic, so repeated trials are not offered.
    expect(validateToolCall('runControllerExperiment', '{"configuration":{"kp":50,"kd":2},"trialCount":3}', arm).ok).toBe(false);
  });
});

describe('skill composition tools', () => {
  const legged = caps({ name: 'Quadruped', bodyKind: 'legged', jointCount: 12 }, { navigation: false, sceneObjects: false, skillAuthoring: true });
  const crouch = {
    name: 'crouch',
    keyframes: [{ seconds: 1, body: { z: -0.08 } }],
    goals: [{ kind: 'baseHeight', below: true, heightM: 0.21, holdS: 1 }, { kind: 'upright', maxTiltRad: 0.35 }],
  };

  it('are offered only where skills can be composed', () => {
    expect(toolsForModel(legged).map((t) => t.name)).toEqual(expect.arrayContaining(['draftSkill', 'evaluateSkillDraft']));
    expect(toolsForModel(caps()).map((t) => t.name)).not.toContain('draftSkill');
    const r = validateToolCall('draftSkill', JSON.stringify(crouch), caps());
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.result.reason).toBe('Cannot do this yet: skills are composed in the Skills tab.');
  });

  it('accept a bounded draft and fill in the final hold', () => {
    const r = validateToolCall('draftSkill', JSON.stringify(crouch), legged);
    expect(r).toEqual({ ok: true, name: 'draftSkill', args: { ...crouch, finalHoldSeconds: 1 } });
    const full = {
      name: "lift front left",
      keyframes: [
        { seconds: 0.8, body: { y: -0.04, x: -0.03 } },
        { seconds: 0.5, body: { y: -0.04, x: -0.03 }, tips: [{ limb: 'front_left', z: 0.08 }], joints: [{ joint: 'front_left_hip_roll', value: 0.1 }] },
      ],
      finalHoldSeconds: 2,
      goals: [{ kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.05, holdS: 1 }, { kind: 'standsAtEnd', maxTiltRad: 0.35, minBaseHeightM: 0.13 }],
    };
    expect(validateToolCall('draftSkill', JSON.stringify(full), legged)).toEqual({ ok: true, name: 'draftSkill', args: full });
    expect(validateToolCall('evaluateSkillDraft', '{}', legged)).toEqual({ ok: true, name: 'evaluateSkillDraft', args: {} });
  });

  it('reject unknown goal kinds, out-of-range values and extra fields before anything runs', () => {
    const bad: unknown[] = [
      { ...crouch, goals: [{ kind: 'flies', minS: 1 }] },
      { ...crouch, goals: [{ kind: 'upright', maxTiltRad: 3 }] },
      { ...crouch, goals: [{ kind: 'upright', maxTiltRad: 0.3, bonus: 1 }] },
      { ...crouch, goals: [] },
      { ...crouch, keyframes: [] },
      { ...crouch, keyframes: Array.from({ length: 9 }, () => ({ seconds: 1 })) },
      { ...crouch, keyframes: [{ seconds: 0.01 }] },
      { ...crouch, keyframes: [{ seconds: 1, body: { z: -0.5 } }] },
      { ...crouch, keyframes: [{ seconds: 1, tips: [{ limb: 'front_left', z: 0.9 }] }] },
      { ...crouch, keyframes: [{ seconds: 1, torque: 5 }] },
      { ...crouch, name: 'rm -rf /' },
      { ...crouch, name: '' },
    ];
    for (const args of bad) {
      const r = validateToolCall('draftSkill', JSON.stringify(args), legged);
      expect(r.ok, JSON.stringify(args).slice(0, 80)).toBe(false);
      if (!r.ok) expect(r.result.status).toBe('rejected');
    }
  });

  it('keep their copies of the skill limits equal to the originals', () => {
    expect(DRAFT_NAME_PATTERN.source).toBe(TAUGHT_NAME_PATTERN.source);
    expect(DRAFT_GOAL_BOUNDS).toEqual(GOAL_BOUNDS);
    expect(DRAFT_POSE_LIMITS).toEqual(COMPOSE_LIMITS);
    expect(MAX_DRAFT_KEYFRAMES).toBe(MAX_COMPOSED_KEYFRAMES);
    expect(DRAFT_GAIT_BOUNDS).toEqual(GAIT_BOUNDS);
    expect([...DRAFT_GAIT_PATTERNS]).toEqual([...GAIT_PATTERNS]);
    expect(6).toBeLessThanOrEqual(MAX_GOALS);
  });

  it('accept a gait instead of keyframes, never both or neither, and fill in its defaults', () => {
    const walk = {
      name: 'walk',
      gait: { pattern: 'trot', dutyFactor: 0.6, cycleS: 0.5, strideM: 0.06, stepHeightM: 0.04, durationS: 4 },
      goals: [{ kind: 'travel', direction: 'forward', minDistanceM: 0.2 }, { kind: 'upright', maxTiltRad: 0.4 }],
    };
    expect(validateToolCall('draftSkill', JSON.stringify(walk), legged)).toEqual({
      ok: true,
      name: 'draftSkill',
      args: { ...walk, gait: { ...walk.gait, lateralM: 0, turnRad: 0, bodyHeightM: 0, feedbackKp: 0.1, feedbackKd: 0.005 }, finalHoldSeconds: 1 },
    });
    const both = validateToolCall('draftSkill', JSON.stringify({ ...walk, keyframes: [{ seconds: 1 }] }), legged);
    expect(both.ok).toBe(false);
    if (!both.ok) expect(both.result.reason).toBe('Invalid arguments for draftSkill: give either keyframes or a gait.');
    expect(validateToolCall('draftSkill', JSON.stringify({ name: 'walk', goals: walk.goals }), legged).ok).toBe(false);
    for (const gait of [{ ...walk.gait, cycleS: 0.1 }, { ...walk.gait, pattern: 'gallop' }, { ...walk.gait, torque: 1 }, { ...walk.gait, strideM: 0.5 }]) {
      expect(validateToolCall('draftSkill', JSON.stringify({ ...walk, gait }), legged).ok, JSON.stringify(gait)).toBe(false);
    }
  });

  it('accept body-angle, foot, swing and reach goals within their bounds', () => {
    const goals = [
      { kind: 'bodyAngle', axis: 'pitch', minRad: -0.6, maxRad: -0.15, holdS: 1 },
      { kind: 'footDown', limbId: 'rear_left', minFraction: 0.9 },
      { kind: 'tipSwing', limbId: 'front_right', axis: 'y', minAmplitudeM: 0.05, minSwings: 4 },
      { kind: 'tipReach', limbId: 'front_right', x: 0.1, y: 0, z: 0.1, toleranceM: 0.03, holdS: 0.5 },
      { kind: 'turn', left: true, minAngleRad: 1.3, maxAngleRad: 1.8 },
      { kind: 'airborne', limbIds: ['front_left', 'rear_right'], minS: 1.5 },
    ];
    expect(validateToolCall('draftSkill', JSON.stringify({ ...crouch, goals }), legged).ok).toBe(true);
    const bad = [{ ...goals[0], axis: 'yaw' }, { ...goals[0], minRad: -2 }, { ...goals[1], minFraction: 1.5 }, { ...goals[2], minSwings: 2.5 }, { ...goals[3], toleranceM: 0 }, { ...goals[5], limbIds: [] }];
    for (const g of bad) expect(validateToolCall('draftSkill', JSON.stringify({ ...crouch, goals: [g] }), legged).ok, JSON.stringify(g)).toBe(false);
  });

  it('describe every goal kind to the model, within the tool size limits', () => {
    const spec = toolsForModel(legged).find((t) => t.name === 'draftSkill')!;
    const text = JSON.stringify(spec.parameters);
    for (const kind of ['upright', 'noFall', 'tipHeight', 'baseHeight', 'airborne', 'baseRise', 'travel', 'turn', 'standsAtEnd', 'bodyAngle', 'footDown', 'tipSwing', 'tipReach']) expect(text).toContain(`"${kind}"`);
    expect(spec.description.length).toBeLessThanOrEqual(1024);
    expect(JSON.stringify(toolsForModel(legged)).length).toBeLessThan(16_000);
  });
});

describe('tools of every tab', () => {
  const names = (features: Partial<RobotCapabilities['features']>, over: Partial<RobotCapabilities> = {}) =>
    toolsForModel(caps({ name: 'Quadruped', bodyKind: 'legged', jointCount: 12, ...over }, { navigation: false, sceneObjects: false, ...features })).map((t) => t.name);

  it('are offered only where they act', () => {
    expect(names({ physicsTools: true })).toEqual(expect.arrayContaining(['pushRobot', 'resetSimulation']));
    expect(names({})).not.toContain('pushRobot');
    expect(names({ kinematicsTools: true })).toEqual(expect.arrayContaining(['moveTipTo', 'setJointAngles']));
    expect(names({ kinematicsTools: true }, { bodyKind: 'wheeled' })).toContain('setJointAngles');
    expect(names({ kinematicsTools: true }, { bodyKind: 'wheeled', targetableLimbs: [] })).not.toContain('moveTipTo');
    expect(names({ kinematicsTools: true }, { bodyKind: 'wheeled', targetableLimbs: ['arm'] })).toContain('moveTipTo');
    expect(names({})).not.toContain('moveTipTo');
    expect(names({ designTools: true })).toContain('buildRobot');
    expect(names({ robotChoice: true })).toContain('selectRobot');
    expect(names({ skillAuthoring: true })).toContain('improveSkillDraft');
    // A learned gait policy needs a body that walks.
    expect(names({ skillAuthoring: true })).toContain('trainGaitPolicy');
    expect(names({ skillAuthoring: true }, { bodyKind: 'arm' })).not.toContain('trainGaitPolicy');
    expect(names({})).not.toContain('trainGaitPolicy');
    expect(validateToolCall('trainGaitPolicy', '{"iterations":5}', caps({ name: 'Quadruped', bodyKind: 'legged' }, { skillAuthoring: true })).ok).toBe(false);
    expect(validateToolCall('trainGaitPolicy', '{"architecture":"mlp"}', caps({ name: 'Quadruped', bodyKind: 'legged' }, { skillAuthoring: true })).ok).toBe(true);
    expect(validateToolCall('trainGaitPolicy', '{"architecture":"gpu"}', caps({ name: 'Quadruped', bodyKind: 'legged' }, { skillAuthoring: true })).ok).toBe(false);
    // A walking controller is trained once per legged body; walking by command needs one.
    const ranges = { forwardMps: [-0.12, 0.15], leftMps: [-0.08, 0.08], turnRadps: [-0.6, 0.6] } as const;
    expect(names({ skillAuthoring: true })).toContain('trainLocomotion');
    expect(names({ skillAuthoring: true })).not.toContain('locomote');
    expect(names({ skillAuthoring: true }, { walking: ranges })).toContain('locomote');
    expect(names({ skillAuthoring: true }, { bodyKind: 'arm' })).not.toContain('trainLocomotion');
    // Navigate trains the walking controller of a legged robot too, never of a wheeled one.
    expect(names({ navigation: true })).toContain('trainLocomotion');
    expect(names({ navigation: true }, { bodyKind: 'wheeled' })).not.toContain('trainLocomotion');
    expect(names({})).not.toContain('trainLocomotion');
    const legged = caps({ name: 'Quadruped', bodyKind: 'legged', walking: ranges }, { skillAuthoring: true });
    expect(validateToolCall('locomote', '{"forwardMps":0.1,"durationS":3}', legged)).toEqual({ ok: true, name: 'locomote', args: { forwardMps: 0.1, durationS: 3 } });
    expect(validateToolCall('locomote', '{"forwardMps":0.1}', legged).ok).toBe(false);
    expect(validateToolCall('locomote', '{"forwardMps":2,"durationS":3}', legged).ok).toBe(false);
    // Or a target, where the body should end up; never both, and never nothing.
    expect(validateToolCall('locomote', '{"forwardM":0.3,"turnRad":1.57}', legged)).toEqual({ ok: true, name: 'locomote', args: { forwardM: 0.3, turnRad: 1.57 } });
    const both = validateToolCall('locomote', '{"forwardM":0.3,"durationS":3}', legged);
    expect(both.ok).toBe(false);
    if (!both.ok) expect(both.result.reason).toBe('Invalid arguments for locomote: give either a target (forwardM, leftM, turnRad) or speeds with durationS.');
    expect(validateToolCall('locomote', '{}', legged).ok).toBe(false);
    expect(validateToolCall('locomote', '{"leftM":3}', legged).ok).toBe(false);
    // Mirroring needs a saved skill to mirror.
    expect(names({ skillAuthoring: true })).not.toContain('mirrorSkill');
    expect(names({ skillAuthoring: true }, { skills: ['lean left (keyframes)'] })).toContain('mirrorSkill');
    const mirror = validateToolCall('mirrorSkill', '{"skill":"lean left"}', caps({ name: 'Quadruped', bodyKind: 'legged', skills: ['lean left (keyframes)'] }, { skillAuthoring: true }));
    expect(mirror).toEqual({ ok: true, name: 'mirrorSkill', args: { skill: 'lean left (keyframes)' } });
    expect(validateToolCall('mirrorSkill', '{"skill":"lean left","name":"lean/right"}', caps({ name: 'Quadruped', bodyKind: 'legged', skills: ['lean left (keyframes)'] }, { skillAuthoring: true })).ok).toBe(false);
    expect(names({ skillSequences: true })).not.toContain('runSkillSequence');
    expect(names({ skillSequences: true }, { skills: ['hop (keyframes)'] })).toContain('runSkillSequence');
    // Every tab's tool list stays within the proxy's limit of 24 tools per request.
    const everything = names({ skillAuthoring: true, skillSequences: true, robotChoice: true, physicsTools: true, kinematicsTools: true, designTools: true, skillEvaluation: true, controllerExperiments: true }, { skills: ['hop (keyframes)'] });
    expect(everything.length).toBeLessThanOrEqual(24);
  });

  it('bound their arguments before anything runs', () => {
    const all = caps({ name: 'Quadruped', bodyKind: 'legged', jointCount: 12, skills: ['hop (keyframes)', 'sit (keyframes)'] }, { physicsTools: true, kinematicsTools: true, designTools: true, robotChoice: true, skillSequences: true });
    expect(validateToolCall('pushRobot', '{"direction":"left","impulseNs":2}', all).ok).toBe(true);
    expect(validateToolCall('pushRobot', '{"direction":"up","impulseNs":2}', all).ok).toBe(false);
    expect(validateToolCall('pushRobot', '{"direction":"left","impulseNs":50}', all).ok).toBe(false);
    expect(validateToolCall('resetSimulation', '{}', all).ok).toBe(true);
    expect(validateToolCall('moveTipTo', '{"limb":"front_left","x":0.3,"y":0.1,"z":0.05}', all).ok).toBe(true);
    expect(validateToolCall('moveTipTo', '{"x":9,"y":0,"z":0}', all).ok).toBe(false);
    expect(validateToolCall('setJointAngles', '{"joints":[{"joint":"front_left_knee","value":-1.2}]}', all).ok).toBe(true);
    expect(validateToolCall('setJointAngles', '{"joints":[]}', all).ok).toBe(false);
    expect(validateToolCall('runSkillSequence', '{"steps":[{"skill":"sit (keyframes)"},{"skill":"hop (keyframes)","times":2}]}', all)).toEqual({
      ok: true,
      name: 'runSkillSequence',
      args: { steps: [{ skill: 'sit (keyframes)', times: 1 }, { skill: 'hop (keyframes)', times: 2 }] },
    });
    expect(validateToolCall('runSkillSequence', '{"steps":[{"skill":"hop (keyframes)","times":9}]}', all).ok).toBe(false);
    const unsaved = validateToolCall('runSkillSequence', '{"steps":[{"skill":"hop (keyframes)"},{"skill":"fly"}]}', all);
    expect(unsaved.ok).toBe(false);
    if (!unsaved.ok) expect(unsaved.result.reason).toBe('No evaluated skill named "fly" is saved for Quadruped.');
    expect(validateToolCall('selectRobot', '{"robot":"hexapod"}', all).ok).toBe(true);
    expect(validateToolCall('selectRobot', '{"robot":"rover"}', all).ok).toBe(true);
    expect(validateToolCall('selectRobot', '{"robot":"spider"}', all).ok).toBe(false);
    expect(validateToolCall('buildRobot', '{"preset":"hexapod"}', all).ok).toBe(true);
    expect(validateToolCall('buildRobot', '{"arm":{"dof":4,"reach":0.8},"name":"Reacher"}', all).ok).toBe(true);
    expect(validateToolCall('buildRobot', '{"arm":{"dof":9,"reach":0.8}}', all).ok).toBe(false);
    const two = validateToolCall('buildRobot', '{"preset":"hexapod","arm":{"dof":4,"reach":0.8}}', all);
    expect(two.ok).toBe(false);
    if (!two.ok) expect(two.result.reason).toBe('Invalid arguments for buildRobot: give exactly one of preset, legged or arm.');
    expect(validateToolCall('buildRobot', '{}', all).ok).toBe(false);
  });
});
