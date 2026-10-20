import { describe, expect, it } from 'vitest';
import type { RobotCapabilities } from '../../src/ai/capabilities.ts';
import { asksForLearning, plainReply, runAiTurn, skillFollowUp, type TurnOptions } from '../../src/ai/loop.ts';
import { MAX_MESSAGES, MAX_TOTAL_CHARS, proxyJsonLength, requestChars, type ChatMessage, type ChatRequest, type ChatResult } from '../../src/ai/protocol.ts';
import type { ToolResult } from '../../src/ai/result.ts';
import { toolsForModel } from '../../src/ai/tools.ts';

const rover: RobotCapabilities = {
  name: 'Rover',
  morphologyHash: 'abc123',
  bodyKind: 'wheeled',
  jointCount: 4,
  skills: [],
  places: [],
  features: { pointing: false, navigation: true, sceneObjects: false, demonstrations: false, controllerExperiments: false, skillEvaluation: false, skillAuthoring: false, skillSequences: false, robotChoice: false, physicsTools: false, kinematicsTools: false, designTools: false },
};

const text = (content: string): ChatResult => ({ content, toolCalls: [], finishReason: 'stop', model: 'fake-model', usage: { promptTokens: 100, completionTokens: 10 } });
const calls = (...c: [string, string, string][]): ChatResult => ({
  content: null,
  toolCalls: c.map(([id, name, args]) => ({ id, name, arguments: args })),
  finishReason: 'tool_calls',
  model: 'fake-model',
  usage: { promptTokens: 120, completionTokens: 20 },
});
const reached: ToolResult = { taskId: 'task-1', status: 'succeeded', reason: 'Reached the goal.', measurements: { groundTruthDistanceToGoal: { value: 0.12, unit: 'm' } }, fallback: null };

function setup(script: (ChatResult | Error)[], over: Partial<TurnOptions> = {}) {
  const requests: ChatRequest[] = [];
  const executed: [string, Record<string, unknown>][] = [];
  const options: TurnOptions = {
    chat: async (r) => {
      requests.push(structuredClone(r));
      const next = script.shift();
      if (!next) throw new Error('script exhausted');
      if (next instanceof Error) throw next;
      return next;
    },
    system: 'You operate a simulated robot.',
    history: [],
    userText: 'Drive to x 1, y 0.5.',
    tools: toolsForModel(rover),
    caps: rover,
    execute: async (name, args) => {
      executed.push([name, args]);
      return name === 'navigateTo' ? reached : { taskId: null, status: 'succeeded', reason: 'Pose estimate.', measurements: {}, fallback: null };
    },
    signal: new AbortController().signal,
    ...over,
  };
  return { options, requests, executed };
}

describe('plainReply', () => {
  it('rewrites LaTeX math as plain text and removes markdown marks', () => {
    expect(plainReply('Raise \\(kd\\) to 10.')).toBe('Raise kd to 10.');
    expect(plainReply('The damping ratio is \\(\\zeta = \\frac{k_d}{2\\sqrt{k_p I}}\\).')).toBe('The damping ratio is zeta = k_d/(2 sqrt(k_p I)).');
    expect(plainReply('\\[ e_{ss} = \\frac{\\tau_g}{k_p} \\approx 2.45^\\circ \\]')).toBe('e_ss = tau_g/k_p ≈ 2.45 deg');
    expect(plainReply('**Gravity compensation** removes the `sag`.\n### Next')).toBe('Gravity compensation removes the sag.\nNext');
  });

  it('leaves plain text, units and products alone', () => {
    const s = 'Gain set B (kp 60, kd 3) measured 27.0 % overshoot; wn*h is 0.148 and the effort limit is 6 N*m.';
    expect(plainReply(s)).toBe(s);
  });

  it('is applied to the reply of a turn', async () => {
    const { options } = setup([text('  Raise **kd** to \\(10\\).  ')]);
    expect((await runAiTurn(options)).text).toBe('Raise kd to 10.');
  });
});

describe('runAiTurn', () => {
  it('does not nudge a successful preview into training or evaluation when the user declined them', () => {
    const preview = { tool: 'draftSkill', arguments: '{}', accepted: true, result: { ...reached, fallback: 'Call trainGaitPolicy now, then evaluateSkillDraft.' } };
    expect(skillFollowUp([preview], 'Try it once, but do not train or evaluate it yet.')).toBeNull();
    expect(skillFollowUp([preview], 'Preview only.')).toBeNull();
  });

  it.each([
    ['trainGaitPolicy', 'Try it, but do not train or evaluate it yet.'],
    ['evaluateSkillDraft', 'Try it, but do not train or evaluate it yet.'],
    ['evaluateSkillDraft', "Evaluate the draft but don't save it."],
    ['improveSkillDraft', 'Preview only.'],
    ['trainLocomotion', 'Do not retrain the robot.'],
    ['trainLocomotion', 'Train a walking controller, but do not save it.'],
    ['draftSkill', 'Walk at 1 m/s. Do not substitute a lower speed if that exceeds the trained range.'],
    ['trainGaitPolicy', 'Walk at 1 m/s. Do not substitute a lower speed if that exceeds the trained range.'],
    ['draftSkill', 'Run the saved skill. Do not compose or teach a replacement.'],
  ])('blocks %s when the request explicitly declines its side effects', async (tool, request) => {
    const robot: RobotCapabilities = { ...rover, bodyKind: 'legged', features: { ...rover.features, skillAuthoring: true } };
    const { options, executed } = setup([calls(['blocked', tool, '{}']), text('No prohibited action ran.')], {
      caps: robot, tools: toolsForModel(robot), userText: request,
    });
    const result = await runAiTurn(options);
    expect(executed).toEqual([]);
    expect(result.steps[0]).toMatchObject({ accepted: false, result: { status: 'rejected' } });
    expect(result.steps[0]!.result.reason).toContain('Not permitted by this request');
    expect(skillFollowUp(result.steps, request)).toBeNull();
  });

  it('removes a tool fallback that contradicts the request restrictions', async () => {
    const robot: RobotCapabilities = { ...rover, bodyKind: 'legged', features: { ...rover.features, skillAuthoring: true } };
    const result: ToolResult = { ...reached, fallback: 'Call evaluateSkillDraft now to save it.' };
    const { options, requests } = setup([calls(['try', 'draftSkill', '{"name":"preview","keyframes":[{"seconds":1}],"goals":[{"kind":"upright","maxTiltRad":0.3}]}']), text('Preview complete.')], {
      caps: robot, tools: toolsForModel(robot), userText: 'Preview only.', execute: async () => result,
    });
    const report = await runAiTurn(options);
    expect(report.steps[0]).toMatchObject({ accepted: true, result: { status: 'succeeded', fallback: null } });
    expect(result.fallback).not.toBeNull();
    expect(JSON.stringify(requests[1]!.messages)).not.toContain('Call evaluateSkillDraft now to save it.');
  });

  it('ends on a tool cancellation without executing queued tools or asking the model to explain it', async () => {
    const executed: string[] = [];
    const steps: string[] = [];
    const { options, requests } = setup([
      calls(['cancelled', 'navigateTo', '{"x":1,"y":0}'], ['later', 'getRobotState', '{}']),
      text('The cancelled task failed; try changing the target.'),
    ], {
      execute: async (name) => {
        executed.push(name);
        return { taskId: 'task-1', status: 'cancelled', reason: 'Stopped by the user.', measurements: {}, fallback: null };
      },
      onStep: (step) => steps.push(step.result.status),
      refresh: async () => { throw new Error('No refresh after cancellation'); },
    });
    const report = await runAiTurn(options);
    expect(report).toMatchObject({ end: 'cancelled', text: null, modelCalls: 1, error: null });
    expect(report.steps).toHaveLength(1);
    expect(executed).toEqual(['navigateTo']);
    expect(steps).toEqual(['cancelled']);
    expect(requests).toHaveLength(1);
    expect(options.signal.aborted).toBe(false);
  });

  it('returns a plain answer and offers only the robot tools', async () => {
    const { options, requests } = setup([text('The rover can drive to goals on the arena floor.')]);
    const r = await runAiTurn(options);
    expect(r).toMatchObject({ end: 'answered', text: 'The rover can drive to goals on the arena floor.', modelCalls: 1, steps: [], tokens: { prompt: 100, completion: 10 } });
    expect(requests[0]?.messages.map((m) => m.role)).toEqual(['system', 'user']);
    expect(requests[0]?.toolChoice).toBe('auto');
    expect(requests[0]?.tools?.map((t) => t.name)).toContain('navigateTo');
    expect(requests[0]?.tools?.map((t) => t.name)).not.toContain('executeSkill');
  });

  it('runs a validated tool call, returns the measured result to the model, then reports', async () => {
    const { options, requests, executed } = setup([calls(['call_1', 'navigateTo', '{"x":1,"y":0.5}']), text('The rover reached the goal, 0.12 m away.')]);
    const steps: string[] = [];
    const r = await runAiTurn({ ...options, onStep: (s) => steps.push(`${s.tool}:${s.result.status}`) });
    expect(executed).toEqual([['navigateTo', { x: 1, y: 0.5 }]]);
    expect(steps).toEqual(['navigateTo:succeeded']);
    expect(r.end).toBe('answered');
    expect(r.steps[0]).toEqual({ tool: 'navigateTo', arguments: '{"x":1,"y":0.5}', accepted: true, result: reached });
    const second = requests[1]!.messages;
    expect(second.slice(2)).toEqual([
      { role: 'assistant', content: null, toolCalls: [{ id: 'call_1', name: 'navigateTo', arguments: '{"x":1,"y":0.5}' }] },
      { role: 'tool', toolCallId: 'call_1', content: JSON.stringify(reached) },
    ]);
    expect(r.tokens).toEqual({ prompt: 220, completion: 30 });
  });

  it('never runs a call that fails validation and tells the model why', async () => {
    const { options, executed, requests } = setup([calls(['call_1', 'navigateTo', '{"x":9,"y":0}']), text('That goal is outside the arena.')]);
    const r = await runAiTurn(options);
    expect(executed).toEqual([]);
    expect(r.steps[0]).toMatchObject({ tool: 'navigateTo', accepted: false, result: { status: 'rejected' } });
    expect(r.steps[0]?.result.reason).toMatch(/^Invalid arguments for navigateTo: x: /);
    const toolMessage = requests[1]!.messages[3];
    expect(toolMessage).toMatchObject({ role: 'tool', toolCallId: 'call_1' });
  });

  it('runs at most one action per model answer', async () => {
    const { options, executed } = setup([calls(['a', 'getRobotState', '{}'], ['b', 'navigateTo', '{"x":1,"y":0}'], ['c', 'navigateTo', '{"x":2,"y":0}']), text('Done.')]);
    const r = await runAiTurn(options);
    expect(executed.map(([n]) => n)).toEqual(['getRobotState', 'navigateTo']);
    expect(r.steps[2]).toMatchObject({ tool: 'navigateTo', accepted: false, result: { status: 'rejected', reason: 'One action at a time: request it again after reading the previous result.' } });
  });

  it('runs a path of walks from one answer in order, and stops at the first that does not succeed', async () => {
    const dog: RobotCapabilities = {
      ...rover, name: 'Quadruped', bodyKind: 'legged', jointCount: 12, walking: { forwardMps: [-0.12, 0.15], leftMps: [-0.08, 0.08], turnRadps: [-0.6, 0.6] },
      features: { ...rover.features, navigation: false, skillAuthoring: true },
    };
    const walked: ToolResult = { taskId: 'task-1', status: 'succeeded', reason: 'Walked.', measurements: {}, fallback: null };
    const outcomes: ToolResult[] = [walked, walked, { ...walked, status: 'failed', reason: 'Ended outside tolerance.' }];
    const ran: string[] = [];
    const path = calls(['a', 'locomote', '{"forwardM":0.2}'], ['b', 'locomote', '{"turnRad":1.57}'], ['c', 'locomote', '{"forwardM":0.2}'], ['d', 'locomote', '{"turnRad":1.57}']);
    const { options } = setup([path, text('Stopped after the second side.')], { caps: dog, tools: toolsForModel(dog), execute: async (name, args) => (ran.push(`${name} ${JSON.stringify(args)}`), outcomes.shift()!) });
    const r = await runAiTurn(options);
    expect(ran).toEqual(['locomote {"forwardM":0.2}', 'locomote {"turnRad":1.57}', 'locomote {"forwardM":0.2}']);
    expect(r.steps.map((s) => `${s.tool}:${s.result.status}`)).toEqual(['locomote:succeeded', 'locomote:succeeded', 'locomote:failed', 'locomote:rejected']);
    expect(r.steps[3]!.result.reason).toBe('Not run: an earlier action in this answer did not succeed; read its result first.');
  });

  it('sanitizes odd tool names and call IDs before they go back to the proxy', async () => {
    const { options, requests } = setup([calls(['id with spaces!', 'set joint torque!', '{}']), text('I cannot set torques.')]);
    const r = await runAiTurn(options);
    expect(r.steps[0]).toMatchObject({ tool: 'setjointtorque', accepted: false, result: { status: 'rejected', reason: 'There is no tool named setjointtorque.' } });
    const [assistant, tool] = requests[1]!.messages.slice(2);
    expect(assistant).toMatchObject({ role: 'assistant', toolCalls: [{ id: 'call_1', name: 'setjointtorque' }] });
    expect(tool).toMatchObject({ role: 'tool', toolCallId: 'call_1' });
  });

  it('stops at the model call budget', async () => {
    const loop = () => calls(['x', 'getRobotState', '{}']);
    const { options } = setup([loop(), loop(), loop(), text('never reached')], { maxModelCalls: 3 });
    const r = await runAiTurn(options);
    expect(r).toMatchObject({ end: 'step-limit', modelCalls: 3, text: null });
    expect(r.steps).toHaveLength(3);
  });

  it('reports request errors, content filtering, cancellation and tool exceptions', async () => {
    const failed = await runAiTurn(setup([new Error('The AI proxy is not running here.')]).options);
    expect(failed).toMatchObject({ end: 'error', error: 'The AI proxy is not running here.', modelCalls: 0 });
    const filtered = await runAiTurn(setup([{ ...text(''), finishReason: 'content_filter' }]).options);
    expect(filtered).toMatchObject({ end: 'filtered', error: 'The provider content filter withheld the answer.' });
    const c = new AbortController();
    c.abort();
    expect((await runAiTurn(setup([text('x')], { signal: c.signal }).options)).end).toBe('cancelled');
    const throwing = setup([calls(['a', 'navigateTo', '{"x":1,"y":0}']), text('It failed.')], {
      execute: async () => {
        throw new Error('the simulation is not loaded');
      },
    });
    const r = await runAiTurn(throwing.options);
    expect(r.steps[0]?.result).toMatchObject({ status: 'failed', reason: 'The tool failed: the simulation is not loaded' });
  });

  it('keeps tool results short in the conversation, as valid JSON', async () => {
    const long: ToolResult = { taskId: null, status: 'succeeded', reason: 'x'.repeat(5000), measurements: { a: { value: 1, unit: 'm' } }, fallback: null };
    const { options, requests } = setup([calls(['a', 'getRobotState', '{}']), text('ok')], { execute: async () => long });
    await runAiTurn(options);
    const tool = requests[1]!.messages[3];
    const content = tool?.role === 'tool' ? tool.content : '';
    expect(content.length).toBeLessThanOrEqual(2500);
    expect(content.length).toBeGreaterThan(2400);
    expect(JSON.parse(content)).toMatchObject({ status: 'succeeded', measurements: 'left out to fit the request limit' });
    expect((JSON.parse(content) as { reason: string }).reason).toMatch(/^x+ \[cut\]$/);
  });

  it('refreshes the tools after an action, so a skill saved during the command can run in it', async () => {
    const before: RobotCapabilities = { ...rover, name: 'Quadruped', bodyKind: 'legged', features: { ...rover.features, navigation: false, skillAuthoring: true } };
    const after: RobotCapabilities = { ...before, skills: ['wave (keyframes)'] };
    const ran: string[] = [];
    let refreshed = 0;
    const { options, requests } = setup([calls(['a', 'evaluateSkillDraft', '{}']), calls(['b', 'executeSkill', '{"skill":"wave (keyframes)"}']), text('It waved.')], {
      caps: before,
      tools: toolsForModel(before),
      execute: async (name) => {
        ran.push(name);
        return { taskId: 'task-1', status: 'succeeded', reason: 'Done.', measurements: {}, fallback: null };
      },
      refresh: async () => {
        refreshed++;
        return { caps: after, tools: toolsForModel(after) };
      },
    });
    const r = await runAiTurn(options);
    expect(requests[0]!.tools!.map((t) => t.name)).not.toContain('executeSkill');
    expect(requests[1]!.tools!.map((t) => t.name)).toContain('executeSkill');
    expect(ran).toEqual(['evaluateSkillDraft', 'executeSkill']);
    expect(r.steps.map((s) => `${s.tool}:${s.result.status}`)).toEqual(['evaluateSkillDraft:succeeded', 'executeSkill:succeeded']);
    expect(refreshed).toBe(2);
  });

  it('reminds the model to revise after a missed goal and to evaluate a draft that met every goal', async () => {
    const legged: RobotCapabilities = { ...rover, name: 'Quadruped', bodyKind: 'legged', features: { ...rover.features, navigation: false, skillAuthoring: true } };
    const results: ToolResult[] = [
      { taskId: 'task-1', status: 'failed', reason: 'Tried the draft: 1 of 2 goals met.', measurements: {}, fallback: null },
      { taskId: 'task-2', status: 'succeeded', reason: 'Tried the draft: 2 of 2 goals met.', measurements: {}, fallback: null },
      { taskId: 'task-3', status: 'succeeded', reason: 'Evaluated: 20 of 20 trials.', measurements: {}, fallback: null },
    ];
    const draft = '{"name":"wave","keyframes":[{"seconds":1}],"goals":[{"kind":"upright","maxTiltRad":0.5}]}';
    const { options, requests } = setup(
      [calls(['a', 'draftSkill', draft]), text('I will revise it.'), calls(['b', 'draftSkill', draft]), text('It worked.'), calls(['c', 'evaluateSkillDraft', '{}']), text('Saved.')],
      { caps: legged, tools: toolsForModel(legged), execute: async () => results.shift()!, followUp: skillFollowUp },
    );
    const r = await runAiTurn(options);
    expect(r.steps.map((s) => `${s.tool}:${s.result.status}`)).toEqual(['draftSkill:failed', 'draftSkill:succeeded', 'evaluateSkillDraft:succeeded']);
    expect(r.text).toBe('Saved.');
    expect(r.modelCalls).toBe(6);
    const last = requests[5]!.messages;
    expect(last.filter((m) => m.role === 'system' && m.content.startsWith('The try missed a goal. Do not answer yet')).length).toBe(1);
    expect(last.filter((m) => m.role === 'system' && m.content.startsWith('The draft met every goal. Call evaluateSkillDraft now')).length).toBe(1);
    expect(last.some((m) => m.role === 'assistant' && m.content === 'I will revise it.')).toBe(true);
  });

  it('reminds at most twice per command, and not after other tools', async () => {
    const legged: RobotCapabilities = { ...rover, name: 'Quadruped', bodyKind: 'legged', features: { ...rover.features, navigation: false, skillAuthoring: true } };
    const missed: ToolResult = { taskId: 'task-1', status: 'failed', reason: 'Tried the draft: 0 of 1 goals met.', measurements: {}, fallback: null };
    const draft = '{"name":"hover","keyframes":[{"seconds":1}],"goals":[{"kind":"airborne","minS":1}]}';
    const { options } = setup(
      [calls(['a', 'draftSkill', draft]), text('Again?'), calls(['b', 'draftSkill', draft]), text('Again?'), calls(['c', 'draftSkill', draft]), text('It cannot hover.')],
      { caps: legged, tools: toolsForModel(legged), execute: async () => missed, followUp: skillFollowUp },
    );
    const r = await runAiTurn(options);
    expect(r.steps).toHaveLength(3);
    expect(r.text).toBe('It cannot hover.');
    expect(skillFollowUp([{ tool: 'navigateTo', arguments: '{}', accepted: true, result: missed }])).toBeNull();
    expect(skillFollowUp([])).toBeNull();
    // A refused action call is to be corrected and made again.
    const refused: ToolResult = { taskId: null, status: 'rejected', reason: 'No evaluated skill named "hop" is saved for Quadruped.', measurements: {}, fallback: null };
    expect(skillFollowUp([{ tool: 'runSkillSequence', arguments: '{}', accepted: false, result: refused }])).toBe(
      'The runSkillSequence call was refused before it ran: No evaluated skill named "hop" is saved for Quadruped. Correct the call and make it again instead of answering, unless the refusal says it cannot be done here.',
    );
    expect(skillFollowUp([{ tool: 'getRobotState', arguments: '{}', accepted: false, result: refused }])).toBeNull();
    // A mirrored skill is a draft: a missed goal asks for a revision, a met one for the evaluation.
    const tried: ToolResult = { taskId: 'task-2', status: 'succeeded', reason: 'Tried the draft "lean right": 2 of 2 goals met.', measurements: {}, fallback: null };
    expect(skillFollowUp([{ tool: 'mirrorSkill', arguments: '{}', accepted: true, result: missed }])).toMatch(/^The try missed a goal\. Do not answer yet: .*\(draft 2 of 3\)/);
    expect(skillFollowUp([{ tool: 'mirrorSkill', arguments: '{}', accepted: true, result: tried }])).toMatch(/^The draft met every goal\. Call evaluateSkillDraft now/);
    // A kept learned gait policy is evaluated next; a rejected one leaves the choice to the model.
    expect(skillFollowUp([{ tool: 'trainGaitPolicy', arguments: '{}', accepted: true, result: tried }])).toMatch(/^The learned feedback policy was kept in the draft\. Call evaluateSkillDraft now/);
    expect(skillFollowUp([{ tool: 'trainGaitPolicy', arguments: '{}', accepted: true, result: missed }])).toBeNull();
    // A request for learning trains a gait draft's policy before the evaluation (seen in the browser check: the model evaluated without training).
    const gait = '{"name":"side step","gait":{"pattern":"trot"},"goals":[]}';
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: gait, accepted: true, result: tried }], 'step right, then train a learned feedback policy')).toMatch(/^The draft met every goal, and the user asked for learning\. Call trainGaitPolicy now/);
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: gait, accepted: true, result: missed }, { tool: 'improveSkillDraft', arguments: '{}', accepted: true, result: tried }], 'a learned side step')).toMatch(/^The draft met every goal, and the user asked for learning/);
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: gait, accepted: true, result: tried }], 'step right')).toMatch(/^The draft met every goal\. Call evaluateSkillDraft now/);
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: '{"name":"wave","keyframes":[]}', accepted: true, result: tried }], 'learn to wave')).toMatch(/^The draft met every goal\. Call evaluateSkillDraft now/);
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: gait, accepted: true, result: tried }, { tool: 'trainGaitPolicy', arguments: '{}', accepted: true, result: missed }, { tool: 'improveSkillDraft', arguments: '{}', accepted: true, result: tried }], 'train it')).toMatch(/^The draft met every goal\. Call evaluateSkillDraft now/);
    // A keyframe draft follows the next step its result names (the bridge decides by the body and the request, D30).
    const balance = '{"name":"two legs","keyframes":[]}';
    const asks: ToolResult = { ...tried, fallback: 'Every goal was met in this one run, open loop; keyframes do not react to a push or a slip. Call trainGaitPolicy now.' };
    const fell: ToolResult = { ...missed, fallback: 'If the poses are right but the body cannot hold them open loop, call trainGaitPolicy.' };
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: balance, accepted: true, result: asks }], 'stand on two legs')).toMatch(/^The draft met every goal open loop, and its result asks for a learned feedback policy\. Call trainGaitPolicy now/);
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: balance, accepted: true, result: fell }], 'stand on two legs')).toMatch(/\(draft 2 of 3\), or call improveSkillDraft if it came close, or call trainGaitPolicy if the poses are right but the body cannot hold them open loop\./);
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: balance, accepted: true, result: missed }], 'stand on two legs')).not.toContain('trainGaitPolicy');
    expect(skillFollowUp([{ tool: 'draftSkill', arguments: balance, accepted: true, result: fell }, { tool: 'trainGaitPolicy', arguments: '{}', accepted: true, result: missed }, { tool: 'improveSkillDraft', arguments: '{}', accepted: true, result: asks }], 'stand on two legs')).toMatch(/^The draft met every goal\. Call evaluateSkillDraft now/);
    // An evaluation with no trial meeting every goal is revised, not the end of the answer, within the three drafts.
    const none: ToolResult = { ...missed, reason: 'Evaluated "learned jump": none of 20 randomized trials met every goal, so it was not saved.' };
    const jumpSteps = [{ tool: 'draftSkill', arguments: balance, accepted: true, result: tried }, { tool: 'evaluateSkillDraft', arguments: '{}', accepted: true, result: none }];
    expect(skillFollowUp(jumpSteps, 'teach a jump')).toMatch(/^No evaluation trial met every goal\. Do not answer yet: revise the draft with draftSkill \(draft 2 of 3\)/);
    const three = [...jumpSteps, ...jumpSteps, ...jumpSteps];
    expect(skillFollowUp(three, 'teach a jump')).toBeNull();
    // A saved skill ends the answer: no extra run to look at (images are never sent, D31).
    const saved: ToolResult = { ...tried, reason: 'Evaluated "learned jump": 20 of 20 randomized trials met every goal.' };
    const evaluated = [{ tool: 'draftSkill', arguments: balance, accepted: true, result: tried }, { tool: 'evaluateSkillDraft', arguments: '{}', accepted: true, result: saved }];
    expect(skillFollowUp(evaluated, 'teach a jump')).toBeNull();
    expect(asksForLearning('walk with a learned controller')).toBe(true);
    expect(asksForLearning('Train a policy for it')).toBe(true);
    expect(asksForLearning('walk forward and turn left')).toBe(false);
    expect(asksForLearning('Try once, but do not train or evaluate it yet.')).toBe(false);
    expect(asksForLearning("Use the learned policy, but don't retrain it.")).toBe(false);
  });

  it('drops the oldest remembered exchanges when a long command would pass the request limits', async () => {
    const history: ChatMessage[] = [];
    for (let k = 0; k < 4; k++) {
      history.push({ role: 'user', content: `request ${k} ${'x'.repeat(900)}` }, { role: 'assistant', content: `reply ${k} ${'y'.repeat(900)}` }, { role: 'system', content: 'Note: the reply above used no tool result; it contains no measurement.' });
    }
    const reading: ToolResult = { taskId: null, status: 'succeeded', reason: 'r'.repeat(5000), measurements: {}, fallback: null };
    const four = () => calls(['a', 'getRobotState', '{}'], ['b', 'getRobotState', '{}'], ['c', 'getRobotState', '{}'], ['d', 'getRobotState', '{}']);
    const { options, requests } = setup([four(), four(), four(), four(), text('ok')], { history, execute: async () => reading });
    const r = await runAiTurn(options);
    expect(r.end).toBe('answered');
    for (const q of requests) {
      expect(requestChars(q)).toBeLessThanOrEqual(MAX_TOTAL_CHARS);
      expect(q.messages.length).toBeLessThanOrEqual(MAX_MESSAGES);
    }
    const remembered = (q: ChatRequest) => q.messages.filter((m) => m.role === 'user' && m.content.startsWith('request ')).length;
    expect(remembered(requests[0]!)).toBe(4);
    expect(remembered(requests.at(-1)!)).toBeLessThan(4);
    // The command itself is never dropped.
    expect(requests.at(-1)!.messages.some((m) => m.role === 'user' && m.content === 'Drive to x 1, y 0.5.')).toBe(true);
  });

  it('counts the tool list as the proxy does, and shortens results already read instead of stopping a long command', async () => {
    // Seen in the browser: a command of ten model calls was refused by the proxy, which counts escaped characters.
    const len = (d: string) => JSON.stringify({ d }).length;
    expect(proxyJsonLength(JSON.stringify({ d: 'a+b' }))).toBe(len('a+b') + 5);
    expect(proxyJsonLength(JSON.stringify({ d: "it's <b>" }))).toBe(len("it's <b>") + 15);
    expect(proxyJsonLength(JSON.stringify({ d: 'say "hi"' }))).toBe(len('say "hi"') + 8);
    expect(proxyJsonLength(JSON.stringify({ d: '\u00e9' }))).toBe(len('\u00e9') + 5);
    const reading: ToolResult = { taskId: null, status: 'succeeded', reason: 'r'.repeat(5000), measurements: {}, fallback: null };
    const four = () => calls(['a', 'getRobotState', '{}'], ['b', 'getRobotState', '{}'], ['c', 'getRobotState', '{}'], ['d', 'getRobotState', '{}']);
    const { options, requests } = setup([four(), four(), four(), four(), four(), four(), text('ok')], { execute: async () => reading });
    const r = await runAiTurn(options);
    expect(r.end).toBe('answered');
    for (const q of requests) expect(requestChars(q)).toBeLessThanOrEqual(MAX_TOTAL_CHARS);
    const last = requests.at(-1)!;
    expect(last.messages.some((m) => m.role === 'tool' && m.content.includes('(shortened after it was read)'))).toBe(true);
    // The results the model has not read yet stay whole.
    const unread = last.messages.slice(last.messages.map((m) => m.role).lastIndexOf('assistant') + 1);
    expect(unread.filter((m) => m.role === 'tool').every((m) => !m.content.includes('shortened'))).toBe(true);
  });

  it('refuses arguments longer than the proxy accepts, saying how long they were', async () => {
    const long = JSON.stringify({ x: 1, y: 0, pad: 'a'.repeat(4100) });
    const { options, executed, requests } = setup([calls(['a', 'navigateTo', long]), text('Too long.')]);
    const r = await runAiTurn(options);
    expect(executed).toEqual([]);
    expect(r.steps[0]!.result.reason).toBe(`The arguments are ${long.length} characters long; at most 4000 fit. Send fewer or simpler poses.`);
    const assistant = requests[1]!.messages[2];
    expect(assistant?.role === 'assistant' && assistant.toolCalls?.[0]?.arguments.length).toBe(4000);
  });
});
