import { describe, expect, it } from 'vitest';
import type { RobotCapabilities } from '../../src/ai/capabilities.ts';
import type { AiAvailability } from '../../src/ai/client.ts';
import { AiController, type BridgeSnapshot, type RobotBridge } from '../../src/ai/controller.ts';
import type { ChatRequest, ChatResult } from '../../src/ai/protocol.ts';
import { rejected } from '../../src/ai/result.ts';
import { TaskManager, type FinalResult, type StopBehavior, type TaskExecutor } from '../../src/ai/tasks.ts';

const rover: RobotCapabilities = {
  name: 'Rover',
  morphologyHash: 'abc123',
  bodyKind: 'wheeled',
  jointCount: 4,
  skills: [],
  places: [],
  features: { pointing: false, navigation: true, sceneObjects: false, demonstrations: false, controllerExperiments: false, skillEvaluation: false, skillAuthoring: false, skillSequences: false, robotChoice: false, physicsTools: false, kinematicsTools: false, designTools: false },
};

const text = (content: string): ChatResult => ({ content, toolCalls: [], finishReason: 'stop', model: 'fake-model', usage: null });
const navigate = (x: number): ChatResult => ({ content: null, toolCalls: [{ id: 'call_1', name: 'navigateTo', arguments: JSON.stringify({ x, y: 0 }) }], finishReason: 'tool_calls', model: 'fake-model', usage: null });

function fakeClient(script: ChatResult[], status: AiAvailability = { state: 'available', provider: 'fake', capabilities: null }) {
  const requests: ChatRequest[] = [];
  let statusCalls = 0;
  return {
    requests,
    statusCalls: () => statusCalls,
    status: async () => {
      statusCalls++;
      return status;
    },
    chat: async (r: ChatRequest) => {
      requests.push(structuredClone(r));
      const next = script.shift();
      if (!next) throw new Error('script exhausted');
      return next;
    },
  };
}

function fakeBridge(run: (signal: AbortSignal) => Promise<FinalResult>) {
  const bridge = {
    key: 'navigate:1',
    unsupported: null as string | null,
    contextKey: () => bridge.key,
    async snapshot(): Promise<BridgeSnapshot | { unsupported: string }> {
      if (bridge.unsupported) return { unsupported: bridge.unsupported };
      return {
        caps: rover,
        summary: 'Robot: Rover in the Navigate tab.',
        query: () => ({ taskId: null, status: 'succeeded', reason: 'Idle.', measurements: {}, fallback: null }),
        action: (tool): TaskExecutor | ReturnType<typeof rejected> => (tool === 'navigateTo' ? { run } : rejected(`${tool} is not available.`)),
      };
    },
  };
  return bridge satisfies RobotBridge;
}

function setup(script: ChatResult[], run: (signal: AbortSignal) => Promise<FinalResult>, status?: AiAvailability) {
  const stops: StopBehavior[] = [];
  const tasks = new TaskManager({ stop: (b) => stops.push(b), body: () => 'wheeled' });
  let t = 0;
  const client = fakeClient(script, status);
  const ai = new AiController(client, tasks, () => t);
  return { ai, tasks, stops, client, bridge: fakeBridge(run), advance: (ms: number) => (t += ms) };
}

const succeed = async (): Promise<FinalResult> => ({ status: 'succeeded', reason: 'Reached the goal.', measurements: { groundTruthDistanceToGoal: { value: 0.1, unit: 'm' } }, fallback: null });

describe('AiController', () => {
  it('refuses without calling the model when AI is unavailable or the view has no tools', async () => {
    const off = setup([], succeed, { state: 'unavailable', reason: 'The AI proxy is not running here.' });
    expect(await off.ai.run('drive to the wall', off.bridge)).toEqual({ kind: 'unavailable', reason: 'The AI proxy is not running here.' });
    const on = setup([], succeed);
    on.bridge.unsupported = 'AI commands act on the Navigate and Skills tabs for now.';
    expect(await on.ai.run('drive', on.bridge)).toEqual({ kind: 'unsupported', reason: 'AI commands act on the Navigate and Skills tabs for now.' });
    expect(on.client.requests).toHaveLength(0);
  });

  it('runs a navigation task through the task manager and keeps the exchange as conversation memory', async () => {
    const s = setup([navigate(1), text('The rover reached the goal.')], succeed);
    const r = await s.ai.run('drive to x 1', s.bridge);
    expect(r.kind).toBe('turn');
    if (r.kind !== 'turn') return;
    expect(r.report.steps[0]?.result).toMatchObject({ taskId: 'task-1', status: 'succeeded', reason: 'Reached the goal.' });
    expect(s.tasks.history[0]).toMatchObject({ id: 'task-1', tool: 'navigateTo', status: 'succeeded' });
    expect(s.client.requests[0]?.messages[0]?.content).toContain('Robot: Rover in the Navigate tab.');
    expect(s.ai.history).toEqual([
      { role: 'user', content: 'drive to x 1' },
      { role: 'assistant', content: 'The rover reached the goal.' },
      { role: 'system', content: 'Note: the reply above was written from tool results measured then (navigateTo succeeded); it is not a current measurement.' },
    ]);
  });

  it('Stop cancels the running task with the body stop and ends the turn', async () => {
    let started!: () => void;
    const running = new Promise<void>((resolve) => (started = resolve));
    const s = setup([navigate(1), text('unused')], (signal) => {
      started();
      return new Promise<FinalResult>((resolve) => signal.addEventListener('abort', () => resolve({ status: 'cancelled', reason: 'late', measurements: {}, fallback: null })));
    });
    const turn = s.ai.run('drive', s.bridge);
    await running;
    expect(s.ai.busy).toBe(true);
    expect(await s.ai.run('again', s.bridge)).toEqual({ kind: 'busy' });
    expect(s.ai.stop()).toMatchObject({ taskId: 'task-1', status: 'cancelled', reason: 'Stopped with the Stop control.', fallback: 'The rover brakes to a stop.' });
    const r = await turn;
    expect(r.kind === 'turn' && r.report.end).toBe('cancelled');
    expect(s.stops).toEqual(['brake']);
    expect(s.ai.busy).toBe(false);
  });

  it('refuses actions planned before the robot or scene changed', async () => {
    const s = setup([navigate(1), text('The scene changed.')], succeed);
    const chat = s.client.chat;
    // The scene changes while the model is choosing its action.
    s.client.chat = async (r) => {
      s.bridge.key = 'navigate:2';
      return chat(r);
    };
    const r = await s.ai.run('drive', s.bridge);
    expect(r.kind === 'turn' && r.report.steps[0]?.result).toMatchObject({ status: 'rejected', reason: 'The robot or scene changed after this command was planned; ask again.' });
    expect(s.tasks.history).toHaveLength(0);
    expect(s.tasks.stateVersion).toBe(1);
  });

  it('forgets the conversation when the context changes, and caches the status for 30 s', async () => {
    const s = setup([text('Hello.'), text('Hi again.')], succeed);
    await s.ai.run('hello', s.bridge);
    expect(s.ai.history).toHaveLength(3);
    expect(s.ai.history[2]).toEqual({ role: 'system', content: 'Note: the reply above used no tool result; it contains no measurement.' });
    s.bridge.key = 'skills:7';
    await s.ai.run('hello again', s.bridge);
    expect(s.client.requests[1]?.messages.map((m) => m.role)).toEqual(['system', 'user']);
    expect(s.client.statusCalls()).toBe(1);
    s.advance(30_000);
    await s.ai.refreshStatus();
    expect(s.client.statusCalls()).toBe(2);
    await s.ai.refreshStatus(true);
    expect(s.client.statusCalls()).toBe(3);
  });

  it('keeps the last four exchanges whole, each reply followed by its evidence note', async () => {
    const n = ['1', '2', '3', '4', '5'];
    const s = setup(n.map((k) => text(`Reply ${k}.`)), succeed);
    for (const k of n) await s.ai.run(`hello ${k}`, s.bridge);
    expect(s.ai.history).toHaveLength(12);
    expect(s.ai.history.filter((m) => m.role === 'user').map((m) => m.content)).toEqual(['hello 2', 'hello 3', 'hello 4', 'hello 5']);
    expect(s.client.requests[4]?.messages.map((m) => m.role)).toEqual(['system', ...Array.from({ length: 4 }, () => ['user', 'assistant', 'system']).flat(), 'user']);
  });
});
