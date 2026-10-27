import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import type { BridgeSnapshot, RobotBridge } from '../../src/ai/controller';
import { LongTermMemory } from '../../src/ai/memory';
import type { ToolResult } from '../../src/ai/result';
import { TaskManager, type TaskExecutor } from '../../src/ai/tasks';
import { toolsForModel } from '../../src/ai/tools';
import { workspaceBridge } from '../../src/ui/ai/bridge';
import { ControlSession } from '../../src/ui/control/session';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

/** What the loop sends the model for one tool result; longer content would be cut. */
const MAX_TOOL_CONTENT = 1500;

async function lab(): Promise<{ s: ControlSession; bridge: RobotBridge }> {
  const s = new ControlSession(() => Promise.resolve(mj));
  await s.init();
  const bridge = workspaceBridge({
    mode: () => 'control',
    nav: () => null,
    skills: () => {
      throw new Error('not used');
    },
    memory: () => new LongTermMemory(),
    control: () => s,
  });
  return { s, bridge };
}

async function snapshotOf(bridge: RobotBridge): Promise<BridgeSnapshot> {
  const snap = await bridge.snapshot();
  if ('unsupported' in snap) throw new Error(snap.unsupported);
  return snap;
}

function executor(x: TaskExecutor | ToolResult): TaskExecutor {
  if (!('run' in x)) throw new Error(x.reason);
  return x;
}

describe('the experiment coach in the Control lab', () => {
  it('offers controller experiments and describes the setup and both gain sets', async () => {
    const { s, bridge } = await lab();
    const snap = await snapshotOf(bridge);
    expect(snap.caps).toMatchObject({ bodyKind: 'arm', features: { controllerExperiments: true, navigation: false } });
    const names = toolsForModel(snap.caps).map((t) => t.name);
    expect(names).toContain('runControllerExperiment');
    expect(names).not.toContain('navigateTo');
    expect(snap.summary).toContain('You are the experiment coach');
    expect(snap.summary).toContain("Gain set A (the learner's): kp 60, kd 3, without gravity compensation. Gain set B: kp 60, kd 3, with gravity compensation.");
    expect(snap.summary).toMatch(/Setup: joint "[^"]+: [^"]+" \(effort limit [\d.]+ N\*m, velocity limit [\d.]+ rad\/s\), step 20 deg, 2 s, target moved at the velocity limit\./);
    expect(snap.query('getRobotState', {}).reason).toContain('No step test results are shown');
    expect(bridge.contextKey()).toMatch(/^control:\d+:1:20:2:rate-limited$/);
    s.dispose();
  });

  it('runs an experiment on gain set B only and reports the measured metrics of both gain sets', async () => {
    const { s, bridge } = await lab();
    const key = bridge.contextKey();
    const snap = await snapshotOf(bridge);
    const r = await executor(snap.action('runControllerExperiment', { configuration: { kp: 120, kd: 8, gravityCompensation: true } })).run(new AbortController().signal);
    expect(r.status).toBe('succeeded');
    // Gain set A is the learner's and stays as it was.
    expect(s.state.a).toEqual({ gains: { kp: 60, kd: 3 }, gravityCompensation: false });
    expect(s.state.b).toEqual({ gains: { kp: 120, kd: 8 }, gravityCompensation: true });
    const { stepA, stepB } = s.state;
    expect(r.measurements['aOvershootPct']).toEqual({ value: Number(stepA!.metrics.overshootPct.toFixed(1)), unit: '%' });
    expect(r.measurements['bSteadyStateError']).toEqual({ value: Number(stepB!.metrics.steadyStateError.toFixed(4)), unit: 'rad' });
    // Physical check: gravity compensation and stiffer gains leave a smaller steady-state error than plain PD.
    expect(Math.abs(r.measurements['bSteadyStateError']!.value)).toBeLessThan(Math.abs(r.measurements['aSteadyStateError']!.value));
    expect(r.reason).toMatch(/^Step test of joint "[^"]+", step 20 deg, 2 s, target moved at the velocity limit\. A \(kp 60, kd 3, without gravity compensation\): rise .+; B \(kp 120, kd 8, with gravity compensation\): rise .+, damping ratio \d+\.\d\d\. Gain set B was kp 60, kd 3, with gravity compensation before this experiment\. This run changed more than one value at once \(from A: kp, kd and gravity compensation; from B's earlier settings: kp and kd\), so it does not show the effect of one change alone\.$/);
    expect(JSON.stringify({ taskId: 'task-1', ...r }).length).toBeLessThan(MAX_TOOL_CONTENT);
    // The coach's experiments change gain set B, which is not part of the context key.
    expect(bridge.contextKey()).toBe(key);
    const state = (await snapshotOf(bridge)).query('getRobotState', {});
    expect(state.reason).toContain('Shown results: Step test of joint');
    expect(state.measurements).toEqual(r.measurements);
    expect(JSON.stringify(state).length).toBeLessThan(MAX_TOOL_CONTENT);
    s.dispose();
  });

  it('without a configuration runs the lab as set, changes nothing and says why plain PD never settles', async () => {
    const { s, bridge } = await lab();
    const snap = await snapshotOf(bridge);
    const before = { a: s.state.a, b: s.state.b, compare: s.state.compare };
    const r = await executor(snap.action('runControllerExperiment', {})).run(new AbortController().signal);
    expect(r.status).toBe('succeeded');
    expect({ a: s.state.a, b: s.state.b, compare: s.state.compare }).toEqual(before);
    expect(r.reason).not.toContain('Gain set B was');
    const [aPart, bPart] = r.reason.split('; B (');
    // Plain PD on a loaded joint rests where kp times the error balances gravity, outside the 2 % band.
    expect(aPart).toMatch(/A \(kp 60, kd 3, without gravity compensation\): .*settling not settled \(came to rest \d+ % of the step off target, outside the 2 % band\)/);
    expect(s.state.stepA!.metrics.settlingTimeS).toBeNull();
    // Gravity compensation removes that sag.
    expect(bPart).toMatch(/^kp 60, kd 3, with gravity compensation\): /);
    expect(bPart).not.toContain('came to rest');
    expect(Math.abs(s.state.stepB!.metrics.steadyStateError)).toBeLessThan(Math.abs(s.state.stepA!.metrics.steadyStateError));

    s.setCompare(false);
    const single = await executor(snap.action('runControllerExperiment', {})).run(new AbortController().signal);
    expect(single.status).toBe('succeeded');
    expect(single.reason).not.toContain('; B (');
    expect(s.state).toMatchObject({ compare: false, stepB: null, stepSetup: { b: null } });
    expect(Object.keys(single.measurements).every((k) => k.startsWith('a'))).toBe(true);
    s.dispose();
  });

  it('one change from the shown gain set B is not flagged as several changes', async () => {
    const { s, bridge } = await lab();
    const snap = await snapshotOf(bridge);
    // B starts as kp 60, kd 3 with gravity compensation; only kd changes.
    const r = await executor(snap.action('runControllerExperiment', { configuration: { kp: 60, kd: 6, gravityCompensation: true } })).run(new AbortController().signal);
    expect(r.status).toBe('succeeded');
    expect(r.reason).toContain('Gain set B was kp 60, kd 3, with gravity compensation before this experiment.');
    expect(r.reason).not.toContain('more than one value');
    expect(s.state.a).toEqual({ gains: { kp: 60, kd: 3 }, gravityCompensation: false });
    s.dispose();
  });

  it('a Stop or a setup change ends the experiment without reporting partial or older results', async () => {
    const { s, bridge } = await lab();
    const snap = await snapshotOf(bridge);
    const tasks = new TaskManager({ stop: () => {}, body: () => 'arm' });
    const run = tasks.start('runControllerExperiment', tasks.stateVersion, executor(snap.action('runControllerExperiment', { configuration: { kp: 90, kd: 5, gravityCompensation: false } })));
    expect(tasks.stopAll()).toMatchObject({ status: 'cancelled', fallback: 'The experiment stopped; its unfinished results were discarded.' });
    expect((await run.done).status).toBe('cancelled');
    await new Promise((r) => setTimeout(r, 20));
    expect(s.state.running).toBeNull();
    expect(s.state.stepA).toBeNull();

    const pending = executor(snap.action('runControllerExperiment', { configuration: { kp: 90, kd: 5, gravityCompensation: false } })).run(new AbortController().signal);
    s.setStep(30);
    expect(await pending).toMatchObject({ status: 'failed', reason: 'The step test did not finish: the lab setup changed or the run was stopped.' });
    s.dispose();
  });
});
