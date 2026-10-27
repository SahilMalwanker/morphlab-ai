import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import type { BridgeSnapshot, RobotBridge } from '../../src/ai/controller';
import { LongTermMemory } from '../../src/ai/memory';
import type { ToolResult } from '../../src/ai/result';
import type { FinalResult, TaskExecutor } from '../../src/ai/tasks';
import { toolsForModel } from '../../src/ai/tools';
import { morphologyHash } from '../../src/core/model/hash';
import { hexapod, quadruped } from '../../src/core/model/presets';
import type { LocomotionController } from '../../src/skills/locomotion';
import { workspaceBridge } from '../../src/ui/ai/bridge';
import { NavSession } from '../../src/ui/nav/session';
import { testLocomotionResult, testWalkingController } from '../nav/walkingFixture';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

async function snapshotOf(bridge: RobotBridge): Promise<BridgeSnapshot> {
  const snap = await bridge.snapshot();
  if ('unsupported' in snap) throw new Error(snap.unsupported);
  return snap;
}

const finish = (x: TaskExecutor | ToolResult): Promise<FinalResult | ToolResult> => ('run' in x ? x.run(new AbortController().signal) : Promise.resolve(x));
const toolNames = (snap: BridgeSnapshot) => toolsForModel(snap.caps).map((t) => t.name);

describe('the AI in Navigate with legged robots', () => {
  it('switches the robot, trains its walking controller from Navigate, and says how it walks', async () => {
    const dogHash = await morphologyHash(quadruped());
    let stored: LocomotionController | null = null;
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    s.attachWalking({
      controllerFor: (h) => (h === dogHash ? stored : null),
      train: async (robot) => {
        stored = testWalkingController(robot, dogHash);
        return { ok: true, controller: stored, result: testLocomotionResult(robot), wallTimeS: 2 };
      },
    });
    const bridge = workspaceBridge({ mode: () => 'navigate', nav: () => s, skills: () => { throw new Error('not used'); }, memory: () => new LongTermMemory(), control: () => null, builtRobot: () => hexapod() });
    let snap = await snapshotOf(bridge);
    expect(toolNames(snap)).toEqual(expect.arrayContaining(['navigateTo', 'selectRobot']));
    expect(toolNames(snap)).not.toContain('trainLocomotion');
    expect(await finish(snap.action('selectRobot', { robot: 'arm3' }))).toMatchObject({
      status: 'rejected',
      reason: 'Three-joint arm cannot navigate: its base is fixed in place, so it cannot move through the arena. Wheeled rover stays loaded.',
    });
    expect(await finish(snap.action('selectRobot', { robot: 'quadruped' }))).toMatchObject({
      status: 'succeeded',
      reason: 'Navigate now shows Quadruped (4 legs; it walks with the trot gait generator steered by the planner; no walking controller is trained for this body yet.) at the start of the map "Single block".',
    });

    snap = await snapshotOf(bridge);
    expect(snap.caps).toMatchObject({ name: 'Quadruped', bodyKind: 'legged', walking: null });
    expect(toolNames(snap)).toContain('trainLocomotion');
    expect(toolNames(snap)).not.toContain('locomote');
    expect(snap.summary).toContain('It walks with the trot gait generator steered by the planner');
    expect(snap.summary).toContain('A request to learn to walk, to train its walking, or for a learned, trained or AI walking controller needs a trainLocomotion call');
    // Seen in the browser: the model answered "learn to walk" without calling the tool; it is reminded once.
    expect(snap.followUp?.([], 'learn to walk')).toMatch(/^The user asked the robot to learn to walk\. Call trainLocomotion now/);
    expect(snap.followUp?.([], 'walk to the goal')).toBeNull();
    expect(snap.followUp?.([], 'how would it learn to walk?')).toBeNull();
    expect(snap.followUp?.([], 'please learn to walk better')).toMatch(/^The user asked the robot to learn to walk/);
    const trained = await finish(snap.action('trainLocomotion', {}));
    expect(trained.status).toBe('succeeded');
    expect(trained.reason).toMatch(/^Trained and saved this body's walking controller .* Quadruped walks planned paths in Navigate with it from the next time it stands\.$/);

    snap = await snapshotOf(bridge);
    expect(snap.caps.walking).toEqual(stored!.ranges);
    expect(snap.summary).toContain('It walks with its learned walking controller (trained 2026-10-16): the calibrated gait generator');
    expect(snap.followUp?.([], 'learn to walk')).toBeNull();
    expect(await finish(snap.action('selectRobot', { robot: 'quadruped' }))).toMatchObject({ status: 'succeeded', reason: 'Navigate already shows Quadruped.' });
    const built = await finish(snap.action('selectRobot', { robot: 'built' }));
    expect(built.reason).toMatch(/^Navigate now shows Hexapod \(6 legs; it walks with the trot gait generator/);
    expect(s.state.robotSource).toBe('custom');
    s.dispose();
  }, 120_000);
});
