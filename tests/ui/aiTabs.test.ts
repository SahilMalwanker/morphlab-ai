import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import type { BridgeSnapshot } from '../../src/ai/controller';
import { LongTermMemory } from '../../src/ai/memory';
import type { ToolResult } from '../../src/ai/result';
import type { TaskExecutor } from '../../src/ai/tasks';
import { toolsForModel } from '../../src/ai/tools';
import { PRESETS } from '../../src/core/model/catalog';
import { quadruped } from '../../src/core/model/presets';
import { workspaceBridge, type WorkspaceAccess } from '../../src/ui/ai/bridge';
import { BuilderSession } from '../../src/ui/builder/session';
import { KinematicsSession } from '../../src/ui/kinematics/session';
import type { Mode } from '../../src/ui/modes';
import { PhysicsSession } from '../../src/ui/physics/session';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const memory = new LongTermMemory([], () => '2026-10-06T12:00:00.000Z');

function bridgeFor(mode: Mode, parts: Partial<WorkspaceAccess>) {
  return workspaceBridge({
    mode: () => mode,
    nav: () => null,
    skills: () => {
      throw new Error('The Skills session is not used here.');
    },
    memory: () => memory,
    control: () => null,
    ...parts,
  });
}

async function snapshot(mode: Mode, parts: Partial<WorkspaceAccess>): Promise<BridgeSnapshot> {
  const snap = await bridgeFor(mode, parts).snapshot();
  if ('unsupported' in snap) throw new Error(snap.unsupported);
  return snap;
}

function immediate(x: TaskExecutor | ToolResult): ToolResult {
  if ('run' in x) throw new Error('Expected an immediate result, got a task.');
  return x;
}

function task(x: TaskExecutor | ToolResult): TaskExecutor {
  if (!('run' in x)) throw new Error(x.reason);
  return x;
}

describe('AI tools in the Kinematics tab (geometry only)', () => {
  it('moves a limb tip by inverse kinematics and reports the solver status; joints stop at their limits', async () => {
    const arm = PRESETS.find((p) => p.id === 'arm3')!.make();
    const k = new KinematicsSession(arm);
    const snap = await snapshot('kinematics', { kinematics: () => k });
    expect(snap.caps.features.kinematicsTools).toBe(true);
    expect(snap.caps.morphologyHash).toMatch(/^[0-9a-f]{64}$/);
    expect(snap.summary).toContain('geometry only');
    const T = k.state.fk.tips[0]!;
    const reached = immediate(snap.action('moveTipTo', { limb: 'arm', x: T[12]! + 0.05, y: T[13]!, z: T[14]! - 0.05 }));
    expect(reached.status).toBe('succeeded');
    expect(reached.reason).toMatch(/^IK for the arm tip: converged\./);
    expect(reached.measurements['residual']!.value).toBeLessThan(0.001);
    const far = immediate(snap.action('moveTipTo', { x: 2.5, y: 0, z: 0.5 }));
    expect(far.status).toBe('failed');
    expect(far.reason).toContain('unreachable');
    expect(immediate(snap.action('moveTipTo', { limb: 'tail', x: 0, y: 0, z: 0.5 })).status).toBe('rejected');
    const set = immediate(snap.action('setJointAngles', { joints: [{ joint: 'shoulder', value: 6 }] }));
    expect(set.reason).toBe('Set 1 joint; shoulder stopped at a joint limit.');
    expect(k.state.q[1]).toBe(1.75);
    expect(immediate(snap.action('setJointAngles', { joints: [{ joint: 'wrist', value: 0.1 }] })).status).toBe('rejected');

    // Wheel-only bodies have no targetable limb in Kinematics.
    k.setRobot(PRESETS.find((p) => p.id === 'rover')!.make(), 'rover');
    const roverSnap = await snapshot('kinematics', { kinematics: () => k });
    expect(roverSnap.caps.bodyKind).toBe('wheeled');
    expect(roverSnap.caps.morphologyHash).toMatch(/^[0-9a-f]{64}$/);
    expect(toolsForModel(roverSnap.caps).map((tool) => tool.name)).not.toContain('moveTipTo');
    expect(roverSnap.query('getRobotState', {}).reason).not.toContain('IK target');
    expect(roverSnap.query('getRobotCapabilities', {}).reason).toContain('Tools here: getRobotState, setJointAngles.');
    expect(immediate(roverSnap.action('moveTipTo', { limb: 'front_left_wheel', x: 0.2, y: 0.1, z: 0 }))).toMatchObject({
      status: 'rejected',
      reason: 'This robot has no limb whose tip can be placed.',
      fallback: null,
    });
    k.setRobot({ ...k.state.description, limbs: [...k.state.description.limbs, ...arm.limbs] });
    const mobileArm = await snapshot('kinematics', { kinematics: () => k });
    expect(mobileArm.caps.bodyKind).toBe('wheeled');
    expect(toolsForModel(mobileArm.caps).map((tool) => tool.name)).toContain('moveTipTo');
  });
});

describe('AI tools in the Physics tab', () => {
  it('pushes the robot and reports its measured response; refuses while paused; resets', async () => {
    const p = new PhysicsSession(() => Promise.resolve(mj), 'quadruped');
    await p.init();
    const snap = await snapshot('physics', { physics: () => p });
    expect(snap.caps.features.physicsTools).toBe(true);
    expect(snap.query('getRobotState', {}).reason).toMatch(/^The simulation is running\. Body height 0\.\d{3} m, tilt \d+\.\d deg \(simulator ground truth\)\.$/);
    const done = task(snap.action('pushRobot', { direction: 'left', impulseNs: 1 })).run(new AbortController().signal);
    // The task watches 2 s of simulated time; step it as the Physics view does.
    const steps = Math.ceil(2.2 / p.runtime!.timestep);
    for (let i = 0; i < steps; i++) p.stepOnce();
    const r = await done;
    expect(r.status).toBe('succeeded');
    expect(r.reason).toMatch(/^Pushed left with 1 N\*s\. Over the next 2\.\d s the body tilted up to \d+\.\d deg and its lowest height was 0\.\d{3} m; it (ended at|fell over)/);
    expect(r.measurements['groundTruthMaxTilt']!.value).toBeGreaterThan(0);
    expect(r.measurements['largestJointErrorFromTarget']!.unit).toBe('rad');

    const cancel = new AbortController();
    const cancelled = task(snap.action('pushRobot', { direction: 'forward', impulseNs: 0.5 })).run(cancel.signal);
    cancel.abort();
    expect((await cancelled).status).toBe('cancelled');

    p.setRunning(false);
    expect(await task(snap.action('pushRobot', { direction: 'forward', impulseNs: 1 })).run(new AbortController().signal)).toMatchObject({ status: 'failed', reason: 'The Physics simulation is paused.' });
    p.setRunning(true);

    expect(immediate(snap.action('resetSimulation', {})).status).toBe('succeeded');
    expect(p.runtime!.time).toBe(0);
    p.dispose();
  });
});

describe('AI tools in the Build tab', () => {
  it('designs robots through the validated generators and changes nothing when a design is refused', async () => {
    const b = new BuilderSession(quadruped());
    const snap = await snapshot('build', { builder: () => b });
    expect(snap.caps.features.designTools).toBe(true);
    const hex = immediate(snap.action('buildRobot', { preset: 'hexapod' }));
    expect(hex.status).toBe('succeeded');
    expect(hex.reason).toMatch(/^Built Hexapod: floating base of 3 kg, 6 legs, 18 joints \(version 2\)\./);
    const legged = immediate(
      snap.action('buildRobot', { legged: { pairs: 3, dofPerLeg: 2, bodyLength: 0.5, bodyWidth: 0.2, bodyHeight: 0.08, bodyMass: 5, thigh: 0.15, shank: 0.15, effort: 15 }, name: 'Strider' }),
    );
    expect(legged.status).toBe('succeeded');
    expect(b.state.robot.name).toBe('Strider');
    expect(b.state.robot.limbs).toHaveLength(6);
    expect(immediate(snap.action('buildRobot', { arm: { dof: 4, reach: 0.8 } })).reason).toMatch(/^Built 4-joint arm: fixed base of 2 kg, 1 arm, 4 joints/);
    const version = b.state.version;
    const robot = b.state.robot;
    const refused = immediate(snap.action('buildRobot', { legged: { pairs: 2, dofPerLeg: 3, bodyLength: 0.4, bodyWidth: 0.2, bodyHeight: 0.08, bodyMass: 4, thigh: 0.16, shank: 0.16, effort: 0 } }));
    expect(refused.status).toBe('rejected');
    expect(refused.reason).toBe('The design was refused: Joint effort limit (N*m) must be greater than 0 and at most 1000.');
    expect(b.state.version).toBe(version);
    expect(b.state.robot).toBe(robot);
  });
});

describe('the bridge across tabs', () => {
  it('keys the conversation to the tab and its robot, and reports a tab without its session', async () => {
    const k = new KinematicsSession(quadruped());
    const bridge = bridgeFor('kinematics', { kinematics: () => k });
    const key = bridge.contextKey();
    expect(key).toMatch(/^kinematics:\d+$/);
    k.setRobot(PRESETS.find((p) => p.id === 'arm3')!.make());
    expect(bridge.contextKey()).not.toBe(key);
    expect(bridgeFor('build', {}).contextKey()).toBe('build');
    expect(bridgeFor('physics', {}).contextKey()).toBe('physics:none');
    expect(await bridgeFor('physics', {}).snapshot()).toEqual({ unsupported: 'The Physics tab is not ready for AI commands yet.' });
  });
});
