import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { morphologyHash } from '../../src/core/model/hash';
import { hexapod, quadruped, rover } from '../../src/core/model/presets';
import type { JointDescription, LimbDescription, RobotDescription } from '../../src/core/model/types';
import { wrapAngle } from '../../src/nav/follower';
import { handleNavJob, type NavWorkerRequest } from '../../src/nav/jobs';
import { TEST_MAPS } from '../../src/nav/maps';
import { DEFAULT_REFEREE, NavReferee } from '../../src/nav/referee';
import type { NavInput } from '../../src/nav/stack';
import { groundTruthPose, type NavRunResult } from '../../src/physics/navigation';
import { JobWorkerClient, type JobMessage } from '../../src/skills/workerClient';
import { NavSession } from '../../src/ui/nav/session';
import { testLocomotionResult, testWalkingController } from '../nav/walkingFixture';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

function run(s: NavSession, seconds: number): void {
  const n = Math.round(seconds / s.runtime!.timestep);
  for (let i = 0; i < n; i++) s.stepOnce();
}

const occupied = (s: NavSession) => s.sensorOnlyStack!.grid.logOdds.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);

/** The rover with every wheel joint changed by `edit`. */
function editedRover(id: string, name: string, edit: (j: JointDescription) => JointDescription): RobotDescription {
  const r = rover();
  return { ...r, id, name, limbs: r.limbs.map((l): LimbDescription => ({ ...l, joints: l.joints.map(edit) })) };
}

/** A rover edited in Build: wheels moved out to 0.25 m (a 0.5 m track). */
const wideRover = (): RobotDescription =>
  editedRover('wide_rover', 'Wide rover', (j) => ({ ...j, origin: { ...j.origin, xyz: [j.origin.xyz[0], Math.sign(j.origin.xyz[1]) * 0.25, j.origin.xyz[2]] } }));

/** The rover with every wheel axis reversed: a positive joint velocity now drives it backward. */
const reversedRover = (): RobotDescription => editedRover('reversed_rover', 'Reversed-axis rover', (j) => ({ ...j, axis: [0, -1, 0] }));

describe('live navigation keeps sensor-derived state across transitions', () => {
  it('remote driving, goal changes, restarts and mode switches never re-initialize the sensor-only estimate', async () => {
    const inputs: { stack: string; input: NavInput }[] = [];
    const s = new NavSession(() => Promise.resolve(mj), (stack, input) => inputs.push({ stack, input }));
    await s.init();
    const gen = s.generation;
    const estimator = s.sensorOnlyStack!;
    // Explicit reset: the estimate starts at the map's declared start pose.
    expect(estimator.pose).toEqual(s.map.start);

    s.setRemote(0.3, 0.6);
    run(s, 2);
    s.stop();
    run(s, 0.3);
    const afterRemote = { ...estimator.pose };
    const truth = groundTruthPose(s.runtime!);
    // Odometry followed the motion from sensors; it is close to, but not copied from, the simulator pose.
    expect(Math.hypot(afterRemote.x - s.map.start.x, afterRemote.y - s.map.start.y)).toBeGreaterThan(0.3);
    expect(Math.hypot(afterRemote.x - truth.x, afterRemote.y - truth.y)).toBeLessThan(0.05);
    expect(afterRemote).not.toEqual({ x: truth.x, y: truth.y, yaw: truth.yaw });
    const mapped = occupied(s);
    expect(mapped).toBeGreaterThan(0);

    s.startAuto();
    expect(s.sensorOnlyStack).toBe(estimator);
    expect(estimator.pose).toEqual(afterRemote);
    run(s, 1);
    expect(s.setGoal(2.5, 1.2)).toBe(true);
    expect(s.sensorOnlyStack).toBe(estimator);
    expect(occupied(s)).toBeGreaterThanOrEqual(mapped);
    s.setMode('ground-truth');
    run(s, 0.5);
    s.setMode('sensor-only');
    expect(s.sensorOnlyStack).toBe(estimator);
    expect(s.state.driving).toBe('auto');
    expect(s.generation).toBe(gen);

    // Only an explicit reset starts over, from the declared start pose.
    s.reset();
    expect(s.generation).toBe(gen + 1);
    expect(s.sensorOnlyStack).not.toBe(estimator);
    expect(s.sensorOnlyStack!.pose).toEqual(s.map.start);
    expect(occupied(s)).toBe(0);

    const sensorInputs = inputs.filter((i) => i.stack === 'sensor-only');
    expect(sensorInputs.length).toBeGreaterThan(100);
    expect(sensorInputs.every((i) => !('groundTruth' in i.input))).toBe(true);
    s.dispose();
  });

  it('after remote driving the sensor-only rover still reaches the goal from its own estimate', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    s.setMap('open');
    s.setRemote(0.3, -0.4);
    run(s, 1.5);
    s.stop();
    s.startAuto();
    for (let i = 0; i < 40 && s.state.driving === 'auto'; i++) run(s, 0.5);
    expect(s.state.collided).toBe(false);
    expect(s.state.reached).toBe(true);
    expect(s.state.message).toMatch(/^Reached the goal/);
    s.dispose();
  });
});

describe('choosing the robot in Navigate', () => {
  it('refuses fixed bodies with the reason and keeps the rover', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    const gen = s.generation;
    expect(s.setPreset('arm3')).toMatchObject({ ok: false, reason: 'Three-joint arm cannot navigate: its base is fixed in place, so it cannot move through the arena. Wheeled rover stays loaded.' });
    expect(s.state.message).toBe('Three-joint arm cannot navigate: its base is fixed in place, so it cannot move through the arena. Wheeled rover stays loaded.');
    expect(s.generation).toBe(gen);
    expect(s.state).toMatchObject({ robotSource: 'rover', robotName: 'Wheeled rover', status: 'ready' });
    s.dispose();
  });

  it('a wider rover from Build drives with its own track: odometry follows the simulator', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    const gen = s.generation;
    expect(s.setRobot(wideRover())).toEqual({ ok: true });
    expect(s.generation).toBe(gen + 1);
    expect(s.state).toMatchObject({ robotSource: 'custom', robotName: 'Wide rover', status: 'ready' });
    expect(s.runtime!.description.limbs[0]!.joints[0]!.origin.xyz[1]).toBe(0.25);
    expect(s.drive.kind === 'wheeled' && s.drive.trackWidth).toBe(0.5);
    s.setRemote(0.3, 0.6);
    run(s, 2);
    s.stop();
    run(s, 0.3);
    const est = s.sensorOnlyStack!.pose;
    const truth = groundTruthPose(s.runtime!);
    expect(Math.hypot(est.x - s.map.start.x, est.y - s.map.start.y)).toBeGreaterThan(0.3);
    // It turned (0.6 rad/s commanded for 2 s), and the IMU heading with the 0.5 m track kept odometry on the simulator.
    expect(Math.abs(truth.yaw)).toBeGreaterThan(0.3);
    expect(Math.hypot(est.x - truth.x, est.y - truth.y)).toBeLessThan(0.05);
    s.dispose();
  });

  it('reversed wheel axes still drive forward: commands and odometry use each wheel sign', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    expect(s.setRobot(reversedRover())).toEqual({ ok: true });
    const x0 = groundTruthPose(s.runtime!).x;
    s.setRemote(0.3, 0);
    run(s, 1.5);
    s.stop();
    run(s, 0.3);
    const truth = groundTruthPose(s.runtime!);
    expect(truth.x - x0).toBeGreaterThan(0.3);
    expect(Math.abs(s.sensorOnlyStack!.pose.x - truth.x)).toBeLessThan(0.05);
    s.dispose();
  });

  it('a batch evaluation starts with the robot selected at that moment', async () => {
    const posted: NavWorkerRequest[] = [];
    const capture = { onmessage: null, onerror: null, postMessage: (r: NavWorkerRequest) => void posted.push(r), terminate() {} };
    const s = new NavSession(() => Promise.resolve(mj), undefined, new JobWorkerClient<NavWorkerRequest>(() => capture as unknown as Worker, 'Navigation worker'));
    await s.init();
    s.setRobot(wideRover());
    const p = s.evaluateAll();
    expect(posted[0]?.robot?.name).toBe('Wide rover');
    expect(s.state.evaluation).toMatchObject({ running: true, robotName: 'Wide rover' });
    s.cancelEvaluation();
    await p;
    s.dispose();
  });

  it('the worker drives the selected robot to the goal (sensor-only, ground-truth referee)', async () => {
    const messages: JobMessage[] = [];
    await handleNavJob(() => Promise.resolve(mj), { id: 7, type: 'evaluate', mode: 'sensor-only', mapIds: ['open', 'block'], robot: wideRover() }, (m) => void messages.push(m));
    const last = messages.at(-1)!;
    if (last.type !== 'result') throw new Error(last.type === 'error' ? last.message : 'no result');
    const results = last.result as NavRunResult[];
    expect(results.map((r) => ({ map: r.mapId, reached: r.reached, collided: r.collided }))).toEqual([
      { map: 'open', reached: true, collided: false },
      { map: 'block', reached: true, collided: false },
    ]);
  }, 120_000);
});

describe('walking legged robots in Navigate', () => {
  it('the quadruped walks and turns under keyboard command; leg odometry follows the simulator', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    expect(s.setPreset('quadruped')).toEqual({ ok: true });
    expect(s.drive.kind).toBe('legged');
    expect(s.state).toMatchObject({ robotSource: 'quadruped', robotName: 'Quadruped', status: 'ready', speedLimit: 0.15 });
    run(s, 0.5);
    const start = groundTruthPose(s.runtime!);
    // Keyboard commands are clamped to the walking limits.
    s.setRemote(1, 0);
    expect(s.state.remote).toEqual({ v: 0.15, omega: 0 });
    run(s, 4);
    s.setRemote(0, 1);
    expect(s.state.remote).toEqual({ v: 0, omega: 0.5 });
    run(s, 2);
    s.stop();
    run(s, 2.5);
    const truth = groundTruthPose(s.runtime!);
    const est = s.sensorOnlyStack!.pose;
    const dist = Math.hypot(truth.x - start.x, truth.y - start.y);
    console.log('Quadruped, keyboard', JSON.stringify({ distM: +dist.toFixed(3), turnRad: +wrapAngle(truth.yaw - start.yaw).toFixed(3), estimateErrorM: +Math.hypot(est.x - truth.x, est.y - truth.y).toFixed(3) }));
    expect(s.state.fallen).toBe(false);
    expect(truth.x - start.x).toBeGreaterThan(0.25);
    expect(wrapAngle(truth.yaw - start.yaw)).toBeGreaterThan(0.5);
    // Leg odometry within a quarter of the distance plus 2 cm (the walking controller's test tolerance); the heading comes from the IMU.
    expect(Math.hypot(est.x - truth.x, est.y - truth.y)).toBeLessThan(0.25 * dist + 0.02);
    expect(Math.abs(wrapAngle(est.yaw - truth.yaw))).toBeLessThan(0.05);
    // Stopped and standing again.
    expect(s.runtime!.groundTruthBaseState().tilt).toBeLessThan(0.2);
    s.dispose();
  }, 120_000);

  it('the quadruped walks to the goal from its own sensor-only estimate', async () => {
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    s.setPreset('quadruped');
    s.setMap('open');
    s.startAuto();
    for (let i = 0; i < 130 && s.state.driving === 'auto'; i++) run(s, 1);
    console.log('Quadruped, live, open floor', JSON.stringify({ message: s.state.message, t: +s.runtime!.time.toFixed(2) }));
    expect(s.state.fallen).toBe(false);
    expect(s.state.collided).toBe(false);
    expect(s.state.reached).toBe(true);
    expect(s.state.message).toMatch(/^Reached the goal/);
    s.dispose();
  }, 300_000);

  it('walks with the walking controller stored for its morphology, and trains one from Navigate', async () => {
    const dog = quadruped(), hex = hexapod();
    const [dogHash, hexHash] = await Promise.all([morphologyHash(dog), morphologyHash(hex)]);
    const stored = testWalkingController(dog, dogHash);
    const trained = testWalkingController(hex, hexHash);
    const progress: number[] = [];
    const s = new NavSession(() => Promise.resolve(mj));
    await s.init();
    s.attachWalking({
      controllerFor: (h) => (h === dogHash ? stored : null),
      train: async (robot, onProgress) => {
        onProgress(0.5, 'Half way');
        progress.push(s.state.walkingJob?.fraction ?? -1);
        return robot.id === 'hexapod' ? { ok: true, controller: trained, result: testLocomotionResult(hex), wallTimeS: 1 } : { ok: false, reason: 'Not this body.' };
      },
    });
    expect(s.state.walker).toBeNull();
    expect(await s.trainWalking()).toEqual({ ok: false, reason: 'Wheeled rover rolls on wheels; walking controllers are trained for legged bodies.' });
    expect(s.setPreset('quadruped')).toEqual({ ok: true });
    await s.refreshWalking();
    expect(s.walkingController).toBe(stored);
    expect(s.state.walker).toMatchObject({ learned: true, text: expect.stringMatching(/^its learned walking controller \(trained 2026-10-16\)/) });
    run(s, 0.5);
    const x0 = groundTruthPose(s.runtime!).x;
    s.setRemote(0.15, 0);
    run(s, 3);
    expect(groundTruthPose(s.runtime!).x - x0).toBeGreaterThan(0.15);
    expect(s.state.fallen).toBe(false);
    s.stop();
    // The hexapod has no walking controller until one is trained from Navigate.
    expect(s.setPreset('hexapod')).toEqual({ ok: true });
    await s.refreshWalking();
    expect(s.state.walker).toMatchObject({ learned: false });
    expect(s.walkingController).toBeNull();
    const r = await s.trainWalking();
    expect(r.ok).toBe(true);
    expect(progress).toEqual([0.5]);
    expect(s.walkingController).toBe(trained);
    expect(s.state).toMatchObject({ walker: { learned: true }, walkingJob: null, message: 'Walking controller trained and saved for Hexapod: 40 of 40 randomized walking commands ended within tolerance. It walks from the next time the robot stands.' });
    s.dispose();
  }, 120_000);

  it('the worker walks the quadruped and the hexapod to the goal (sensor-only, ground-truth referee)', async () => {
    for (const robot of [quadruped(), hexapod()]) {
      const messages: JobMessage[] = [];
      await handleNavJob(() => Promise.resolve(mj), { id: 8, type: 'evaluate', mode: 'sensor-only', mapIds: ['open', 'block'], robot }, (m) => void messages.push(m));
      const last = messages.at(-1)!;
      if (last.type !== 'result') throw new Error(last.type === 'error' ? last.message : 'no result');
      const results = last.result as NavRunResult[];
      console.log(robot.name, JSON.stringify(results.map((r) => ({ map: r.mapId, outcome: r.outcome, t: r.timeS, d: r.finalDistanceM, est: r.estimateErrorM, replans: r.replans }))));
      expect(results.map((r) => ({ map: r.mapId, reached: r.reached, outcome: r.outcome }))).toEqual([
        { map: 'open', reached: true, outcome: 'reached' },
        { map: 'block', reached: true, outcome: 'reached' },
      ]);
      // Leg odometry with rolling feet: within 0.1 m of ground truth after about 3 m (without the correction the quadruped ended 0.23 to 0.27 m off).
      expect(Math.max(...results.map((r) => r.estimateErrorM))).toBeLessThan(0.1);
    }
  }, 600_000);
});

describe('navigation referee (shared by the live view and the batch evaluation)', () => {
  const map = TEST_MAPS.find((m) => m.id === 'block')!;
  it('ends runs on arrival, short stops, collisions, falls, missing paths, time limits and numerical faults', () => {
    const goal = map.goal;
    const at = (x: number, y: number) => ({ x, y, yaw: 0 });
    expect(new NavReferee(map, goal, 0).check(1, at(goal[0] - 0.1, goal[1]), 'reached', true)?.outcome).toBe('reached');
    expect(new NavReferee(map, goal, 0).check(1, at(goal[0] - 0.6, goal[1]), 'reached', true)?.outcome).toBe('stopped-short');
    expect(new NavReferee(map, goal, 0).check(1, at(1.5, 0), 'following', true)?.outcome).toBe('collision');
    expect(new NavReferee(map, goal, 0).check(1, at(0, 0), 'following', true, false)).toMatchObject({ outcome: 'fell', message: 'Fell over (detected from simulator state).' });
    expect(new NavReferee(map, goal, 0).check(1, at(0, 0), 'following', false)?.outcome).toBe('numerical-instability');
    const r = new NavReferee(map, goal, 10);
    expect(r.check(10.5, at(0, 0), 'no-path', true)).toBeNull();
    expect(r.check(12.0, at(0, 0), 'following', true)).toBeNull();
    expect(r.check(12.5, at(0, 0), 'no-path', true)).toBeNull();
    expect(r.check(12.5 + DEFAULT_REFEREE.noPathTimeoutS + 0.01, at(0, 0), 'no-path', true)?.outcome).toBe('no-path');
    expect(new NavReferee(map, goal, 10).check(10 + DEFAULT_REFEREE.timeLimitS, at(0, 0), 'following', true)?.outcome).toBe('timeout');
  });
});

/** Runs navigation jobs in-process with the handler the real worker uses. */
class InProcessNavWorker {
  onmessage: ((e: MessageEvent<JobMessage>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  terminated = false;
  postMessage(req: NavWorkerRequest): void {
    void handleNavJob(() => Promise.resolve(mj), req, (m) => {
      if (!this.terminated) this.onmessage?.({ data: m } as MessageEvent<JobMessage>);
    });
  }
  terminate(): void {
    this.terminated = true;
  }
}

describe('batch navigation evaluation in a worker', () => {
  it('keeps the mode it started with and reports every map', async () => {
    const s = new NavSession(() => Promise.resolve(mj), undefined, new JobWorkerClient<NavWorkerRequest>(() => new InProcessNavWorker() as unknown as Worker, 'Navigation worker'));
    await s.init();
    const p = s.evaluateAll();
    expect(s.state.evaluation).toMatchObject({ running: true, mode: 'sensor-only' });
    // Switching the live mode does not change the batch that is already running.
    s.setMode('ground-truth');
    const results = await p;
    expect(results).toHaveLength(TEST_MAPS.length);
    expect(results.every((r) => r.mode === 'sensor-only')).toBe(true);
    expect(s.state.evaluation).toMatchObject({ running: false, progress: 1, mode: 'sensor-only' });
    expect(s.state.evaluation.results).toBe(results);
    s.dispose();
  }, 120_000);

  it('can be cancelled and the worker is terminated', async () => {
    let terminated = 0;
    const silent = { onmessage: null, onerror: null, postMessage() {}, terminate: () => void terminated++ };
    const s = new NavSession(() => Promise.resolve(mj), undefined, new JobWorkerClient<NavWorkerRequest>(() => silent as unknown as Worker, 'Navigation worker'));
    await s.init();
    const p = s.evaluateAll();
    expect(s.state.evaluation.running).toBe(true);
    s.cancelEvaluation();
    expect(await p).toEqual([]);
    expect(terminated).toBe(1);
    expect(s.state.evaluation).toMatchObject({ running: false, detail: `Cancelled after 0 of ${TEST_MAPS.length} maps` });
    // Disposing the session terminates a running worker too.
    void s.evaluateAll();
    s.dispose();
    expect(terminated).toBe(2);
  });
});
