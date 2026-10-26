import type { MainModule } from '@mujoco/mujoco';
import { PRESETS, type PresetId } from '../../core/model/catalog';
import { morphologyHash } from '../../core/model/hash';
import { rover } from '../../core/model/presets';
import type { RobotDescription } from '../../core/model/types';
import { validateRobot } from '../../core/model/validate';
import { driveGeometry, navConfigFor, odometryFor, wheelTargets, type DriveGeometry, type DriveLimits } from '../../nav/drive';
import type { Pose2 } from '../../nav/follower';
import { capSpeed, wheelSpeeds } from '../../nav/follower';
import { LEG_GAINS, LeggedDriver } from '../../nav/legged';
import { rectOverlapsBox, TEST_MAPS, type NavMap } from '../../nav/maps';
import { DEFAULT_REFEREE, NavReferee } from '../../nav/referee';
import { NavStack, RAY_ANGLES, RAY_MAX_RANGE, type NavInput, type NavStackConfig } from '../../nav/stack';
import { createNavRuntime, groundTruthPose, groundTruthUpright, NAV_CONTROL_DECIMATION, wheelTorques, type NavRunResult } from '../../physics/navigation';
import type { MujocoRuntime, SensorReadings } from '../../physics/runtime';
import type { NavWorkerRequest } from '../../nav/jobs';
import { NavWorkerClient } from '../../nav/workerClient';
import { locomotionProblem, type LocomotionController, type LocomotionResult } from '../../skills/locomotion';
import type { JobWorkerClient } from '../../skills/workerClient';
import { Store } from '../store';

export type NavMode = 'sensor-only' | 'ground-truth';

/** Where the Navigate robot came from: a preset, or the robot handed over from Build. */
export type NavRobotSource = PresetId | 'custom';

export type WalkingTraining = { ok: false; reason: string; cancelled?: boolean } | { ok: true; controller: LocomotionController; result: LocomotionResult; wallTimeS: number };

/** Walking controllers, trained in the AI Playground's workers and shared with the Skills tab. */
export interface NavWalkingAccess {
  /** The stored walking controller of this morphology, or null. */
  controllerFor(morphologyHash: string): LocomotionController | null;
  /** Trains and saves one for this body; aborting cancels it. */
  train(robot: RobotDescription, onProgress: (fraction: number, detail: string) => void, signal: AbortSignal): Promise<WalkingTraining>;
}

/** How a legged body walks, for people and the AI. */
export function walkerInfo(c: LocomotionController | null): { learned: boolean; text: string } {
  if (!c) return { learned: false, text: 'the trot gait generator steered by the planner; no walking controller is trained for this body yet.' };
  const e = c.evaluation;
  return {
    learned: true,
    text: `its learned walking controller (trained ${c.createdAt.slice(0, 10)}): the calibrated gait generator ${c.policy ? 'with a learned feedback policy' : 'without a feedback policy (the policy did not beat the calibration)'}; ${e.final.successes} of ${e.trials} randomized walking commands ended within tolerance.`,
  };
}

export interface NavState {
  status: 'loading' | 'ready' | 'error';
  error: string | null;
  /** The body being driven and where it came from. */
  robotSource: NavRobotSource;
  robotName: string;
  mapId: string;
  mode: NavMode;
  driving: 'idle' | 'auto' | 'remote';
  simTime: number;
  /** Simulator pose (ground truth), shown for reference only. */
  truth: Pose2 | null;
  /** The navigation controller's own estimate (sensor-only mode), or null. */
  estimate: Pose2 | null;
  goal: [number, number];
  navStatus: string;
  replans: number;
  reached: boolean;
  collided: boolean;
  /** A walking body fell over (simulator state); Reset is required. */
  fallen: boolean;
  /** How a legged body walks: its learned walking controller or the gait generator alone; null for wheeled bodies. */
  walker: { learned: boolean; text: string } | null;
  /** Why the learned feedback policy stopped correcting the walk (the calibrated generator walks on), or null. */
  walkerFault: string | null;
  /** A walking controller being trained for this body from Navigate. */
  walkingJob: { fraction: number; detail: string } | null;
  /** The simulation produced non-finite state; Reset is required. */
  unstable: boolean;
  ranges: number[];
  /** Keyboard remote command: forward speed (m/s) and turn rate (rad/s). */
  remote: { v: number; omega: number };
  /** Commanded cap on the body speed while driving to a goal, m/s (within the body's speed limit range). */
  speedLimit: number;
  /** Body speed from simulator ground truth between display updates, m/s (display only). */
  truthSpeed: number | null;
  message: string | null;
  /** Batch evaluation in the worker; `mode`, `robotName` and `walker` are those it was started with. */
  evaluation: { running: boolean; progress: number; results: NavRunResult[]; mode: NavMode | null; robotName: string | null; walker: string | null; detail: string };
}

const CONTROL_DECIMATION = NAV_CONTROL_DECIMATION;
const MAX_SUBSTEPS = 25;

/** Sees every input handed to a navigation stack (tests audit what each stack receives). */
export type NavInputAudit = (stack: 'sensor-only' | 'ground-truth', input: NavInput) => void;

/**
 * Navigate: a wheeled or legged body (the rover unless another is chosen)
 * in one of the test maps, driven by the navigation stack (sensor-only or
 * labeled ground-truth mode) or from the keyboard. Wheeled bodies steer by
 * wheel speeds; legged bodies walk with the walking controller trained for
 * their morphology, or the gait generator. Ground truth is
 * used by the referee and for display; the sensor-only controller never
 * receives it.
 *
 * Both stacks are created only at an explicit reset (robot or map load, or
 * Reset), from the map's declared start pose. From then on they observe
 * every control update, whatever is driving, so odometry and the map carry
 * over remote driving, goal changes and restarts.
 */
export class NavSession {
  readonly store: Store<NavState>;
  private mj: MainModule | null = null;
  private rt: MujocoRuntime | null = null;
  private sensorStack: NavStack | null = null;
  private truthStack: NavStack | null = null;
  private referee: NavReferee | null = null;
  private robotDesc: RobotDescription = rover();
  private driveGeom: DriveGeometry = NavSession.mustDrive(this.robotDesc);
  private cfg: NavStackConfig = navConfigFor(this.driveGeom);
  private target: number[] = [];
  /** Walks a legged body; null for wheeled ones. */
  private legs: LeggedDriver | null = null;
  private walkingAccess: NavWalkingAccess | null = null;
  /** The legged body's walking controller (validated for its morphology), or null. */
  private controller: LocomotionController | null = null;
  private trainAbort: AbortController | null = null;
  private velocityLimit: number[] = [];
  private accumulator = 0;
  private stepCount = 0;
  private lastPublish = 0;
  private initPromise: Promise<void> | null = null;
  /** Increases at every explicit reset; the only time the stacks are re-created. */
  generation = 0;

  constructor(private readonly loadModule: () => Promise<MainModule>, private readonly audit?: NavInputAudit, private readonly worker: JobWorkerClient<NavWorkerRequest> = new NavWorkerClient()) {
    const map = TEST_MAPS[1]!;
    this.store = new Store<NavState>({
      status: 'loading', error: null, robotSource: 'rover', robotName: this.robotDesc.name, mapId: map.id, mode: 'sensor-only', driving: 'idle', simTime: 0, truth: null, estimate: null, goal: map.goal,
      navStatus: 'idle', replans: 0, reached: false, collided: false, fallen: false, walker: null, walkerFault: null, walkingJob: null, unstable: false, ranges: [], remote: { v: 0, omega: 0 }, speedLimit: this.driveGeom.limits.speedLimitRange[1], truthSpeed: null, message: null,
      evaluation: { running: false, progress: 0, results: [], mode: null, robotName: null, walker: null, detail: '' },
    });
  }

  private static mustDrive(robot: RobotDescription): DriveGeometry {
    const r = driveGeometry(robot);
    if (!r.ok) throw new Error(`${robot.name} cannot navigate: ${r.reason}.`);
    return r.drive;
  }

  get state(): NavState {
    return this.store.get();
  }

  get map(): NavMap {
    return TEST_MAPS.find((m) => m.id === this.state.mapId)!;
  }

  get runtime(): MujocoRuntime | null {
    return this.rt;
  }

  /** The body being driven (at its description pose, not where it is now). */
  get robot(): RobotDescription {
    return this.robotDesc;
  }

  /** Wheel layout or legs, footprint and sensor mount of the body being driven. */
  get drive(): DriveGeometry {
    return this.driveGeom;
  }

  /** Speed limits, keyboard speeds and the run time limit of the body being driven. */
  get limits(): DriveLimits {
    return this.driveGeom.limits;
  }

  /** True when the body can be driven now: loaded, not collided, fallen or unstable. */
  get drivable(): boolean {
    const s = this.state;
    return s.status === 'ready' && !s.collided && !s.fallen && !s.unstable;
  }

  /** The stack that drives in the current mode (for display: path and map). */
  get navStack(): NavStack | null {
    return this.state.mode === 'sensor-only' ? this.sensorStack : this.truthStack;
  }

  /** The sensor-only estimator and map, which exist from the last reset on. */
  get sensorOnlyStack(): NavStack | null {
    return this.sensorStack;
  }

  init(): Promise<void> {
    this.initPromise ??= (async () => {
      try {
        this.mj = await this.loadModule();
        this.build();
      } catch (e) {
        this.store.update({ status: 'error', error: `Physics engine failed to load: ${e instanceof Error ? e.message : String(e)}` });
      }
    })();
    return this.initPromise;
  }

  /** Explicit reset: rebuilds the robot at the map's declared start pose and starts both stacks from it. */
  private build(): void {
    if (!this.mj) return;
    this.rt?.dispose();
    this.rt = null;
    const map = this.map;
    try {
      this.rt = createNavRuntime(this.mj, map, this.robotDesc, this.driveGeom);
    } catch (e) {
      this.sensorStack = this.truthStack = null;
      this.store.update({ status: 'error', error: `${this.robotDesc.name} could not be built in the arena: ${e instanceof Error ? e.message : String(e)}`, driving: 'idle' });
      return;
    }
    this.velocityLimit = this.rt.description.limbs.flatMap((l) => l.joints.map((j) => j.limit.velocity));
    const d = this.driveGeom;
    this.sensorStack = new NavStack('sensor-only', map.start, odometryFor(this.robotDesc, d, map.start), null, this.cfg);
    this.truthStack = new NavStack('ground-truth', map.start, odometryFor(this.robotDesc, d, map.start), map.boxes, this.cfg);
    this.generation++;
    this.referee = null;
    this.target = d.kind === 'wheeled' ? d.wheels.map(() => 0) : [];
    this.legs = d.kind === 'legged' ? new LeggedDriver(this.robotDesc, d, LEG_GAINS, this.controller) : null;
    this.accumulator = 0;
    this.stepCount = 0;
    // Readings from before a reset or map change are never shown or reported as current.
    this.lastRanges = [];
    this.store.update({ status: 'ready', error: null, driving: 'idle', simTime: 0, estimate: null, goal: map.goal, navStatus: 'idle', replans: 0, reached: false, collided: false, fallen: false, unstable: false, message: null, remote: { v: 0, omega: 0 }, truthSpeed: null });
    this.lastSpeedSample = null;
    this.publish(true);
  }

  setMap(id: string): void {
    if (!TEST_MAPS.some((m) => m.id === id)) return;
    this.store.update({ mapId: id });
    this.build();
  }

  /** Loads a preset by id; see setRobot. */
  setPreset(id: PresetId): { ok: true } | { ok: false; reason: string } {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return { ok: false, reason: `There is no preset "${id}".` };
    return this.setRobot(p.make(), id);
  }

  /**
   * Drives another body: a preset or the robot from Build. Wheel-only bodies
   * the skid-steering controller can drive and legged bodies the gait
   * generator can walk are accepted; anything else is refused with the
   * reason and the current robot stays. Like a map change, this is an
   * explicit reset; the speed limit returns to the new body's maximum.
   */
  setRobot(robot: RobotDescription, source: NavRobotSource = 'custom'): { ok: true } | { ok: false; reason: string } {
    if (validateRobot(robot).length > 0) return this.refuseRobot(robot.name, 'the robot description is not valid');
    const r = driveGeometry(robot);
    if (!r.ok) return this.refuseRobot(robot.name, r.reason);
    if (this.state.driving === 'remote') this.stop();
    this.robotDesc = robot;
    this.driveGeom = r.drive;
    this.cfg = navConfigFor(r.drive);
    this.controller = null;
    this.store.update({ robotSource: source, robotName: robot.name, speedLimit: r.drive.limits.speedLimitRange[1], walker: r.drive.kind === 'legged' ? walkerInfo(null) : null });
    this.build();
    void this.refreshWalking();
    return { ok: true };
  }

  private refuseRobot(name: string, reason: string): { ok: false; reason: string } {
    const text = `${name} cannot navigate: ${reason}. ${this.robotDesc.name} stays loaded.`;
    this.store.update({ message: text });
    return { ok: false, reason: text };
  }

  /** Connects the walking controllers trained in the AI Playground (App wires the Skills tab's). */
  attachWalking(access: NavWalkingAccess | null): void {
    this.walkingAccess = access;
    void this.refreshWalking();
  }

  /** The walking controller the legged body walks with, or null (the gait generator, or a wheeled body). */
  get walkingController(): LocomotionController | null {
    return this.driveGeom.kind === 'legged' ? this.controller : null;
  }

  /** Looks up the stored walking controller of this body; it takes over the next time the body stands. */
  async refreshWalking(): Promise<void> {
    const robot = this.robotDesc;
    if (this.driveGeom.kind !== 'legged') {
      this.controller = null;
      this.store.update({ walker: null });
      return;
    }
    const hash = await morphologyHash(robot);
    if (robot !== this.robotDesc) return;
    const c = this.walkingAccess?.controllerFor(hash) ?? null;
    this.useWalking(c && locomotionProblem(c, robot, hash) === null ? c : null);
  }

  private useWalking(c: LocomotionController | null): void {
    this.controller = c;
    this.legs?.setController(c);
    this.store.update({ walker: walkerInfo(c) });
  }

  /**
   * Trains a walking controller for this legged body in the AI Playground's
   * workers (four to eight minutes) and walks with it from the next time the
   * body stands. It is saved for this morphology in every view.
   */
  async trainWalking(signal?: AbortSignal): Promise<WalkingTraining> {
    if (signal?.aborted) return { ok: false, reason: 'Cancelled before it started.', cancelled: true };
    const robot = this.robotDesc;
    if (this.driveGeom.kind !== 'legged') return { ok: false, reason: `${robot.name} rolls on wheels; walking controllers are trained for legged bodies.` };
    if (!this.walkingAccess) return { ok: false, reason: 'Walking controllers cannot be trained from this view.' };
    if (this.state.walkingJob) return { ok: false, reason: 'A walking controller is already being trained.' };
    const abort = new AbortController();
    this.trainAbort = abort;
    const onAbort = (): void => abort.abort();
    signal?.addEventListener('abort', onAbort, { once: true });
    this.store.update({ walkingJob: { fraction: 0, detail: 'Starting' } });
    try {
      const r = await this.walkingAccess.train(robot, (fraction, detail) => this.store.update({ walkingJob: { fraction, detail } }), abort.signal);
      const current = robot === this.robotDesc;
      if (r.ok && current) this.useWalking(r.controller);
      else if (r.ok) void this.refreshWalking();
      const e = r.ok ? r.controller.evaluation : null;
      const done = e ? `Walking controller trained and saved for ${robot.name}: ${e.final.successes} of ${e.trials} randomized walking commands ended within tolerance.` : '';
      this.store.update({ message: !r.ok ? r.reason : current ? `${done} It walks from the next time the robot stands.` : `${done} ${this.robotDesc.name} is loaded now.` });
      return r.ok ? r : { ...r, cancelled: r.cancelled || abort.signal.aborted };
    } finally {
      signal?.removeEventListener('abort', onAbort);
      this.trainAbort = null;
      this.store.update({ walkingJob: null });
    }
  }

  cancelWalkingTraining(): void {
    this.trainAbort?.abort();
  }

  /** Switches the controller inputs; both stacks have been observing, so neither starts over. */
  setMode(mode: NavMode): void {
    if (mode === this.state.mode) return;
    const wasAuto = this.state.driving === 'auto';
    this.navStack?.clearGoal();
    this.store.update({ mode });
    if (wasAuto) this.startAuto();
  }

  reset(): void {
    this.build();
  }

  /** Inside the arena and clear of obstacles for the robot's footprint. */
  isFreeFloor(x: number, y: number): boolean {
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < -0.4 || x > 3.4 || y < -1.9 || y > 1.9) return false;
    const [hx, hy] = this.driveGeom.halfExtents;
    return !this.map.boxes.some((b) => rectOverlapsBox(x, y, 0, hx, hy, b));
  }

  /** Where the controller believes the robot is: its sensor-only estimate, or ground truth in ground-truth mode. */
  believedPosition(): { x: number; y: number; source: 'sensor-estimate' | 'ground-truth' } | null {
    const s = this.state;
    const p = s.mode === 'sensor-only' ? s.estimate : s.truth;
    return p ? { x: p.x, y: p.y, source: s.mode === 'sensor-only' ? 'sensor-estimate' : 'ground-truth' } : null;
  }

  /** Sets the speed cap for driving to a goal; takes effect at the next control update, also during a run. */
  setSpeedLimit(v: number): boolean {
    const [lo, hi] = this.limits.speedLimitRange;
    if (!Number.isFinite(v) || v < lo - 1e-9 || v > hi + 1e-9) return false;
    this.store.update({ speedLimit: Math.round(v * 100) / 100 });
    return true;
  }

  /** One step slower or faster (0.1 m/s, or 0.05 m/s for a little; walking bodies 0.05 and 0.02 m/s), within the body's range; works during a run. */
  changeSpeed(change: 'slower' | 'faster', amount: 'little' | 'normal'): { ok: true; description: string } | { ok: false; reason: string } {
    const legged = this.driveGeom.kind === 'legged';
    const step = legged ? (amount === 'little' ? 0.02 : 0.05) : amount === 'little' ? 0.05 : 0.1;
    const [lo, hi] = this.limits.speedLimitRange;
    const before = this.state.speedLimit;
    const next = Math.min(hi, Math.max(lo, Math.round((before + (change === 'faster' ? step : -step)) * 100) / 100));
    if (Math.abs(next - before) < 1e-9) {
      return { ok: false, reason: `The speed limit is already at its ${change === 'faster' ? 'maximum' : 'minimum'}, ${before.toFixed(2)} m/s.` };
    }
    this.setSpeedLimit(next);
    return { ok: true, description: `Speed limit ${next.toFixed(2)} m/s (was ${before.toFixed(2)} m/s); the path is unchanged.` };
  }

  /** Sets a goal on the floor; refused inside or right next to obstacles and outside the arena. */
  setGoal(x: number, y: number): boolean {
    if (!this.isFreeFloor(x, y)) {
      this.store.update({ message: 'That goal is outside the arena or too close to an obstacle.' });
      return false;
    }
    this.store.update({ goal: [x, y], message: null });
    if (this.state.driving === 'auto') this.startAuto();
    return true;
  }

  /** Drives to the goal from the current estimate and map; nothing is re-initialized from the simulator. */
  startAuto(): void {
    const stack = this.navStack;
    if (!this.rt || !stack || !this.drivable) return;
    stack.setGoal(this.state.goal);
    this.referee = new NavReferee(this.map, this.state.goal, this.rt.time, { ...DEFAULT_REFEREE, timeLimitS: this.limits.timeLimitS }, this.driveGeom.halfExtents);
    this.store.update({ driving: 'auto', reached: false, navStatus: 'planning', message: null });
  }

  /** Keyboard remote command; switches to remote driving. The estimator keeps observing. */
  setRemote(v: number, omega: number): void {
    if (this.state.collided || this.state.fallen || this.state.unstable) return;
    const { remoteSpeed, remoteTurn } = this.limits;
    const vv = Math.max(-remoteSpeed, Math.min(remoteSpeed, Number.isFinite(v) ? v : 0));
    const ww = Math.max(-remoteTurn, Math.min(remoteTurn, Number.isFinite(omega) ? omega : 0));
    this.sensorStack?.clearGoal();
    this.truthStack?.clearGoal();
    this.referee = null;
    this.store.update({ driving: 'remote', remote: { v: vv, omega: ww } });
  }

  /** Stops driving (a walking body stops stepping and stands); `message` says why when the stop was not requested by the user. */
  stop(message?: string): void {
    this.sensorStack?.clearGoal();
    this.truthStack?.clearGoal();
    this.referee = null;
    this.halt();
    this.store.update({ driving: 'idle', remote: { v: 0, omega: 0 }, ...(message ? { message } : {}) });
  }

  private halt(): void {
    this.target = this.target.map(() => 0);
    this.legs?.setCommand(0, 0);
  }

  private endRun(message: string, patch: Partial<NavState> = {}): void {
    this.sensorStack?.clearGoal();
    this.truthStack?.clearGoal();
    this.referee = null;
    this.halt();
    this.store.update({ driving: 'idle', remote: { v: 0, omega: 0 }, message, ...patch });
  }

  stepOnce(): void {
    const rt = this.rt;
    const sensorStack = this.sensorStack, truthStack = this.truthStack;
    if (!rt || !sensorStack || !truthStack || this.state.collided || this.state.fallen || this.state.unstable) return;
    const s = this.state;
    const d = this.driveGeom;
    const legs = this.legs;
    rt.controlStep((sensors: SensorReadings) => {
      if (this.stepCount % CONTROL_DECIMATION === 0) {
        const t = rt.time;
        const sensorInput: NavInput = { t, sensors };
        const truthInput: NavInput = { t, sensors, groundTruth: { pose: groundTruthPose(rt) } };
        this.audit?.('sensor-only', sensorInput);
        this.audit?.('ground-truth', truthInput);
        const auto = s.driving === 'auto';
        const sensorCmd = auto && s.mode === 'sensor-only' ? sensorStack.update(sensorInput) : (sensorStack.observe(sensorInput), null);
        const truthCmd = auto && s.mode === 'ground-truth' ? truthStack.update(truthInput) : (truthStack.observe(truthInput), null);
        const cmd = sensorCmd ?? truthCmd;
        if (d.kind === 'wheeled') {
          if (cmd) {
            const c = capSpeed(wheelSpeeds(cmd.v, cmd.omega, d.wheelRadius, d.trackWidth), s.speedLimit, d.wheelRadius);
            this.target = wheelTargets(d.wheels, c.left, c.right);
          } else if (s.driving === 'remote') {
            const w = wheelSpeeds(s.remote.v, s.remote.omega, d.wheelRadius, d.trackWidth);
            this.target = wheelTargets(d.wheels, w.left, w.right);
          } else {
            this.target = d.wheels.map(() => 0);
          }
        } else if (legs) {
          // The cap scales speed and turn rate together, so the path curvature is unchanged.
          const k = cmd && Math.abs(cmd.v) > s.speedLimit ? s.speedLimit / Math.abs(cmd.v) : 1;
          if (cmd) legs.setCommand(cmd.v * k, cmd.omega * k);
          else if (s.driving === 'remote') legs.setCommand(s.remote.v, s.remote.omega);
          else legs.setCommand(0, 0);
        }
        this.lastRanges = Array.from(sensors.ranges);
      }
      return legs ? legs.torques(rt.time, rt.timestep, sensors, rt.layout.effort) : wheelTorques(this.target, sensors, rt.layout.effort, this.velocityLimit);
    });
    this.stepCount++;
    // Referee (ground truth): the same rules as the batch evaluation.
    if (rt.hasNonFiniteState()) {
      this.endRun('The simulation became numerically unstable and was stopped. Reset to continue.', { unstable: true });
      return;
    }
    if (legs && !groundTruthUpright(rt, d)) {
      this.endRun('The robot fell over (detected from simulator state). Reset to continue.', { fallen: true });
      return;
    }
    const gt = groundTruthPose(rt);
    if (NavReferee.collides(this.map, gt, d.halfExtents)) {
      this.endRun('Collision with an obstacle (detected from simulator state). Reset to continue.', { collided: true });
      return;
    }
    const stack = this.navStack;
    if (s.driving === 'auto' && this.referee && stack) {
      const verdict = this.referee.check(rt.time, gt, stack.status, true);
      if (verdict) this.endRun(verdict.message, { reached: verdict.outcome === 'reached' });
    }
  }

  private lastRanges: number[] = [];
  private lastSpeedSample: { t: number; x: number; y: number } | null = null;

  /** Advances by wall time with a fixed-step accumulator (capped per frame). */
  advance(wallDt: number): void {
    if (!this.rt) return;
    this.accumulator += wallDt;
    let n = 0;
    while (this.accumulator >= this.rt.timestep && n < MAX_SUBSTEPS) {
      this.stepOnce();
      this.accumulator -= this.rt.timestep;
      n++;
    }
    if (n === MAX_SUBSTEPS) this.accumulator = 0;
    this.publish(false);
  }

  private publish(force: boolean): void {
    const rt = this.rt;
    if (!rt) return;
    const now = performance.now();
    if (!force && now - this.lastPublish < 100) return;
    this.lastPublish = now;
    const s = this.state;
    const stack = this.navStack;
    const truth = groundTruthPose(rt);
    const prev = this.lastSpeedSample;
    const dt = prev ? rt.time - prev.t : 0;
    const truthSpeed = prev && dt > 1e-6 ? Math.hypot(truth.x - prev.x, truth.y - prev.y) / dt : prev ? s.truthSpeed : null;
    this.lastSpeedSample = { t: rt.time, x: truth.x, y: truth.y };
    this.store.update({
      simTime: rt.time,
      truth,
      truthSpeed,
      estimate: s.mode === 'sensor-only' && this.sensorStack ? { ...this.sensorStack.pose } : null,
      navStatus: s.driving === 'auto' && stack ? stack.status : s.driving === 'remote' ? 'remote control' : s.reached ? 'reached' : 'idle',
      replans: stack?.replans ?? s.replans,
      ranges: this.lastRanges,
      walkerFault: this.legs?.fault ?? null,
    });
  }

  /** World-frame segments of the current range rays, for display. */
  raySegments(): [number, number, number, number][] {
    const rt = this.rt;
    if (!rt) return [];
    const p = groundTruthPose(rt);
    const c = Math.cos(p.yaw), s = Math.sin(p.yaw);
    const [rx, ry] = this.driveGeom.rayOrigin;
    const ox = p.x + c * rx - s * ry, oy = p.y + s * rx + c * ry;
    return RAY_ANGLES.map((a, i) => {
      const r = this.lastRanges[i];
      const len = r !== undefined && r >= 0 && r <= RAY_MAX_RANGE ? r : RAY_MAX_RANGE;
      return [ox, oy, ox + len * Math.cos(p.yaw + a), oy + len * Math.sin(p.yaw + a)];
    });
  }

  /**
   * Runs every test map in a Web Worker with the controller mode, robot and
   * walking controller chosen now; changing them later does not affect the
   * running batch. Results appear as maps finish.
   */
  async evaluateAll(): Promise<NavRunResult[]> {
    if (this.state.evaluation.running) return this.state.evaluation.results;
    const mode = this.state.mode;
    const robot = this.robotDesc;
    const robotName = robot.name;
    const walking = this.walkingController;
    const walker = this.driveGeom.kind === 'legged' ? (walking ? 'learned walking controller' : 'gait generator') : null;
    this.store.update({ evaluation: { running: true, progress: 0, results: [], mode, robotName, walker, detail: 'Starting' } });
    try {
      const r = await this.worker.run<NavRunResult[]>({ type: 'evaluate', mode, mapIds: TEST_MAPS.map((m) => m.id), robot, ...(walking ? { walking } : {}) }, (progress, detail, partial) => {
        this.store.update({ evaluation: { running: true, progress, results: Array.isArray(partial) ? (partial as NavRunResult[]) : this.state.evaluation.results, mode, robotName, walker, detail } });
      });
      this.store.update({ evaluation: { running: false, progress: 1, results: r.result, mode, robotName, walker, detail: `Finished in ${r.wallTimeS.toFixed(1)} s` } });
      return r.result;
    } catch (e) {
      const text = e instanceof Error ? e.message : String(e);
      const ev = this.state.evaluation;
      this.store.update({ evaluation: { ...ev, running: false, detail: text === 'Cancelled.' ? `Cancelled after ${ev.results.length} of ${TEST_MAPS.length} maps` : `Failed: ${text}` } });
      return ev.results;
    }
  }

  cancelEvaluation(): void {
    this.worker.cancel();
  }

  dispose(): void {
    this.trainAbort?.abort();
    this.worker.dispose();
    this.rt?.dispose();
    this.rt = null;
  }
}
