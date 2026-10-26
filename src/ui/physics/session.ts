import type { MainModule } from '@mujoco/mujoco';
import { speedRatio, TargetRateLimiter } from '../../core/control/limits';
import { pdControl, pdStabilityMargin, suggestGains, type PdGains, type PdResult } from '../../core/control/pd';
import { Telemetry } from '../../core/control/telemetry';
import { buildKinematicModel, clampToLimits, type KinematicModel } from '../../core/kinematics/fk';
import type { Mat4 } from '../../core/math/mat4';
import { normalize, type Vec3 } from '../../core/math/vec3';
import { homeConfiguration, PRESETS, type PresetId } from '../../core/model/catalog';
import type { RobotDescription } from '../../core/model/types';
import { jointUnits } from '../../core/model/units';
import { validateRobot } from '../../core/model/validate';
import { MujocoRuntime, type SensorReadings } from '../../physics/runtime';
import { Store } from '../store';

export const PHYSICS_PRESETS: PresetId[] = ['quadruped', 'arm3', 'arm6', 'hexapod', 'rover'];
/** Robot source: a preset id, or 'custom' for a robot sent from the builder. */
export type PhysicsSource = PresetId | 'custom';
export const PUSH_DURATION_S = 0.05;
const TELEMETRY_DECIMATION = 5;
const MAX_SUBSTEPS = 25;
/** Driver update period in physics steps (100 Hz at a 2 ms step). */
const DRIVER_DECIMATION = 5;

/** High-level controller: sensor readings in, joint targets out (or null to keep the last targets). */
export type PhysicsDriver = (t: number, s: SensorReadings) => number[] | null;

/** Why an observed run ended early. */
export type RunAbortCause = 'interrupted' | 'controller-error' | 'unstable';

/** Watches every physics step of a run; `abort` is called once when the run cannot continue. */
export interface StepObserver {
  step(rt: MujocoRuntime, sensors: SensorReadings): void;
  abort(reason: string, cause: RunAbortCause): void;
}

export const DEFAULT_GAINS: Partial<Record<PresetId, PdGains>> = {
  quadruped: { kp: 40, kd: 1 },
  hexapod: { kp: 30, kd: 0.8 },
  arm3: { kp: 60, kd: 3 },
  arm6: { kp: 60, kd: 3 },
};

export interface PhysicsEvent {
  t: number;
  text: string;
}

export interface Stability {
  ok: boolean;
  wnH: number;
  dampingNumber: number;
  /** Damping ratio of the selected joint with its current inertia estimate. */
  zetaSelected: number;
  minInertia: number;
}

export interface PhysicsState {
  status: 'loading' | 'ready' | 'error';
  error: string | null;
  presetId: PhysicsSource;
  description: RobotDescription | null;
  running: boolean;
  simTime: number;
  realTimeFactor: number;
  stepMs: number;
  gains: PdGains;
  gainError: string | null;
  stability: Stability | null;
  qCmd: number[];
  selectedJoint: number;
  base: { height: number; tilt: number } | null;
  saturationFraction: number;
  /** Share of the last second in which the selected joint moved faster than its velocity limit (measured). */
  overspeedFraction: number;
  /** Largest measured speed divided by the velocity limit over all joints in the last second. */
  maxSpeedRatio: number;
  warnings: Record<string, number>;
  unstable: boolean;
  events: PhysicsEvent[];
}

/**
 * Framework-free controller for physics mode. Runs the TypeScript PD law at
 * the physics rate using encoder readings, writes torques to MuJoCo motors and
 * records commanded, measured and applied signals.
 */
export class PhysicsSession {
  readonly store: Store<PhysicsState>;
  telemetry: Telemetry | null = null;
  private mj: MainModule | null = null;
  private rt: MujocoRuntime | null = null;
  private km: KinematicModel | null = null;
  private inertias: number[] = [];
  private accumulator = 0;
  private stepCount = 0;
  private pushRemaining = 0;
  private pushForce: Vec3 = [0, 0, 0];
  private pd: PdResult | null = null;
  private satWindow: number[] = [];
  private lastPublish = 0;
  private wallAccum = 0;
  private simAccum = 0;
  private stepTimeAccum = 0;
  private stepTimeCount = 0;
  private pending: RobotDescription | null = null;
  private driver: PhysicsDriver | null = null;
  private driverQ: number[] = [];
  private driverStart = 0;
  private observer: StepObserver | null = null;
  private limiter: TargetRateLimiter | null = null;
  private continuous: boolean[] = [];
  private velocityLimit: number[] = [];
  private overspeedWindow: number[] = [];
  private speedWindow: number[] = [];

  constructor(private readonly loadModule: () => Promise<MainModule>, presetId: PresetId = 'quadruped') {
    this.store = new Store<PhysicsState>({
      status: 'loading', error: null, presetId, description: null, running: true, simTime: 0, realTimeFactor: 0, stepMs: 0,
      gains: DEFAULT_GAINS[presetId] ?? { kp: 40, kd: 1 }, gainError: null, stability: null, qCmd: [], selectedJoint: 0,
      base: null, saturationFraction: 0, overspeedFraction: 0, maxSpeedRatio: 0, warnings: {}, unstable: false, events: [],
    });
  }

  get state(): PhysicsState {
    return this.store.get();
  }

  get runtime(): MujocoRuntime | null {
    return this.rt;
  }

  private initPromise: Promise<void> | null = null;

  /** Loads the engine and builds the first robot. Repeated calls share one load. */
  init(): Promise<void> {
    this.initPromise ??= this.load();
    return this.initPromise;
  }

  private async load(): Promise<void> {
    try {
      this.mj = await this.loadModule();
    } catch (e) {
      this.store.update({ status: 'error', error: `Physics engine failed to load: ${e instanceof Error ? e.message : String(e)}` });
      return;
    }
    if (this.pending) this.build(this.pending, 'custom');
    else this.setPreset(this.state.presetId === 'custom' ? 'quadruped' : this.state.presetId);
    this.pending = null;
  }

  private build(desc: RobotDescription, source: PhysicsSource): void {
    if (!this.mj) return;
    this.abortRun('the robot was replaced');
    const q0 = homeConfiguration(desc);
    this.rt?.dispose();
    this.rt = null;
    try {
      this.rt = MujocoRuntime.create(this.mj, desc, { initialQ: q0 });
    } catch (e) {
      this.store.update({ status: 'error', error: `Model failed to compile: ${e instanceof Error ? e.message : String(e)}` });
      return;
    }
    this.km = buildKinematicModel(desc);
    this.inertias = this.rt.jointInertias();
    this.driver = null;
    this.observer = null;
    this.driverQ = q0.slice();
    const joints = desc.limbs.flatMap((l) => l.joints);
    this.continuous = joints.map((j) => j.type === 'continuous');
    this.velocityLimit = joints.map((j) => j.limit.velocity);
    this.limiter = new TargetRateLimiter(this.velocityLimit, this.continuous, q0);
    this.overspeedWindow = [];
    this.speedWindow = [];
    const n = q0.length;
    const channels: string[] = [];
    for (let j = 0; j < n; j++) channels.push(`qCmd${j}`, `qMeas${j}`, `tauCmd${j}`, `tauApplied${j}`);
    channels.push('baseHeight', 'baseTilt');
    this.telemetry = new Telemetry(channels, 1200);
    this.pd = null;
    this.accumulator = 0;
    this.stepCount = 0;
    this.pushRemaining = 0;
    this.satWindow = [];
    const events = [{ t: 0, text: `Loaded ${desc.name} (${n} actuated joints).` }];
    let gains = source !== 'custom' ? DEFAULT_GAINS[source] : undefined;
    if (!gains) {
      const minInertia = Math.min(...this.inertias);
      const s = suggestGains(desc.base.kind === 'floating', minInertia, this.rt.timestep);
      gains = s.gains;
      if (s.reduced) events.unshift({ t: 0, text: `Default gains reduced to kp ${gains.kp.toPrecision(3)}, kd ${gains.kd.toPrecision(3)} to stay within the stability bound for the lightest joint (${minInertia.toFixed(4)} ${this.inertiaUnit(minInertia)}).` });
    }
    this.store.set({
      ...this.state,
      status: 'ready', error: null, presetId: source, description: desc, simTime: 0, gains, gainError: null,
      stability: this.stabilityFor(gains, 0), qCmd: q0, selectedJoint: 0, unstable: false, warnings: {},
      events,
    });
    this.publish(true);
  }

  setPreset(presetId: PresetId): void {
    const preset = PRESETS.find((p) => p.id === presetId);
    if (!preset) throw new RangeError(`Unknown preset ${presetId}`);
    if (!this.mj) {
      this.pending = null;
      this.store.update({ presetId });
      return;
    }
    this.build(preset.make(), presetId);
  }

  /** Simulates a robot from the builder. Invalid descriptions are refused with a message. */
  loadRobot(desc: RobotDescription): boolean {
    const issues = validateRobot(desc);
    if (issues.length) {
      this.log(`Refused robot: ${issues[0]!.path}: ${issues[0]!.message}.`);
      return false;
    }
    if (!this.mj) {
      this.pending = desc;
      this.store.update({ presetId: 'custom', description: desc });
      return true;
    }
    this.build(desc, 'custom');
    return true;
  }

  /** Unit of a joint inertia value: kg*m^2 for rotating joints, kg for prismatic ones. */
  private inertiaUnit(value: number): string {
    const i = this.inertias.indexOf(value);
    return jointUnits(this.km?.joints[i]?.type ?? 'revolute').inertia;
  }

  private stabilityFor(g: PdGains, selected: number): Stability | null {
    if (!this.rt || this.inertias.length === 0) return null;
    const minInertia = Math.min(...this.inertias);
    const m = pdStabilityMargin(g, minInertia, this.rt.timestep);
    const Isel = this.inertias[selected] ?? minInertia;
    return { ok: m.ok, wnH: m.wnH, dampingNumber: m.dampingNumber, zetaSelected: g.kp > 0 ? g.kd / (2 * Math.sqrt(g.kp * Isel)) : Infinity, minInertia };
  }

  /** Applies new gains immediately (no model rebuild). Gains beyond the stability bound are refused. */
  setGains(g: PdGains): boolean {
    if (!Number.isFinite(g.kp) || !Number.isFinite(g.kd) || g.kp < 0 || g.kd < 0) {
      this.store.update({ gainError: 'Gains must be finite and non-negative.' });
      return false;
    }
    const st = this.stabilityFor(g, this.state.selectedJoint);
    if (st && !st.ok) {
      this.store.update({
        gainError: `Refused: with the lightest joint (${st.minInertia.toFixed(4)} ${this.inertiaUnit(st.minInertia)}) and a ${(this.rt!.timestep * 1000).toFixed(0)} ms step these gains exceed the explicit stability bound (wn*h = ${st.wnH.toFixed(2)}, limit 0.2; kd*h/I = ${st.dampingNumber.toFixed(2)}, limit 1).`,
      });
      return false;
    }
    this.store.update({ gains: g, gainError: null, stability: st });
    return true;
  }

  selectJoint(j: number): void {
    if (j < 0 || j >= this.state.qCmd.length) return;
    this.store.update({ selectedJoint: j, stability: this.stabilityFor(this.state.gains, j) });
  }

  /** Commanded joint target; clamped to the joint limits. */
  setTarget(j: number, value: number): void {
    if (!this.km || !Number.isFinite(value) || j < 0 || j >= this.state.qCmd.length) return;
    const q = this.state.qCmd.slice();
    q[j] = value;
    this.store.update({ qCmd: clampToLimits(this.km, q) });
  }

  /** Commanded targets for every joint at once; clamped to the joint limits. */
  setTargets(q: readonly number[]): void {
    if (!this.km || q.length !== this.state.qCmd.length || !q.every(Number.isFinite)) return;
    this.store.update({ qCmd: clampToLimits(this.km, q) });
  }

  setRunning(on: boolean): void {
    this.store.update({ running: on });
  }

  /** Schedules an impulse (N*s) along a world direction, applied over PUSH_DURATION_S. */
  push(direction: Vec3, impulseNs: number): void {
    if (!this.rt || !Number.isFinite(impulseNs) || impulseNs <= 0) return;
    const d = normalize(direction);
    const f = impulseNs / PUSH_DURATION_S;
    this.pushForce = [d[0] * f, d[1] * f, d[2] * f];
    this.pushRemaining = PUSH_DURATION_S;
    const where = this.state.description?.base.kind === 'floating' ? 'base' : 'last link';
    this.log(`Push ${impulseNs.toFixed(1)} N*s along (${d.map((v) => v.toFixed(0)).join(', ')}) at the ${where} (${f.toFixed(0)} N for ${PUSH_DURATION_S * 1000} ms).`);
  }

  reset(): void {
    if (!this.rt || !this.state.description) return;
    this.abortRun('the simulation was reset');
    this.rt.reset();
    this.telemetry?.clear();
    this.pushRemaining = 0;
    this.accumulator = 0;
    this.stepCount = 0;
    this.satWindow = [];
    this.overspeedWindow = [];
    this.speedWindow = [];
    const home = homeConfiguration(this.state.description);
    this.driverQ = home.slice();
    this.driverStart = 0;
    this.limiter?.reset(home);
    this.store.update({ qCmd: home, unstable: false, simTime: 0 });
    this.log('Reset to the initial state.');
    this.publish(true);
  }

  /**
   * Hands joint targets to a high-level controller called every
   * DRIVER_DECIMATION steps (100 Hz) with sensor readings only and the time
   * since it was attached. Pass null to return to manual targets.
   */
  setDriver(driver: PhysicsDriver | null, label?: string): void {
    this.driver = driver;
    this.driverStart = this.rt?.time ?? 0;
    this.driverQ = this.state.qCmd.slice();
    if (label) this.log(label);
  }

  get hasDriver(): boolean {
    return this.driver !== null;
  }

  /** Called after every physics step of a run (referee measurements, not control). */
  setStepObserver(observer: StepObserver | null): void {
    this.observer = observer;
  }

  /** Ends the observed run, if any: detaches the driver and tells the observer why. */
  private abortRun(reason: string, cause: RunAbortCause = 'interrupted'): void {
    const o = this.observer;
    if (!o) return;
    this.observer = null;
    this.driver = null;
    o.abort(reason, cause);
  }

  private log(text: string): void {
    const events = [{ t: this.rt?.time ?? 0, text }, ...this.state.events].slice(0, 30);
    this.store.update({ events });
  }

  private pushBody(): number {
    const L = this.rt!.layout;
    return L.floating ? L.baseBodyId : L.linkBodyIds[L.linkBodyIds.length - 1]!;
  }

  /** One physics step with the PD controller in the loop. */
  stepOnce(): void {
    const rt = this.rt;
    const limiter = this.limiter;
    if (!rt || !limiter || this.state.unstable) return;
    const t0 = performance.now();
    const tSample = rt.time;
    const gains = this.state.gains;
    let requested = this.driver ? this.driverQ : this.state.qCmd;
    const gainList = requested.map(() => gains);
    if (this.pushRemaining > 1e-9) {
      rt.setBodyWrench(this.pushBody(), this.pushForce);
      this.pushRemaining -= rt.timestep;
    } else {
      rt.clearWrenches();
    }
    let driverError: string | null = null;
    const s = rt.controlStep((sensors) => {
      if (this.driver && this.stepCount % DRIVER_DECIMATION === 0) {
        try {
          const q = this.driver(rt.time - this.driverStart, sensors);
          if (q && q.length === requested.length && q.every(Number.isFinite)) this.driverQ = requested = clampToLimits(this.km!, q);
        } catch (e) {
          driverError = e instanceof Error ? e.message : String(e);
        }
      }
      // Targets move toward the request no faster than each joint's velocity limit.
      limiter.step(requested, rt.timestep);
      this.pd = pdControl(gainList, rt.layout.effort, { qCmd: limiter.qCmd, qMeas: sensors.q, qdMeas: sensors.qd, continuous: this.continuous }, this.pd ?? undefined);
      return this.pd.tauApplied;
    });
    if (!this.pd) return;
    this.stepTimeAccum += performance.now() - t0;
    this.stepTimeCount++;
    this.stepCount++;
    if (driverError) {
      this.log(`The controller stopped with an error: ${driverError}`);
      this.abortRun(`the controller failed: ${driverError}`, 'controller-error');
      // Without an observed run the failing driver would still be attached; return to the manual targets.
      this.driver = null;
    }
    if (rt.hasNonFiniteState()) {
      this.store.update({ unstable: true, running: false });
      this.log('The simulation became numerically unstable and was paused. Reset to continue.');
      this.abortRun('the simulation became numerically unstable', 'unstable');
      return;
    }
    if (this.observer) {
      try {
        this.observer.step(rt, s);
      } catch (e) {
        this.abortRun(`the run monitor failed: ${e instanceof Error ? e.message : String(e)}`, 'controller-error');
      }
    }
    const sel = this.state.selectedJoint;
    this.satWindow.push(this.pd.saturated[sel] ?? 0);
    if (this.satWindow.length > 500) this.satWindow.shift();
    const vSel = this.velocityLimit[sel];
    this.overspeedWindow.push(vSel !== undefined && Math.abs(s.qd[sel]!) > vSel ? 1 : 0);
    if (this.overspeedWindow.length > 500) this.overspeedWindow.shift();
    this.speedWindow.push(speedRatio(s.qd, this.velocityLimit));
    if (this.speedWindow.length > 500) this.speedWindow.shift();
    if (this.stepCount % TELEMETRY_DECIMATION === 0 && this.telemetry) {
      const n = requested.length;
      const v = new Float64Array(4 * n + 2);
      for (let j = 0; j < n; j++) {
        v[4 * j] = limiter.qCmd[j]!;
        v[4 * j + 1] = s.q[j]!;
        v[4 * j + 2] = this.pd.tauCmd[j]!;
        v[4 * j + 3] = this.pd.tauApplied[j]!;
      }
      const b = rt.groundTruthBaseState();
      v[4 * n] = b.pos[2];
      v[4 * n + 1] = b.tilt;
      this.telemetry.push(tSample, v);
    }
  }

  /** Advances simulated time by the elapsed wall time using a fixed-step accumulator. */
  advance(wallDt: number): number {
    if (!this.rt || !this.state.running || this.state.unstable) {
      this.publish(false);
      return 0;
    }
    this.accumulator += wallDt;
    let n = 0;
    while (this.accumulator >= this.rt.timestep && n < MAX_SUBSTEPS) {
      this.stepOnce();
      this.accumulator -= this.rt.timestep;
      n++;
    }
    if (n === MAX_SUBSTEPS) this.accumulator = 0;
    this.wallAccum += wallDt;
    this.simAccum += n * this.rt.timestep;
    this.publish(false);
    return n;
  }

  /** Throttled publication of slow-changing readouts to the UI store (about 10 Hz). */
  private publish(force: boolean): void {
    const rt = this.rt;
    if (!rt) return;
    const now = performance.now();
    if (!force && now - this.lastPublish < 100) return;
    this.lastPublish = now;
    const b = rt.groundTruthBaseState();
    const sat = this.satWindow.length ? this.satWindow.reduce((a, c) => a + c, 0) / this.satWindow.length : 0;
    const over = this.overspeedWindow.length ? this.overspeedWindow.reduce((a, c) => a + c, 0) / this.overspeedWindow.length : 0;
    this.store.update({
      simTime: rt.time,
      ...(this.driver ? { qCmd: this.driverQ.slice() } : {}),
      base: rt.layout.floating ? { height: b.pos[2], tilt: b.tilt } : null,
      saturationFraction: sat,
      overspeedFraction: over,
      maxSpeedRatio: this.speedWindow.length ? Math.max(...this.speedWindow) : 0,
      realTimeFactor: this.wallAccum > 0 ? this.simAccum / this.wallAccum : 0,
      stepMs: this.stepTimeCount ? this.stepTimeAccum / this.stepTimeCount : 0,
      warnings: rt.warningCounts(),
    });
    this.wallAccum = 0;
    this.simAccum = 0;
    this.stepTimeAccum = 0;
    this.stepTimeCount = 0;
  }

  /** Ground-truth body transforms for rendering (simulator state). */
  renderTransforms(): { base: Mat4; links: Mat4[] } | null {
    if (!this.rt) return null;
    return { base: this.rt.groundTruthBaseTransform(), links: this.rt.groundTruthLinkTransforms() };
  }

  dispose(): void {
    this.abortRun('the simulation was closed');
    this.rt?.dispose();
    this.rt = null;
  }
}
