import type { MainModule } from '@mujoco/mujoco';
import { pdStabilityMargin, type PdGains } from '../../core/control/pd';
import { PRESETS, type PresetId } from '../../core/model/catalog';
import type { RobotDescription } from '../../core/model/types';
import { validateRobot } from '../../core/model/validate';
import { experimentRobot, runChallenge, stepTest, type ChallengeId, type ChallengeResult, type ControllerSpec, type Run, type StepCommand, type StepTestResult } from '../../physics/experiments';
import { MujocoRuntime } from '../../physics/runtime';
import { Store } from '../store';

export type GainSet = 'a' | 'b';

export interface StabilityNote {
  ok: boolean;
  wnH: number;
  dampingNumber: number;
}

/** The setup a step-test result was produced with. */
export interface StepRunSetup {
  joint: number;
  /** Step as entered: degrees (revolute) or metres (prismatic). */
  step: number;
  durationS: number;
  command: StepCommand;
  a: ControllerSpec;
  b: ControllerSpec | null;
}

export interface ControlState {
  status: 'loading' | 'ready' | 'error';
  error: string | null;
  robot: RobotDescription;
  /** Preset id or 'custom' for a builder robot. */
  source: string;
  onStand: boolean;
  joint: number;
  /** Step size in degrees (revolute) or metres (prismatic). */
  step: number;
  durationS: number;
  /** How the step is commanded: at the joint's velocity limit, or as an ideal jump (analytical reference). */
  command: StepCommand;
  a: ControllerSpec;
  b: ControllerSpec;
  compare: boolean;
  /** Explicit-integration stability check for each gain set (lightest joint). */
  stability: Record<GainSet, StabilityNote> | null;
  running: { label: string; progress: number } | null;
  stepA: StepTestResult | null;
  stepB: StepTestResult | null;
  /** Setup of the shown step-test results. */
  stepSetup: StepRunSetup | null;
  challenges: Partial<Record<ChallengeId, { a: ChallengeResult; b: ChallengeResult | null; setup: { a: ControllerSpec; b: ControllerSpec | null } }>>;
  message: string | null;
}

export const CONTROL_PRESETS: PresetId[] = ['arm3', 'arm6', 'quadruped', 'hexapod'];
const DEFAULT_A: ControllerSpec = { gains: { kp: 60, kd: 3 }, gravityCompensation: false };
const DEFAULT_B: ControllerSpec = { gains: { kp: 60, kd: 3 }, gravityCompensation: true };

/** Time budget per animation frame for experiment runs, ms. */
const SLICE_MS = 12;

/**
 * Control lab: runs recorded experiments (step tests and challenges) on a
 * fresh simulator instance in time slices on the main thread, and keeps the
 * results for side-by-side comparison of two controllers.
 */
export class ControlSession {
  readonly store: Store<ControlState>;
  private mj: MainModule | null = null;
  private initPromise: Promise<void> | null = null;
  private active: Run<unknown> | null = null;
  /** Incremented by cancel(); an experiment stores results only if it is unchanged since the start. */
  private generation = 0;
  private minInertia = Number.NaN;
  private timestep = 0.002;

  constructor(private readonly loadModule: () => Promise<MainModule>, robot: RobotDescription = PRESETS.find((p) => p.id === 'arm3')!.make()) {
    this.store = new Store<ControlState>({
      status: 'loading', error: null, robot, source: robot.id, onStand: experimentRobot(robot).onStand, joint: 1, step: 20, durationS: 2, command: 'rate-limited',
      a: DEFAULT_A, b: DEFAULT_B, compare: true, stability: null, running: null, stepA: null, stepB: null, stepSetup: null, challenges: {}, message: null,
    });
  }

  get state(): ControlState {
    return this.store.get();
  }

  init(): Promise<void> {
    this.initPromise ??= (async () => {
      try {
        this.mj = await this.loadModule();
        this.prepare(this.state.robot);
        this.store.update({ status: 'ready' });
      } catch (e) {
        this.store.update({ status: 'error', error: `Physics engine failed to load: ${e instanceof Error ? e.message : String(e)}` });
      }
    })();
    return this.initPromise;
  }

  /** Measures the lightest joint inertia of the experiment robot for the stability notes. */
  private prepare(robot: RobotDescription): void {
    if (!this.mj) return;
    const rt = MujocoRuntime.create(this.mj, experimentRobot(robot).robot);
    try {
      this.minInertia = Math.min(...rt.jointInertias());
      this.timestep = rt.timestep;
    } finally {
      rt.dispose();
    }
    this.updateStability();
  }

  private updateStability(): void {
    if (!Number.isFinite(this.minInertia)) return;
    const s = this.state;
    const note = (g: PdGains): StabilityNote => pdStabilityMargin(g, this.minInertia, this.timestep);
    this.store.update({ stability: { a: note(s.a.gains), b: note(s.b.gains) } });
  }

  setPreset(id: PresetId): void {
    const p = PRESETS.find((x) => x.id === id);
    if (p) this.setRobot(p.make(), id);
  }

  setRobot(robot: RobotDescription, source = 'custom'): boolean {
    const issues = validateRobot(robot);
    if (issues.length) {
      this.store.update({ message: `Refused robot: ${issues[0]!.path}: ${issues[0]!.message}.` });
      return false;
    }
    this.cancel();
    const dof = robot.limbs.reduce((n, l) => n + l.joints.length, 0);
    this.store.update({
      robot, source, onStand: experimentRobot(robot).onStand, joint: Math.min(1, dof - 1), stepA: null, stepB: null, stepSetup: null, challenges: {}, message: null,
    });
    this.prepare(robot);
    return true;
  }

  // A run in progress belongs to the setup it started with; changing the setup cancels it.

  setJoint(j: number): void {
    const dof = this.state.robot.limbs.reduce((n, l) => n + l.joints.length, 0);
    if (j < 0 || j >= dof || j === this.state.joint) return;
    this.cancel();
    this.store.update({ joint: j, stepA: null, stepB: null, stepSetup: null });
  }

  setStep(v: number): void {
    if (!Number.isFinite(v) || v === 0 || v === this.state.step) return;
    this.cancel();
    this.store.update({ step: v });
  }

  setDuration(v: number): void {
    if (!Number.isFinite(v) || v < 0.5 || v > 10 || v === this.state.durationS) return;
    this.cancel();
    this.store.update({ durationS: v });
  }

  setCommand(c: StepCommand): void {
    if (c === this.state.command) return;
    this.cancel();
    this.store.update({ command: c });
  }

  /** Updates one gain set. Gains must be finite and non-negative; unstable gains are allowed but flagged. */
  setController(which: GainSet, patch: Partial<PdGains> & { gravityCompensation?: boolean }): boolean {
    const cur = this.state[which];
    const gains = { kp: patch.kp ?? cur.gains.kp, kd: patch.kd ?? cur.gains.kd };
    if (![gains.kp, gains.kd].every((v) => Number.isFinite(v) && v >= 0)) {
      this.store.update({ message: 'Gains must be finite and non-negative.' });
      return false;
    }
    this.cancel();
    this.store.update({ [which]: { gains, gravityCompensation: patch.gravityCompensation ?? cur.gravityCompensation }, message: null } as Partial<ControlState>);
    this.updateStability();
    return true;
  }

  setCompare(on: boolean): void {
    if (on === this.state.compare) return;
    this.cancel();
    this.store.update({ compare: on });
  }

  private jointIsAngular(): boolean {
    const j = this.state.robot.limbs.flatMap((l) => l.joints)[this.state.joint];
    return j?.type !== 'prismatic';
  }

  /** Runs `run` in slices of about SLICE_MS per frame, reporting progress. */
  private async slice<T>(label: string, run: Run<T>): Promise<T | null> {
    this.stopActive();
    this.active = run as Run<unknown>;
    this.store.update({ running: { label, progress: 0 } });
    try {
      for (;;) {
        const t0 = performance.now();
        let r = run.next();
        while (!r.done && performance.now() - t0 < SLICE_MS) r = run.next();
        if (this.active !== run) return null;
        if (r.done) return r.value;
        this.store.update({ running: { label, progress: r.value } });
        await new Promise((res) => setTimeout(res, 0));
        if (this.active !== run) return null;
      }
    } finally {
      if (this.active === run) {
        this.active = null;
        this.store.update({ running: null });
      }
    }
  }

  private stopActive(): void {
    const run = this.active;
    this.active = null;
    run?.return(undefined as never);
    if (run) this.store.update({ running: null });
  }

  /** Stops the running experiment; results of experiments started before are discarded. */
  cancel(): void {
    this.generation++;
    this.stopActive();
  }

  async runStepTest(): Promise<void> {
    const mj = this.mj;
    if (!mj) return;
    this.cancel();
    const gen = this.generation;
    const s = this.state;
    const step = this.jointIsAngular() ? (s.step * Math.PI) / 180 : s.step;
    const setup: StepRunSetup = { joint: s.joint, step: s.step, durationS: s.durationS, command: s.command, a: s.a, b: s.compare ? s.b : null };
    const cfg = { joint: s.joint, step, settleS: 0.5, durationS: s.durationS, command: s.command };
    try {
      const a = await this.slice('Step test, gain set A', stepTest(mj, s.robot, { ...cfg, controller: setup.a }));
      if (!a || gen !== this.generation) return;
      this.store.update({ stepA: a, stepB: null, stepSetup: { ...setup, b: null } });
      if (!setup.b) return;
      const b = await this.slice('Step test, gain set B', stepTest(mj, s.robot, { ...cfg, controller: setup.b }));
      if (b && gen === this.generation) this.store.update({ stepB: b, stepSetup: setup });
    } catch (e) {
      this.store.update({ message: e instanceof Error ? e.message : String(e) });
    }
  }

  async runChallenge(id: ChallengeId): Promise<void> {
    const mj = this.mj;
    if (!mj) return;
    this.cancel();
    const gen = this.generation;
    const s = this.state;
    try {
      const a = await this.slice(`Challenge, gain set A`, runChallenge(mj, s.robot, id, s.a));
      if (!a || gen !== this.generation) return;
      let b: ChallengeResult | null = null;
      if (s.compare && a.applicable) {
        b = await this.slice(`Challenge, gain set B`, runChallenge(mj, s.robot, id, s.b));
        if (!b || gen !== this.generation) return;
      }
      this.store.update({ challenges: { ...this.state.challenges, [id]: { a, b, setup: { a: s.a, b: b ? s.b : null } } } });
    } catch (e) {
      this.store.update({ message: e instanceof Error ? e.message : String(e) });
    }
  }

  dispose(): void {
    this.cancel();
  }
}
