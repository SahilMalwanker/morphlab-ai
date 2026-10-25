import { baseTransform, buildKinematicModel, clampToLimits, forwardKinematics, type FKResult, type KinematicModel } from '../../core/kinematics/fk';
import { solveIK, type IKResult } from '../../core/kinematics/ik';
import { manipulability, positionJacobian, type Manipulability } from '../../core/kinematics/jacobian';
import { jointsAtLimits, sampleWorkspace, type LimitContact } from '../../core/kinematics/workspace';
import { mulberry32 } from '../../core/math/rng';
import { isFiniteVec, type Vec3 } from '../../core/math/vec3';
import { homeConfiguration } from '../../core/model/catalog';
import type { RobotDescription } from '../../core/model/types';
import { validateRobot } from '../../core/model/validate';
import { Store } from '../store';

export interface Overlays {
  frames: boolean;
  axes: boolean;
  ellipsoid: boolean;
  workspace: boolean;
}

export interface KinematicsState {
  description: RobotDescription;
  /** Preset id the robot came from, or 'custom' (chain editor or builder). */
  source: string;
  model: KinematicModel;
  q: number[];
  limbIndex: number;
  target: Vec3;
  followTarget: boolean;
  ik: IKResult | null;
  fk: FKResult;
  manip: Manipulability | null;
  limits: LimitContact[];
  overlays: Overlays;
  workspace: Float32Array | null;
  inputError: string | null;
}

/** Limbs that can be targeted with position IK (wheels cannot). */
export function targetableLimbs(desc: RobotDescription): number[] {
  return desc.limbs.map((l, i) => (l.role === 'wheel' ? -1 : i)).filter((i) => i >= 0);
}

function tipOf(fk: FKResult, limb: number): Vec3 {
  const T = fk.tips[limb]!;
  return [T[12]!, T[13]!, T[14]!];
}

/**
 * Framework-free controller for the kinematics playground. All numbers come
 * from src/core; the UI and renderer only read the published state.
 */
export class KinematicsSession {
  readonly store: Store<KinematicsState>;

  constructor(desc: RobotDescription, source = desc.id) {
    this.store = new Store(KinematicsSession.initialState(desc, source));
  }

  private static initialState(desc: RobotDescription, source: string, overlays?: Overlays): KinematicsState {
    const model = buildKinematicModel(desc);
    const q = homeConfiguration(desc);
    const fk = forwardKinematics(model, q, baseTransform(model));
    const limbIndex = targetableLimbs(desc)[0] ?? -1;
    const target: Vec3 = limbIndex >= 0 ? tipOf(fk, limbIndex) : [0, 0, 0];
    const state: KinematicsState = {
      description: desc,
      source,
      model,
      q,
      limbIndex,
      target,
      followTarget: true,
      ik: null,
      fk,
      manip: null,
      limits: [],
      overlays: overlays ?? { frames: false, axes: false, ellipsoid: false, workspace: false },
      workspace: null,
      inputError: null,
    };
    return KinematicsSession.derive(state);
  }

  private static derive(s: KinematicsState): KinematicsState {
    const fk = forwardKinematics(s.model, s.q, baseTransform(s.model));
    const manip = s.limbIndex >= 0 ? manipulability(positionJacobian(s.model, s.limbIndex, fk), s.model.limbs[s.limbIndex]!.count) : null;
    const limits = s.limbIndex >= 0 ? jointsAtLimits(s.model, s.limbIndex, s.q) : [];
    return { ...s, fk, manip, limits };
  }

  get state(): KinematicsState {
    return this.store.get();
  }

  private commit(patch: Partial<KinematicsState>): void {
    this.store.set(KinematicsSession.derive({ ...this.state, ...patch }));
  }

  /** Replaces the robot. Invalid descriptions are rejected with a message and not applied. */
  setRobot(desc: RobotDescription, source = 'custom'): string[] {
    const issues = validateRobot(desc);
    if (issues.length > 0) {
      const msgs = issues.map((i) => `${i.path}: ${i.message}`);
      this.store.update({ inputError: msgs[0]! });
      return msgs;
    }
    const next = KinematicsSession.initialState(desc, source, this.state.overlays);
    this.store.set(next.overlays.workspace ? { ...next, workspace: this.computeWorkspace(next) } : next);
    return [];
  }

  setJoint(index: number, value: number): void {
    if (!Number.isFinite(value)) {
      this.store.update({ inputError: 'Joint value must be a finite number.' });
      return;
    }
    const q = this.state.q.slice();
    if (index < 0 || index >= q.length) return;
    q[index] = value;
    this.commit({ q: clampToLimits(this.state.model, q), ik: null, inputError: null });
  }

  selectLimb(limbIndex: number): void {
    if (!targetableLimbs(this.state.description).includes(limbIndex)) return;
    const next = { ...this.state, limbIndex, ik: null, inputError: null, target: tipOf(this.state.fk, limbIndex) };
    this.store.set(KinematicsSession.derive(next.overlays.workspace ? { ...next, workspace: this.computeWorkspace(next) } : next));
  }

  setFollowTarget(on: boolean): void {
    this.store.update({ followTarget: on });
    if (on) this.solve();
  }

  /** Moves the IK target; non-finite input is rejected and the scene is left unchanged. */
  setTarget(p: readonly number[]): void {
    if (p.length !== 3 || !isFiniteVec(p)) {
      this.store.update({ inputError: 'Target coordinates must be finite numbers in metres.' });
      return;
    }
    this.store.update({ target: [p[0]!, p[1]!, p[2]!], inputError: null });
    if (this.state.followTarget) this.solve();
  }

  solve(): IKResult | null {
    const s = this.state;
    if (s.limbIndex < 0) return null;
    const ik = solveIK(s.model, s.limbIndex, s.target, s.q, { restarts: 2, seed: 1 });
    this.commit({ q: ik.q, ik });
    return ik;
  }

  /** Returns joints to the home pose and places the target at the tip. */
  reset(): void {
    const s = this.state;
    const q = homeConfiguration(s.description);
    const fk = forwardKinematics(s.model, q, baseTransform(s.model));
    this.commit({ q, ik: null, inputError: null, target: s.limbIndex >= 0 ? tipOf(fk, s.limbIndex) : s.target });
  }

  setOverlay(name: keyof Overlays, on: boolean): void {
    const overlays = { ...this.state.overlays, [name]: on };
    const workspace = name === 'workspace' ? (on ? this.computeWorkspace(this.state) : null) : this.state.workspace;
    this.store.update({ overlays, workspace });
  }

  private computeWorkspace(s: KinematicsState): Float32Array | null {
    if (s.limbIndex < 0) return null;
    return sampleWorkspace(s.model, s.limbIndex, s.q, 4000, mulberry32(12345), baseTransform(s.model));
  }
}
