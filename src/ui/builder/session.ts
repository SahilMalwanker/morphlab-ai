import {
  clampStance,
  DEFAULT_LEGGED,
  describeIssues,
  editLimb,
  generateArm,
  generateLegged,
  mirrorPartner,
  patchJoint,
  placeOnGround,
  removeLimb,
  setJointCount,
  type JointPatch,
  type LeggedSpec,
} from '../../core/model/builder';
import { PRESETS, type PresetId } from '../../core/model/catalog';
import { canonicalJson, morphologyHash } from '../../core/model/hash';
import type { BaseDescription, LimbDescription, RobotDescription } from '../../core/model/types';
import type { Vec3 } from '../../core/math/vec3';
import { exportBundle, importBundle } from '../../persistence/bundle';
import { Store } from '../store';

export interface HistoryEntry {
  version: number;
  /** SHA-256 morphology hash; null while it is being computed. */
  hash: string | null;
  change: string;
}

export interface BuilderState {
  robot: RobotDescription;
  version: number;
  /** Incremented on every edit attempt, accepted or refused (lets fields revert). */
  attempt: number;
  hash: string | null;
  history: HistoryEntry[];
  selectedLimb: string;
  selectedJoint: number;
  mirror: boolean;
  /** Messages from the last refused edit; the robot was not changed. */
  issues: string[];
  notice: string | null;
  legged: LeggedSpec;
  arm: { dof: number; reach: number };
}

const MAX_HISTORY = 30;

/** Optional storage hook: every accepted version is saved (IndexedDB in the app). */
export interface BuilderPersistence {
  saveRobot(robot: RobotDescription): Promise<void>;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Framework-free robot builder. Every edit produces a candidate description
 * that must pass validation before it replaces the current robot; each
 * accepted edit is a new version with its own morphology hash.
 */
export class BuilderSession {
  readonly store: Store<BuilderState>;
  private persistence: BuilderPersistence | null = null;

  /** Connects storage; the current robot is saved right away and after every accepted edit. */
  attachPersistence(p: BuilderPersistence | null): void {
    this.persistence = p;
    this.save(this.state.robot);
  }

  private save(robot: RobotDescription): void {
    this.persistence?.saveRobot(robot).catch((e) => this.store.update({ notice: `The robot could not be saved: ${errorText(e)}` }));
  }

  /** Loads a stored robot as a new version, keeping it byte-identical (no re-seating). */
  loadRobot(robot: RobotDescription, label = `Loaded saved robot ${robot.name}`): boolean {
    return this.apply(() => robot, label, false);
  }

  constructor(initial: RobotDescription) {
    this.store = new Store<BuilderState>({
      robot: initial,
      version: 1,
      attempt: 0,
      hash: null,
      history: [{ version: 1, hash: null, change: `Started from ${initial.name}` }],
      selectedLimb: initial.limbs[0]?.id ?? '',
      selectedJoint: 0,
      mirror: true,
      issues: [],
      notice: null,
      legged: { ...DEFAULT_LEGGED },
      arm: { dof: 4, reach: 0.8 },
    });
    this.hashVersion(initial, 1);
  }

  get state(): BuilderState {
    return this.store.get();
  }

  get limb(): LimbDescription | undefined {
    return this.state.robot.limbs.find((l) => l.id === this.state.selectedLimb);
  }

  get partner(): LimbDescription | undefined {
    const id = mirrorPartner(this.state.robot, this.state.selectedLimb);
    return id ? this.state.robot.limbs.find((l) => l.id === id) : undefined;
  }

  private hashVersion(robot: RobotDescription, version: number): void {
    void morphologyHash(robot).then((hash) => {
      const s = this.state;
      this.store.update({
        hash: s.version === version ? hash : s.hash,
        history: s.history.map((h) => (h.version === version ? { ...h, hash } : h)),
      });
    });
  }

  /**
   * Validates and applies a candidate. `ground` marks geometric edits: the
   * stance is clamped into edited limits and the robot is re-seated on the
   * ground (neither the stance nor the initial pose is morphology).
   */
  private apply(make: () => RobotDescription, change: string, ground: boolean): boolean {
    const attempt = this.state.attempt + 1;
    let next: RobotDescription;
    try {
      next = make();
    } catch (e) {
      this.store.update({ issues: [`${errorText(e).replace(/\.?$/, '.')}`], notice: null, attempt });
      return false;
    }
    // Submitting a value that is already there is not an edit.
    if (canonicalJson(next) === canonicalJson(this.state.robot)) {
      if (this.state.issues.length || this.state.notice) this.store.update({ issues: [], notice: null });
      return true;
    }
    if (ground) {
      const c = clampStance(next);
      next = c.robot;
      if (c.changed) change += '; stance clamped to the new limits';
    }
    const issues = describeIssues(next);
    if (issues.length) {
      this.store.update({ issues, notice: null, attempt });
      return false;
    }
    if (ground) next = placeOnGround(next);
    const s = this.state;
    const version = s.version + 1;
    const selectedLimb = next.limbs.some((l) => l.id === s.selectedLimb) ? s.selectedLimb : next.limbs[0]!.id;
    const limb = next.limbs.find((l) => l.id === selectedLimb)!;
    this.store.update({
      robot: next,
      version,
      attempt,
      hash: null,
      history: [{ version, hash: null, change }, ...s.history].slice(0, MAX_HISTORY),
      selectedLimb,
      selectedJoint: Math.min(s.selectedJoint, limb.joints.length - 1),
      issues: [],
      notice: null,
    });
    this.hashVersion(next, version);
    this.save(next);
    return true;
  }

  startFromPreset(id: PresetId): boolean {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return false;
    return this.apply(() => p.make(), `Loaded preset ${p.label}`, false);
  }

  generateLegged(spec: LeggedSpec): boolean {
    this.store.update({ legged: spec });
    return this.apply(() => generateLegged(spec), `Generated ${2 * spec.pairs} legs with ${spec.dofPerLeg} joint(s) each`, false);
  }

  generateArm(dof: number, reach: number): boolean {
    this.store.update({ arm: { dof, reach } });
    return this.apply(() => generateArm(dof, reach), `Generated a ${dof}-joint arm with ${reach} m reach`, false);
  }

  rename(name: string): boolean {
    return this.apply(() => ({ ...this.state.robot, name: name.trim() }), `Renamed to ${name.trim()}`, false);
  }

  setBase(patch: Partial<Pick<BaseDescription, 'kind' | 'size' | 'mass'>>): boolean {
    const r = this.state.robot;
    const what = Object.keys(patch).join(', ');
    return this.apply(() => ({ ...r, base: { ...r.base, ...patch } }), `Base ${what} changed`, true);
  }

  private limbLabel(): string {
    return this.limb?.name ?? this.state.selectedLimb;
  }

  updateJoint(patch: JointPatch): boolean {
    const s = this.state;
    const jointName = this.limb?.joints[s.selectedJoint]?.name ?? `joint ${s.selectedJoint + 1}`;
    const change = `${this.limbLabel()}, ${jointName}: ${Object.keys(patch).join(', ')}${s.mirror && this.partner ? ' (mirrored)' : ''}`;
    return this.apply(
      () => editLimb(s.robot, s.selectedLimb, (l) => ({ ...l, joints: l.joints.map((j, i) => (i === s.selectedJoint ? patchJoint(j, patch) : j)) }), s.mirror),
      change,
      true,
    );
  }

  updateTip(patch: { xyz?: Vec3; radius?: number }): boolean {
    const s = this.state;
    return this.apply(
      () => editLimb(s.robot, s.selectedLimb, (l) => ({ ...l, tip: { xyz: patch.xyz ?? l.tip.xyz, radius: patch.radius ?? l.tip.radius } }), s.mirror),
      `${this.limbLabel()}: tip ${Object.keys(patch).join(', ')}${s.mirror && this.partner ? ' (mirrored)' : ''}`,
      true,
    );
  }

  setJointCount(n: number): boolean {
    const s = this.state;
    return this.apply(() => setJointCount(s.robot, s.selectedLimb, n, s.mirror), `${this.limbLabel()}: ${n} joint(s)${s.mirror && this.partner ? ' (mirrored)' : ''}`, true);
  }

  removeSelectedLimb(): boolean {
    const s = this.state;
    const partner = s.mirror ? this.partner : undefined;
    return this.apply(() => removeLimb(s.robot, s.selectedLimb, s.mirror), `Removed ${this.limbLabel()}${partner ? ` and ${partner.name}` : ''}`, true);
  }

  selectLimb(id: string): void {
    if (this.state.robot.limbs.some((l) => l.id === id)) this.store.update({ selectedLimb: id, selectedJoint: 0, issues: [] });
  }

  selectJoint(i: number): void {
    const n = this.limb?.joints.length ?? 0;
    if (i >= 0 && i < n) this.store.update({ selectedJoint: i, issues: [] });
  }

  setMirror(on: boolean): void {
    this.store.update({ mirror: on });
  }

  /** Canonical bundle text for the current robot. */
  exportText(exportedAt: string = new Date().toISOString()): string {
    return exportBundle([this.state.robot], exportedAt);
  }

  /** Reports an import refused before reading the file (for example, too large). */
  refuseImport(message: string): void {
    this.store.update({ issues: [message], notice: null });
  }

  /** Loads the first valid robot of a bundle; everything rejected is reported. */
  importText(text: string): boolean {
    const res = importBundle(text);
    if (res.errors.length) {
      this.store.update({ issues: res.errors, notice: null });
      return false;
    }
    const rejected = res.rejected.flatMap((r) => r.reasons.map((m) => `${r.kind === 'robot' ? 'Robot' : 'Skill'} ${r.index + 1}: ${m}`));
    const skillNote = res.skills.length ? ` The file also has ${res.skills.length} skill${res.skills.length === 1 ? '' : 's'}; skills are imported in the Skills tab.` : '';
    const robot = res.robots[0];
    if (!robot) {
      this.store.update({ issues: [...(rejected.length ? rejected : ['The file contains no robots.']), ...(skillNote ? [skillNote.trim()] : [])], notice: null });
      return false;
    }
    const ok = this.apply(() => robot, `Imported ${robot.name}`, false);
    const extra = res.robots.length > 1 ? ` The file has ${res.robots.length} robots; the first was loaded.` : '';
    this.store.update({ notice: `Imported ${robot.name}.${extra}${skillNote}`, issues: rejected });
    return ok;
  }
}
