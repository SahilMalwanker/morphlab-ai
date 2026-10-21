import { forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { solveIK } from '../core/kinematics/ik';
import { identity, type Quat } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';

export interface FootLiftParams {
  /** Body shift toward the support-triangle centroid, m. */
  shiftM: number;
  /** Foot lift height above its stance position, m. */
  liftHeightM: number;
  holdS: number;
  settleS: number;
  shiftS: number;
  liftS: number;
  lowerS: number;
  unshiftS: number;
  /** Touch force above which the lowered foot counts as landed, N. */
  touchdownN: number;
  /** Abort and return to stance when the IMU tilt exceeds this, rad. */
  abortTiltRad: number;
}

export const DEFAULT_FOOT_LIFT: FootLiftParams = {
  shiftM: 0.05,
  liftHeightM: 0.06,
  holdS: 2.0,
  settleS: 0.5,
  shiftS: 1.0,
  liftS: 0.5,
  lowerS: 0.6,
  unshiftS: 1.0,
  touchdownN: 2,
  abortTiltRad: 0.5,
};

/** Accepted range of every state-machine parameter, inclusive (used by the editor, import and storage). */
export const FOOT_LIFT_BOUNDS: Record<keyof FootLiftParams, [number, number]> = {
  shiftM: [0, 0.1],
  liftHeightM: [0.01, 0.15],
  holdS: [0.5, 5],
  settleS: [0, 5],
  shiftS: [0.2, 3],
  liftS: [0.1, 2],
  lowerS: [0.1, 2],
  unshiftS: [0.1, 5],
  touchdownN: [0, 50],
  abortTiltRad: [0.05, 1.5],
};

/** Problems with a state-machine parameter set, one message per parameter; empty when valid. */
export function footLiftParamProblems(p: Record<string, unknown>): string[] {
  const out: string[] = [];
  for (const [key, [lo, hi]] of Object.entries(FOOT_LIFT_BOUNDS)) {
    const v = p[key];
    if (typeof v !== 'number' || !Number.isFinite(v)) out.push(`${key} is missing or not a finite number`);
    else if (v < lo || v > hi) out.push(`${key} = ${v} is outside ${lo} to ${hi}`);
  }
  return out;
}

export type FootLiftPhase = 'settle' | 'shift' | 'lift' | 'hold' | 'lower' | 'unshift' | 'done' | 'aborted';

/** Observation available in sensor-only mode. Deliberately has no ground-truth fields. */
export interface FootLiftObservation {
  q: ArrayLike<number>;
  touch: readonly number[];
  imuQuat: Quat;
  /** Joint velocities from the encoders, rad/s (m/s for sliding joints). */
  qd?: ArrayLike<number>;
  /** IMU angular rate in the body frame, rad/s. */
  gyro?: Vec3;
}

function smooth(x: number): number {
  const s = Math.min(1, Math.max(0, x));
  return 0.5 - 0.5 * Math.cos(Math.PI * s);
}

/** Tilt from vertical encoded by an orientation quaternion [w, x, y, z]. */
export function tiltFromQuat(q: Quat): number {
  const [w, x, y, z] = q;
  const r22 = 1 - 2 * (x * x + y * y);
  const n = w * w + x * x + y * y + z * z;
  return Math.acos(Math.max(-1, Math.min(1, r22 / (n || 1))));
}

/**
 * Hand-designed state machine (method label: state-machine). Shifts the body
 * over the support triangle of the other legs, lifts one foot with leg IK in
 * the base frame, holds, lowers until the touch sensor reports contact, then
 * shifts back. Inputs are joint encoders, IMU orientation and foot touch only.
 */
export class FootLiftController {
  readonly nominalFeet: Vec3[];
  readonly shift: Vec3;
  private targets: number[];
  private phase: FootLiftPhase = 'settle';
  private phaseStart = 0;

  constructor(
    private readonly km: KinematicModel,
    readonly limbIndex: number,
    private readonly stanceQ: number[],
    readonly params: FootLiftParams = DEFAULT_FOOT_LIFT,
  ) {
    if (!km.limbs[limbIndex]) throw new RangeError(`No limb ${limbIndex}`);
    const fk = forwardKinematics(km, stanceQ, identity());
    this.nominalFeet = fk.tips.map((T) => [T[12]!, T[13]!, T[14]!] as Vec3);
    const support = this.nominalFeet.filter((_, i) => i !== limbIndex);
    if (support.length < 3) throw new RangeError('Foot lift needs at least three supporting legs');
    const cx = support.reduce((s, p) => s + p[0], 0) / support.length;
    const cy = support.reduce((s, p) => s + p[1], 0) / support.length;
    const n = Math.hypot(cx, cy) || 1;
    this.shift = [(cx / n) * params.shiftM, (cy / n) * params.shiftM, 0];
    this.targets = stanceQ.slice();
  }

  get currentPhase(): FootLiftPhase {
    return this.phase;
  }

  private enter(phase: FootLiftPhase, t: number): void {
    this.phase = phase;
    this.phaseStart = t;
  }

  /** Advances the state machine and returns joint position targets (rad). */
  update(t: number, obs: FootLiftObservation): number[] {
    const p = this.params;
    if (this.phase !== 'aborted' && this.phase !== 'done' && tiltFromQuat(obs.imuQuat) > p.abortTiltRad) this.enter('aborted', t);
    const dt = t - this.phaseStart;
    switch (this.phase) {
      case 'settle': if (dt >= p.settleS) this.enter('shift', t); break;
      case 'shift': if (dt >= p.shiftS) this.enter('lift', t); break;
      case 'lift': if (dt >= p.liftS) this.enter('hold', t); break;
      case 'hold': if (dt >= p.holdS) this.enter('lower', t); break;
      case 'lower':
        if ((dt >= p.lowerS && (obs.touch[this.limbIndex] ?? 0) > p.touchdownN) || dt >= p.lowerS + 0.5) this.enter('unshift', t);
        break;
      case 'unshift': if (dt >= p.unshiftS) this.enter('done', t); break;
      default: break;
    }
    const e = t - this.phaseStart;
    let alpha = 0, beta = 0;
    switch (this.phase) {
      case 'shift': alpha = smooth(e / p.shiftS); break;
      case 'lift': alpha = 1; beta = smooth(e / p.liftS); break;
      case 'hold': alpha = 1; beta = 1; break;
      case 'lower': alpha = 1; beta = 1 - smooth(e / p.lowerS); break;
      case 'unshift': alpha = 1 - smooth(e / p.unshiftS); break;
      default: break;
    }
    if (this.phase === 'aborted') return this.stanceQ.slice();
    this.km.limbs.forEach((_, li) => {
      const nom = this.nominalFeet[li]!;
      const target: Vec3 = [nom[0] - alpha * this.shift[0], nom[1] - alpha * this.shift[1], nom[2] + (li === this.limbIndex ? beta * p.liftHeightM : 0)];
      const res = solveIK(this.km, li, target, this.targets, { T_W_B: identity(), maxIterations: 30, tolerance: 1e-5 });
      if (res.status !== 'invalid-input') this.targets = res.q;
    });
    return this.targets.slice();
  }
}
