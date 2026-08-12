import { choleskySolve } from '../math/linalg';
import { multiply, type Mat4 } from '../math/mat4';
import { mulberry32 } from '../math/rng';
import { cross, isFiniteVec, type Vec3 } from '../math/vec3';
import { baseTransform, clampToLimits, jointMotion, type KinematicModel } from './fk';

/**
 * 'unreachable' is reported only when the target lies beyond a proven upper
 * bound on the limb's reach. 'stalled' means the solver stopped improving
 * (a local minimum or joint limits) without any proof that the target is
 * out of reach.
 */
export type IKStatus = 'converged' | 'unreachable' | 'stalled' | 'max-iterations' | 'invalid-input';

export interface IKOptions {
  /** Position tolerance, m. */
  tolerance?: number;
  /** Iterations per attempt. */
  maxIterations?: number;
  /** Largest joint change per iteration (rad or m, infinity norm). */
  maxStep?: number;
  /** Initial damping, m. */
  lambda?: number;
  /** Extra attempts from seeded random configurations when the first attempt fails. */
  restarts?: number;
  seed?: number;
  T_W_B?: Mat4;
}

export interface IKResult {
  status: IKStatus;
  /** Full joint vector (all limbs); always within joint limits. */
  q: number[];
  /** Distance between the target and the tip at `q`, m. */
  residual: number;
  iterations: number;
  restartsUsed: number;
  /** True when the target lies outside a proven upper bound on the limb reach. */
  beyondReach: boolean;
  message: string;
}

interface LimbPose {
  links: Mat4[];
  tip: Vec3;
}

function limbPose(model: KinematicModel, limbIndex: number, q: ArrayLike<number>, T_W_B: Mat4): LimbPose {
  const limb = model.limbs[limbIndex]!;
  const links: Mat4[] = [];
  let T = T_W_B;
  for (let i = limb.start; i < limb.start + limb.count; i++) {
    const j = model.joints[i]!;
    T = multiply(multiply(T, j.T_parent_joint), jointMotion(j, q[i]!));
    links.push(T);
  }
  const E = multiply(T, limb.T_last_tip);
  return { links, tip: [E[12]!, E[13]!, E[14]!] };
}

function limbJacobian(model: KinematicModel, limbIndex: number, pose: LimbPose): Float64Array {
  const limb = model.limbs[limbIndex]!;
  const n = limb.count;
  const J = new Float64Array(3 * n);
  for (let c = 0; c < n; c++) {
    const j = model.joints[limb.start + c]!;
    const T = pose.links[c]!;
    const a: Vec3 = [
      T[0]! * j.axis[0] + T[4]! * j.axis[1] + T[8]! * j.axis[2],
      T[1]! * j.axis[0] + T[5]! * j.axis[1] + T[9]! * j.axis[2],
      T[2]! * j.axis[0] + T[6]! * j.axis[1] + T[10]! * j.axis[2],
    ];
    const col = j.type === 'prismatic' ? a : cross(a, [pose.tip[0] - T[12]!, pose.tip[1] - T[13]!, pose.tip[2] - T[14]!]);
    J[c] = col[0];
    J[n + c] = col[1];
    J[2 * n + c] = col[2];
  }
  return J;
}

function dist(a: Vec3, b: Readonly<Vec3>): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

/** Damped least-squares step over the free joints: dq = Jf^T (Jf Jf^T + lambda^2 I)^-1 e. */
function dlsStep(J: Float64Array, n: number, e: Vec3, lambda: number, free: boolean[]): Float64Array {
  const A = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let s = r === c ? lambda * lambda : 0;
      for (let k = 0; k < n; k++) if (free[k]) s += J[r * n + k]! * J[c * n + k]!;
      A[r * 3 + c] = s;
    }
  }
  const y = choleskySolve(A, e, 3) ?? new Float64Array(3);
  const dq = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    if (!free[k]) continue;
    dq[k] = J[k]! * y[0]! + J[n + k]! * y[1]! + J[2 * n + k]! * y[2]!;
  }
  return dq;
}

interface Attempt {
  q: number[];
  residual: number;
  iterations: number;
  stalled: boolean;
}

function attempt(model: KinematicModel, limbIndex: number, target: Readonly<Vec3>, qStart: number[], T_W_B: Mat4, o: Required<Omit<IKOptions, 'T_W_B' | 'seed' | 'restarts'>>): Attempt {
  const limb = model.limbs[limbIndex]!;
  const n = limb.count;
  let q = clampToLimits(model, qStart);
  let pose = limbPose(model, limbIndex, q, T_W_B);
  let err = dist(pose.tip, target);
  let lambda = o.lambda;
  let iterations = 0;
  let slowSteps = 0;
  for (; iterations < o.maxIterations; iterations++) {
    if (err < o.tolerance) return { q, residual: err, iterations, stalled: false };
    const J = limbJacobian(model, limbIndex, pose);
    const e: Vec3 = [target[0] - pose.tip[0], target[1] - pose.tip[1], target[2] - pose.tip[2]];
    const free = new Array<boolean>(n).fill(true);
    let dq = dlsStep(J, n, e, lambda, free);
    // Active set: drop joints held at a limit that the step would push further out.
    for (let pass = 0; pass < n; pass++) {
      let changed = false;
      for (let k = 0; k < n; k++) {
        if (!free[k]) continue;
        const jt = model.joints[limb.start + k]!;
        const v = q[limb.start + k]!;
        if ((v <= jt.lower + 1e-12 && dq[k]! < 0) || (v >= jt.upper - 1e-12 && dq[k]! > 0)) {
          free[k] = false;
          changed = true;
        }
      }
      if (!changed) break;
      dq = dlsStep(J, n, e, lambda, free);
    }
    let maxAbs = 0;
    for (let k = 0; k < n; k++) maxAbs = Math.max(maxAbs, Math.abs(dq[k]!));
    const s = maxAbs > o.maxStep ? o.maxStep / maxAbs : 1;
    const qNew = q.slice();
    for (let k = 0; k < n; k++) qNew[limb.start + k] = q[limb.start + k]! + s * dq[k]!;
    const qClamped = clampToLimits(model, qNew);
    const poseNew = limbPose(model, limbIndex, qClamped, T_W_B);
    const errNew = dist(poseNew.tip, target);
    if (errNew < err) {
      slowSteps = err - errNew < 1e-9 * Math.max(err, 1e-3) ? slowSteps + 1 : 0;
      q = qClamped;
      pose = poseNew;
      err = errNew;
      lambda = Math.max(1e-6, lambda * 0.5);
      if (slowSteps >= 8) return { q, residual: err, iterations: iterations + 1, stalled: true };
    } else {
      lambda *= 4;
      if (lambda > 1e3) return { q, residual: err, iterations: iterations + 1, stalled: true };
    }
  }
  return { q, residual: err, iterations, stalled: err >= o.tolerance && lambda > 1 };
}

/**
 * Position-only inverse kinematics for one limb using damped least squares
 * with Levenberg-Marquardt damping adaptation, step clamping, joint-limit
 * clamping and an active set for joints held at limits.
 */
export function solveIK(model: KinematicModel, limbIndex: number, target: Readonly<Vec3>, q0: ArrayLike<number>, options: IKOptions = {}): IKResult {
  const T_W_B = options.T_W_B ?? baseTransform(model);
  const limb = model.limbs[limbIndex];
  const qIn = Array.from(q0);
  if (!limb || !isFiniteVec(target) || target.length !== 3 || qIn.length !== model.dof || !isFiniteVec(qIn)) {
    const q = qIn.length === model.dof && isFiniteVec(qIn) ? clampToLimits(model, qIn) : new Array<number>(model.dof).fill(0);
    return { status: 'invalid-input', q: clampToLimits(model, q), residual: NaN, iterations: 0, restartsUsed: 0, beyondReach: false, message: 'Target or joint vector is invalid (wrong size or not finite).' };
  }
  const o = {
    tolerance: options.tolerance ?? 1e-5,
    maxIterations: options.maxIterations ?? 200,
    maxStep: options.maxStep ?? 0.3,
    lambda: options.lambda ?? 0.01,
  };
  const firstOrigin = multiply(T_W_B, model.joints[limb.start]!.T_parent_joint);
  const beyondReach = dist([firstOrigin[12]!, firstOrigin[13]!, firstOrigin[14]!], target) > limb.reachBound;

  let best = attempt(model, limbIndex, target, qIn, T_W_B, o);
  let iterations = best.iterations;
  let restartsUsed = 0;
  const restarts = beyondReach ? 0 : options.restarts ?? 0;
  if (best.residual >= o.tolerance && restarts > 0) {
    const rng = mulberry32(options.seed ?? 1);
    for (let r = 0; r < restarts && best.residual >= o.tolerance; r++) {
      const qStart = qIn.slice();
      for (let k = 0; k < limb.count; k++) {
        const j = model.joints[limb.start + k]!;
        const lo = Number.isFinite(j.lower) ? j.lower : -Math.PI;
        const hi = Number.isFinite(j.upper) ? j.upper : Math.PI;
        qStart[limb.start + k] = lo + (hi - lo) * rng();
      }
      const a = attempt(model, limbIndex, target, qStart, T_W_B, o);
      iterations += a.iterations;
      restartsUsed++;
      // Prefer continuity: switch branches only for a solution or a clearly better (> 1 mm) closest point.
      if (a.residual < o.tolerance || a.residual < best.residual - 1e-3) best = a;
    }
  }
  const converged = best.residual < o.tolerance;
  const status: IKStatus = converged ? 'converged' : beyondReach ? 'unreachable' : best.stalled ? 'stalled' : 'max-iterations';
  const mm = (best.residual * 1000).toFixed(1);
  const message = converged
    ? 'Target reached.'
    : beyondReach
      ? `Target is outside the maximum reach of this limb; closest point is ${mm} mm away.`
      : status === 'stalled'
        ? `The solver stopped improving ${mm} mm from the target${restartsUsed ? ` after ${restartsUsed} restarts` : ''} (a local minimum or joint limits). This does not prove the target is out of reach.`
        : `Iteration limit reached; remaining error ${mm} mm.`;
  return { status, q: best.q, residual: best.residual, iterations, restartsUsed, beyondReach, message };
}
