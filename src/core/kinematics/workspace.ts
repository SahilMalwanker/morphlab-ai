import type { Mat4 } from '../math/mat4';
import { forwardKinematics, randomConfiguration, type KinematicModel } from './fk';

/**
 * Samples tip positions of one limb over uniformly random joint values within
 * limits (other limbs keep `q`). Returns xyz triples in the world frame.
 */
export function sampleWorkspace(model: KinematicModel, limbIndex: number, q: number[], count: number, rng: () => number, T_W_B?: Mat4): Float32Array {
  const limb = model.limbs[limbIndex];
  if (!limb) throw new RangeError(`No limb ${limbIndex}`);
  const out = new Float32Array(3 * count);
  for (let k = 0; k < count; k++) {
    const r = randomConfiguration(model, rng);
    const qk = q.slice();
    for (let i = limb.start; i < limb.start + limb.count; i++) qk[i] = r[i]!;
    const T = forwardKinematics(model, qk, T_W_B).tips[limbIndex]!;
    out[3 * k] = T[12]!;
    out[3 * k + 1] = T[13]!;
    out[3 * k + 2] = T[14]!;
  }
  return out;
}

export interface LimitContact {
  jointIndex: number;
  side: 'lower' | 'upper';
}

/** Joints of a limb sitting at (within tol of) a position limit. */
export function jointsAtLimits(model: KinematicModel, limbIndex: number, q: number[], tol = 1e-6): LimitContact[] {
  const limb = model.limbs[limbIndex]!;
  const out: LimitContact[] = [];
  for (let i = limb.start; i < limb.start + limb.count; i++) {
    const j = model.joints[i]!;
    if (q[i]! <= j.lower + tol) out.push({ jointIndex: i, side: 'lower' });
    else if (q[i]! >= j.upper - tol) out.push({ jointIndex: i, side: 'upper' });
  }
  return out;
}
