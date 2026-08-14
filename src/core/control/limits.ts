/** Angle wrapped to (-pi, pi]. */
export function wrapToPi(a: number): number {
  const w = a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));
  return w === -Math.PI ? Math.PI : w;
}

/**
 * Bounded target trajectory: the commanded joint target moves toward the
 * requested one no faster than the joint's declared velocity limit, and the
 * resulting commanded velocity is reported for the PD damping term.
 * Continuous joints move the shorter way around.
 */
export class TargetRateLimiter {
  readonly qCmd: Float64Array;
  readonly qdCmd: Float64Array;

  constructor(private readonly vmax: readonly number[], private readonly continuous: readonly boolean[], q0: ArrayLike<number>) {
    if (vmax.length !== q0.length || continuous.length !== q0.length) throw new RangeError('Limiter sizes do not match the joint vector');
    if (!vmax.every((v) => Number.isFinite(v) && v > 0)) throw new RangeError('Velocity limits must be positive and finite');
    this.qCmd = Float64Array.from(q0);
    this.qdCmd = new Float64Array(q0.length);
  }

  /** Jumps to a configuration (for resets); the commanded velocity becomes zero. */
  reset(q0: ArrayLike<number>): void {
    this.qCmd.set(Array.from(q0));
    this.qdCmd.fill(0);
  }

  /** Advances by dt toward `requested`; returns true when any joint was held back by its limit. */
  step(requested: ArrayLike<number>, dt: number): boolean {
    let limited = false;
    for (let i = 0; i < this.qCmd.length; i++) {
      const r = requested[i]!;
      if (!Number.isFinite(r) || !(dt > 0)) {
        this.qdCmd[i] = 0;
        continue;
      }
      const cur = this.qCmd[i]!;
      const err = this.continuous[i] ? wrapToPi(r - cur) : r - cur;
      const maxStep = this.vmax[i]! * dt;
      const d = err > maxStep ? maxStep : err < -maxStep ? -maxStep : err;
      if (d !== err) limited = true;
      this.qCmd[i] = this.continuous[i] ? wrapToPi(cur + d) : cur + d;
      this.qdCmd[i] = d / dt;
    }
    return limited;
  }
}

/** Largest ratio of measured speed to the velocity limit over all joints (above 1 means overspeed). */
export function speedRatio(qd: ArrayLike<number>, vmax: readonly number[]): number {
  let r = 0;
  for (let i = 0; i < vmax.length; i++) {
    const v = Math.abs(qd[i]!);
    if (Number.isFinite(v)) r = Math.max(r, v / vmax[i]!);
  }
  return r;
}
