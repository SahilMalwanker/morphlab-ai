import { wrapToPi } from './limits';

export interface PdGains {
  /** N*m/rad (revolute) or N/m (prismatic). */
  kp: number;
  /** N*m*s/rad (revolute) or N*s/m (prismatic). */
  kd: number;
}

export interface PdResult {
  /** Torque requested by the control law before limits. */
  tauCmd: Float64Array;
  /** Torque after saturation; the value sent to the actuator. */
  tauApplied: Float64Array;
  saturated: Uint8Array;
  /** True when an input was non-finite; the affected joints receive zero torque. */
  fault: boolean;
}

export interface PdInput {
  qCmd: ArrayLike<number>;
  qdCmd?: ArrayLike<number>;
  qMeas: ArrayLike<number>;
  qdMeas: ArrayLike<number>;
  tauFeedforward?: ArrayLike<number>;
  /** Joints without position limits (continuous): their position error is wrapped to (-pi, pi]. */
  continuous?: ArrayLike<boolean>;
}

/**
 * Joint-space PD law with saturation:
 * tauCmd = kp (qCmd - qMeas) + kd (qdCmd - qdMeas) + tauFf; tauApplied = clamp(tauCmd, +-effort).
 */
export function pdControl(gains: readonly PdGains[], effort: ArrayLike<number>, input: PdInput, out?: PdResult): PdResult {
  const n = gains.length;
  const res = out ?? { tauCmd: new Float64Array(n), tauApplied: new Float64Array(n), saturated: new Uint8Array(n), fault: false };
  res.fault = false;
  for (let i = 0; i < n; i++) {
    const g = gains[i]!;
    const raw = input.qCmd[i]! - input.qMeas[i]!;
    const e = input.continuous?.[i] ? wrapToPi(raw) : raw;
    const ed = (input.qdCmd?.[i] ?? 0) - input.qdMeas[i]!;
    const tau = g.kp * e + g.kd * ed + (input.tauFeedforward?.[i] ?? 0);
    const lim = effort[i]!;
    if (!Number.isFinite(tau) || !Number.isFinite(lim)) {
      res.tauCmd[i] = 0;
      res.tauApplied[i] = 0;
      res.saturated[i] = 0;
      res.fault = true;
      continue;
    }
    const applied = tau > lim ? lim : tau < -lim ? -lim : tau;
    res.tauCmd[i] = tau;
    res.tauApplied[i] = applied;
    res.saturated[i] = applied !== tau ? 1 : 0;
  }
  return res;
}

/** Natural frequency and damping ratio of a PD-controlled joint with effective inertia I. */
export function secondOrderParameters(g: PdGains, inertia: number): { wn: number; zeta: number } {
  const wn = Math.sqrt(g.kp / inertia);
  return { wn, zeta: g.kd / (2 * Math.sqrt(g.kp * inertia)) };
}

/** Gains for a desired natural frequency and damping ratio. */
export function gainsFor(wn: number, zeta: number, inertia: number): PdGains {
  const kp = inertia * wn * wn;
  return { kp, kd: 2 * zeta * Math.sqrt(kp * inertia) };
}

/**
 * Conservative explicit-integration check for a PD joint with timestep h:
 * the PD torque is explicit in the integrator, so require wn*h < 0.2 and kd*h/I < 1.
 */
export function pdStabilityMargin(g: PdGains, inertia: number, h: number): { ok: boolean; wnH: number; dampingNumber: number } {
  const { wn } = secondOrderParameters(g, inertia);
  const wnH = wn * h;
  const dampingNumber = (g.kd * h) / inertia;
  return { ok: wnH < 0.2 && dampingNumber < 1, wnH, dampingNumber };
}

/**
 * Starting gains for a robot without tuned defaults: 40 / 1 for floating
 * bases, 60 / 3 for fixed bases, halved until the explicit-stability check
 * passes for the lightest joint.
 */
export function suggestGains(floating: boolean, minInertia: number, h: number): { gains: PdGains; reduced: boolean } {
  let gains: PdGains = floating ? { kp: 40, kd: 1 } : { kp: 60, kd: 3 };
  let reduced = false;
  for (let i = 0; i < 40 && !pdStabilityMargin(gains, minInertia, h).ok; i++) {
    gains = { kp: gains.kp / 2, kd: gains.kd / 2 };
    reduced = true;
  }
  return { gains, reduced };
}
