export interface StepMetrics {
  /** 10 % to 90 % rise time, s; null if the 90 % level is never reached. */
  riseTimeS: number | null;
  /** Time of the extreme value beyond the reference, s; null without overshoot. */
  peakTimeS: number | null;
  /** Overshoot as a percentage of the step size (>= 0). */
  overshootPct: number;
  /** Time after which the response stays within the band, s; null if it never settles. */
  settlingTimeS: number | null;
  /** Reference minus the mean of the last 10 % of samples. */
  steadyStateError: number;
}

function crossing(t: ArrayLike<number>, y: ArrayLike<number>, level: number, sign: number): number | null {
  for (let i = 1; i < t.length; i++) {
    const a = sign * (y[i - 1]! - level), b = sign * (y[i]! - level);
    if (a < 0 && b >= 0) return t[i - 1]! + ((t[i]! - t[i - 1]!) * -a) / (b - a);
  }
  return null;
}

/** Step-response metrics for a step from y0 to yRef applied at t[0]. */
export function stepMetrics(t: ArrayLike<number>, y: ArrayLike<number>, y0: number, yRef: number, band = 0.02): StepMetrics {
  if (t.length !== y.length || t.length < 2) throw new RangeError('Need matching time and value arrays');
  const delta = yRef - y0;
  if (delta === 0) throw new RangeError('Step size must be non-zero');
  const sign = Math.sign(delta);
  const t0 = t[0]!;
  const r10 = crossing(t, y, y0 + 0.1 * delta, sign);
  const r90 = crossing(t, y, y0 + 0.9 * delta, sign);
  let peakIdx = -1, peakVal = -Infinity;
  for (let i = 0; i < y.length; i++) {
    const v = sign * (y[i]! - yRef);
    if (v > peakVal) { peakVal = v; peakIdx = i; }
  }
  const overshootPct = Math.max(0, (peakVal / Math.abs(delta)) * 100);
  const tol = band * Math.abs(delta);
  let lastOutside = -1;
  for (let i = 0; i < y.length; i++) if (Math.abs(y[i]! - yRef) > tol) lastOutside = i;
  const settlingTimeS = lastOutside === y.length - 1 ? null : lastOutside < 0 ? 0 : t[lastOutside + 1]! - t0;
  const n0 = Math.floor(0.9 * y.length);
  let mean = 0;
  for (let i = n0; i < y.length; i++) mean += y[i]!;
  mean /= y.length - n0;
  return {
    riseTimeS: r10 !== null && r90 !== null ? r90 - r10 : null,
    peakTimeS: overshootPct > 0 ? t[peakIdx]! - t0 : null,
    overshootPct,
    settlingTimeS,
    steadyStateError: yRef - mean,
  };
}

/**
 * True when the response comes to rest outside the settling band: over the
 * final 10 % of the record it varies by less than the band, but its mean is
 * further than the band from yRef, so it never settles. A joint that sags
 * under a static load does this; a response still moving or ringing does not.
 */
export function restsOffTarget(y: ArrayLike<number>, y0: number, yRef: number, band = 0.02): boolean {
  const tol = band * Math.abs(yRef - y0);
  const n0 = Math.floor(0.9 * y.length);
  if (!(tol > 0) || n0 >= y.length) return false;
  let mean = 0, lo = Infinity, hi = -Infinity;
  for (let i = n0; i < y.length; i++) {
    const v = y[i]!;
    mean += v;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  mean /= y.length - n0;
  return Number.isFinite(mean) && hi - lo <= tol && Math.abs(yRef - mean) > tol;
}

/** Unit step response of y'' + 2 zeta wn y' + wn^2 y = wn^2 u, from rest. */
export function secondOrderStep(t: number, wn: number, zeta: number): number {
  if (!(wn > 0) || !(zeta >= 0)) throw new RangeError('wn must be positive and zeta non-negative');
  if (zeta < 1) {
    const wd = wn * Math.sqrt(1 - zeta * zeta);
    return 1 - Math.exp(-zeta * wn * t) * (Math.cos(wd * t) + (zeta / Math.sqrt(1 - zeta * zeta)) * Math.sin(wd * t));
  }
  if (zeta === 1) return 1 - Math.exp(-wn * t) * (1 + wn * t);
  const r = Math.sqrt(zeta * zeta - 1);
  const p1 = -wn * (zeta - r), p2 = -wn * (zeta + r);
  return 1 + (p2 * Math.exp(p1 * t) - p1 * Math.exp(p2 * t)) / (p1 - p2);
}

/** Metrics of the ideal continuous second-order system, sampled finely. */
export function analyticStepMetrics(wn: number, zeta: number, horizonS: number, band = 0.02): StepMetrics {
  const n = 20000;
  const t = new Float64Array(n + 1), y = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) {
    t[i] = (horizonS * i) / n;
    y[i] = secondOrderStep(t[i]!, wn, zeta);
  }
  return stepMetrics(t, y, 0, 1, band);
}

/**
 * Response of the linear joint model I y'' + kd y' + kp y = kp r(t), from
 * rest at y0, to an arbitrary command sequence held constant over each step
 * of length h (as the simulation applies it). Classical Runge-Kutta 4.
 * Returns y at the start of every step, aligned with `r`.
 */
export function linearResponse(kp: number, kd: number, inertia: number, h: number, r: ArrayLike<number>, y0: number): Float64Array {
  if (!(kp >= 0) || !(kd >= 0) || !(inertia > 0) || !(h > 0) || !Number.isFinite(y0)) throw new RangeError('Need kp, kd >= 0, a positive inertia and step, and a finite start');
  const acc = (p: number, v: number, u: number) => (kp * (u - p) - kd * v) / inertia;
  const y = new Float64Array(r.length);
  let p = y0, v = 0;
  for (let i = 0; i < r.length; i++) {
    y[i] = p;
    const u = r[i]!;
    const k1p = v, k1v = acc(p, v, u);
    const k2p = v + 0.5 * h * k1v, k2v = acc(p + 0.5 * h * k1p, v + 0.5 * h * k1v, u);
    const k3p = v + 0.5 * h * k2v, k3v = acc(p + 0.5 * h * k2p, v + 0.5 * h * k2v, u);
    const k4p = v + h * k3v, k4v = acc(p + h * k3p, v + h * k3v, u);
    p += (h / 6) * (k1p + 2 * k2p + 2 * k3p + k4p);
    v += (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
  }
  return y;
}
