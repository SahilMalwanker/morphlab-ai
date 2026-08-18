import { describe, expect, it } from 'vitest';
import { TargetRateLimiter, wrapToPi } from '../../src/core/control/limits';
import { pdControl, pdStabilityMargin, gainsFor, secondOrderParameters } from '../../src/core/control/pd';
import { analyticStepMetrics, linearResponse, restsOffTarget, secondOrderStep, stepMetrics } from '../../src/core/control/stepMetrics';
import { Telemetry } from '../../src/core/control/telemetry';
import { jointUnits, sharedGainUnits } from '../../src/core/model/units';

describe('PD law', () => {
  it('saturates at the effort limit and reports it', () => {
    const r = pdControl([{ kp: 100, kd: 1 }], [2], { qCmd: [1], qMeas: [0], qdMeas: [0] });
    expect(r.tauCmd[0]).toBe(100);
    expect(r.tauApplied[0]).toBe(2);
    expect(r.saturated[0]).toBe(1);
  });

  it('outputs zero torque and flags a fault on non-finite input', () => {
    const r = pdControl([{ kp: 10, kd: 1 }], [5], { qCmd: [Number.NaN], qMeas: [0], qdMeas: [0] });
    expect(r.tauApplied[0]).toBe(0);
    expect(r.fault).toBe(true);
  });

  it('gainsFor inverts secondOrderParameters', () => {
    const g = gainsFor(20, 0.7, 0.05);
    const p = secondOrderParameters(g, 0.05);
    expect(p.wn).toBeCloseTo(20, 12);
    expect(p.zeta).toBeCloseTo(0.7, 12);
  });

  it('stability margin flags gains that are too stiff for the timestep', () => {
    expect(pdStabilityMargin({ kp: 40, kd: 1 }, 0.0115, 0.002).ok).toBe(true);
    expect(pdStabilityMargin({ kp: 400, kd: 1 }, 0.0115, 0.002).ok).toBe(false);
    expect(pdStabilityMargin({ kp: 40, kd: 10 }, 0.0115, 0.002).ok).toBe(false);
  });
});

describe('step metrics', () => {
  it('match closed-form overshoot and peak time of an underdamped system', () => {
    const wn = 12, zeta = 0.3;
    const m = analyticStepMetrics(wn, zeta, 3);
    expect(m.overshootPct).toBeCloseTo(100 * Math.exp((-zeta * Math.PI) / Math.sqrt(1 - zeta * zeta)), 2);
    expect(m.peakTimeS!).toBeCloseTo(Math.PI / (wn * Math.sqrt(1 - zeta * zeta)), 3);
    expect(Math.abs(m.steadyStateError)).toBeLessThan(0.01);
  });

  it('rise time of a first-order-like overdamped response and no overshoot', () => {
    const m = analyticStepMetrics(10, 2, 5);
    expect(m.overshootPct).toBe(0);
    expect(m.peakTimeS).toBeNull();
    expect(m.riseTimeS).toBeGreaterThan(0);
  });

  it('reports null settling and rise when the response never gets there', () => {
    const t = [0, 1, 2, 3], y = [0, 0.1, 0.2, 0.3];
    const m = stepMetrics(t, y, 0, 1);
    expect(m.settlingTimeS).toBeNull();
    expect(m.riseTimeS).toBeNull();
  });

  it('handles negative steps symmetrically', () => {
    const t = Array.from({ length: 2001 }, (_, i) => i * 0.0015);
    const pos = stepMetrics(t, t.map((x) => secondOrderStep(x, 15, 0.4)), 0, 1);
    const neg = stepMetrics(t, t.map((x) => -secondOrderStep(x, 15, 0.4)), 0, -1);
    expect(neg.overshootPct).toBeCloseTo(pos.overshootPct, 10);
    expect(neg.settlingTimeS).toBeCloseTo(pos.settlingTimeS!, 10);
  });

  it('first-order response: rise time tau ln 9 and 2 % settling time tau ln 50 (closed form)', () => {
    const tau = 0.25, dt = 1e-4;
    const t = Array.from({ length: 30001 }, (_, i) => i * dt);
    const m = stepMetrics(t, t.map((x) => 1 - Math.exp(-x / tau)), 0, 1);
    expect(m.riseTimeS!).toBeCloseTo(tau * Math.log(9), 5);
    expect(Math.abs(m.settlingTimeS! - tau * Math.log(50))).toBeLessThanOrEqual(dt);
    expect(m.overshootPct).toBe(0);
    expect(m.steadyStateError).toBeLessThan(1e-4);
  });

  it('steady-state error of a response that settles short of the reference', () => {
    const t = Array.from({ length: 20001 }, (_, i) => i * 1e-3);
    const y = t.map((x) => 0.8 * (1 - Math.exp(-x)));
    const m = stepMetrics(t, y, 0, 1);
    expect(m.steadyStateError).toBeCloseTo(0.2, 6);
    expect(m.settlingTimeS).toBeNull();
    expect(restsOffTarget(y, 0, 1)).toBe(true);
    expect(restsOffTarget(y.map((v) => -v), 0, -1)).toBe(true);
  });

  it('tells a response resting off target from one still ringing, still moving or inside the band', () => {
    const t = Array.from({ length: 10001 }, (_, i) => i * 1e-3);
    // Undamped, one period per second: the last 10 % of the 10 s record is one full period, so its mean is on target.
    const ringing = t.map((x) => secondOrderStep(x, 2 * Math.PI, 0));
    expect(stepMetrics(t, ringing, 0, 1).settlingTimeS).toBeNull();
    expect(Math.abs(stepMetrics(t, ringing, 0, 1).steadyStateError)).toBeLessThan(2e-3);
    expect(restsOffTarget(ringing, 0, 1)).toBe(false);
    // Time constant 5 s: about 15 % short over the last second, but still rising 3 % of the step during it.
    const creeping = t.map((x) => 1 - Math.exp(-x / 5));
    expect(stepMetrics(t, creeping, 0, 1).settlingTimeS).toBeNull();
    expect(restsOffTarget(creeping, 0, 1)).toBe(false);
    // A 1.5 % static error settles inside the 2 % band.
    const close = t.map((x) => 0.985 * (1 - Math.exp(-5 * x)));
    expect(stepMetrics(t, close, 0, 1).settlingTimeS).not.toBeNull();
    expect(restsOffTarget(close, 0, 1)).toBe(false);
  });

  it('underdamped rise time and settling time agree with bisection on the exact response', () => {
    const wn = 10, zeta = 0.3;
    const y = (x: number) => secondOrderStep(x, wn, zeta);
    const solve = (f: (x: number) => number, a: number, b: number) => {
      for (let i = 0; i < 200; i++) {
        const m = (a + b) / 2;
        if (Math.sign(f(m)) === Math.sign(f(a))) a = m;
        else b = m;
      }
      return (a + b) / 2;
    };
    const tPeak = Math.PI / (wn * Math.sqrt(1 - zeta * zeta));
    const rise = solve((x) => y(x) - 0.9, 0, tPeak) - solve((x) => y(x) - 0.1, 0, tPeak);
    // Settling: last time |y - 1| = 0.02, found by scanning half-periods from the end.
    const td = Math.PI / (wn * Math.sqrt(1 - zeta * zeta));
    let settle = 0;
    for (let k = 1; k < 40; k++) {
      const a = (k - 1) * td, b = k * td;
      const ext = Math.abs(y(k * td) - 1);
      if (ext > 0.02) settle = solve((x) => Math.abs(y(x) - 1) - 0.02, b, b + td / 2) || a;
    }
    const m = analyticStepMetrics(wn, zeta, 3);
    expect(m.riseTimeS!).toBeCloseTo(rise, 4);
    expect(Math.abs(m.settlingTimeS! - settle)).toBeLessThan(3 / 20000 + 1e-9);
  });
});

describe('telemetry ring buffer', () => {
  it('keeps the newest samples in order once full', () => {
    const t = new Telemetry(['a'], 3);
    for (let i = 0; i < 5; i++) t.push(i, [10 * i]);
    expect(t.length).toBe(3);
    expect([t.time(0), t.time(1), t.time(2)]).toEqual([2, 3, 4]);
    expect(t.value(2, 0)).toBe(40);
  });
});

describe('linear joint model', () => {
  it('a constant command reproduces the closed-form second-order step', () => {
    const kp = 8, kd = 0.3, I = 0.05, h = 0.002;
    const { wn, zeta } = secondOrderParameters({ kp, kd }, I);
    const n = 2000;
    const y = linearResponse(kp, kd, I, h, new Float64Array(n).fill(0.5), 0.1);
    // RK4 at a 2 ms step: global error O(h^4), about 1e-9 here.
    for (let i = 0; i < n; i += 97) expect(y[i]!).toBeCloseTo(0.1 + 0.4 * secondOrderStep(i * h, wn, zeta), 7);
  });

  it('rejects non-physical parameters', () => {
    expect(() => linearResponse(1, 1, 0, 0.002, [1], 0)).toThrow(RangeError);
    expect(() => linearResponse(-1, 1, 1, 0.002, [1], 0)).toThrow(RangeError);
  });
});

describe('target rate limiter and continuous joints', () => {
  it('never moves a target faster than the velocity limit and reports when it held back', () => {
    const lim = new TargetRateLimiter([2, 0.5], [false, false], [0, 0]);
    const h = 0.002;
    let held = 0;
    for (let k = 0; k < 600; k++) {
      const prev = Array.from(lim.qCmd);
      if (lim.step([1, -0.1], h)) held++;
      expect(Math.abs(lim.qCmd[0]! - prev[0]!)).toBeLessThanOrEqual(2 * h + 1e-15);
      expect(Math.abs(lim.qCmd[1]! - prev[1]!)).toBeLessThanOrEqual(0.5 * h + 1e-15);
    }
    expect(lim.qCmd[0]).toBeCloseTo(1, 12);
    expect(lim.qCmd[1]).toBeCloseTo(-0.1, 12);
    // Joint 0 needs 0.5 s (250 steps of 4 mrad) at 2 rad/s; the last step may land exactly on the target.
    expect(held).toBeGreaterThanOrEqual(249);
    expect(held).toBeLessThanOrEqual(250);
    expect(() => new TargetRateLimiter([0], [false], [0])).toThrow(RangeError);
  });

  it('continuous joints take the shorter way around and the PD error is wrapped', () => {
    expect(wrapToPi(3 * Math.PI / 2)).toBeCloseTo(-Math.PI / 2, 12);
    expect(wrapToPi(-Math.PI)).toBe(Math.PI);
    const lim = new TargetRateLimiter([1], [true], [3]);
    lim.step([-3], 0.002);
    // From 3 rad to -3 rad the short way is +0.28 rad through pi, not -6 rad.
    expect(lim.qdCmd[0]).toBeCloseTo(1, 12);
    const wrapped = pdControl([{ kp: 10, kd: 0 }], [100], { qCmd: [-3], qMeas: [3], qdMeas: [0], continuous: [true] });
    expect(wrapped.tauCmd[0]).toBeCloseTo(10 * (2 * Math.PI - 6), 9);
    const plain = pdControl([{ kp: 10, kd: 0 }], [100], { qCmd: [-3], qMeas: [3], qdMeas: [0] });
    expect(plain.tauCmd[0]).toBeCloseTo(-60, 9);
  });
});

describe('joint units', () => {
  it('names rotation and sliding quantities in their own SI units', () => {
    expect(jointUnits('revolute')).toMatchObject({ position: 'rad', effort: 'N*m', stiffness: 'N*m/rad', damping: 'N*m*s/rad', inertia: 'kg*m^2', display: { unit: 'deg' } });
    expect(jointUnits('continuous').effort).toBe('N*m');
    expect(jointUnits('prismatic')).toMatchObject({ position: 'm', velocity: 'm/s', effort: 'N', effortName: 'Force', stiffness: 'N/m', damping: 'N*s/m', inertia: 'kg', display: { unit: 'm', scale: 1 } });
    expect(sharedGainUnits(['revolute', 'revolute']).stiffness).toBe('N*m/rad');
    expect(sharedGainUnits(['prismatic']).damping).toBe('N*s/m');
    expect(sharedGainUnits(['revolute', 'prismatic']).stiffness).toBe('N*m/rad; N/m for prismatic joints');
  });
});
