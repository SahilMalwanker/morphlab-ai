import { describe, expect, it } from 'vitest';
import type { Quat } from '../../src/core/math/mat4';
import type { Vec3 } from '../../src/core/math/vec3';
import { buildKinematicModel } from '../../src/core/kinematics/fk';
import { quadruped } from '../../src/core/model/presets';
import { driveGeometry, navConfigFor, type LeggedDrive } from '../../src/nav/drive';
import { calibratedMotion, GAIT_DECIMATION, LeggedDriver, LeggedWalker } from '../../src/nav/legged';
import { NavStack, RAY_ANGLES, type NavInput, type PoseEstimator } from '../../src/nav/stack';
import type { SensorReadings } from '../../src/physics/runtime';
import { walkPolicy } from '../../src/skills/locomotion';
import { LegOdometry } from '../../src/skills/odometry';
import { WALK_COMMAND_OUTPUTS, WALK_POLICY_INPUTS } from '../../src/skills/walkPolicy';
import { testWalkingController } from './walkingFixture';

function quadDrive(): LeggedDrive {
  const g = driveGeometry(quadruped());
  if (!g.ok || g.drive.kind !== 'legged') throw new Error('quadruped not legged');
  return g.drive;
}

function sensors(imuQuat: Quat, centerRange: number, q: number[] = []): SensorReadings {
  return {
    q: Float64Array.from(q),
    qd: new Float64Array(q.length),
    imuQuat,
    gyro: [0, 0, 0],
    accel: [0, 0, 9.81],
    touch: [],
    ranges: RAY_ANGLES.map((a) => (a === 0 ? centerRange : -1)),
  };
}

const atOrigin: PoseEstimator = { update: () => ({ x: 0, y: 0, yaw: 0 }) };
const occupied = (s: NavStack) => s.grid.logOdds.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);

describe('range rays on a walking body', () => {
  // Nose down by 0.15 rad: a rotation about +y, [w, x, y, z].
  const pitched: Quat = [Math.cos(0.075), 0, Math.sin(0.075), 0];
  const level: Quat = [1, 0, 0, 0];
  const cfg = navConfigFor(quadDrive());
  const [rx, , rz] = cfg.rayOrigin;
  // Closed form: the mount sits at 0.15 - rz + (-sin p) rx + cos p rz; the center ray points down by p - elevation.
  const p = 0.15, e = 0.035;
  const mount = 0.15 - rz - Math.sin(p) * rx + Math.cos(p) * rz;
  const floorHit = mount / Math.sin(p - e);

  it('a hit the IMU attitude puts on the floor is not mapped as an obstacle', () => {
    expect(floorHit).toBeGreaterThan(0.5);
    expect(floorHit).toBeLessThan(2);
    const filtered = new NavStack('sensor-only', { x: 0, y: 0, yaw: 0 }, atOrigin, null, cfg);
    const unfiltered = new NavStack('sensor-only', { x: 0, y: 0, yaw: 0 }, atOrigin, null, { ...cfg, rayHeightM: undefined });
    const input: NavInput = { t: 0, sensors: sensors(pitched, floorHit) };
    filtered.observe(input);
    unfiltered.observe(input);
    expect(occupied(filtered)).toBe(0);
    expect(occupied(unfiltered)).toBeGreaterThan(0);
  });

  it('an obstacle ahead of a level body is still mapped', () => {
    const s = new NavStack('sensor-only', { x: 0, y: 0, yaw: 0 }, atOrigin, null, cfg);
    s.observe({ t: 0, sensors: sensors(level, 1.0) });
    expect(occupied(s)).toBeGreaterThan(0);
  });
});

describe('leg odometry with rolling feet', () => {
  it('tracks the contact point, which moves the body further than the foot centers do (closed form)', () => {
    // Straight quadruped legs (0.32 m, 0.022 m foot spheres) swinging from -0.2 to 0.2 rad while the feet roll without slipping:
    // the body moves L (sin 0.2 - sin -0.2) relative to the foot centers, and the centers roll forward r * 0.4 rad.
    const km = buildKinematicModel(quadruped());
    const L = 0.32, r = 0.022;
    const walk = (rolling: boolean) => {
      const odometry = new LegOdometry(km, [0, 1, 2, 3], rolling);
      for (let i = 0; i <= 400; i++) {
        const q = -0.2 + (0.4 * i) / 400;
        odometry.update(i * 0.01, { q: Array.from({ length: 4 }, () => [0, q, 0]).flat(), touch: [5, 5, 5, 5], imuQuat: [1, 0, 0, 0], gyro: [0, 0, 0] });
      }
      return odometry.x;
    };
    const legs = L * (Math.sin(0.2) - Math.sin(-0.2));
    expect(walk(false)).toBeCloseTo(legs, 6);
    expect(walk(true)).toBeCloseTo(legs + r * 0.4, 5);
  });
});

describe('navigation stack without a path', () => {
  it('stands still once a replan fails, instead of following the stale path until the next replan', () => {
    const start = { x: 0, y: 0, yaw: 0 };
    const s = new NavStack('ground-truth', start, atOrigin, []);
    const at = (t: number): NavInput => ({ t, sensors: sensors([1, 0, 0, 0], -1), groundTruth: { pose: start } });
    s.setGoal([2, 0]);
    expect(s.update(at(0)).v).toBeGreaterThan(0);
    expect(s.status).toBe('following');
    // The goal becomes enclosed: the next replan fails.
    s.grid.markBoxes([{ cx: 2, cy: 0, hx: 0.6, hy: 0.6 }]);
    expect(s.update(at(0.5))).toEqual({ v: 0, omega: 0, done: false });
    expect(s.status).toBe('no-path');
    // Between replans the old path is not followed.
    expect(s.update(at(0.52))).toEqual({ v: 0, omega: 0, done: false });
    expect(s.status).toBe('no-path');
  });
});

describe('open-ended walking with the gait generator', () => {
  const d = quadDrive();
  const km = buildKinematicModel(quadruped());
  const obs = { q: d.stanceQ, touch: [0, 0, 0, 0], imuQuat: [1, 0, 0, 0] as Quat };
  const dt = 0.002 * GAIT_DECIMATION;
  const moved = (q: number[]) => Math.max(...q.map((v, i) => Math.abs(v - d.stanceQ[i]!)));

  it('stands until commanded, steps past the generator length while commanded, and returns to its stance after a stop', () => {
    const w = new LeggedWalker(km, d.stanceQ, d.gait);
    let t = 0;
    for (; t < 1; t += dt) expect(w.update(t, obs)).toEqual(d.stanceQ);
    expect(w.stepping).toBe(false);
    w.setCommand(0.12, 0.2);
    // Twice the 20 s generator length: the clock is kept in its steady window, so stepping never fades out.
    let peak = 0;
    for (; t < 41; t += dt) {
      const q = w.update(t, obs);
      if (t > 39) peak = Math.max(peak, moved(q));
    }
    expect(w.stepping).toBe(true);
    expect(peak).toBeGreaterThan(0.05);
    w.setCommand(0, 0);
    let stoppedAt = -1;
    let last: number[] = [];
    for (; t < 46; t += dt) {
      last = w.update(t, obs);
      if (!w.stepping && stoppedAt < 0) stoppedAt = t;
    }
    // Slowing down (at most 0.5 s), at most two more cycles, then the fade-out (0.5 s).
    expect(stoppedAt).toBeGreaterThan(41);
    expect(stoppedAt).toBeLessThan(41 + 0.5 + 2 * d.gait.cycleS + 0.5 + 0.05);
    expect(last).toEqual(d.stanceQ);
  });
});

describe('walking with a learned walking controller', () => {
  const robot = quadruped();
  const d = quadDrive();
  const km = buildKinematicModel(robot);
  const J = d.stanceQ.length;
  const HASH = 'a'.repeat(64);
  const zeros = (n: number) => new Array<number>(n).fill(0);
  const ones = (n: number) => new Array<number>(n).fill(1);
  const obs = { q: d.stanceQ, touch: [5, 5, 5, 5], imuQuat: [Math.cos(0.03), 0.02, Math.sin(0.03), 0] as Quat, gyro: [0.2, -0.1, 0.05] as Vec3 };
  const commandPolicy = (out: Vec3) => {
    const theta = new Float64Array(WALK_COMMAND_OUTPUTS * (WALK_POLICY_INPUTS + 1));
    out.forEach((v, k) => (theta[WALK_COMMAND_OUTPUTS * WALK_POLICY_INPUTS + k] = v));
    return walkPolicy(theta, J, zeros(WALK_POLICY_INPUTS), ones(WALK_POLICY_INPUTS));
  };

  it('walks exactly like the gait generator with a calibration that changes nothing and no policy or a zero policy', () => {
    const plain = new LeggedWalker(km, d.stanceQ, d.gait);
    const calibrated = new LeggedWalker(km, d.stanceQ, d.gait, testWalkingController(robot, HASH));
    const zero = new LeggedWalker(km, d.stanceQ, d.gait, testWalkingController(robot, HASH, 1, commandPolicy([0, 0, 0])));
    expect(calibrated.controller).not.toBeNull();
    for (const w of [plain, calibrated, zero]) w.setCommand(0.1, 0.2);
    for (let k = 0; k < 300; k++) {
      const q = plain.update(k * 0.01, obs);
      expect(calibrated.update(k * 0.01, obs)).toEqual(q);
      expect(zero.update(k * 0.01, obs)).toEqual(q);
    }
  });

  it('asks the generator for the stride the calibration says walks the commanded speed', () => {
    // A calibration that measured the generator walking twice as fast as asked halves the command per cycle.
    const halved = calibratedMotion(testWalkingController(robot, HASH, 2).calibration, [0.1, 0, 0.4], 0.5);
    expect(halved[0]).toBeCloseTo(0.025, 12);
    expect(halved[1]).toBe(0);
    expect(halved[2]).toBeCloseTo(0.1, 12);
    // Within the generator's bounds per cycle.
    expect(calibratedMotion(testWalkingController(robot, HASH, 0.1).calibration, [0.1, 0, 0], 0.5)[0]).toBe(0.3);
  });

  it('changes the walk when its policy corrects the command', () => {
    const plain = new LeggedWalker(km, d.stanceQ, d.gait);
    const turning = new LeggedWalker(km, d.stanceQ, d.gait, testWalkingController(robot, HASH, 1, commandPolicy([0, 0, 1])));
    plain.setCommand(0.1, 0);
    turning.setCommand(0.1, 0);
    let largest = 0;
    for (let k = 0; k < 200; k++) {
      const a = plain.update(k * 0.01, obs), b = turning.update(k * 0.01, obs);
      largest = Math.max(largest, ...a.map((v, i) => Math.abs(v - b[i]!)));
    }
    expect(largest).toBeGreaterThan(0.01);
  });

  it('refuses a policy that does not fit the body', () => {
    const six = walkPolicy(new Float64Array(21 * (WALK_POLICY_INPUTS + 1)), 18, zeros(WALK_POLICY_INPUTS), ones(WALK_POLICY_INPUTS), true);
    expect(() => new LeggedWalker(km, d.stanceQ, d.gait, testWalkingController(robot, HASH, 1, six))).toThrow(/^The policy maps 18 inputs to 21 outputs/);
  });

  it('takes a walking controller given while walking only once the body stands again', () => {
    const driver = new LeggedDriver(robot, d);
    const s = sensors([1, 0, 0, 0], -1, d.stanceQ);
    const effort = d.stanceQ.map(() => 12);
    const h = 0.002;
    let k = 0;
    const steps = (seconds: number) => {
      for (const end = k + Math.round(seconds / h); k < end; k++) driver.torques(k * h, h, s, effort);
    };
    driver.setCommand(0.1, 0);
    steps(2);
    const base = testWalkingController(robot, HASH);
    const c = { ...base, evaluation: { ...base.evaluation, conditions: { ...base.evaluation.conditions, gains: { kp: 30, kd: 0.8 } } } };
    driver.setController(c);
    steps(1);
    expect(driver.controller).toBeNull();
    expect(driver.jointGains).toEqual({ kp: 40, kd: 1 });
    driver.setCommand(0, 0);
    steps(3);
    expect(driver.controller).toBe(c);
    expect(driver.jointGains).toEqual({ kp: 30, kd: 0.8 });
    driver.setController(null);
    steps(0.02);
    expect(driver.jointGains).toEqual({ kp: 40, kd: 1 });
  });
});
