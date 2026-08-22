import { buildKinematicModel, forwardKinematics } from '../core/kinematics/fk';
import { identity, rpyToRotation } from '../core/math/mat4';
import { homeConfiguration } from '../core/model/catalog';
import type { RobotDescription } from '../core/model/types';
import { DEFAULT_GAIT, GAIT_BOUNDS, gaitProblems, legIndices, type GaitParams } from '../skills/gait';
import type { Pose2 } from './follower';
import { legOdometry } from './legged';
import { OBSTACLE_HEIGHT } from './maps';
import { DEFAULT_NAV, wheelOdometry, type NavStackConfig, type PoseEstimator, type WheelRef } from './stack';

export type { WheelRef };

/** Speed and time limits of a body in Navigate. */
export interface DriveLimits {
  /** Allowed speed limits for driving to a goal, m/s; the upper end is the path follower's maximum. */
  speedLimitRange: readonly [number, number];
  /** Path follower turn-rate cap, rad/s. */
  maxTurnRate: number;
  /** Keyboard driving: forward speed (m/s) and turn rate (rad/s). */
  remoteSpeed: number;
  remoteTurn: number;
  /** A run to a goal ends after this long, s. */
  timeLimitS: number;
}

export const WHEELED_LIMITS: DriveLimits = { speedLimitRange: [0.1, DEFAULT_NAV.follower.maxSpeed], maxTurnRate: DEFAULT_NAV.follower.maxTurnRate, remoteSpeed: 0.4, remoteTurn: 1.6, timeLimitS: 60 };

/**
 * Legged bodies walk with the gait generator at up to 0.075 m stride and
 * 0.25 rad turn per 0.5 s cycle, inside the ranges the gait sweep measured
 * upright (D25). Walking is slower than driving, so runs get twice as long.
 */
export const LEGGED_LIMITS: DriveLimits = { speedLimitRange: [0.05, 0.15], maxTurnRate: 0.5, remoteSpeed: 0.15, remoteTurn: 0.5, timeLimitS: 120 };

interface BodyGeometry {
  /** Footprint half extents plus a 2 cm margin (legged: plus room for the steps), m. */
  halfExtents: [number, number];
  /** Planner clearance: half the footprint diagonal plus 1.5 cm, rounded up to 1 cm, m. */
  robotRadiusM: number;
  /** Range-ray mount in the base frame, m: at the front of the footprint (legged: ahead of the feet, 0.15 m above the floor). */
  rayOrigin: [number, number, number];
  /** Base height when the wheels or feet rest on the floor, m. */
  restHeight: number;
  limits: DriveLimits;
}

/** A skid-steered wheeled body: what steering, odometry, planning and the referee need. */
export interface WheeledDrive extends BodyGeometry {
  kind: 'wheeled';
  /** Every wheel in joint-vector order. */
  wheels: WheelRef[];
  wheelRadius: number;
  /** Distance between the mean left and mean right wheel contact lines, m. */
  trackWidth: number;
}

/** A legged body that walks with the gait generator. */
export interface LeggedDrive extends BodyGeometry {
  kind: 'legged';
  /** Limb indices of the legs. */
  legs: number[];
  /** Joint vector the body stands in (its home configuration). */
  stanceQ: number[];
  /** Gait generator settings; stride and turn are set from the body command while walking. */
  gait: GaitParams;
}

/** Derived from a robot description: how Navigate moves, senses and judges this body. */
export type DriveGeometry = WheeledDrive | LeggedDrive;

export type DriveCheck = { ok: true; drive: DriveGeometry } | { ok: false; reason: string };

const FOOTPRINT_MARGIN_M = 0.02;
const AXIS_TOLERANCE = 0.999;
/** Rays mounted higher than this pass over the arena obstacles, m. */
const MAX_RAY_HEIGHT_M = OBSTACLE_HEIGHT - 0.05;
/** Legged bodies carry their rays this high above the floor, half the obstacle height, m. */
export const LEG_RAY_HEIGHT_M = OBSTACLE_HEIGHT / 2;
/** Room for the feet stepping beyond their stance positions (half a stride plus turning), m. */
const STEP_ROOM_M = 0.04;
/** Sums like 0.14 + 0.06 are not exact in floating point; geometry is kept to the micrometre. */
const um = (v: number): number => Math.round(v * 1e6) / 1e6;
const clearance = (half: readonly [number, number]): number => Math.ceil((Math.hypot(...half) + 0.015) * 100) / 100;

/**
 * Checks whether Navigate can move this body and derives its geometry.
 * Wheel-only, free-standing bodies with wheels on both sides spinning about
 * the body's left-right axis are driven by skid steering. Free-standing
 * bodies with at least four legs walk with the gait generator. Everything
 * else is refused with the reason.
 */
export function driveGeometry(robot: RobotDescription): DriveCheck {
  if (robot.base.kind === 'fixed') return { ok: false, reason: 'its base is fixed in place, so it cannot move through the arena' };
  const wheelLimbs = robot.limbs.filter((l) => l.role === 'wheel');
  if (wheelLimbs.length === 0) return leggedGeometry(robot);
  if (wheelLimbs.length !== robot.limbs.length) return { ok: false, reason: 'it mixes wheels with legs or arms; Navigate drives bodies that roll on wheels only or walk on legs' };
  return wheeledGeometry(robot);
}

function wheeledGeometry(robot: RobotDescription): DriveCheck {
  const wheels: WheelRef[] = [];
  const left: number[] = [], right: number[] = [];
  let index = 0, frontX = robot.base.size[0] / 2, halfX = robot.base.size[0] / 2, halfY = robot.base.size[1] / 2, lowest = -robot.base.size[2] / 2;
  let radius: number | null = null;
  for (const limb of robot.limbs) {
    const j = limb.joints[0];
    const w = limb.wheel;
    if (!j || !w || limb.joints.length !== 1) return { ok: false, reason: `the ${limb.name.toLowerCase()} is not a single-joint wheel` };
    radius ??= w.radius;
    if (Math.abs(w.radius - radius) > 1e-9) return { ok: false, reason: 'its wheels differ in radius; the skid-steering controller assumes one wheel radius' };
    const R = rpyToRotation(j.origin.rpy);
    const n = Math.hypot(...j.axis);
    const axisY = (R[3]! * j.axis[0] + R[4]! * j.axis[1] + R[5]! * j.axis[2]) / n;
    if (Math.abs(axisY) < AXIS_TOLERANCE) return { ok: false, reason: `the ${limb.name.toLowerCase()} does not spin about the body's left-right axis` };
    const [x, y, z] = j.origin.xyz;
    if (Math.abs(y) < 0.01) return { ok: false, reason: `the ${limb.name.toLowerCase()} sits on the centerline; skid steering needs wheels on the left and right` };
    const side = y > 0 ? 'left' : 'right';
    (side === 'left' ? left : right).push(y);
    wheels.push({ index, side, sign: axisY > 0 ? 1 : -1 });
    index += limb.joints.length;
    frontX = Math.max(frontX, x + w.radius);
    halfX = Math.max(halfX, Math.abs(x) + w.radius);
    halfY = Math.max(halfY, Math.abs(y) + w.width / 2);
    lowest = Math.min(lowest, z - w.radius);
  }
  if (left.length === 0 || right.length === 0 || radius === null) return { ok: false, reason: 'skid steering needs at least one wheel on each side' };
  if (lowest >= -robot.base.size[2] / 2 - 0.005) return { ok: false, reason: 'its body would rest on the floor instead of its wheels' };
  const restHeight = um(-lowest);
  if (restHeight > MAX_RAY_HEIGHT_M) {
    return { ok: false, reason: `its range rays would start ${restHeight.toFixed(2)} m above the floor and pass over the ${OBSTACLE_HEIGHT} m obstacles` };
  }
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  const halfExtents: [number, number] = [um(halfX + FOOTPRINT_MARGIN_M), um(halfY + FOOTPRINT_MARGIN_M)];
  return {
    ok: true,
    drive: {
      kind: 'wheeled',
      wheels,
      wheelRadius: radius,
      trackWidth: um(mean(left) - mean(right)),
      halfExtents,
      robotRadiusM: clearance(halfExtents),
      rayOrigin: [um(frontX), 0, 0],
      restHeight,
      limits: WHEELED_LIMITS,
    },
  };
}

function leggedGeometry(robot: RobotDescription): DriveCheck {
  const legs = legIndices(robot);
  if (legs.length === 0) return { ok: false, reason: 'it has neither wheels nor legs to move with' };
  if (legs.length < 4) return { ok: false, reason: `it has ${legs.length === 1 ? 'one leg' : `${legs.length} legs`}; walking in Navigate needs at least four, so the body stays up while half of them step` };
  const short = legs.map((i) => robot.limbs[i]!).find((l) => l.joints.length < 2);
  if (short) return { ok: false, reason: `the ${short.name.toLowerCase()} has one joint; each leg needs at least two to step` };
  const gait: GaitParams = { ...DEFAULT_GAIT, strideM: 0, lateralM: 0, turnRad: 0, durationS: GAIT_BOUNDS.durationS[1] };
  const problem = gaitProblems(gait, robot)[0];
  if (problem) return { ok: false, reason: `the walking gait does not fit it: ${problem.replace(/\.$/, '')}` };
  const stanceQ = homeConfiguration(robot);
  const km = buildKinematicModel(robot);
  const fk = forwardKinematics(km, stanceQ, identity());
  const [bx, by, bz] = robot.base.size;
  let front = bx / 2, back = -bx / 2, leftY = by / 2, rightY = -by / 2;
  let otherLow = -bz / 2;
  const feet: number[] = [];
  const extend = (T: ArrayLike<number>, r: number) => {
    front = Math.max(front, T[12]! + r);
    back = Math.min(back, T[12]! - r);
    leftY = Math.max(leftY, T[13]! + r);
    rightY = Math.min(rightY, T[13]! - r);
  };
  robot.limbs.forEach((limb, li) => {
    const start = km.limbs[li]!.start;
    limb.joints.forEach((j, ji) => {
      const T = fk.links[start + ji]!;
      extend(T, j.link.radius);
      otherLow = Math.min(otherLow, T[14]! - j.link.radius);
    });
    const tip = fk.tips[li]!;
    extend(tip, limb.tip.radius);
    if (limb.role === 'leg') feet.push(tip[14]! - limb.tip.radius);
    else otherLow = Math.min(otherLow, tip[14]! - limb.tip.radius);
  });
  const low = Math.min(...feet), high = Math.max(...feet);
  if (high - low > 0.02) return { ok: false, reason: `in its stance its feet are not level (they differ by ${(high - low).toFixed(2)} m), so not every foot reaches the floor` };
  if (otherLow < low + 0.005) return { ok: false, reason: 'in its stance its body or a knee would touch the floor before its feet' };
  const restHeight = um(-low);
  const halfExtents: [number, number] = [um(Math.max(front, -back) + STEP_ROOM_M + FOOTPRINT_MARGIN_M), um(Math.max(leftY, -rightY) + STEP_ROOM_M + FOOTPRINT_MARGIN_M)];
  return {
    ok: true,
    drive: {
      kind: 'legged',
      legs,
      stanceQ,
      gait,
      halfExtents,
      robotRadiusM: clearance(halfExtents),
      // Ahead of the stepping feet, so no ray can see the body's own legs, and half the obstacle height above the floor.
      rayOrigin: [um(front + STEP_ROOM_M + 0.01), 0, um(LEG_RAY_HEIGHT_M - restHeight)],
      restHeight,
      limits: LEGGED_LIMITS,
    },
  };
}

/** Navigation settings for this body; the rover gets exactly DEFAULT_NAV. */
export function navConfigFor(d: DriveGeometry): NavStackConfig {
  const cfg: NavStackConfig = {
    ...DEFAULT_NAV,
    robotRadiusM: d.robotRadiusM,
    softRangeM: Math.max(DEFAULT_NAV.softRangeM, d.robotRadiusM + 0.4),
    rayOrigin: d.rayOrigin,
  };
  if (d.kind === 'wheeled') return cfg;
  return { ...cfg, follower: { ...DEFAULT_NAV.follower, maxSpeed: d.limits.speedLimitRange[1], maxTurnRate: d.limits.maxTurnRate }, rayHeightM: LEG_RAY_HEIGHT_M };
}

/** Sensor-only odometry for this body from the declared start pose: wheel encoders or legs, with the IMU heading. */
export function odometryFor(robot: RobotDescription, d: DriveGeometry, start: Pose2): PoseEstimator {
  return d.kind === 'wheeled' ? wheelOdometry(start, d.wheels, d.wheelRadius) : legOdometry(start, buildKinematicModel(robot), d.legs);
}

/** Joint velocity targets for every wheel from the left and right forward wheel speeds, rad/s. */
export function wheelTargets(wheels: readonly WheelRef[], left: number, right: number): number[] {
  return wheels.map((w) => w.sign * (w.side === 'left' ? left : right));
}

/** The body at the map's start pose, with its wheels or feet at most 3 cm above the floor. */
export function placeAt(robot: RobotDescription, d: DriveGeometry, start: { x: number; y: number; yaw: number }): RobotDescription {
  const z = Math.min(Math.max(robot.base.pose.xyz[2], d.restHeight + 0.002), d.restHeight + 0.03);
  return { ...robot, base: { ...robot.base, pose: { xyz: [start.x, start.y, z], rpy: [0, 0, start.yaw] } } };
}
