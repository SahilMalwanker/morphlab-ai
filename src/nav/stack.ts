import { quatToRotation } from '../core/math/mat4';
import type { SensorReadings } from '../physics/runtime';
import { astar } from './astar';
import { DEFAULT_FOLLOWER, PurePursuit, WheelImuOdometry, yawFromQuat, type FollowerConfig, type Pose2 } from './follower';
import { OccupancyGrid, type Box2, type GridSpec } from './grid';
import { ARENA } from './maps';

/** Range ray directions in the base frame (yaw angles, rad), mounted at the front of the body. */
export const RAY_ANGLES = [-80, -60, -40, -20, 0, 20, 40, 60, 80].map((d) => (d * Math.PI) / 180);
export const RAY_ORIGIN: [number, number, number] = [0.2, 0, 0];
export const RAY_MAX_RANGE = 2.0;
/** Rays tilt up by this angle so a slightly pitched body does not see the floor as an obstacle, rad. */
export const RAY_ELEVATION = 0.035;

/** What a navigation controller may see. Ground truth is present only in the labeled ground-truth mode. */
export interface NavInput {
  /** Time since the run started, s. */
  t: number;
  sensors: SensorReadings;
  groundTruth?: { pose: Pose2 };
}

/** Body command: forward speed (m/s) and turn rate (rad/s, positive left). Wheels or legs turn it into actuator targets. */
export interface NavCommand {
  v: number;
  omega: number;
  done: boolean;
}

/** Sensor-only pose estimate in the map frame, updated at every control update. */
export interface PoseEstimator {
  update(input: NavInput, dt: number): Pose2;
}

/** A range hit the IMU attitude places lower than this above the floor is the floor, not an obstacle, m. */
export const FLOOR_HIT_M = 0.04;

/** One wheel joint: its index in the joint vector, its side, and +1 when a positive joint velocity drives the body forward. */
export interface WheelRef {
  index: number;
  side: 'left' | 'right';
  sign: 1 | -1;
}

/** The rover's wheels: front left, front right, rear left, rear right. */
export const ROVER_WHEELS: readonly WheelRef[] = [
  { index: 0, side: 'left', sign: 1 },
  { index: 1, side: 'right', sign: 1 },
  { index: 2, side: 'left', sign: 1 },
  { index: 3, side: 'right', sign: 1 },
];

export interface NavStackConfig {
  /** Cells closer than this to an obstacle are blocked (the body's half-diagonal plus a margin), m. */
  robotRadiusM: number;
  /** Cells closer than this pay an extra cost that grows toward the obstacle, m. */
  softRangeM: number;
  softWeight: number;
  replanIntervalS: number;
  follower: FollowerConfig;
  /** Range-ray mount in the base frame, m. */
  rayOrigin: readonly [number, number, number];
  /**
   * Height of the ray mount above the floor while standing, m. When set, a
   * hit that the IMU attitude puts on the floor (a pitching walker) clears
   * the cells up to it instead of marking an obstacle.
   */
  rayHeightM?: number;
}

export const DEFAULT_NAV: NavStackConfig = {
  robotRadiusM: 0.32, softRangeM: 0.75, softWeight: 3, replanIntervalS: 0.5,
  follower: { ...DEFAULT_FOLLOWER, lookaheadM: 0.3, maxSpeed: 0.4 },
  rayOrigin: RAY_ORIGIN,
};

export function arenaGridSpec(resolution = 0.05): GridSpec {
  return {
    originX: ARENA.minX,
    originY: ARENA.minY,
    resolution,
    width: Math.round((ARENA.maxX - ARENA.minX) / resolution),
    height: Math.round((ARENA.maxY - ARENA.minY) / resolution),
  };
}

export type NavStatus = 'idle' | 'planning' | 'following' | 'reached' | 'no-path';

/** Wheel odometry with the IMU heading for a skid-steered body (sensor only). */
export function wheelOdometry(start: Pose2, wheels: readonly WheelRef[], wheelRadius: number): PoseEstimator {
  const left = wheels.filter((w) => w.side === 'left');
  const right = wheels.filter((w) => w.side === 'right');
  if (left.length === 0 || right.length === 0) throw new RangeError('Skid steering needs wheels on both sides');
  const odometry = new WheelImuOdometry(start, wheelRadius);
  const side = (qd: ArrayLike<number>, ws: readonly WheelRef[]) => {
    let sum = 0;
    for (const w of ws) sum += w.sign * qd[w.index]!;
    return sum / ws.length;
  };
  return { update: (input, dt) => odometry.update(dt, side(input.sensors.qd, left), side(input.sensors.qd, right), yawFromQuat(input.sensors.imuQuat)) };
}

/**
 * Mapping, planning and path following for a wheeled or legged body. In
 * sensor-only mode the pose comes from the body's odometry (wheel encoders
 * or legs) and the IMU, and the map from range rays. In ground-truth mode the simulator pose and a known
 * map are used. The pose estimate and the map persist for the life of the
 * stack; goals can be set, changed and cleared without touching them.
 */
export class NavStack {
  readonly grid: OccupancyGrid;
  pose: Pose2;
  path: [number, number][] = [];
  replans = 0;
  status: NavStatus = 'idle';
  goal: [number, number] | null = null;
  private follower: PurePursuit | null = null;
  private lastPlan = -Infinity;
  private lastT: number | null = null;

  constructor(
    readonly mode: 'sensor-only' | 'ground-truth',
    /** The declared start pose; the only pose the sensor-only stack is ever given. */
    start: Pose2,
    /** Sensor-only odometry started at `start` (unused in ground-truth mode). */
    private readonly odometry: PoseEstimator,
    knownMap: Box2[] | null,
    private readonly cfg: NavStackConfig = DEFAULT_NAV,
  ) {
    this.grid = new OccupancyGrid(arenaGridSpec());
    if (mode === 'ground-truth') {
      if (!knownMap) throw new RangeError('Ground-truth mode needs the known map');
      this.grid.markBoxes(knownMap);
    }
    this.pose = { ...start };
  }

  /** Starts (or restarts) navigation to a goal; the pose estimate and the map are kept. */
  setGoal(goal: readonly [number, number]): void {
    this.goal = [goal[0], goal[1]];
    this.path = [];
    this.follower = null;
    this.lastPlan = -Infinity;
    this.status = 'planning';
  }

  /** Stops navigating; the pose estimate and the map are kept. */
  clearGoal(): void {
    this.goal = null;
    this.path = [];
    this.follower = null;
    this.status = 'idle';
  }

  private estimatePose(input: NavInput, dt: number): Pose2 {
    if (this.mode === 'ground-truth') {
      if (!input.groundTruth) throw new Error('Ground-truth mode requires the simulator pose');
      return input.groundTruth.pose;
    }
    return this.odometry.update(input, dt);
  }

  private integrateRays(input: NavInput): void {
    const p = this.pose;
    const c = Math.cos(p.yaw), s = Math.sin(p.yaw);
    const [rx, ry, rz] = this.cfg.rayOrigin;
    const ox = p.x + c * rx - s * ry, oy = p.y + s * rx + c * ry;
    const h = this.cfg.rayHeightM;
    // Third row of the IMU rotation (sensor): how high the mount sits and how far up each ray points in the world.
    const R = h === undefined ? null : quatToRotation(input.sensors.imuQuat);
    const mount = R && h !== undefined ? h - rz + R[6]! * rx + R[7]! * ry + R[8]! * rz : 0;
    RAY_ANGLES.forEach((a, i) => {
      const r = input.sensors.ranges[i];
      if (r === undefined || !Number.isFinite(r)) return;
      const flat = r >= 0 ? r * Math.cos(RAY_ELEVATION) : r;
      if (R && r >= 0 && r <= RAY_MAX_RANGE) {
        const up = (R[6]! * Math.cos(a) + R[7]! * Math.sin(a)) * Math.cos(RAY_ELEVATION) + R[8]! * Math.sin(RAY_ELEVATION);
        if (mount + r * up < FLOOR_HIT_M) {
          this.grid.integrateRay(ox, oy, p.yaw + a, -1, flat);
          return;
        }
      }
      this.grid.integrateRay(ox, oy, p.yaw + a, flat, RAY_MAX_RANGE);
    });
  }

  private plan(goalXY: [number, number]): boolean {
    const dist = this.grid.distanceToOccupied();
    const { robotRadiusM: r, softRangeM: soft, softWeight } = this.cfg;
    const blocked = new Uint8Array(dist.length);
    const cost = new Float32Array(dist.length);
    for (let i = 0; i < dist.length; i++) {
      const d = dist[i]!;
      if (d <= r + 1e-6) blocked[i] = 1;
      else if (d < soft) cost[i] = softWeight * ((soft - d) / (soft - r)) ** 2;
    }
    const { width, height } = this.grid.spec;
    const start = this.grid.worldToCell(this.pose.x, this.pose.y);
    const goal = this.grid.worldToCell(goalXY[0], goalXY[1]);
    // The robot's own cell and its immediate neighbours are free by definition.
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const x = start[0] + dx, y = start[1] + dy;
      if (x >= 0 && y >= 0 && x < width && y < height) blocked[y * width + x] = 0;
    }
    const res = astar(blocked, width, height, start, goal, cost);
    if (!res) return false;
    this.path = res.path.map(([ix, iy]) => this.grid.cellCenter(ix, iy));
    this.path[this.path.length - 1] = [goalXY[0], goalXY[1]];
    if (this.follower) this.follower.setPath(this.path);
    else this.follower = new PurePursuit(this.path, this.cfg.follower);
    this.replans++;
    return true;
  }

  /** Updates the pose estimate and, in sensor-only mode, the map. Call on every control update. */
  observe(input: NavInput): void {
    const dt = this.lastT === null ? 0 : Math.max(0, input.t - this.lastT);
    this.lastT = input.t;
    this.pose = this.estimatePose(input, dt);
    if (this.mode === 'sensor-only') this.integrateRays(input);
  }

  /** One control update (50 to 100 Hz): observe, then plan and follow toward the goal, if any. */
  update(input: NavInput): NavCommand {
    this.observe(input);
    const goal = this.goal;
    if (!goal || this.status === 'idle') return { v: 0, omega: 0, done: false };
    if (this.status === 'reached') return { v: 0, omega: 0, done: true };
    if (input.t - this.lastPlan >= this.cfg.replanIntervalS || !this.follower) {
      this.lastPlan = input.t;
      if (!this.plan(goal)) {
        this.status = 'no-path';
        return { v: 0, omega: 0, done: false };
      }
    } else if (this.status === 'no-path') {
      // The latest plan failed on the current map: stand still until a replan succeeds instead of following the stale path.
      return { v: 0, omega: 0, done: false };
    }
    const cmd = this.follower!.command(this.pose);
    if (cmd.done) {
      this.status = 'reached';
      return { v: 0, omega: 0, done: true };
    }
    this.status = 'following';
    return { v: cmd.v, omega: cmd.omega, done: false };
  }
}
