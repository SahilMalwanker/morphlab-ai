import type { MainModule } from '@mujoco/mujoco';
import { rover } from '../core/model/presets';
import type { RobotDescription } from '../core/model/types';
import { driveGeometry, navConfigFor, odometryFor, placeAt, wheelTargets, type DriveGeometry } from '../nav/drive';
import { wheelSpeeds, type Pose2 } from '../nav/follower';
import { LEG_GAINS, LeggedDriver } from '../nav/legged';
import { OBSTACLE_HEIGHT, type NavMap } from '../nav/maps';
import { DEFAULT_REFEREE, NavReferee, ROVER_HALF_EXTENTS, type NavOutcome, type NavVerdict } from '../nav/referee';
import { NavStack, RAY_ANGLES, RAY_ELEVATION, RAY_ORIGIN, type NavInput } from '../nav/stack';
import type { LocomotionController } from '../skills/locomotion';
import type { Run } from './experiments';
import type { BoxObstacle, RangeRay } from './mjcf';
import { MujocoRuntime, type SensorReadings } from './runtime';

export { ROVER_HALF_EXTENTS };
/** Wheel velocity gain, N*m per rad/s. */
export const WHEEL_KV = 0.6;
export const NAV_CONTROL_DECIMATION = 10;
const CONTROL_DECIMATION = NAV_CONTROL_DECIMATION;
/** A walking body has fallen once it tilts this far from upright, rad. */
export const FALL_TILT_RAD = 0.8;

export const rangeRays = (origin: readonly [number, number, number] = RAY_ORIGIN): RangeRay[] =>
  RAY_ANGLES.map((a, i) => ({ name: `r${i}`, pos: [origin[0], origin[1], origin[2]], dir: [Math.cos(a) * Math.cos(RAY_ELEVATION), Math.sin(a) * Math.cos(RAY_ELEVATION), Math.sin(RAY_ELEVATION)] }));

export function mapObstacles(map: NavMap): BoxObstacle[] {
  return map.boxes.map((b, i) => ({ name: `b${i}`, pos: [b.cx, b.cy, OBSTACLE_HEIGHT / 2], halfSize: [b.hx, b.hy, OBSTACLE_HEIGHT / 2] }));
}

/** The body's drive geometry; throws, naming the reason, when the navigation stack cannot drive it. */
export function navDrive(robot: RobotDescription): DriveGeometry {
  const r = driveGeometry(robot);
  if (!r.ok) throw new RangeError(`${robot.name} cannot navigate: ${r.reason}.`);
  return r.drive;
}

/** Builds a wheeled or legged body (the rover by default) in a test map with its range rays attached; legged bodies start in their stance. */
export function createNavRuntime(mj: MainModule, map: NavMap, robot: RobotDescription = rover(), drive: DriveGeometry = navDrive(robot)): MujocoRuntime {
  const rays = rangeRays(drive.rayOrigin);
  const joints = robot.limbs.reduce((n, l) => n + l.joints.length, 0);
  const initialQ = drive.kind === 'legged' ? drive.stanceQ.slice() : new Array<number>(joints).fill(0);
  const rt = MujocoRuntime.create(mj, placeAt(robot, drive, map.start), { initialQ, obstacles: mapObstacles(map), rangeRays: rays });
  rt.attachRangeSensors(rays.map((r) => r.name));
  return rt;
}

/** Ground truth (referee only): false once a walking body has tipped over or sunk to half its standing height. */
export function groundTruthUpright(rt: MujocoRuntime, drive: DriveGeometry): boolean {
  const b = rt.groundTruthBaseState();
  return b.tilt < FALL_TILT_RAD && b.pos[2] > 0.5 * drive.restHeight;
}

/** Ground truth (referee and labeled ground-truth mode only): planar pose of the base. */
export function groundTruthPose(rt: MujocoRuntime): Pose2 {
  const T = rt.groundTruthBaseTransform();
  return { x: T[12]!, y: T[13]!, yaw: Math.atan2(T[1]!, T[0]!) };
}

/**
 * Wheel velocity control with saturation at the effort limit. Wheel speed
 * targets are clamped to each wheel's declared velocity limit first.
 * One target per wheel actuator, in joint order.
 */
export function wheelTorques(target: readonly number[], s: SensorReadings, effort: readonly number[], velocityLimit?: readonly number[]): number[] {
  return target.map((w, i) => {
    const v = velocityLimit ? Math.max(-velocityLimit[i]!, Math.min(velocityLimit[i]!, w)) : w;
    return Math.max(-effort[i]!, Math.min(effort[i]!, WHEEL_KV * (v - s.qd[i]!)));
  });
}

export interface NavRunConfig {
  mode: 'sensor-only' | 'ground-truth';
  timeLimitS?: number;
  /** The wheeled or legged body to move; the rover when absent. */
  robot?: RobotDescription;
  /** A legged body's walking controller (trained in the AI Playground); the gait generator walks alone without it. */
  walking?: LocomotionController | null;
  /** Sees every input handed to the navigation controller (tests use it to audit inputs). */
  spy?: (input: NavInput) => void;
}

export interface NavRunResult {
  mapId: string;
  mode: 'sensor-only' | 'ground-truth';
  outcome: NavOutcome;
  reached: boolean;
  collided: boolean;
  failure: string | null;
  timeS: number;
  /** Ground-truth distance from the goal at the end, m. */
  finalDistanceM: number;
  /** Distance between the controller's pose estimate and ground truth at the end, m. */
  estimateErrorM: number;
  replans: number;
  /** Ground-truth path (every 0.1 s). */
  trace: [number, number][];
  lastPlan: [number, number][];
}

/**
 * Drives a wheeled body (the rover by default) or walks a legged one to the
 * map goal. The controller gets sensor readings only (plus the simulator
 * pose in the labeled ground-truth mode); the shared referee uses ground
 * truth to end the run.
 */
export function* runNavigation(mj: MainModule, map: NavMap, cfg: NavRunConfig): Run<NavRunResult> {
  const robot = cfg.robot ?? rover();
  const drive = navDrive(robot);
  const rt = createNavRuntime(mj, map, robot, drive);
  try {
    const stack = new NavStack(cfg.mode, map.start, odometryFor(robot, drive, map.start), cfg.mode === 'ground-truth' ? map.boxes : null, navConfigFor(drive));
    stack.setGoal(map.goal);
    const timeLimitS = cfg.timeLimitS ?? drive.limits.timeLimitS;
    const referee = new NavReferee(map, map.goal, 0, { ...DEFAULT_REFEREE, timeLimitS }, drive.halfExtents);
    const velocityLimit = rt.description.limbs.flatMap((l) => l.joints.map((j) => j.limit.velocity));
    const h = rt.timestep;
    const total = Math.round(timeLimitS / h);
    const legs = drive.kind === 'legged' ? new LeggedDriver(robot, drive, LEG_GAINS, cfg.walking ?? null) : null;
    let target = drive.kind === 'wheeled' ? drive.wheels.map(() => 0) : [];
    const trace: [number, number][] = [];
    let verdict: NavVerdict | null = null;
    for (let k = 0; !verdict; k++) {
      const t = k * h;
      rt.controlStep((s) => {
        if (k % CONTROL_DECIMATION === 0) {
          const input: NavInput = cfg.mode === 'ground-truth' ? { t, sensors: s, groundTruth: { pose: groundTruthPose(rt) } } : { t, sensors: s };
          cfg.spy?.(input);
          const cmd = stack.update(input);
          if (drive.kind === 'wheeled') {
            const w = wheelSpeeds(cmd.v, cmd.omega, drive.wheelRadius, drive.trackWidth);
            target = wheelTargets(drive.wheels, w.left, w.right);
          } else legs!.setCommand(cmd.v, cmd.omega);
        }
        return legs ? legs.torques(t, h, s, rt.layout.effort) : wheelTorques(target, s, rt.layout.effort, velocityLimit);
      });
      const finite = !rt.hasNonFiniteState();
      const gt = finite ? groundTruthPose(rt) : { x: Number.NaN, y: Number.NaN, yaw: Number.NaN };
      if (k % 50 === 0 && finite) trace.push([gt.x, gt.y]);
      verdict = referee.check(rt.time, gt, stack.status, finite, !legs || !finite || groundTruthUpright(rt, drive));
      if (k % 200 === 0) yield Math.min(1, k / total);
    }
    const gt = groundTruthPose(rt);
    const reached = verdict.outcome === 'reached';
    return {
      mapId: map.id,
      mode: cfg.mode,
      outcome: verdict.outcome,
      reached,
      collided: verdict.outcome === 'collision',
      failure: reached ? null : verdict.message,
      timeS: +rt.time.toFixed(2),
      finalDistanceM: +verdict.finalDistanceM.toFixed(3),
      estimateErrorM: +Math.hypot(stack.pose.x - gt.x, stack.pose.y - gt.y).toFixed(3),
      replans: stack.replans,
      trace,
      lastPlan: stack.path,
    };
  } finally {
    rt.dispose();
  }
}
