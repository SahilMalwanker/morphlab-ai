export interface Pose2 {
  x: number;
  y: number;
  /** Heading about +Z, rad. */
  yaw: number;
}

export function wrapAngle(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

export interface FollowerConfig {
  lookaheadM: number;
  maxSpeed: number;
  maxTurnRate: number;
  goalToleranceM: number;
  /** Turn in place when the heading error exceeds this, rad. */
  turnInPlaceRad: number;
}

export const DEFAULT_FOLLOWER: FollowerConfig = { lookaheadM: 0.35, maxSpeed: 0.45, maxTurnRate: 1.8, goalToleranceM: 0.15, turnInPlaceRad: 0.9 };

/**
 * Pure pursuit path follower: steers toward the point one lookahead
 * distance ahead on the path. `progress` is the index of the path vertex
 * reached so far and only moves forward.
 */
export class PurePursuit {
  private progress = 0;
  constructor(private path: [number, number][], private readonly cfg: FollowerConfig = DEFAULT_FOLLOWER) {
    if (path.length === 0) throw new RangeError('Path is empty');
  }

  setPath(path: [number, number][]): void {
    if (path.length === 0) throw new RangeError('Path is empty');
    this.path = path;
    this.progress = 0;
  }

  get goal(): [number, number] {
    return this.path[this.path.length - 1]!;
  }

  /** Linear speed (m/s) and turn rate (rad/s) toward the path, or done when at the goal. */
  command(pose: Pose2): { v: number; omega: number; done: boolean } {
    const c = this.cfg;
    const P = this.path;
    const [gx, gy] = this.goal;
    if (Math.hypot(gx - pose.x, gy - pose.y) < c.goalToleranceM) return { v: 0, omega: 0, done: true };
    // Closest point on the path segments ahead of the progress marker.
    let seg = this.progress, segT = 0, bestD = Infinity;
    const last = Math.min(P.length - 1, this.progress + 40);
    for (let i = this.progress; i <= last; i++) {
      const a = P[i]!, b = P[Math.min(i + 1, P.length - 1)]!;
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const len2 = dx * dx + dy * dy;
      const t = len2 > 0 ? Math.max(0, Math.min(1, ((pose.x - a[0]) * dx + (pose.y - a[1]) * dy) / len2)) : 0;
      const d = Math.hypot(a[0] + t * dx - pose.x, a[1] + t * dy - pose.y);
      if (d < bestD) { bestD = d; seg = i; segT = t; }
    }
    this.progress = seg;
    // Walk the lookahead distance forward along the path from that point.
    let target: [number, number] = [gx, gy];
    let remaining = c.lookaheadM;
    for (let i = seg; i < P.length - 1; i++) {
      const a = P[i]!, b = P[i + 1]!;
      const segLen = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const start = i === seg ? segT * segLen : 0;
      if (segLen - start >= remaining) {
        const t = (start + remaining) / segLen;
        target = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        break;
      }
      remaining -= segLen - start;
    }
    const alpha = wrapAngle(Math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw);
    const clampW = (w: number) => Math.max(-c.maxTurnRate, Math.min(c.maxTurnRate, w));
    if (Math.abs(alpha) > c.turnInPlaceRad) return { v: 0, omega: clampW(2.5 * alpha), done: false };
    const dGoal = Math.hypot(gx - pose.x, gy - pose.y);
    const v = Math.min(c.maxSpeed, 0.8 * dGoal + 0.05) * Math.cos(alpha);
    const L = Math.max(0.05, Math.hypot(target[0] - pose.x, target[1] - pose.y));
    return { v, omega: clampW((2 * v * Math.sin(alpha)) / L), done: false };
  }
}

/** Skid-steer kinematics: body speed and turn rate to left and right wheel angular velocities. */
export function wheelSpeeds(v: number, omega: number, wheelRadius: number, trackWidth: number): { left: number; right: number } {
  return { left: (v - (omega * trackWidth) / 2) / wheelRadius, right: (v + (omega * trackWidth) / 2) / wheelRadius };
}

/** Caps the body speed of wheel targets; both wheels scale together, so the path curvature is unchanged. */
export function capSpeed(cmd: { left: number; right: number }, limit: number, wheelRadius: number): { left: number; right: number } {
  const v = Math.abs((wheelRadius * (cmd.left + cmd.right)) / 2);
  if (!(v > limit)) return { left: cmd.left, right: cmd.right };
  const k = limit / v;
  return { left: cmd.left * k, right: cmd.right * k };
}

/**
 * Dead reckoning from sensors only: wheel encoder speeds give forward speed,
 * the IMU orientation gives heading (relative to the heading at start).
 */
export class WheelImuOdometry {
  pose: Pose2;
  private yaw0: number | null = null;
  constructor(start: Pose2, private readonly wheelRadius: number) {
    this.pose = { ...start };
  }
  update(dt: number, leftWheelSpeed: number, rightWheelSpeed: number, imuYaw: number): Pose2 {
    if (this.yaw0 === null) this.yaw0 = imuYaw - this.pose.yaw;
    const yaw = wrapAngle(imuYaw - this.yaw0);
    const v = (this.wheelRadius * (leftWheelSpeed + rightWheelSpeed)) / 2;
    const mid = wrapAngle(this.pose.yaw + wrapAngle(yaw - this.pose.yaw) / 2);
    this.pose = { x: this.pose.x + v * Math.cos(mid) * dt, y: this.pose.y + v * Math.sin(mid) * dt, yaw };
    return this.pose;
  }
}

/** Heading about +Z from an orientation quaternion [w, x, y, z]. */
export function yawFromQuat(q: readonly number[]): number {
  const [w, x, y, z] = q as [number, number, number, number];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}
