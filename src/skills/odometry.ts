import { forwardKinematics, type KinematicModel } from '../core/kinematics/fk';
import { identity, quatToRotation, type Mat4 } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import type { FootLiftObservation } from './footLift';

/** Normal force above which a foot counts as on the ground, N. */
const CONTACT_N = 1;
/** Time constant of the velocity estimate's low-pass filter, s. */
const VELOCITY_TAU_S = 0.05;

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

/** R v and R^T v for the rotation part of a column-major transform. */
const rotate = (T: Mat4, v: Readonly<Vec3>): Vec3 => [T[0]! * v[0] + T[4]! * v[1] + T[8]! * v[2], T[1]! * v[0] + T[5]! * v[1] + T[9]! * v[2], T[2]! * v[0] + T[6]! * v[1] + T[10]! * v[2]];
const rotateBack = (T: Mat4, v: Readonly<Vec3>): Vec3 => [T[0]! * v[0] + T[1]! * v[1] + T[2]! * v[2], T[4]! * v[0] + T[5]! * v[1] + T[6]! * v[2], T[8]! * v[0] + T[9]! * v[1] + T[10]! * v[2]];

/**
 * Sensor-only estimate of how the body has moved since a walk started (D29):
 * heading from the IMU orientation, and body velocity from the feet on the
 * ground (leg odometry). A planted foot does not move, so the body moves at
 * -(dp/dt + w x p) in its own frame, where p is the foot position from
 * forward kinematics of the joint encoders and w the gyro rate. Feet count as
 * planted when their touch sensor reads more than 1 N now and at the previous
 * update. It never reads simulator state; slipping feet make it drift.
 *
 * With `rollingFeet`, p is the point of the foot sphere that touched the
 * ground at the previous update, carried along with the foot: a sphere that
 * rolls as the shank pitches through its stance moves the body further than
 * its center does (D37). Without it (the default, used by the walking
 * controllers) p is the sphere's center.
 */
export class LegOdometry {
  /** Forward and left of where the walk started, m, in the start frame. */
  x = 0;
  y = 0;
  /** Heading change since the start, rad (positive left). */
  heading = 0;
  /** Body-frame velocity, forward and left, m/s, low-pass filtered. */
  vx = 0;
  vy = 0;
  private yaw0: number | null = null;
  private prevYaw = 0;
  private prev: { t: number; feet: Vec3[]; contact: boolean[]; tips: Mat4[]; down: Vec3 } | null = null;
  private readonly base = identity();
  private readonly radius: number[];

  constructor(private readonly km: KinematicModel, private readonly legs: readonly number[], private readonly rollingFeet = false) {
    this.radius = legs.map((li) => km.description.limbs[li]?.tip.radius ?? 0);
  }

  /** The estimate as forward and left, m, and heading change, rad, in the start frame. */
  get pose(): Vec3 {
    return [this.x, this.y, this.heading];
  }

  update(t: number, obs: FootLiftObservation): void {
    const R = quatToRotation(obs.imuQuat);
    const yaw = Math.atan2(R[3]!, R[0]!);
    if (this.yaw0 === null) {
      this.yaw0 = yaw;
      this.prevYaw = yaw;
    }
    const yaw0 = this.yaw0 ?? yaw;
    this.heading += wrap(yaw - this.prevYaw);
    this.prevYaw = yaw;
    const tips = forwardKinematics(this.km, obs.q, this.base).tips;
    const legTips = this.legs.map((li) => tips[li]!);
    const feet = legTips.map((T): Vec3 => [T[12]!, T[13]!, T[14]!]);
    const contact = this.legs.map((li) => (obs.touch[li] ?? 0) > CONTACT_N);
    // World down in the body frame (third row of the IMU rotation, negated).
    const down: Vec3 = [-R[6]!, -R[7]!, -R[8]!];
    const prev = this.prev;
    if (prev && t > prev.t) {
      const dt = t - prev.t;
      const w = obs.gyro ?? [0, 0, 0];
      let sx = 0, sy = 0, sz = 0, n = 0;
      for (let k = 0; k < feet.length; k++) {
        if (!contact[k] || !prev.contact[k]) continue;
        let p = feet[k]!, p0 = prev.feet[k]!;
        if (this.rollingFeet) {
          // The sphere point that touched the ground at the previous update, then and now.
          const r = this.radius[k]!;
          const d0: Vec3 = [r * prev.down[0], r * prev.down[1], r * prev.down[2]];
          const d = rotate(legTips[k]!, rotateBack(prev.tips[k]!, d0));
          p0 = [p0[0] + d0[0], p0[1] + d0[1], p0[2] + d0[2]];
          p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
        }
        sx -= (p[0] - p0[0]) / dt + (w[1] * p[2] - w[2] * p[1]);
        sy -= (p[1] - p0[1]) / dt + (w[2] * p[0] - w[0] * p[2]);
        sz -= (p[2] - p0[2]) / dt + (w[0] * p[1] - w[1] * p[0]);
        n++;
      }
      if (n > 0) {
        const vb: Vec3 = [sx / n, sy / n, sz / n];
        const vwx = R[0]! * vb[0] + R[1]! * vb[1] + R[2]! * vb[2];
        const vwy = R[3]! * vb[0] + R[4]! * vb[1] + R[5]! * vb[2];
        const c = Math.cos(yaw0), s = Math.sin(yaw0);
        this.x += (c * vwx + s * vwy) * dt;
        this.y += (-s * vwx + c * vwy) * dt;
        const alpha = dt / (VELOCITY_TAU_S + dt);
        this.vx += alpha * (vb[0] - this.vx);
        this.vy += alpha * (vb[1] - this.vy);
      }
    }
    this.prev = { t, feet, contact, tips: legTips, down };
  }
}
