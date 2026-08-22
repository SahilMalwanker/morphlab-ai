import type { Pose2 } from './follower';
import { rectOverlapsBox, type NavMap } from './maps';
import type { NavStatus } from './stack';

/** Rover footprint half extents including the wheels, m. */
export const ROVER_HALF_EXTENTS: [number, number] = [0.22, 0.21];

export interface NavRefereeConfig {
  /** Ground-truth distance to the goal that counts as arrival, m. */
  reachedToleranceM: number;
  /** A run fails once the planner has had no path for this long, s. */
  noPathTimeoutS: number;
  /** A run fails when it has not arrived within this time, s. */
  timeLimitS: number;
}

export const DEFAULT_REFEREE: NavRefereeConfig = { reachedToleranceM: 0.25, noPathTimeoutS: 2, timeLimitS: 60 };

export type NavOutcome = 'reached' | 'collision' | 'fell' | 'no-path' | 'timeout' | 'stopped-short' | 'numerical-instability';

export interface NavVerdict {
  outcome: NavOutcome;
  /** Ground-truth distance from the goal when the run ended, m (NaN for numerical instability). */
  finalDistanceM: number;
  message: string;
}

/**
 * Ground-truth referee for a navigation run, shared by the live view and the
 * batch evaluation so both end runs by the same rules. It never feeds the
 * controller; it only reads the simulator pose and the controller's status.
 */
export class NavReferee {
  private noPathSince = -1;
  private readonly startT: number;

  constructor(
    readonly map: NavMap,
    readonly goal: readonly [number, number],
    startT: number,
    readonly cfg: NavRefereeConfig = DEFAULT_REFEREE,
    /** Footprint half extents of the body being judged, m. */
    readonly halfExtents: readonly [number, number] = ROVER_HALF_EXTENTS,
  ) {
    this.startT = startT;
  }

  /** True when the footprint at this pose overlaps an obstacle. */
  static collides(map: NavMap, p: Pose2, halfExtents: readonly [number, number] = ROVER_HALF_EXTENTS): boolean {
    return map.boxes.some((b) => rectOverlapsBox(p.x, p.y, p.yaw, halfExtents[0], halfExtents[1], b));
  }

  /** Checks one physics step; returns a verdict once the run is over, otherwise null. `upright` is false once a walking body has fallen (ground truth). */
  check(t: number, truth: Pose2, status: NavStatus, finiteState: boolean, upright = true): NavVerdict | null {
    const d = Math.hypot(truth.x - this.goal[0], truth.y - this.goal[1]);
    if (!finiteState) return { outcome: 'numerical-instability', finalDistanceM: Number.NaN, message: 'The simulation became numerically unstable.' };
    if (!upright) return { outcome: 'fell', finalDistanceM: d, message: 'Fell over (detected from simulator state).' };
    if (NavReferee.collides(this.map, truth, this.halfExtents)) return { outcome: 'collision', finalDistanceM: d, message: 'Collision with an obstacle (detected from simulator state).' };
    if (status === 'reached') {
      return d < this.cfg.reachedToleranceM
        ? { outcome: 'reached', finalDistanceM: d, message: `Reached the goal (${(d * 100).toFixed(0)} cm from it, simulator ground truth).` }
        : { outcome: 'stopped-short', finalDistanceM: d, message: `Stopped ${d.toFixed(2)} m from the goal: the pose estimate drifted.` };
    }
    if (status === 'no-path') {
      if (this.noPathSince < 0) this.noPathSince = t;
      if (t - this.noPathSince > this.cfg.noPathTimeoutS) return { outcome: 'no-path', finalDistanceM: d, message: `No path to the goal on the current map for ${this.cfg.noPathTimeoutS} s.` };
    } else {
      this.noPathSince = -1;
    }
    if (t - this.startT >= this.cfg.timeLimitS) return { outcome: 'timeout', finalDistanceM: d, message: `Did not reach the goal within ${this.cfg.timeLimitS} s.` };
    return null;
  }
}
