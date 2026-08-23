import { describe, expect, it } from 'vitest';
import { mulberry32 } from '../../src/core/math/rng';
import { astar, octile } from '../../src/nav/astar';
import { capSpeed, PurePursuit, WheelImuOdometry, wheelSpeeds, wrapAngle, yawFromQuat } from '../../src/nav/follower';
import { bresenham, OccupancyGrid } from '../../src/nav/grid';
import { rectOverlapsBox } from '../../src/nav/maps';

/** Independent reference: Dijkstra with a linear scan for the minimum (no heap, no heuristic). */
function dijkstra(blocked: Uint8Array, w: number, h: number, s: [number, number], t: [number, number], cost?: Float32Array): number | null {
  const n = w * h;
  const dist = new Float64Array(n).fill(Infinity);
  const done = new Uint8Array(n);
  const free = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && blocked[y * w + x] === 0;
  if (!free(...s) || !free(...t)) return null;
  dist[s[1] * w + s[0]] = 0;
  for (;;) {
    let u = -1, best = Infinity;
    for (let i = 0; i < n; i++) if (!done[i] && dist[i]! < best) { best = dist[i]!; u = i; }
    if (u < 0) return null;
    if (u === t[1] * w + t[0]) return best;
    done[u] = 1;
    const ux = u % w, uy = Math.floor(u / w);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const x = ux + dx, y = uy + dy;
      if (!free(x, y)) continue;
      if (dx && dy && (!free(ux + dx, uy) || !free(ux, uy + dy))) continue;
      const c = (dx && dy ? Math.SQRT2 : 1) * (1 + (cost ? cost[y * w + x]! : 0));
      if (best + c < dist[y * w + x]!) dist[y * w + x] = best + c;
    }
  }
}

describe('A* planner', () => {
  it('finds shortest paths: cost equals an independent Dijkstra on 300 random maps', () => {
    const rng = mulberry32(5);
    let solved = 0, unreachable = 0;
    for (let trial = 0; trial < 300; trial++) {
      const w = 8 + Math.floor(rng() * 20), h = 8 + Math.floor(rng() * 20);
      const blocked = new Uint8Array(w * h).map(() => (rng() < 0.28 ? 1 : 0));
      const s: [number, number] = [Math.floor(rng() * w), Math.floor(rng() * h)];
      const t: [number, number] = [Math.floor(rng() * w), Math.floor(rng() * h)];
      const ref = dijkstra(blocked, w, h, s, t);
      const r = astar(blocked, w, h, s, t);
      if (ref === null) {
        expect(r).toBeNull();
        unreachable++;
        continue;
      }
      solved++;
      expect(r!.cost).toBeCloseTo(ref, 9);
      // The path is valid: starts and ends right, free cells, unit or diagonal steps, no corner cutting, cost matches.
      expect(r!.path[0]).toEqual(s);
      expect(r!.path.at(-1)).toEqual(t);
      let c = 0;
      for (let i = 1; i < r!.path.length; i++) {
        const [ax, ay] = r!.path[i - 1]!, [bx, by] = r!.path[i]!;
        const dx = bx - ax, dy = by - ay;
        expect(Math.max(Math.abs(dx), Math.abs(dy))).toBe(1);
        expect(blocked[by * w + bx]).toBe(0);
        if (dx && dy) {
          expect(blocked[ay * w + bx]).toBe(0);
          expect(blocked[by * w + ax]).toBe(0);
        }
        c += dx && dy ? Math.SQRT2 : 1;
      }
      expect(c).toBeCloseTo(r!.cost, 9);
    }
    expect(solved).toBeGreaterThan(100);
    expect(unreachable).toBeGreaterThan(10);
  });

  it('stays optimal with a non-negative cell cost layer (proximity costs)', () => {
    const rng = mulberry32(9);
    for (let trial = 0; trial < 150; trial++) {
      const w = 6 + Math.floor(rng() * 15), h = 6 + Math.floor(rng() * 15);
      const blocked = new Uint8Array(w * h).map(() => (rng() < 0.2 ? 1 : 0));
      const cost = new Float32Array(w * h).map(() => 3 * rng());
      const s: [number, number] = [Math.floor(rng() * w), Math.floor(rng() * h)];
      const t: [number, number] = [Math.floor(rng() * w), Math.floor(rng() * h)];
      const ref = dijkstra(blocked, w, h, s, t, cost);
      const r = astar(blocked, w, h, s, t, cost);
      if (ref === null) expect(r).toBeNull();
      else expect(r!.cost).toBeCloseTo(ref, 4);
    }
  });

  it('uses the octile distance on an empty grid and refuses blocked endpoints', () => {
    const empty = new Uint8Array(100);
    expect(astar(empty, 10, 10, [0, 0], [9, 4])!.cost).toBeCloseTo(octile(0, 0, 9, 4), 12);
    const b = new Uint8Array(100);
    b[4 * 10 + 9] = 1;
    expect(astar(b, 10, 10, [0, 0], [9, 4])).toBeNull();
  });
});

describe('occupancy grid', () => {
  it('Bresenham lines are 8-connected and include both ends', () => {
    const cells = bresenham(0, 0, 7, -3);
    expect(cells[0]).toEqual([0, 0]);
    expect(cells.at(-1)).toEqual([7, -3]);
    for (let i = 1; i < cells.length; i++) expect(Math.max(Math.abs(cells[i]![0] - cells[i - 1]![0]), Math.abs(cells[i]![1] - cells[i - 1]![1]))).toBe(1);
  });

  it('a ray marks free space up to the hit and an occupied cell at the hit', () => {
    const g = new OccupancyGrid({ originX: 0, originY: 0, resolution: 0.1, width: 30, height: 10 });
    for (let k = 0; k < 3; k++) g.integrateRay(0.05, 0.55, 0, 1.5, 2);
    expect(g.isOccupied(...g.worldToCell(1.55, 0.55))).toBe(true);
    expect(g.value(...g.worldToCell(0.75, 0.55))).toBeLessThan(0);
    expect(g.value(...g.worldToCell(2.5, 0.55))).toBe(0);
    // No hit within range: everything along the ray becomes free.
    g.integrateRay(0.05, 0.95, 0, -1, 2);
    expect(g.isOccupied(...g.worldToCell(1.95, 0.95))).toBe(false);
  });

  it('inflation grows obstacles by the robot radius and out-of-bounds is blocked', () => {
    const g = new OccupancyGrid({ originX: 0, originY: 0, resolution: 0.1, width: 20, height: 20 });
    g.markBoxes([{ cx: 1, cy: 1, hx: 0.1, hy: 0.1 }]);
    const m = g.inflated(0.3);
    expect(m[10 * 20 + 13]).toBe(1);
    expect(m[10 * 20 + 15]).toBe(0);
    expect(g.isOccupied(-1, 0)).toBe(true);
  });
});

describe('path following and odometry', () => {
  it('pure pursuit steers toward the path and stops at the goal', () => {
    const f = new PurePursuit([[0, 0], [1, 0], [2, 0]]);
    const ahead = f.command({ x: 0, y: 0.1, yaw: 0 });
    expect(ahead.v).toBeGreaterThan(0);
    expect(ahead.omega).toBeLessThan(0);
    const behind = f.command({ x: 0.5, y: 0, yaw: Math.PI });
    expect(behind.v).toBe(0);
    expect(Math.abs(behind.omega)).toBeGreaterThan(0);
    expect(f.command({ x: 1.95, y: 0, yaw: 0 }).done).toBe(true);
  });

  it('skid-steer wheel speeds and odometry are consistent', () => {
    const w = wheelSpeeds(0.4, 0, 0.06, 0.34);
    expect(w.left).toBeCloseTo(0.4 / 0.06, 12);
    const turn = wheelSpeeds(0, 1, 0.06, 0.34);
    expect(turn.left).toBeCloseTo(-turn.right, 12);
    const odo = new WheelImuOdometry({ x: 0, y: 0, yaw: 0 }, 0.06);
    for (let i = 0; i < 100; i++) odo.update(0.01, w.left, w.right, 0.3);
    expect(odo.pose.x).toBeCloseTo(0.4, 9);
    expect(odo.pose.yaw).toBeCloseTo(0, 12);
    expect(wrapAngle(3 * Math.PI)).toBeCloseTo(Math.PI, 12);
    expect(yawFromQuat([Math.cos(0.35), 0, 0, Math.sin(0.35)])).toBeCloseTo(0.7, 12);
  });

  it('the speed cap scales both wheels, keeping the curvature, and leaves slower commands alone', () => {
    const r = 0.06, track = 0.34;
    const cmd = wheelSpeeds(0.4, 0.5, r, track);
    const capped = capSpeed(cmd, 0.2, r);
    // Body speed v = r (left + right) / 2 and turn rate omega = r (right - left) / track.
    expect((r * (capped.left + capped.right)) / 2).toBeCloseTo(0.2, 12);
    expect((r * (capped.right - capped.left)) / track).toBeCloseTo(0.25, 12);
    expect(capSpeed(cmd, 0.45, r)).toEqual(cmd);
    const back = capSpeed(wheelSpeeds(-0.4, 0, r, track), 0.1, r);
    expect((r * (back.left + back.right)) / 2).toBeCloseTo(-0.1, 12);
    // Turning in place has no body speed to cap.
    const spin = wheelSpeeds(0, 1.5, r, track);
    expect(capSpeed(spin, 0.1, r)).toEqual(spin);
  });

  it('rectangle-box overlap matches simple cases', () => {
    const b = { cx: 1, cy: 0, hx: 0.1, hy: 0.5 };
    expect(rectOverlapsBox(0.7, 0, 0, 0.22, 0.21, b)).toBe(true);
    expect(rectOverlapsBox(0.6, 0, 0, 0.22, 0.21, b)).toBe(false);
    expect(rectOverlapsBox(0.62, 0, Math.PI / 4, 0.22, 0.21, b)).toBe(true);
  });
});
