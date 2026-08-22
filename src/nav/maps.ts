import type { Box2 } from './grid';
import type { Pose2 } from './follower';

export interface NavMap {
  id: string;
  name: string;
  start: Pose2;
  goal: [number, number];
  boxes: Box2[];
}

/** Obstacle height used for every map box, m. */
export const OBSTACLE_HEIGHT = 0.3;

/** World area covered by the maps and the occupancy grid. */
export const ARENA = { minX: -0.6, minY: -2.1, maxX: 3.6, maxY: 2.1 } as const;

const wall = (x0: number, y0: number, x1: number, y1: number): Box2 => ({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, hx: Math.abs(x1 - x0) / 2, hy: Math.abs(y1 - y0) / 2 });

/** Five test maps of increasing difficulty; the rover starts at the origin facing +X. */
export const TEST_MAPS: NavMap[] = [
  { id: 'open', name: 'Open floor', start: { x: 0, y: 0, yaw: 0 }, goal: [3, 0.6], boxes: [] },
  { id: 'block', name: 'Single block', start: { x: 0, y: 0, yaw: 0 }, goal: [3, 0], boxes: [wall(1.3, -0.45, 1.7, 0.45)] },
  { id: 'slalom', name: 'Slalom', start: { x: 0, y: 0, yaw: 0 }, goal: [3, 0], boxes: [wall(0.85, -0.3, 1.15, 1.2), wall(1.95, -1.2, 2.25, 0.3)] },
  { id: 'gap', name: 'Wall with a gap', start: { x: 0, y: 0, yaw: 0 }, goal: [3, 1], boxes: [wall(1.4, -2.1, 1.6, -0.4), wall(1.4, 0.4, 1.6, 2.1)] },
  { id: 'trap', name: 'U-shaped trap', start: { x: 0, y: 0, yaw: 0 }, goal: [3, 0], boxes: [wall(2.0, -0.7, 2.2, 0.7), wall(1.1, 0.5, 2.2, 0.7), wall(1.1, -0.7, 2.2, -0.5)] },
];

/** Separating-axis test between an oriented rectangle and an axis-aligned box. */
export function rectOverlapsBox(cx: number, cy: number, yaw: number, hx: number, hy: number, b: Box2): boolean {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  const axes: [number, number][] = [[1, 0], [0, 1], [c, s], [-s, c]];
  const corners = (pts: [number, number][]) => pts;
  const rect = corners([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]].map(([u, v]) => [cx + c * u! - s * v!, cy + s * u! + c * v!] as [number, number]));
  const box: [number, number][] = [[b.cx - b.hx, b.cy - b.hy], [b.cx + b.hx, b.cy - b.hy], [b.cx + b.hx, b.cy + b.hy], [b.cx - b.hx, b.cy + b.hy]];
  for (const [ax, ay] of axes) {
    const pr = rect.map(([x, y]) => x * ax + y * ay), pb = box.map(([x, y]) => x * ax + y * ay);
    if (Math.max(...pr) < Math.min(...pb) || Math.max(...pb) < Math.min(...pr)) return false;
  }
  return true;
}
