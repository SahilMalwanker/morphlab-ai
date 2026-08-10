/** A point in the ground plane (x, y), m. */
export type Point2 = readonly [number, number];

const cross = (o: Point2, a: Point2, b: Point2): number => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

/** Convex hull, counterclockwise, without collinear points (monotone chain); fewer than three points come back sorted. */
export function convexHull(points: readonly Point2[]): Point2[] {
  const s = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (s.length < 3) return s;
  const half = (pts: Point2[]): Point2[] => {
    const h: Point2[] = [];
    for (const p of pts) {
      while (h.length >= 2 && cross(h[h.length - 2]!, h[h.length - 1]!, p) <= 0) h.pop();
      h.push(p);
    }
    return h;
  };
  const lower = half(s);
  const upper = half(s.reverse());
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/**
 * Static stability margin: the distance of `p` (the centre of mass projected
 * on the ground) inside (positive) or outside (negative) the convex hull of
 * the ground contacts, and the vector from `p` to the nearest hull point.
 * One or two contacts have no area, so the margin is never positive there.
 * Null without contacts.
 */
export function supportMargin(p: Point2, contacts: readonly Point2[]): { marginM: number; nearest: [number, number] } | null {
  if (contacts.length === 0) return null;
  const hull = convexHull(contacts);
  const n = hull.length;
  let best = Number.POSITIVE_INFINITY;
  let near: [number, number] = [hull[0]![0], hull[0]![1]];
  for (let i = 0; i < n; i++) {
    const a = hull[i]!, b = hull[(i + 1) % n]!;
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const len2 = ex * ex + ey * ey;
    const u = len2 > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2)) : 0;
    const qx = a[0] + u * ex, qy = a[1] + u * ey;
    const d = Math.hypot(p[0] - qx, p[1] - qy);
    if (d < best) {
      best = d;
      near = [qx, qy];
    }
  }
  const inside = n >= 3 && hull.every((a, i) => cross(a, hull[(i + 1) % n]!, p) > 0);
  return { marginM: inside ? best : -best, nearest: [near[0] - p[0], near[1] - p[1]] };
}
