import type { Cell } from './grid';

export interface PathResult {
  path: Cell[];
  /** Path length in cells (1 per straight step, sqrt(2) per diagonal). */
  cost: number;
  expanded: number;
}

class MinHeap {
  private readonly keys: number[] = [];
  private readonly vals: number[] = [];
  get size(): number {
    return this.keys.length;
  }
  push(key: number, val: number): void {
    const k = this.keys, v = this.vals;
    let i = k.length;
    k.push(key);
    v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p]! <= key) break;
      k[i] = k[p]!;
      v[i] = v[p]!;
      i = p;
    }
    k[i] = key;
    v[i] = val;
  }
  pop(): number {
    const k = this.keys, v = this.vals;
    const top = v[0]!;
    const lastK = k.pop()!, lastV = v.pop()!;
    if (k.length > 0) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        let mk = lastK;
        if (l < k.length && k[l]! < mk) { m = l; mk = k[l]!; }
        if (r < k.length && k[r]! < mk) { m = r; mk = k[r]!; }
        if (m === i) break;
        k[i] = k[m]!;
        v[i] = v[m]!;
        i = m;
      }
      k[i] = lastK;
      v[i] = lastV;
    }
    return top;
  }
}

const SQRT2 = Math.SQRT2;
const MOVES: [number, number, number][] = [
  [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
  [1, 1, SQRT2], [1, -1, SQRT2], [-1, 1, SQRT2], [-1, -1, SQRT2],
];

/** Octile distance: exact cost between cells on an empty 8-connected grid (admissible, consistent). */
export function octile(ax: number, ay: number, bx: number, by: number): number {
  const dx = Math.abs(ax - bx), dy = Math.abs(ay - by);
  return Math.max(dx, dy) + (SQRT2 - 1) * Math.min(dx, dy);
}

/**
 * A* on an 8-connected grid. Diagonal moves may not cut corners: both
 * orthogonal neighbours must be free. A step into cell k costs its length
 * (1 or sqrt 2) times (1 + cellCost[k]); with non-negative costs the octile
 * heuristic stays admissible, so paths remain optimal for that cost. Returns
 * null when the goal cannot be reached, or when start or goal is blocked.
 */
export function astar(blocked: Uint8Array, width: number, height: number, start: Cell, goal: Cell, cellCost?: Float32Array): PathResult | null {
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < width && y < height;
  const free = (x: number, y: number) => inside(x, y) && blocked[y * width + x] === 0;
  if (!free(start[0], start[1]) || !free(goal[0], goal[1])) return null;
  const n = width * height;
  const g = new Float64Array(n).fill(Infinity);
  const parent = new Int32Array(n).fill(-1);
  const closed = new Uint8Array(n);
  const s = start[1] * width + start[0];
  const t = goal[1] * width + goal[0];
  g[s] = 0;
  const open = new MinHeap();
  open.push(octile(start[0], start[1], goal[0], goal[1]), s);
  let expanded = 0;
  while (open.size > 0) {
    const cur = open.pop();
    if (closed[cur]) continue;
    closed[cur] = 1;
    expanded++;
    if (cur === t) break;
    const cx = cur % width, cy = (cur - cx) / width;
    for (const [dx, dy, c] of MOVES) {
      const nx = cx + dx, ny = cy + dy;
      if (!free(nx, ny)) continue;
      if (dx !== 0 && dy !== 0 && (!free(cx + dx, cy) || !free(cx, cy + dy))) continue;
      const k = ny * width + nx;
      if (closed[k]) continue;
      const ng = g[cur]! + c * (1 + (cellCost ? Math.max(0, cellCost[k]!) : 0));
      if (ng < g[k]! - 1e-12) {
        g[k] = ng;
        parent[k] = cur;
        open.push(ng + octile(nx, ny, goal[0], goal[1]), k);
      }
    }
  }
  if (!Number.isFinite(g[t]!)) return null;
  const path: Cell[] = [];
  for (let k = t; k !== -1; k = parent[k]!) path.push([k % width, Math.floor(k / width)]);
  path.reverse();
  return { path, cost: g[t]!, expanded };
}
