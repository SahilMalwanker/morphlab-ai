/** Axis-aligned box obstacle on the floor, in world metres. */
export interface Box2 {
  cx: number;
  cy: number;
  hx: number;
  hy: number;
}

export interface GridSpec {
  /** World position of the corner of cell (0, 0), m. */
  originX: number;
  originY: number;
  /** Cell edge length, m. */
  resolution: number;
  width: number;
  height: number;
}

export type Cell = [number, number];

const L_OCC = 0.9;
const L_FREE = -0.4;
const L_MAX = 5;
const OCCUPIED = 0.85;

/**
 * Log-odds occupancy grid. Cells start unknown (0); range rays mark the
 * cells they pass through as free and the cell where they hit as occupied.
 */
export class OccupancyGrid {
  readonly logOdds: Float32Array;

  constructor(readonly spec: GridSpec) {
    if (!(spec.resolution > 0) || !Number.isInteger(spec.width) || !Number.isInteger(spec.height) || spec.width < 1 || spec.height < 1) {
      throw new RangeError('Grid needs a positive resolution and whole-number dimensions');
    }
    this.logOdds = new Float32Array(spec.width * spec.height);
  }

  inBounds(ix: number, iy: number): boolean {
    return ix >= 0 && iy >= 0 && ix < this.spec.width && iy < this.spec.height;
  }

  worldToCell(x: number, y: number): Cell {
    return [Math.floor((x - this.spec.originX) / this.spec.resolution), Math.floor((y - this.spec.originY) / this.spec.resolution)];
  }

  cellCenter(ix: number, iy: number): [number, number] {
    return [this.spec.originX + (ix + 0.5) * this.spec.resolution, this.spec.originY + (iy + 0.5) * this.spec.resolution];
  }

  value(ix: number, iy: number): number {
    return this.inBounds(ix, iy) ? this.logOdds[iy * this.spec.width + ix]! : L_MAX;
  }

  isOccupied(ix: number, iy: number): boolean {
    return this.value(ix, iy) > OCCUPIED;
  }

  private add(ix: number, iy: number, d: number): void {
    if (!this.inBounds(ix, iy)) return;
    const k = iy * this.spec.width + ix;
    this.logOdds[k] = Math.max(-L_MAX, Math.min(L_MAX, this.logOdds[k]! + d));
  }

  /**
   * Integrates one range ray from (x0, y0) along `angle` (world frame). A
   * reading of -1 or beyond `maxRange` means nothing was hit within range.
   */
  integrateRay(x0: number, y0: number, angle: number, range: number, maxRange: number): void {
    const hit = range >= 0 && range <= maxRange;
    const len = hit ? range : maxRange;
    const [ax, ay] = this.worldToCell(x0, y0);
    const [bx, by] = this.worldToCell(x0 + len * Math.cos(angle), y0 + len * Math.sin(angle));
    const cells = bresenham(ax, ay, bx, by);
    for (let i = 0; i < cells.length - 1; i++) this.add(cells[i]![0], cells[i]![1], L_FREE);
    const last = cells[cells.length - 1]!;
    this.add(last[0], last[1], hit ? L_OCC : L_FREE);
  }

  /** Marks every cell that overlaps a box as occupied (a known map). */
  markBoxes(boxes: Box2[]): void {
    for (const b of boxes) {
      const [x0, y0] = this.worldToCell(b.cx - b.hx, b.cy - b.hy);
      const [x1, y1] = this.worldToCell(b.cx + b.hx - 1e-9, b.cy + b.hy - 1e-9);
      for (let iy = y0; iy <= y1; iy++) for (let ix = x0; ix <= x1; ix++) if (this.inBounds(ix, iy)) this.logOdds[iy * this.spec.width + ix] = L_MAX;
    }
  }

  /** Occupied cells grown by `radiusM` (robot footprint), as a blocked mask. Out-of-bounds counts as blocked. */
  inflated(radiusM: number): Uint8Array {
    const d = this.distanceToOccupied();
    const out = new Uint8Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = d[i]! <= radiusM + 1e-6 ? 1 : 0;
    return out;
  }

  /**
   * Approximate Euclidean distance (m) from each cell centre to the nearest
   * occupied cell centre, by a two-pass chamfer transform (weights 1 and
   * sqrt 2; at most about 8 % above the true distance).
   */
  distanceToOccupied(): Float32Array {
    const { width: w, height: h, resolution } = this.spec;
    const d = new Float32Array(w * h);
    for (let i = 0; i < d.length; i++) d[i] = this.logOdds[i]! > OCCUPIED ? 0 : Infinity;
    const D = Math.SQRT2;
    const relax = (i: number, j: number, c: number) => {
      const v = d[j]! + c;
      if (v < d[i]!) d[i] = v;
    };
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (x > 0) relax(i, i - 1, 1);
        if (y > 0) {
          relax(i, i - w, 1);
          if (x > 0) relax(i, i - w - 1, D);
          if (x < w - 1) relax(i, i - w + 1, D);
        }
      }
    }
    for (let y = h - 1; y >= 0; y--) {
      for (let x = w - 1; x >= 0; x--) {
        const i = y * w + x;
        if (x < w - 1) relax(i, i + 1, 1);
        if (y < h - 1) {
          relax(i, i + w, 1);
          if (x < w - 1) relax(i, i + w + 1, D);
          if (x > 0) relax(i, i + w - 1, D);
        }
      }
    }
    for (let i = 0; i < d.length; i++) d[i] = d[i]! * resolution;
    return d;
  }
}

/** Cells on the line from a to b, inclusive (integer Bresenham). */
export function bresenham(x0: number, y0: number, x1: number, y1: number): Cell[] {
  const cells: Cell[] = [];
  const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
  let err = dx + dy, x = x0, y = y0;
  for (;;) {
    cells.push([x, y]);
    if (x === x1 && y === y1) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x += sx; }
    if (e2 <= dx) { err += dx; y += sy; }
  }
  return cells;
}
