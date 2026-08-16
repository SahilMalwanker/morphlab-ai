import { describe, expect, it } from 'vitest';
import { convexHull, supportMargin } from '../../src/core/math/support';

describe('support polygon', () => {
  const square = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]] as const;

  it('hull drops interior points and runs counterclockwise', () => {
    const h = convexHull(square);
    expect(h).toHaveLength(4);
    let area2 = 0;
    for (let i = 0; i < h.length; i++) area2 += h[i]![0] * h[(i + 1) % h.length]![1] - h[(i + 1) % h.length]![0] * h[i]![1];
    expect(area2).toBeCloseTo(2, 12);
  });

  it('is positive inside by the distance to the nearest edge', () => {
    const r = supportMargin([0.3, 0.5], square)!;
    expect(r.marginM).toBeCloseTo(0.3, 12);
    expect(r.nearest[0]).toBeCloseTo(-0.3, 12);
    expect(r.nearest[1]).toBeCloseTo(0, 12);
  });

  it('is negative outside and points to the nearest hull point, a corner here', () => {
    const r = supportMargin([1.3, 1.4], square)!;
    expect(r.marginM).toBeCloseTo(-0.5, 12);
    expect(r.nearest[0]).toBeCloseTo(-0.3, 12);
    expect(r.nearest[1]).toBeCloseTo(-0.4, 12);
  });

  it('a triangle of three feet: the centroid is inside by its inradius', () => {
    // Right triangle with legs 3 and 4: inradius (3 + 4 - 5) / 2 = 1, incenter (1, 1).
    expect(supportMargin([1, 1], [[0, 0], [3, 0], [0, 4]])!.marginM).toBeCloseTo(1, 12);
  });

  it('two feet or one have no area: never positive', () => {
    expect(supportMargin([0.5, 0.2], [[0, 0], [1, 0]])!.marginM).toBeCloseTo(-0.2, 12);
    expect(supportMargin([0.5, 0], [[0, 0], [1, 0]])!.marginM).toBeCloseTo(0, 12);
    expect(supportMargin([3, 4], [[0, 0]])!.marginM).toBeCloseTo(-5, 12);
    expect(supportMargin([0, 0], [])).toBeNull();
  });
});
