// Legged navigation sweep (D37): the quadruped and the hexapod walking the five test maps in both controller modes. A measurement, not a pass/fail claim.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { hexapod, quadruped } from '../../src/core/model/presets';
import { TEST_MAPS } from '../../src/nav/maps';
import { runToEnd } from '../../src/physics/experiments';
import { runNavigation } from '../../src/physics/navigation';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

for (const make of [quadruped, hexapod]) {
  for (const mode of ['sensor-only', 'ground-truth'] as const) {
    it(`${make().name}, ${mode}: the five test maps (measured)`, () => {
      const t0 = performance.now();
      const rows = TEST_MAPS.map((map) => {
        const r = runToEnd(runNavigation(mj, map, { mode, robot: make() }));
        return { map: r.mapId, outcome: r.outcome, timeS: r.timeS, finalDistanceM: r.finalDistanceM, estimateErrorM: r.estimateErrorM, replans: r.replans };
      });
      console.table(rows);
      console.log('Legged navigation', make().name, mode, JSON.stringify({ rows, reached: rows.filter((r) => r.outcome === 'reached').length, wallS: +((performance.now() - t0) / 1000).toFixed(1) }));
      expect(rows).toHaveLength(TEST_MAPS.length);
    });
  }
}
