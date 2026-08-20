import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { TEST_MAPS, type NavMap } from '../../src/nav/maps';
import { RAY_ANGLES, RAY_ELEVATION, RAY_MAX_RANGE, RAY_ORIGIN, type NavInput } from '../../src/nav/stack';
import { createNavRuntime, groundTruthPose, runNavigation, type NavRunResult } from '../../src/physics/navigation';
import { runToEnd } from '../../src/physics/experiments';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const SENSOR_KEYS = ['accel', 'gyro', 'imuQuat', 'q', 'qd', 'ranges', 'touch'];

describe('range rays', () => {
  it('measure the distance to a known box face (closed-form ray-plane intersection)', () => {
    const faceX = 1.4;
    const map: NavMap = { id: 'probe', name: 'probe', start: { x: 0, y: 0, yaw: 0 }, goal: [3, 0], boxes: [{ cx: faceX + 0.1, cy: 0, hx: 0.1, hy: 0.8 }] };
    const rt = createNavRuntime(mj, map);
    try {
      let ranges: number[] = [];
      for (let k = 0; k < 400; k++) rt.controlStep((s) => ((ranges = s.ranges), [0, 0, 0, 0]));
      const p = groundTruthPose(rt);
      const ox = p.x + Math.cos(p.yaw) * RAY_ORIGIN[0];
      let hits = 0;
      RAY_ANGLES.forEach((a, i) => {
        const heading = p.yaw + a;
        const horizontal = (faceX - ox) / Math.cos(heading);
        const y = p.y + Math.sin(p.yaw) * RAY_ORIGIN[0] + horizontal * Math.sin(heading);
        const expected = horizontal / Math.cos(RAY_ELEVATION);
        if (Math.abs(y) < 0.75 && expected <= RAY_MAX_RANGE) {
          hits++;
          expect(ranges[i]!).toBeCloseTo(expected, 2);
        } else expect(ranges[i]! < 0 || ranges[i]! > RAY_MAX_RANGE).toBe(true);
      });
      expect(hits).toBeGreaterThanOrEqual(3);
    } finally {
      rt.dispose();
    }
  });
});

describe('sensor-only versus ground-truth inputs', () => {
  it('the sensor-only controller receives sensor readings only, never simulator state', () => {
    const seen = new Set<string>();
    let calls = 0;
    const spy = (input: NavInput) => {
      calls++;
      seen.add(Object.keys(input).sort().join(','));
      seen.add(`sensors:${Object.keys(input.sensors).sort().join(',')}`);
    };
    runToEnd(runNavigation(mj, TEST_MAPS[1]!, { mode: 'sensor-only', timeLimitS: 3, spy }));
    expect(calls).toBeGreaterThan(100);
    expect([...seen].sort()).toEqual(['sensors,t', `sensors:${SENSOR_KEYS.join(',')}`]);
  });

  it('ground-truth mode adds the labeled simulator pose and nothing else', () => {
    const seen = new Set<string>();
    runToEnd(runNavigation(mj, TEST_MAPS[1]!, { mode: 'ground-truth', timeLimitS: 1, spy: (i) => seen.add(Object.keys(i).sort().join(',')) }));
    expect([...seen]).toEqual(['groundTruth,sensors,t']);
  });
});

describe('rover navigation on the five test maps', () => {
  const results: Record<string, NavRunResult[]> = { 'sensor-only': [], 'ground-truth': [] };
  for (const mode of ['sensor-only', 'ground-truth'] as const) {
    it(`${mode}: reaches the goal without collision on at least 4 of 5 maps`, () => {
      for (const map of TEST_MAPS) results[mode]!.push(runToEnd(runNavigation(mj, map, { mode })));
      console.log(mode, JSON.stringify(results[mode]!.map((r) => ({ map: r.mapId, reached: r.reached, failure: r.failure, t: r.timeS, d: r.finalDistanceM, est: r.estimateErrorM, replans: r.replans }))));
      const ok = results[mode]!.filter((r) => r.reached && !r.collided);
      expect(ok.length).toBeGreaterThanOrEqual(4);
      expect(results[mode]!.every((r) => !r.collided)).toBe(true);
    });
  }
});
