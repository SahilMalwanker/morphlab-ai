import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel, forwardKinematics, randomConfiguration } from '../../src/core/kinematics/fk';
import { mulberry32 } from '../../src/core/math/rng';
import { arm3, hexapod, quadruped, quadrupedStance, radialLegged, rover } from '../../src/core/model/presets';
import { generateMjcf } from '../../src/physics/mjcf';
import { MujocoRuntime } from '../../src/physics/runtime';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

describe('MJCF generation', () => {
  it('emits radians, implicitfast, motors with effort limits and encoder sensors', () => {
    const xml = generateMjcf(quadruped());
    expect(xml).toContain('<compiler angle="radian" autolimits="true"/>');
    expect(xml).toContain('integrator="implicitfast"');
    expect(xml.match(/<motor /g)?.length).toBe(12);
    expect(xml).toContain('<motor name="act_front_left_knee" joint="front_left_knee" ctrlrange="-12 12"/>');
    expect(xml.match(/<jointpos /g)?.length).toBe(12);
    expect(xml.match(/<touch /g)?.length).toBe(4);
    expect(xml).toContain('<freejoint name="root"/>');
  });

  it('rejects invalid descriptions before generating XML', () => {
    const bad = arm3();
    bad.limbs[0]!.joints[0]!.limit.effort = 0;
    expect(() => generateMjcf(bad)).toThrow(/Effort limit must be positive/);
  });
});

describe('MuJoCo forward kinematics agrees with the core', () => {
  for (const make of [arm3, quadruped, hexapod, () => radialLegged(5, 4), rover]) {
    const desc = make();
    it(`link transforms match for ${desc.name}`, () => {
      const km = buildKinematicModel(desc);
      const rng = mulberry32(17);
      const rt = MujocoRuntime.create(mj, desc, { initialQ: randomConfiguration(km, rng) });
      try {
        for (let trial = 0; trial < 5; trial++) {
          const q = randomConfiguration(km, rng);
          rt.setState(q, { xyz: [0.1 * trial, -0.2, 0.5], rpy: [0.1, -0.2, 0.3 * trial] });
          const T_W_B = rt.groundTruthBaseTransform();
          const fk = forwardKinematics(km, q, T_W_B);
          const sim = rt.groundTruthLinkTransforms();
          sim.forEach((T, i) => {
            for (let k = 0; k < 16; k++) expect(Math.abs(T[k]! - fk.links[i]![k]!)).toBeLessThan(1e-9);
          });
          const tips = rt.groundTruthTipPositions();
          tips.forEach((p, li) => {
            if (Number.isNaN(p[0])) return;
            const T = fk.tips[li]!;
            for (let r = 0; r < 3; r++) expect(Math.abs(p[r]! - T[12 + r]!)).toBeLessThan(1e-9);
          });
          const s = rt.readSensors();
          q.forEach((v, i) => expect(s.q[i]).toBeCloseTo(v, 12));
        }
      } finally {
        rt.dispose();
      }
    });
  }
});

describe('ground truth: parts other than the feet touching the floor', () => {
  it('reports the body box and the joints within 5 mm of the floor, never the feet', () => {
    const stance = quadrupedStance();
    const rt = MujocoRuntime.create(mj, quadruped(), { initialQ: stance });
    try {
      rt.setState(stance, { xyz: [0, 0, 0.267], rpy: [0, 0, 0] });
      expect(rt.groundTruthTouchingParts()).toEqual([]);
      // Box bottom 2 mm above the floor (the box is 0.08 m tall).
      rt.setState(stance, { xyz: [0, 0, 0.042], rpy: [0, 0, 0] });
      expect(rt.groundTruthTouchingParts()).toContain('body');
      // On its side: half the box width (0.1 m) below the centre.
      rt.setState(stance, { xyz: [0, 0, 0.1], rpy: [Math.PI / 2, 0, 0] });
      expect(rt.groundTruthTouchingParts()).toContain('body');
      // Thighs straight down and shins folded up: each knee 0.16 m below its hip, 1 mm above the floor once the 0.022 m thigh radius is counted.
      const kneel = [0, 0, -2.7, 0, 0, -2.7, 0, 0, -2.7, 0, 0, -2.7];
      rt.setState(kneel, { xyz: [0, 0, 0.183], rpy: [0, 0, 0] });
      expect(rt.groundTruthTouchingParts()).toEqual(['front left leg knee', 'front right leg knee', 'rear left leg knee', 'rear right leg knee']);
    } finally {
      rt.dispose();
    }
  });
});

describe('ground truth: centre of mass', () => {
  it('lies on the mirror plane of a symmetric stance and moves rigidly with the base', () => {
    const stance = quadrupedStance();
    const rt = MujocoRuntime.create(mj, quadruped(), { initialQ: stance });
    try {
      rt.setState(stance, { xyz: [0, 0, 0.267], rpy: [0, 0, 0] });
      const a = rt.groundTruthCenterOfMass();
      expect(a[1]).toBeCloseTo(0, 9);
      expect(Math.abs(a[0])).toBeLessThan(0.2);
      // The 4 kg base dominates: the centre of mass is below the base centre (legs hang down) but above the knees.
      expect(a[2]).toBeLessThan(0.267);
      expect(a[2]).toBeGreaterThan(0.15);
      rt.setState(stance, { xyz: [0.3, -0.2, 0.267], rpy: [0, 0, 0] });
      const b = rt.groundTruthCenterOfMass();
      expect(b[0] - a[0]).toBeCloseTo(0.3, 9);
      expect(b[1] - a[1]).toBeCloseTo(-0.2, 9);
      expect(b[2] - a[2]).toBeCloseTo(0, 9);
    } finally {
      rt.dispose();
    }
  });
});

describe('runtime lifecycle', () => {
  it('dispose is idempotent and use-after-dispose is rejected', () => {
    const rt = MujocoRuntime.create(mj, quadruped(), { initialQ: quadrupedStance() });
    rt.dispose();
    rt.dispose();
    expect(() => rt.reset()).toThrow(/after dispose/);
  });

  it('reset restores the initial state exactly', () => {
    const rt = MujocoRuntime.create(mj, quadruped(), { initialQ: quadrupedStance() });
    try {
      const before = Array.from(rt.data.qpos as Float64Array);
      for (let i = 0; i < 200; i++) rt.step();
      rt.reset();
      expect(Array.from(rt.data.qpos as Float64Array)).toEqual(before);
      expect(rt.time).toBe(0);
    } finally {
      rt.dispose();
    }
  });

  it('rejects non-finite joint values when setting state', () => {
    const rt = MujocoRuntime.create(mj, arm3(), { initialQ: [0, 0.5, 1] });
    try {
      expect(() => rt.setState([0, Number.NaN, 1])).toThrow(/not finite/);
    } finally {
      rt.dispose();
    }
  });
});
