import { describe, expect, it } from 'vitest';
import fixtures from '../fixtures/mr_oracle.json';
import { buildKinematicModel, forwardKinematics, randomConfiguration, tipPosition } from '../../src/core/kinematics/fk';
import { manipulability, positionJacobianAt } from '../../src/core/kinematics/jacobian';
import { mulberry32 } from '../../src/core/math/rng';
import type { Vec3 } from '../../src/core/math/vec3';
import { arm3, hexapod, quadruped, serialChain } from '../../src/core/model/presets';
import type { JointType, RobotDescription } from '../../src/core/model/types';

interface OracleJoint { type: string; xyz: number[]; rpy: number[]; axis: number[]; lower: number; upper: number }
interface OracleChain { base: { xyz: number[]; rpy: number[] }; joints: OracleJoint[]; tip: number[] }

/** Maps an oracle chain spec onto a RobotDescription (pure data mapping, no kinematics). */
function describeChain(id: string, c: OracleChain): RobotDescription {
  return {
    schemaVersion: 1,
    id,
    name: id,
    base: { kind: 'fixed', size: [0.1, 0.1, 0.05], mass: 1, pose: { xyz: c.base.xyz as Vec3, rpy: c.base.rpy as Vec3 } },
    limbs: [
      {
        id: 'arm',
        name: 'Arm',
        role: 'arm',
        joints: c.joints.map((j, i) => ({
          id: `j${i + 1}`,
          name: `j${i + 1}`,
          type: j.type as JointType,
          origin: { xyz: j.xyz as Vec3, rpy: j.rpy as Vec3 },
          axis: j.axis as Vec3,
          limit: { lower: j.lower, upper: j.upper, effort: 10, velocity: 5 },
          link: { radius: 0.02, mass: 0.2 },
          damping: 0,
          armature: 0,
        })),
        tip: { xyz: c.tip as Vec3, radius: 0.01 },
      },
    ],
  };
}

const chains = fixtures.chains as Record<string, OracleChain>;

describe('forward kinematics vs ModernRobotics oracle (FKinSpace)', () => {
  for (const [name, chain] of Object.entries(chains)) {
    it(`matches every fixture configuration of ${name} to 1e-9`, () => {
      const model = buildKinematicModel(describeChain(name, chain));
      const cases = fixtures.cases.filter((c) => c.chain === name);
      expect(cases.length).toBeGreaterThanOrEqual(13);
      for (const c of cases) {
        const T = forwardKinematics(model, c.q).tips[0]!;
        for (let r = 0; r < 4; r++) {
          for (let col = 0; col < 4; col++) expect(Math.abs(T[col * 4 + r]! - c.T[r]![col]!)).toBeLessThan(1e-9);
        }
      }
    });
  }
});

describe('position Jacobian', () => {
  for (const [name, chain] of Object.entries(chains)) {
    it(`matches the oracle space Jacobian of ${name} to 1e-9`, () => {
      const model = buildKinematicModel(describeChain(name, chain));
      for (const c of fixtures.cases.filter((x) => x.chain === name)) {
        const J = positionJacobianAt(model, 0, c.q);
        const n = c.q.length;
        for (let r = 0; r < 3; r++) {
          for (let k = 0; k < n; k++) expect(Math.abs(J[r * n + k]! - c.positionJacobian[r]![k]!)).toBeLessThan(1e-9);
        }
      }
    });
  }

  it('matches central finite differences on multi-limb robots', () => {
    const rng = mulberry32(11);
    for (const desc of [quadruped(), hexapod(), serialChain({ axes: ['z', 'y', 'x', 'y', 'z'], lengths: [0.2, 0.25, 0.1, 0.15, 0.05] })]) {
      const model = buildKinematicModel(desc);
      for (let t = 0; t < 5; t++) {
        const q = randomConfiguration(model, rng);
        model.limbs.forEach((limb, li) => {
          const J = positionJacobianAt(model, li, q);
          const h = 1e-6;
          for (let k = 0; k < limb.count; k++) {
            const qp = q.slice(); qp[limb.start + k]! += h;
            const qm = q.slice(); qm[limb.start + k]! -= h;
            const pp = tipPosition(model, li, qp), pm = tipPosition(model, li, qm);
            for (let r = 0; r < 3; r++) expect(Math.abs(J[r * limb.count + k]! - (pp[r]! - pm[r]!) / (2 * h))).toBeLessThan(1e-6);
          }
        });
      }
    }
  });
});

describe('closed-form references', () => {
  it('planar 2R matches x = l1 c1 + l2 c12, y = l1 s1 + l2 s12 to 1e-12', () => {
    const l1 = 0.4, l2 = 0.3;
    const desc = describeChain('planar', {
      base: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
      joints: [
        { type: 'revolute', xyz: [0, 0, 0], rpy: [0, 0, 0], axis: [0, 0, 1], lower: -3, upper: 3 },
        { type: 'revolute', xyz: [l1, 0, 0], rpy: [0, 0, 0], axis: [0, 0, 1], lower: -3, upper: 3 },
      ],
      tip: [l2, 0, 0],
    });
    const model = buildKinematicModel(desc);
    const rng = mulberry32(5);
    for (let k = 0; k < 50; k++) {
      const q1 = 6 * rng() - 3, q2 = 6 * rng() - 3;
      const p = tipPosition(model, 0, [q1, q2]);
      expect(Math.abs(p[0] - (l1 * Math.cos(q1) + l2 * Math.cos(q1 + q2)))).toBeLessThan(1e-12);
      expect(Math.abs(p[1] - (l1 * Math.sin(q1) + l2 * Math.sin(q1 + q2)))).toBeLessThan(1e-12);
      expect(Math.abs(p[2])).toBeLessThan(1e-12);
    }
  });

  it('three-joint arm matches its closed form to 1e-12', () => {
    const model = buildKinematicModel(arm3());
    const rng = mulberry32(9);
    for (let k = 0; k < 50; k++) {
      const q = randomConfiguration(model, rng);
      const [q1, q2, q3] = q as [number, number, number];
      const r = 0.35 * Math.sin(q2) + 0.3 * Math.sin(q2 + q3);
      const expected = [r * Math.cos(q1), r * Math.sin(q1), 0.18 + 0.35 * Math.cos(q2) + 0.3 * Math.cos(q2 + q3)];
      const p = tipPosition(model, 0, q);
      p.forEach((v, i) => expect(Math.abs(v - expected[i]!)).toBeLessThan(1e-12));
    }
  });
});

describe('input validation', () => {
  it('rejects non-finite joint values and wrong vector sizes', () => {
    const model = buildKinematicModel(arm3());
    expect(() => forwardKinematics(model, [0, Number.NaN, 0])).toThrow(/not finite/);
    expect(() => forwardKinematics(model, [0, 0])).toThrow(/Expected 3/);
  });

  it('rejects an invalid robot description', () => {
    const bad = arm3();
    bad.limbs[0]!.joints[1]!.axis = [0, 0, 0];
    expect(() => buildKinematicModel(bad)).toThrow(/axis must not be zero/);
  });
});

describe('manipulability', () => {
  it('is near zero for the fully stretched arm and positive when bent', () => {
    const model = buildKinematicModel(arm3());
    const stretched = manipulability(positionJacobianAt(model, 0, [0, 0, 0]), 3);
    const bent = manipulability(positionJacobianAt(model, 0, [0, 0.6, 1.0]), 3);
    expect(stretched.inverseCondition).toBeLessThan(1e-9);
    expect(bent.measure).toBeGreaterThan(1e-3);
  });
});
