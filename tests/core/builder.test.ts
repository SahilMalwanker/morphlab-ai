import { describe, expect, it } from 'vitest';
import { buildKinematicModel, forwardKinematics, randomConfiguration } from '../../src/core/kinematics/fk';
import { identity, type Mat4 } from '../../src/core/math/mat4';
import { mulberry32 } from '../../src/core/math/rng';
import {
  DEFAULT_LEGGED,
  describeIssues,
  editLimb,
  generateArm,
  generateLegged,
  mirrorLimb,
  mirrorPartner,
  patchJoint,
  removeLimb,
  restingBaseHeight,
  setJointCount,
} from '../../src/core/model/builder';
import { homeConfiguration, PRESETS } from '../../src/core/model/catalog';
import { morphologyHash } from '../../src/core/model/hash';
import { hexapod, joint, quadruped, rover } from '../../src/core/model/presets';
import type { LimbDescription, RobotDescription } from '../../src/core/model/types';
import { validateRobot } from '../../src/core/model/validate';

function single(limbs: LimbDescription[]): RobotDescription {
  return { schemaVersion: 1, id: 't', name: 'Test', base: { kind: 'fixed', size: [0.1, 0.1, 0.1], mass: 1, pose: { xyz: [0, 0, 0], rpy: [0, 0, 0] } }, limbs };
}

/** M T M with M = diag(1, -1, 1): the reflection of a rigid transform across the XZ plane. */
function reflect(T: Mat4): number[] {
  const out = Array.from(T);
  // Column-major: element (r, c) at c * 4 + r. Flip sign where exactly one of r, c is 1.
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) if ((r === 1) !== (c === 1)) out[c * 4 + r] = -out[c * 4 + r]!;
  return out;
}

const skewLimb: LimbDescription = {
  id: 'skew',
  name: 'Skewed limb',
  role: 'arm',
  joints: [
    joint({ id: 's1', name: 'S1', xyz: [0.1, 0.2, 0.05], rpy: [0.3, -0.4, 0.7], axis: [0.2, 0.9, -0.3], lower: -2, upper: 2, effort: 5, radius: 0.02, mass: 0.2 }),
    joint({ id: 's2', name: 'S2', type: 'prismatic', xyz: [0.05, -0.1, 0.2], rpy: [-0.5, 0.2, 0.1], axis: [0.3, -0.5, 0.8], lower: -0.1, upper: 0.2, effort: 5, radius: 0.02, mass: 0.2 }),
    joint({ id: 's3', name: 'S3', xyz: [0, 0.07, 0.15], rpy: [0.1, 0.9, -1.2], axis: [1, 0, 0], lower: -2, upper: 2, effort: 5, radius: 0.02, mass: 0.2 }),
  ],
  tip: { xyz: [0.02, -0.03, 0.1], radius: 0.01 },
};

describe('mirror symmetry', () => {
  const sources: [string, LimbDescription][] = [
    ['quadruped front-left leg', quadruped().limbs[0]!],
    ['hexapod front-left leg (yawed mount)', hexapod().limbs[0]!],
    ['skewed limb with a prismatic joint', skewLimb],
  ];
  for (const [label, src] of sources) {
    it(`${label}: every frame of the mirrored limb is the reflection of the original at the same joint angles`, () => {
      const m = mirrorLimb(src, 'mirror', 'Mirror', src.joints.map((_, i) => `m${i}`));
      const desc = single([src, m]);
      const km = buildKinematicModel(desc);
      const rng = mulberry32(7);
      const n = src.joints.length;
      for (let trial = 0; trial < 50; trial++) {
        const qa = randomConfiguration(km, rng).slice(0, n);
        const fk = forwardKinematics(km, [...qa, ...qa], identity());
        for (let i = 0; i < n; i++) {
          const want = reflect(fk.links[i]!);
          const got = Array.from(fk.links[n + i]!);
          got.forEach((v, k) => expect(v).toBeCloseTo(want[k]!, 12));
        }
        reflect(fk.tips[0]!).forEach((v, k) => expect(fk.tips[1]![k]).toBeCloseTo(v, 12));
      }
    });
  }

  it('finds mirror partners from mount positions', () => {
    expect(mirrorPartner(quadruped(), 'front_left')).toBe('front_right');
    expect(mirrorPartner(quadruped(), 'rear_right')).toBe('rear_left');
    expect(mirrorPartner(hexapod(), 'middle_left')).toBe('middle_right');
    expect(mirrorPartner(rover(), 'wheel_fl')).toBe('wheel_fr');
    expect(mirrorPartner(PRESETS[0]!.make(), 'arm')).toBeNull();
  });

  it('mirrored edits regenerate the partner as the exact mirror image', () => {
    const q = quadruped();
    const edited = editLimb(q, 'front_left', (l) => ({ ...l, joints: l.joints.map((j, i) => (i === 2 ? patchJoint(j, { lower: -2.0, xyz: [0.01, 0, -0.18] }) : j)) }), true);
    const fl = edited.limbs.find((l) => l.id === 'front_left')!;
    const fr = edited.limbs.find((l) => l.id === 'front_right')!;
    expect(fr.joints.map((j) => j.id)).toEqual(['front_right_hip_roll', 'front_right_hip_pitch', 'front_right_knee']);
    expect(fr).toEqual(mirrorLimb(fl, 'front_right', 'Front right leg', fr.joints.map((j) => j.id)));
    expect(fr.joints[2]!.limit.lower).toBe(-2.0);
    const unmirrored = editLimb(q, 'front_left', (l) => ({ ...l, joints: l.joints.map((j, i) => (i === 2 ? patchJoint(j, { lower: -2.0 }) : j)) }), false);
    expect(unmirrored.limbs.find((l) => l.id === 'front_right')).toEqual(q.limbs[1]);
  });
});

describe('generators', () => {
  it('legged generator: every combination is valid, mirrored and resting on the ground', () => {
    for (let pairs = 1; pairs <= 4; pairs++) {
      for (let dof = 1; dof <= 4; dof++) {
        const d = generateLegged({ ...DEFAULT_LEGGED, pairs, dofPerLeg: dof });
        expect(validateRobot(d)).toEqual([]);
        expect(d.limbs).toHaveLength(2 * pairs);
        expect(d.limbs.every((l) => l.joints.length === dof)).toBe(true);
        for (let k = 0; k < pairs; k++) {
          const left = d.limbs[2 * k]!;
          const right = d.limbs[2 * k + 1]!;
          expect(mirrorPartner(d, left.id)).toBe(right.id);
          expect(right).toEqual(mirrorLimb(left, right.id, right.name, right.joints.map((j) => j.id)));
        }
        // Independent check: the lowest foot sphere touches z = 2 mm at the home configuration.
        const km = buildKinematicModel(d);
        const fk = forwardKinematics(km, homeConfiguration(d));
        const lowest = Math.min(...fk.tips.map((T, i) => T[14]! - d.limbs[i]!.tip.radius));
        expect(lowest).toBeCloseTo(0.002, 9);
      }
    }
  });

  it('legged generator rejects out-of-range specs with specific messages', () => {
    expect(() => generateLegged({ ...DEFAULT_LEGGED, pairs: 5 })).toThrow('Leg pairs must be a whole number from 1 to 4.');
    expect(() => generateLegged({ ...DEFAULT_LEGGED, thigh: -0.1 })).toThrow('Upper leg length (m) must be greater than 0 and at most 1.');
  });

  it('arm generator builds 1 to 7 joints with the requested reach', () => {
    for (let dof = 1; dof <= 7; dof++) {
      const d = generateArm(dof, 0.8);
      expect(validateRobot(d)).toEqual([]);
      const limb = d.limbs[0]!;
      const total = limb.joints.slice(1).reduce((s, j) => s + Math.hypot(...j.origin.xyz), 0) + Math.hypot(...limb.tip.xyz);
      expect(total).toBeCloseTo(0.8, 12);
    }
    expect(() => generateArm(8, 0.8)).toThrow('Arm joints must be a whole number from 1 to 7.');
  });

  it('resting height accounts for wheels and the base box', () => {
    const r = rover();
    const z = restingBaseHeight(r, homeConfiguration(r));
    // Wheel centre 0.02 m below the base frame, wheel radius 0.06 m.
    expect(z).toBeCloseTo(0.08 + 0.002, 12);
  });
});

describe('limb edits', () => {
  it('changes joint count and keeps the description valid', () => {
    const q = quadruped();
    const two = setJointCount(q, 'front_left', 2, true);
    expect(validateRobot(two)).toEqual([]);
    expect(two.limbs[0]!.joints).toHaveLength(2);
    expect(two.limbs[1]!.joints).toHaveLength(2);
    expect(two.limbs[0]!.tip.xyz).toEqual([0, 0, -0.16]);
    expect(Object.keys(two.stance!).some((k) => k.startsWith('front_left_knee'))).toBe(false);
    const five = setJointCount(two, 'front_left', 5, false);
    expect(validateRobot(five)).toEqual([]);
    expect(five.limbs[0]!.joints).toHaveLength(5);
    expect(new Set(five.limbs.flatMap((l) => l.joints.map((j) => j.id))).size).toBe(five.limbs.reduce((s, l) => s + l.joints.length, 0));
    expect(() => setJointCount(q, 'front_left', 0, false)).toThrow('A limb needs at least one joint');
    expect(() => setJointCount(rover(), 'wheel_fl', 2, false)).toThrow('A wheel has exactly one joint');
  });

  it('removes limbs (with the mirror partner) but never the last one', () => {
    const q = quadruped();
    const two = removeLimb(q, 'rear_left', true);
    expect(two.limbs.map((l) => l.id)).toEqual(['front_left', 'front_right']);
    expect(Object.keys(two.stance!).every((k) => k.startsWith('front'))).toBe(true);
    expect(() => removeLimb(removeLimb(two, 'front_left', true), 'x', false)).toThrow();
    expect(() => removeLimb(PRESETS[0]!.make(), 'arm', false)).toThrow('A robot needs at least one limb');
  });

  it('describes invalid edits by limb and joint name', () => {
    const q = quadruped();
    const bad = editLimb(q, 'front_left', (l) => ({ ...l, joints: l.joints.map((j, i) => (i === 2 ? patchJoint(j, { lower: 0.5, upper: 0.2 }) : j)) }), false);
    expect(describeIssues(bad)).toContain('Front left leg, Knee: Lower limit must be below upper limit.');
    const heavy = { ...q, base: { ...q.base, mass: 500 } };
    expect(describeIssues(heavy)).toEqual(['Base: Base mass must be at most 200 kg.']);
    const stance = { ...q, stance: { ...q.stance, front_left_knee: 0.5 } };
    expect(describeIssues(stance)).toEqual(['Front left leg, Knee: Stance position must lie within the joint limits.']);
  });
});

describe('morphology hash', () => {
  it('changes with every physical edit and ignores names, initial pose and stance', async () => {
    const base = quadruped();
    const h0 = await morphologyHash(base);
    const kneeEdit = (p: Parameters<typeof patchJoint>[1]) =>
      editLimb(base, 'front_left', (l) => ({ ...l, joints: l.joints.map((j, i) => (i === 2 ? patchJoint(j, p) : j)) }), false);
    const physical: [string, RobotDescription][] = [
      ['lower limit', kneeEdit({ lower: -2.6 })],
      ['upper limit', kneeEdit({ upper: -0.2 })],
      ['effort', kneeEdit({ effort: 11 })],
      ['velocity', kneeEdit({ velocity: 19 })],
      ['origin', kneeEdit({ xyz: [0, 0, -0.17] })],
      ['origin rpy', kneeEdit({ rpy: [0, 0.01, 0] })],
      ['axis', kneeEdit({ axis: [0, -1, 0] })],
      ['link radius', kneeEdit({ radius: 0.02 })],
      ['link mass', kneeEdit({ mass: 0.13 })],
      ['damping', kneeEdit({ damping: 0.06 })],
      ['armature', kneeEdit({ armature: 0.02 })],
      ['tip', editLimb(base, 'front_left', (l) => ({ ...l, tip: { ...l.tip, radius: 0.03 } }), false)],
      ['base size', { ...base, base: { ...base.base, size: [0.41, 0.2, 0.08] } }],
      ['base mass', { ...base, base: { ...base.base, mass: 4.1 } }],
      ['base kind', { ...base, base: { ...base.base, kind: 'fixed' } }],
      ['joint count', setJointCount(base, 'front_left', 2, false)],
      ['limb removed', removeLimb(base, 'rear_left', false)],
    ];
    const seen = new Set([h0]);
    for (const [label, d] of physical) {
      const h = await morphologyHash(d);
      expect(seen.has(h), label).toBe(false);
      seen.add(h);
    }
    const cosmetic: RobotDescription[] = [
      { ...base, name: 'Renamed' },
      { ...base, base: { ...base.base, pose: { xyz: [0, 0, 0.3], rpy: [0, 0, 0] } } },
      { ...base, stance: { ...base.stance, front_left_knee: -1.2 } },
      editLimb(base, 'front_left', (l) => ({ ...l, name: 'Renamed leg' }), false),
    ];
    for (const d of cosmetic) expect(await morphologyHash(d)).toBe(h0);
  });
});
