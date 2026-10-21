import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel, forwardKinematics } from '../../src/core/kinematics/fk';
import { identity } from '../../src/core/math/mat4';
import { PRESETS } from '../../src/core/model/catalog';
import { hexapod, quadruped, quadrupedStance } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { evaluateSkill } from '../../src/skills/evaluation';
import { DEFAULT_GAIT, patternPhases, type GaitParams } from '../../src/skills/gait';
import { mirrorGait, mirrorGoal, mirrorMap, mirrorName, mirrorQ, mirrorTaught, type MirrorMap } from '../../src/skills/mirror';
import type { SkillGoal } from '../../src/skills/taught';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const arm3 = PRESETS.find((p) => p.id === 'arm3')!.make();
const arm6 = PRESETS.find((p) => p.id === 'arm6')!.make();

function mapOf(robot: RobotDescription): MirrorMap {
  const r = mirrorMap(robot);
  if (!r.ok) throw new Error(r.reason);
  return r.map;
}

/** Limb tip positions in the base frame. */
function tips(robot: RobotDescription, q: number[]): number[][] {
  return forwardKinematics(buildKinematicModel(robot), q, identity()).tips.map((T) => [T[12]!, T[13]!, T[14]!]);
}

describe('mirror images of bodies', () => {
  it('pairs the legs of the legged presets with their partners on the other side', () => {
    // Quadruped limb order: front left, front right, rear left, rear right.
    const quad = mapOf(quadruped());
    expect(quad.limbs).toEqual([1, 0, 3, 2]);
    // Both sides' hip roll axes point forward, so a mirrored roll takes the opposite angle; pitch axes are their own mirror image.
    expect(quad.joints.slice(0, 6)).toEqual([{ index: 3, sign: -1 }, { index: 4, sign: 1 }, { index: 5, sign: 1 }, { index: 0, sign: -1 }, { index: 1, sign: 1 }, { index: 2, sign: 1 }]);
    // Hexapod limb order: front, middle and rear on the left, then on the right; coxa yaw flips, femur and tibia pitch keep their angles.
    const hex = mapOf(hexapod());
    expect(hex.limbs).toEqual([3, 4, 5, 0, 1, 2]);
    expect(hex.joints.slice(0, 3)).toEqual([{ index: 9, sign: -1 }, { index: 10, sign: 1 }, { index: 11, sign: 1 }]);
  });

  it('maps an arm on the plane of symmetry onto itself, flipping joints whose axis lies in that plane', () => {
    const m = mapOf(arm3);
    expect(m.limbs).toEqual([0]);
    // The base yaw turns about z (in the plane): mirrored, it turns the other way; shoulder and elbow pitch about y and keep their angles.
    expect(m.joints).toEqual([{ index: 0, sign: -1 }, { index: 1, sign: 1 }, { index: 2, sign: 1 }]);
  });

  it('puts every limb tip of a mirrored pose at the mirror image of its partner tip, in any pose', () => {
    for (const robot of [quadruped(), hexapod(), arm3, arm6]) {
      const m = mapOf(robot);
      const n = robot.limbs.reduce((k, l) => k + l.joints.length, 0);
      for (let trial = 0; trial < 5; trial++) {
        const q = Array.from({ length: n }, (_, j) => 0.6 * Math.sin(2.3 * j + 1.1 * trial + 0.2));
        const a = tips(robot, q);
        const b = tips(robot, mirrorQ(m, q));
        a.forEach((t, l) => {
          const p = b[m.limbs[l]!]!;
          expect(p[0]).toBeCloseTo(t[0]!, 9);
          expect(p[1]).toBeCloseTo(-t[1]!, 9);
          expect(p[2]).toBeCloseTo(t[2]!, 9);
        });
        // Mirroring twice gives the pose back.
        mirrorQ(m, mirrorQ(m, q)).forEach((v, j) => expect(v).toBeCloseTo(q[j]!, 12));
      }
    }
  });

  it('refuses a body that is not mirror symmetric, and says why', () => {
    const dog = quadruped();
    const moved = structuredClone(dog);
    moved.limbs[1]!.joints[0]!.origin.xyz = [0.19, -0.1, 0];
    const r = mirrorMap(moved);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe('Quadruped is not mirror symmetric: the Front left leg has no partner on the other side.');
    const bent = structuredClone(dog);
    bent.limbs[1]!.joints[1]!.axis = [0, 0, 1];
    const b = mirrorMap(bent);
    expect(b.ok).toBe(false);
    if (!b.ok) expect(b.reason).toContain('has no mirror image in the');
  });
});

describe('mirror images of skills', () => {
  const dog = quadruped();
  const m = mapOf(dog);

  it('swaps left and right in names, and returns null for a name without a side', () => {
    expect(mirrorName('turn left 90 degrees')).toBe('turn right 90 degrees');
    expect(mirrorName('wave front right leg')).toBe('wave front left leg');
    expect(mirrorName('wave_front_right_leg')).toBe('wave_front_left_leg');
    expect(mirrorName('Lean Left')).toBe('Lean Right');
    expect(mirrorName('spin clockwise')).toBe('spin counterclockwise');
    expect(mirrorName('spin counter-clockwise')).toBe('spin clockwise');
    expect(mirrorName('outright leftover')).toBeNull();
    expect(mirrorName('hop')).toBeNull();
  });

  it('turns goals to the other side, and leaves forward, height and tilt goals as they are', () => {
    const goals: SkillGoal[] = [
      { kind: 'travel', direction: 'left', minDistanceM: 0.1, maxDistanceM: 0.3 },
      { kind: 'travel', direction: 'forward', minDistanceM: 0.2 },
      { kind: 'turn', left: true, minAngleRad: 1.3, maxAngleRad: 1.8 },
      { kind: 'bodyAngle', axis: 'roll', minRad: -0.5, maxRad: -0.1, holdS: 1 },
      { kind: 'bodyAngle', axis: 'pitch', minRad: -0.6, maxRad: -0.15, holdS: 1 },
      { kind: 'tipSwing', limbId: 'front_right', axis: 'y', minAmplitudeM: 0.04, minSwings: 4 },
      { kind: 'footDown', limbId: 'rear_left', minFraction: 0.9 },
      { kind: 'tipReach', limbId: 'front_left', x: 0.1, y: 0.05, z: 0.08, toleranceM: 0.03, holdS: 0.5 },
      { kind: 'upright', maxTiltRad: 0.35 },
      { kind: 'airborne', limbIds: ['front_left', 'rear_right'], minS: 1 },
      { kind: 'airborne', minS: 0.05 },
    ];
    expect(goals.map((g) => mirrorGoal(g, dog, m))).toEqual([
      { kind: 'travel', direction: 'right', minDistanceM: 0.1, maxDistanceM: 0.3 },
      { kind: 'travel', direction: 'forward', minDistanceM: 0.2 },
      { kind: 'turn', left: false, minAngleRad: 1.3, maxAngleRad: 1.8 },
      { kind: 'bodyAngle', axis: 'roll', minRad: 0.1, maxRad: 0.5, holdS: 1 },
      { kind: 'bodyAngle', axis: 'pitch', minRad: -0.6, maxRad: -0.15, holdS: 1 },
      { kind: 'tipSwing', limbId: 'front_left', axis: 'y', minAmplitudeM: 0.04, minSwings: 4 },
      { kind: 'footDown', limbId: 'rear_right', minFraction: 0.9 },
      { kind: 'tipReach', limbId: 'front_right', x: 0.1, y: -0.05, z: 0.08, toleranceM: 0.03, holdS: 0.5 },
      { kind: 'upright', maxTiltRad: 0.35 },
      { kind: 'airborne', limbIds: ['front_right', 'rear_left'], minS: 1 },
      { kind: 'airborne', minS: 0.05 },
    ]);
  });

  it('steps and turns a gait the other way, each leg taking its partner phase, and mirrors back to the original', () => {
    const side: GaitParams = { ...DEFAULT_GAIT, strideM: 0.03, lateralM: 0.04, turnRad: -0.05, heldUp: ['front_right'] };
    const g = mirrorGait(side, dog, m);
    expect(g).toMatchObject({ strideM: 0.03, lateralM: -0.04, turnRad: 0.05, heldUp: ['front_left'] });
    // A trot moves diagonal pairs together; its mirror image starts with the other pair.
    expect(patternPhases(dog, 'trot')).toEqual([0, 0.5, 0.5, 0]);
    expect(g.phases).toEqual([0.5, 0, 0, 0.5]);
    expect(mirrorGait(g, dog, m)).toEqual(side);
    // Patterns that are their own mirror image need no phases.
    expect(mirrorGait({ ...DEFAULT_GAIT, pattern: 'bound' }, dog, m).phases).toBeUndefined();
  });

  it('mirrors keyframes joint by joint and refuses targets the partner joint cannot take', () => {
    const stance = quadrupedStance();
    const jointIds = dog.limbs.flatMap((l) => l.joints.map((j) => j.id));
    const lifted = stance.slice();
    lifted[1] = 1.2; // front left hip pitch
    const r = mirrorTaught(dog, { goals: [{ kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.05, holdS: 1 }], motion: { jointIds, keyframes: [{ t: 1, q: lifted }], endHoldS: 1 }, gait: null });
    if (!r.ok) throw new Error(r.reason);
    expect(r.goals).toEqual([{ kind: 'tipHeight', limbId: 'front_right', minHeightM: 0.05, holdS: 1 }]);
    expect(r.motion!.keyframes[0]!.q[4]).toBe(1.2);
    expect(r.motion!.keyframes[0]!.q[1]).toBe(stance[1]);
    const arm = mirrorTaught(arm3, { goals: [], motion: { jointIds: ['base_yaw', 'shoulder', 'elbow'], keyframes: [{ t: 1, q: [0.4, 0.3, 0.2] }], endHoldS: 1 }, gait: null });
    if (!arm.ok) throw new Error(arm.reason);
    expect(arm.motion!.keyframes[0]!.q).toEqual([-0.4, 0.3, 0.2]);
    const asym = structuredClone(arm3);
    asym.limbs[0]!.joints[0]!.limit = { ...asym.limbs[0]!.joints[0]!.limit, lower: -0.1 };
    const refused = mirrorTaught(asym, { goals: [], motion: { jointIds: ['base_yaw', 'shoulder', 'elbow'], keyframes: [{ t: 1, q: [0.4, 0.3, 0.2] }], endHoldS: 1 }, gait: null });
    expect(refused).toEqual({ ok: false, reason: 'keyframe 1: the mirrored target of joint base_yaw (-0.400) is outside its limits.' });
  });
});

describe('mirror images in the simulator (MuJoCo, fixed conditions)', () => {
  it('a side step and its mirror image move and turn by the same amounts to opposite sides', () => {
    const dog = quadruped();
    const stance = quadrupedStance();
    const m = mapOf(dog);
    const mass = dog.base.mass;
    // One trial with friction, mass and joint offsets fixed, so the only difference is the side.
    const opts = { trials: 1, frictionRange: [1, 1] as [number, number], baseMassRange: [mass, mass] as [number, number], jointNoiseRad: 0, seed: 5 };
    const side: GaitParams = { ...DEFAULT_GAIT, strideM: 0, lateralM: 0.04, durationS: 3 };
    const goals: SkillGoal[] = [{ kind: 'travel', direction: 'left', minDistanceM: 0.01 }, { kind: 'turn', left: true, minAngleRad: 0.05 }];
    const run = (gait: GaitParams, g: SkillGoal[]) => evaluateSkill(mj, dog, -1, stance, { kind: 'gait', gait }, opts, { goals: g }).results[0]!.goals!.map((x) => x.measured);
    const [left, turn] = run(side, goals);
    const [right, mirroredTurn] = run(mirrorGait(side, dog, m), goals.map((g) => mirrorGoal(g, dog, m)));
    const seen = `left ${left} m, turn ${turn} rad; mirrored: right ${right} m, turn ${mirroredTurn} rad`;
    expect(left!, seen).toBeGreaterThan(0.1);
    expect(Math.abs(right! - left!), seen).toBeLessThan(0.01 + 0.05 * Math.abs(left!));
    expect(Math.abs(mirroredTurn! - turn!), seen).toBeLessThan(0.02 + 0.05 * Math.abs(turn!));
  });
});
