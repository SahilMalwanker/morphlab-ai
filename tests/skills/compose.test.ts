import { describe, expect, it } from 'vitest';
import { buildKinematicModel, forwardKinematics } from '../../src/core/kinematics/fk';
import { fromXyzRpy, identity, type Mat4 } from '../../src/core/math/mat4';
import type { Vec3 } from '../../src/core/math/vec3';
import { homeConfiguration, PRESETS } from '../../src/core/model/catalog';
import { arm3, quadruped, rover } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { composeKeyframes, describeBody, type ComposedFrame } from '../../src/skills/compose';

const tipsAt = (robot: RobotDescription, q: number[], T: Mat4 = identity()): Vec3[] =>
  forwardKinematics(buildKinematicModel(robot), q, T).tips.map((t) => [t[12]!, t[13]!, t[14]!] as Vec3);

function compose(robot: RobotDescription, frames: ComposedFrame[]) {
  const r = composeKeyframes(robot, frames);
  if (!r.ok) throw new Error(r.problems.join(' '));
  return r;
}

const close = (a: Vec3, b: Vec3, tol: number) => a.forEach((v, k) => expect(Math.abs(v - b[k]!)).toBeLessThan(tol));

describe('composed poses', () => {
  const quad = quadruped();
  const home = homeConfiguration(quad);
  const homeTips = tipsAt(quad, home);

  it('moves the body with every foot kept where it stands', () => {
    for (const body of [{ z: -0.08 }, { x: 0.03, pitch: 0.2 }, { y: -0.04, roll: 0.15, yaw: 0.1 }]) {
      const r = compose(quad, [{ seconds: 1, body }]);
      const T_B0_B1 = fromXyzRpy([body.x ?? 0, body.y ?? 0, body.z ?? 0], [body.roll ?? 0, body.pitch ?? 0, body.yaw ?? 0]);
      // Feet seen from the home body frame, with the body moved: unchanged within the IK tolerance.
      tipsAt(quad, r.keyframes[0]!.q, T_B0_B1).forEach((p, i) => close(p, homeTips[i]!, 2e-4));
      expect(r.notes).toEqual([]);
    }
  });

  it('moves one tip from where it rests and leaves the other limbs at home', () => {
    const r = compose(quad, [{ seconds: 0.5, tips: [{ limb: 'front_left', z: 0.08, x: 0.02 }] }]);
    const q = r.keyframes[0]!.q;
    close(tipsAt(quad, q)[0]!, [homeTips[0]![0] + 0.02, homeTips[0]![1], homeTips[0]![2] + 0.08], 2e-4);
    expect(q.slice(3)).toEqual(home.slice(3));
    expect(r.keyframes[0]!.durationS).toBe(0.5);
  });

  it('combines a body move with a lifted foot, both in the home body frame', () => {
    const r = compose(quad, [{ seconds: 1, body: { y: -0.04 }, tips: [{ limb: 'front_left', z: 0.06 }] }]);
    const T_B0_B1 = fromXyzRpy([0, -0.04, 0], [0, 0, 0]);
    const tips = tipsAt(quad, r.keyframes[0]!.q, T_B0_B1);
    close(tips[0]!, [homeTips[0]![0], homeTips[0]![1], homeTips[0]![2] + 0.06], 2e-4);
    close(tips[3]!, homeTips[3]!, 2e-4);
  });

  it('sets joint targets last, inside their limits', () => {
    const r = compose(quad, [{ seconds: 1, tips: [{ limb: 'front_left', z: 0.05 }], joints: [{ joint: 'front_left_knee', value: -2 }] }]);
    expect(r.keyframes[0]!.q[2]).toBe(-2);
  });

  it('warns about an empty middle frame without changing its return-home meaning', () => {
    const result = compose(quad, [{ seconds: 0.7, body: { pitch: 0.3 } }, { seconds: 1.5 }, { seconds: 1, body: { pitch: 0 } }]);
    expect(result.keyframes[1]!.q).toEqual(home);
    expect(result.keyframes[2]!.q).toEqual(home);
    expect(result.notes).toContain('keyframes[1]: a pose with only seconds returns to home, not a hold; repeat the previous body, tips and joints to hold that pose');
    const finalReturn = compose(quad, [{ seconds: 0.7, body: { pitch: 0.3 } }, { seconds: 1 }]);
    expect(finalReturn.notes).toEqual([]);
    expect(finalReturn.keyframes[1]!.q).toEqual(home);
  });

  it('warns on every lifted-foot frame that drops an earlier body shift', () => {
    const shift = { y: -0.04 };
    const tips = [{ limb: 'front_left', z: 0.06 }];
    const result = compose(quad, [{ seconds: 0.5, body: shift }, { seconds: 0.5, tips }, { seconds: 0.5, tips }, { seconds: 0.5 }]);
    expect(result.notes.filter((note) => note.includes('body offset'))).toEqual([
      'keyframes[1]: the body offset from keyframes[0] is omitted while a foot is lifted; each pose starts from home, so repeat body in every lifted-foot frame that needs that weight shift',
      'keyframes[2]: the body offset from keyframes[0] is omitted while a foot is lifted; each pose starts from home, so repeat body in every lifted-foot frame that needs that weight shift',
    ]);
    expect(result.keyframes[1]!.q.slice(3)).toEqual(home.slice(3));
    expect(compose(quad, [{ seconds: 0.5, body: shift }, { seconds: 0.5, body: shift, tips }]).notes).toEqual([]);
    expect(compose(quad, [{ seconds: 0.5, body: shift }, { seconds: 0.5, body: { y: 0 }, tips }, { seconds: 0.5, tips }]).notes).toEqual([]);
    expect(compose(quad, [{ seconds: 0.5, body: shift }, { seconds: 0.5 }, { seconds: 0.5, tips }]).notes.some((note) => note.includes('body offset'))).toBe(false);
  });

  it('reports a tip it cannot reach with the distance, instead of hiding it', () => {
    const r = compose(quad, [{ seconds: 1, tips: [{ limb: 'front_left', z: -0.3 }] }]);
    expect(r.notes).toHaveLength(1);
    expect(r.notes[0]).toMatch(/^keyframes\[0\]: the front left leg tip stops \d+ mm from its target \(out of its reach\)$/);
    const km = buildKinematicModel(quad);
    r.keyframes[0]!.q.forEach((v, j) => {
      expect(v).toBeGreaterThanOrEqual(km.joints[j]!.lower);
      expect(v).toBeLessThanOrEqual(km.joints[j]!.upper);
    });
  });

  it('warns when body-angle targets all oppose the goal without changing the composed motion', () => {
    for (const axis of ['pitch', 'roll'] as const) {
      for (const sign of [-1, 1]) {
        const frames = [{ seconds: 1, body: { [axis]: -sign * 0.2 } }, { seconds: 1, body: { [axis]: 0 } }];
        const goal = { kind: 'bodyAngle' as const, axis, minRad: sign > 0 ? 0.1 : -0.3, maxRad: sign > 0 ? 0.3 : -0.1, holdS: 1 };
        const result = composeKeyframes(quad, frames, [goal]);
        if (!result.ok) throw new Error(result.problems.join(' '));
        expect(result.notes.some((note) => note.includes(`body.${axis} sign conflict`))).toBe(true);
        expect(result.keyframes).toEqual(compose(quad, frames).keyframes);
        const consistent = composeKeyframes(quad, [{ seconds: 1, body: { [axis]: sign * 0.2 } }], [goal]);
        if (!consistent.ok) throw new Error(consistent.problems.join(' '));
        expect(consistent.notes).toEqual([]);
        const crossing = composeKeyframes(quad, [...frames, { seconds: 1, body: { [axis]: sign * 0.2 } }], [goal]);
        if (!crossing.ok) throw new Error(crossing.problems.join(' '));
        expect(crossing.notes.some((note) => note.includes('sign conflict'))).toBe(false);
      }
    }
  });

  it('places an arm tip by IK on a fixed base', () => {
    const arm = arm3();
    const armHome = tipsAt(arm, homeConfiguration(arm))[0]!;
    const r = compose(arm, [{ seconds: 1, tips: [{ limb: 'arm', y: 0.1, z: -0.05 }] }, { seconds: 1, tips: [{ limb: 'arm', y: -0.1, z: -0.05 }] }]);
    close(tipsAt(arm, r.keyframes[0]!.q)[0]!, [armHome[0], armHome[1] + 0.1, armHome[2] - 0.05], 2e-4);
    close(tipsAt(arm, r.keyframes[1]!.q)[0]!, [armHome[0], armHome[1] - 0.1, armHome[2] - 0.05], 2e-4);
  });

  it('refuses what the body does not have or cannot do, naming each problem', () => {
    const bad = (robot: RobotDescription, frames: ComposedFrame[]) => {
      const r = composeKeyframes(robot, frames);
      expect(r.ok).toBe(false);
      return r.ok ? [] : r.problems;
    };
    expect(bad(quad, [{ seconds: 1, tips: [{ limb: 'tail', z: 0.1 }] }])).toEqual(['keyframes[0].tips[0].limb: Quadruped has no limb "tail"; its limbs are front_left, front_right, rear_left, rear_right.']);
    expect(bad(quad, [{ seconds: 1, joints: [{ joint: 'front_left_knee', value: -3 }] }])).toEqual(['keyframes[0].joints[0].value: "front_left_knee" takes -2.70 to -0.15 rad, not -3.']);
    expect(bad(quad, [{ seconds: 1, joints: [{ joint: 'neck', value: 0 }] }])).toEqual(['keyframes[0].joints[0].joint: Quadruped has no joint "neck".']);
    expect(bad(quad, [{ seconds: 0.05, body: { z: 0.5 } }])).toEqual(['keyframes[0].seconds must be from 0.1 to 10 s.', 'keyframes[0].body.z must be a number from -0.2 to 0.2 m.']);
    expect(bad(quad, [{ seconds: 1, tips: [{ limb: 'front_left', z: 0.1 }, { limb: 'front_left', z: 0.2 }] }])).toEqual(['keyframes[0].tips[1].limb: "front_left" is listed twice in one pose.']);
    expect(bad(arm3(), [{ seconds: 1, body: { z: -0.05 } }])).toEqual(['keyframes[0].body: Three-joint arm has a fixed base, so its body cannot move; move limb tips instead.']);
    expect(bad(rover(), [{ seconds: 1, tips: [{ limb: 'wheel_fl', z: 0.1 }] }])).toEqual(['keyframes[0].tips[0].limb: "wheel_fl" is a wheel; it has no tip to place.']);
    expect(bad(quad, [])).toEqual(['keyframes must hold 1 to 8 poses.']);
    expect(bad(quad, Array.from({ length: 9 }, () => ({ seconds: 1 })))).toEqual(['keyframes must hold 1 to 8 poses.']);
    expect(bad(quad, [{ seconds: 1, body: { z: Number.NaN } }])).toEqual(['keyframes[0].body.z must be a number from -0.2 to 0.2 m.']);
  });

  it('describes the body in the numbers poses are composed from', () => {
    const d = describeBody(quad);
    expect(d).toContain('Body: a floating base of 0.40 x 0.20 x 0.08 m and 4 kg; in the home stance it rests 0.269 m above the ground.');
    expect(d).toContain('front_left (Front left leg, leg with a foot): tip at home (0.170, 0.130, -0.245) m, 0.002 m above the ground; effort limit 12 N*m; joints front_left_hip_roll (axis +x, -0.70 to 0.70, home 0.00); front_left_hip_pitch (axis +y, -1.80 to 1.80, home 0.70); front_left_knee (axis +y, -2.70 to -0.15, home -1.40).');
    expect(describeBody(arm3())).toMatch(/^Body: a fixed base of 0\.16 x 0\.16 x 0\.06 m \(it never moves\)\. .*arm \(Arm, arm\): tip at home/);
    for (const p of PRESETS) expect(describeBody(p.make()).length, p.id).toBeLessThan(2400);
  });
});
