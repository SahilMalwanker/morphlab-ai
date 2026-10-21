import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel } from '../../src/core/kinematics/fk';
import { PRESETS } from '../../src/core/model/catalog';
import { morphologyHash } from '../../src/core/model/hash';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { checkSkillCompatibility, createSkillArtifact, parseSkillArtifact, skillLabel, specFromArtifact } from '../../src/skills/artifact';
import { DEFAULT_EVALUATION, evaluateSkill } from '../../src/skills/evaluation';
import {
  defaultGoal, GoalJudge, goalKindsFor, goalProblems, goalReward, goalRobotProblems, goalsProblems, goalText, homeTipClearance, KeyframeController, matchTaughtSkill, motionProblems,
  type JudgeSample, type KeyframeMotion, type SkillGoal,
} from '../../src/skills/taught';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const desc = quadruped();
const stance = quadrupedStance();
const ids = desc.limbs.flatMap((l) => l.joints.map((j) => j.id));
/** Every leg folded: hip pitch 1.1 rad, knee -2.2 rad, so each foot stays under its hip and the body drops about 10 cm. */
const crouchQ = desc.limbs.flatMap(() => [0, 1.1, -2.2]);

describe('keyframe motion (open loop)', () => {
  it('blends from the stance through each keyframe with a cosine profile, then holds the last pose', () => {
    const km = buildKinematicModel(desc);
    const far = ids.map((_, i) => (i % 3 === 2 ? -3 : 0.2));
    const c = new KeyframeController(km, stance, { jointIds: ids, keyframes: [{ t: 1, q: crouchQ }, { t: 2, q: far }], endHoldS: 0.5 });
    expect(c.durationS).toBe(2.5);
    expect(c.update(0)).toEqual(stance);
    expect(c.phase).toBe('moving to keyframe 1');
    // Halfway in time is halfway in the cosine blend.
    const mid = c.update(0.5);
    expect(mid[1]).toBeCloseTo(0.9, 12);
    expect(mid[2]).toBeCloseTo(-1.8, 12);
    // A quarter of the way the blend has covered (1 - cos(pi / 4)) / 2 of the change.
    expect(c.update(0.25)[1]).toBeCloseTo(0.7 + 0.4 * (0.5 - 0.5 * Math.cos(Math.PI / 4)), 12);
    expect(c.update(1)).toEqual(crouchQ);
    expect(c.phase).toBe('moving to keyframe 2');
    // Targets beyond a joint limit are clamped to it (knee lower limit -2.7 rad).
    const held = c.update(2.2);
    expect(held[2]).toBe(-2.7);
    expect(held[1]).toBe(0.2);
    expect(c.phase).toBe('holding the last pose');
    expect(c.finished).toBe(false);
    c.update(2.51);
    expect(c.finished).toBe(true);
  });

  it('refuses keyframes for other joints and an empty motion', () => {
    const km = buildKinematicModel(desc);
    expect(() => new KeyframeController(km, stance, { jointIds: ['a'], keyframes: [{ t: 1, q: [0] }], endHoldS: 0 })).toThrow("The keyframes are for other joints than this robot's");
    expect(() => new KeyframeController(km, stance, { jointIds: ids, keyframes: [], endHoldS: 1 })).toThrow('A keyframe motion needs at least one keyframe');
  });
});

const limbs = [
  { id: 'arm', name: 'Arm', foot: false },
  { id: 'f1', name: 'Foot one', foot: true },
  { id: 'f2', name: 'Foot two', foot: true },
];
const sample = (t: number, o: Partial<JudgeSample> = {}): JudgeSample => ({
  t, basePos: [0, 0, 0.25], tiltRad: 0.05, yawRad: 0, pitchRad: 0, rollRad: 0, tipClearanceM: [0, 0, 0], tipPos: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], tipContact: [false, true, true], ...o,
});

describe('goal judge (synthetic ground truth)', () => {
  it('judges a raised limb tip by its longest continuous stretch above the height', () => {
    const goal: SkillGoal = { kind: 'tipHeight', limbId: 'arm', minHeightM: 0.05, holdS: 1 };
    const longer: SkillGoal = { ...goal, holdS: 1.5 };
    const j = new GoalJudge([goal, longer], limbs);
    for (let i = 0; i <= 300; i++) {
      const up = i >= 50 && i <= 120 ? 0.08 : i >= 150 && i <= 280 ? 0.06 : 0;
      j.add(sample(i * 0.01, { tipClearanceM: [up, 0, 0] }));
    }
    const v = j.verdict({ unstable: false });
    const [met, missed] = v.goals;
    expect(met!.met).toBe(true);
    expect(met!.measured).toBeCloseTo(1.3, 6);
    expect(met!.score).toBe(1);
    expect(met!.text).toBe('arm tip at least 0.05 m up for 1.30 s of 1 s required (highest 0.080 m)');
    expect(missed!.met).toBe(false);
    // Half for reaching the height (capped at 1), half for the share of the hold.
    expect(missed!.score).toBeCloseTo(0.5 + 0.5 * (1.3 / 1.5), 6);
    expect(v.success).toBe(false);
  });

  it('reports the share of the goals that hold at this moment, for the dense reward of a feedback policy (D30)', () => {
    const goals: SkillGoal[] = [
      { kind: 'tipHeight', limbId: 'arm', minHeightM: 0.05, holdS: 1 },
      { kind: 'upright', maxTiltRad: 0.3 },
      { kind: 'bodyAngle', axis: 'pitch', minRad: -1, maxRad: -0.5, holdS: 1 },
      // Judged only at the end or over the whole run: never counted at a moment.
      { kind: 'standsAtEnd', maxTiltRad: 0.35, minBaseHeightM: 0.1 },
      { kind: 'travel', direction: 'forward', minDistanceM: 0.1 },
    ];
    const j = new GoalJudge(goals, limbs);
    expect(j.instant).toBe(0);
    j.add(sample(0, { tipClearanceM: [0.08, 0, 0], tiltRad: 0.1, pitchRad: -0.7 }));
    expect(j.instant).toBe(1);
    j.add(sample(0.01, { tipClearanceM: [0.08, 0, 0], tiltRad: 0.6, pitchRad: -0.7 }));
    expect(j.instant).toBeCloseTo(2 / 3, 12);
    j.add(sample(0.02, { tipClearanceM: [0.01, 0, 0], tiltRad: 0.6, pitchRad: 0 }));
    expect(j.instant).toBe(0);
    // No goal that can be judged at a moment: nothing holds.
    const none = new GoalJudge([{ kind: 'travel', direction: 'forward', minDistanceM: 0.1 }], limbs);
    none.add(sample(0));
    expect(none.instant).toBe(0);
    // A hold already met counts as holding afterwards, so holding a pose longer earns nothing more.
    const brief = new GoalJudge([{ kind: 'tipHeight', limbId: 'arm', minHeightM: 0.05, holdS: 0.02 }, { kind: 'upright', maxTiltRad: 0.3 }], limbs);
    for (const t of [0, 0.01, 0.02]) brief.add(sample(t, { tipClearanceM: [0.08, 0, 0] }));
    brief.add(sample(0.03, { tipClearanceM: [0, 0, 0] }));
    expect(brief.instant).toBe(1);
    brief.add(sample(0.04, { tiltRad: 0.5 }));
    expect(brief.instant).toBe(0.5);
    expect(brief.verdict({ unstable: false }).goals[0]!.met).toBe(true);
  });

  it('counts the body as airborne only while every foot is off the ground', () => {
    const j = new GoalJudge([{ kind: 'airborne', minS: 0.05 }, { kind: 'airborne', minS: 0.2 }], limbs);
    for (let i = 0; i <= 300; i++) {
      const contact = i >= 100 && i <= 110 ? [false, false, false] : i >= 200 && i <= 260 ? [false, false, true] : [false, true, true];
      j.add(sample(i * 0.01, { tipContact: contact }));
    }
    const [a, b] = j.verdict({ unstable: false }).goals;
    expect(a!.met).toBe(true);
    expect(a!.measured).toBeCloseTo(0.1, 6);
    expect(b!.met).toBe(false);
    expect(b!.score).toBeCloseTo(0.5, 6);
    // Without feet, nothing can show the body left the ground.
    const footless = new GoalJudge([{ kind: 'airborne', minS: 0.05 }], [{ id: 'arm', name: 'Arm', foot: false }]);
    footless.add(sample(0, { tipContact: [false] }));
    footless.add(sample(1, { tipContact: [false] }));
    expect(footless.verdict({ unstable: false }).goals[0]!.met).toBe(false);
    // A body that tipped onto its back has every foot in the air, but it is not jumping.
    const flipped = new GoalJudge([{ kind: 'airborne', minS: 0.05 }], limbs);
    for (let i = 0; i <= 100; i++) flipped.add(sample(i * 0.01, { tipContact: [false, false, false], tiltRad: i < 3 ? 0.5 : 2.8 }));
    const [f] = flipped.verdict({ unstable: false }).goals;
    expect(f!.met).toBe(false);
    expect(f!.measured).toBeCloseTo(0.02, 6);
    expect(f!.text).toBe('every foot off the ground with the body upright for at most 0.020 s at once (needed 0.05 s)');
  });

  it('counts named feet off the ground together, whatever the other feet do (a stand on two legs, D30)', () => {
    const four = [
      { id: 'fl', name: 'Front left leg', foot: true },
      { id: 'fr', name: 'Front right leg', foot: true },
      { id: 'rl', name: 'Rear left leg', foot: true },
      { id: 'rr', name: 'Rear right leg', foot: true },
    ];
    const at = (t: number, contact: boolean[]) => sample(t, { tipContact: contact, tipClearanceM: [0, 0, 0, 0], tipPos: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] });
    const goal: SkillGoal = { kind: 'airborne', limbIds: ['fl', 'rr'], minS: 0.5 };
    const j = new GoalJudge([goal, { kind: 'airborne', minS: 0.05 }], four);
    // Front left up alone, then rear right up alone: one after the other is not together.
    for (let i = 0; i <= 100; i++) j.add(at(i * 0.01, [false, true, true, true]));
    for (let i = 101; i <= 200; i++) j.add(at(i * 0.01, [true, true, true, false]));
    let [two, all] = j.verdict({ unstable: false }).goals;
    expect(two!.met).toBe(false);
    expect(two!.measured).toBe(0);
    // Both up together for 0.6 s on the other two feet.
    for (let i = 201; i <= 260; i++) j.add(at(i * 0.01, [false, true, true, false]));
    [two, all] = j.verdict({ unstable: false }).goals;
    expect(two!.met).toBe(true);
    expect(two!.measured).toBeCloseTo(0.59, 6);
    expect(two!.text).toBe('the feet of the front left leg and rear right leg off the ground with the body upright for at most 0.590 s at once (needed 0.5 s)');
    expect(all!.met).toBe(false);
    expect(goalText(goal, (id) => four.find((l) => l.id === id)!.name.toLowerCase())).toBe(
      'the feet of the front left leg and rear right leg leave the ground together (the others may stay down), the body upright, for 0.5 s',
    );
    expect(goalProblems({ kind: 'airborne', minS: 0.5, limbIds: [] })).toEqual(['goal.limbIds must list 1 to 8 different limbs when present.']);
    expect(goalProblems({ kind: 'airborne', minS: 0.5, limbIds: ['fl', 'fl'] })).toEqual(['goal.limbIds must list 1 to 8 different limbs when present.']);
    expect(goalProblems(goal)).toEqual([]);
    expect(goalRobotProblems([{ kind: 'airborne', minS: 0.5, limbIds: ['front_left', 'tail'] }], desc)).toEqual(['The robot has no limb "tail".']);
  });

  it('fails a free-standing run whose body or knees touch the ground, or that tips over, whatever its goals say', () => {
    const goal: SkillGoal = { kind: 'tipHeight', limbId: 'arm', minHeightM: 0.05, holdS: 0.1 };
    const run = (parts: (t: number) => string[], tilt: (t: number) => number = () => 0.05, goals: SkillGoal[] = [goal]) => {
      const j = new GoalJudge(goals, limbs, 0.25, true);
      for (let i = 0; i <= 100; i++) j.add(sample(i * 0.01, { tipClearanceM: [0.08, 0, 0], groundParts: parts(i * 0.01), tiltRad: tilt(i * 0.01) }));
      return { v: j.verdict({ unstable: false }), instant: j.instant };
    };
    const clean = run(() => []);
    expect(clean.v.success).toBe(true);
    expect(clean.v.checks).toEqual([
      { kind: 'feetOnly', met: true, text: 'only the feet touched the ground' },
      { kind: 'noTipOver', met: true, text: 'never tipped over (largest tilt 3 deg)' },
    ]);
    // A knee down for 0.1 s fails the run although its one goal was met; one grazing step does not.
    const knee = run((t) => (t >= 0.5 && t <= 0.6 ? ['front left leg knee'] : []));
    expect(knee.v.goals[0]!.met).toBe(true);
    expect(knee.v.success).toBe(false);
    expect(knee.v.checks[0]).toEqual({ kind: 'feetOnly', met: false, text: 'the front left leg knee (0.10 s) touched the ground; only the feet may' });
    expect(run((t) => (t === 0.5 ? ['body'] : [])).v.success).toBe(true);
    // The moment share counts the checks: the body is down in the last sample (the goal and the tilt hold, 2 of 3).
    expect(run((t) => (t >= 0.9 ? ['body'] : [])).instant).toBeCloseTo(2 / 3, 12);
    // Tipped past 90 deg and back.
    const over = run(() => [], (t) => (t > 0.4 && t < 0.6 ? 1.8 : 0.05));
    expect(over.v.success).toBe(false);
    expect(over.v.checks[1]).toEqual({ kind: 'noTipOver', met: false, text: 'tipped over (largest tilt 103 deg, past 90 deg)' });
    // Lying down may put the body on the ground.
    const lie: SkillGoal = { kind: 'baseHeight', below: true, heightM: 0.1, holdS: 0.1 };
    const low = run(() => ['body'], () => 0.05, [lie]);
    expect(low.v.checks[0]).toEqual({ kind: 'feetOnly', met: true, text: 'the body may touch the ground (a goal asks it to lie low)' });
    // The reward counts a failed check like a missed goal.
    expect(goalReward(knee.v, false)).toBeLessThan(goalReward(clean.v, false) - 1);
    // A fixed base is not checked.
    expect(new GoalJudge([goal], limbs).verdict({ unstable: false }).checks).toEqual([]);
  });

  it('measures travel along the start heading and turning as unwrapped yaw', () => {
    const goals: SkillGoal[] = [
      { kind: 'travel', direction: 'forward', minDistanceM: 0.25 },
      { kind: 'travel', direction: 'left', minDistanceM: 0.2 },
      { kind: 'travel', direction: 'backward', minDistanceM: 0.1 },
      { kind: 'turn', left: true, minAngleRad: 3.5 },
      { kind: 'turn', left: false, minAngleRad: 0.5 },
    ];
    const j = new GoalJudge(goals, limbs);
    // Facing +y at the start, ending 0.3 m along +y and 0.1 m along -x, after turning 4 rad to the left.
    for (let i = 0; i <= 40; i++) {
      const yaw = Math.atan2(Math.sin(Math.PI / 2 + 0.1 * i), Math.cos(Math.PI / 2 + 0.1 * i));
      j.add(sample(i * 0.1, { yawRad: yaw, basePos: [(-0.1 * i) / 40, (0.3 * i) / 40, 0.25] }));
    }
    const [fwd, left, back, turnLeft, turnRight] = j.verdict({ unstable: false }).goals;
    expect(fwd!.met).toBe(true);
    expect(fwd!.measured).toBeCloseTo(0.3, 9);
    expect(left!.met).toBe(false);
    expect(left!.measured).toBeCloseTo(0.1, 9);
    expect(left!.score).toBeCloseTo(0.5, 9);
    expect(back!.measured).toBeCloseTo(-0.3, 9);
    expect(back!.score).toBe(0);
    expect(turnLeft!.met).toBe(true);
    expect(turnLeft!.measured).toBeCloseTo(4, 9);
    expect(turnRight!.met).toBe(false);
    expect(turnRight!.measured).toBeCloseTo(-4, 9);
  });

  it('judges body height, a rise from the resting height, upright, falling and the end of the run', () => {
    const goals: SkillGoal[] = [
      { kind: 'baseHeight', below: true, heightM: 0.2, holdS: 1 },
      { kind: 'baseHeight', below: false, heightM: 0.3, holdS: 0.05 },
      { kind: 'baseRise', minRiseM: 0.05 },
      { kind: 'baseRise', minRiseM: 0.1 },
      { kind: 'noFall', minBaseHeightM: 0.1 },
      { kind: 'noFall', minBaseHeightM: 0.2 },
      { kind: 'upright', maxTiltRad: 0.35 },
      { kind: 'standsAtEnd', maxTiltRad: 0.35, minBaseHeightM: 0.2 },
    ];
    // Resting height 0.25 m; the body starts 1 cm above it and drops (that drop is not a rise).
    const j = new GoalJudge(goals, limbs, 0.25);
    for (let i = 0; i <= 400; i++) {
      const z = i < 5 ? 0.26 : i >= 100 && i <= 250 ? 0.15 : i >= 270 && i <= 280 ? 0.32 : 0.25;
      j.add(sample(i * 0.01, { basePos: [0, 0, z], tiltRad: i < 100 ? 1.0 : 0.1 }));
    }
    const r = j.verdict({ unstable: false }).goals;
    expect(r.map((g) => g.met)).toEqual([true, true, true, false, true, false, false, true]);
    expect(r[0]!.measured).toBeCloseTo(1.5, 6);
    expect(r[1]!.measured).toBeCloseTo(0.1, 6);
    expect(r[2]!.measured).toBeCloseTo(0.07, 9);
    expect(r[2]!.text).toBe('body rose 0.070 m above its resting height of 0.250 m (needed 0.05 m)');
    expect(r[3]!.score).toBeCloseTo(0.7, 9);
    expect(r[5]!.score).toBeCloseTo(0.75, 9);
    expect(r[6]!.text).toBe('largest tilt 57 deg (limit 20 deg)');
    expect(r[7]!.text).toBe('in the last 0.5 s: largest tilt 6 deg (limit 20 deg), lowest body height 0.250 m (minimum 0.2 m)');
  });

  it('a body that never comes to rest shows no rise; an unstable or empty run never succeeds', () => {
    const j = new GoalJudge([{ kind: 'baseRise', minRiseM: 0.01 }, { kind: 'baseHeight', below: false, heightM: 0.27, holdS: 0 }], limbs, 0.25);
    for (let i = 0; i <= 100; i++) j.add(sample(i * 0.01, { basePos: [0, 0, 0.28] }));
    const [rise, above] = j.verdict({ unstable: false }).goals;
    expect(rise!.met).toBe(false);
    expect(rise!.text).toBe('the body never came to rest at its resting height of 0.250 m, so no rise was measured');
    expect(above!.met).toBe(false);
    const upright = new GoalJudge([{ kind: 'upright', maxTiltRad: 0.35 }], limbs);
    expect(upright.verdict({ unstable: false }).success).toBe(false);
    upright.add(sample(0));
    expect(upright.verdict({ unstable: false }).success).toBe(true);
    expect(upright.verdict({ unstable: true }).success).toBe(false);
  });

  it('rewards progress toward every goal, weighs the worst goal, rewards success and penalizes an unstable run', () => {
    const j = new GoalJudge([{ kind: 'upright', maxTiltRad: 0.35 }, { kind: 'baseRise', minRiseM: 0.1 }], limbs);
    j.add(sample(0));
    j.add(sample(0.1, { basePos: [0, 0, 0.3] }));
    const v = j.verdict({ unstable: false });
    expect(v.goals[0]!.score).toBe(1);
    expect(v.goals[1]!.score).toBeCloseTo(0.5, 9);
    // Half the mean (0.75) and half the worst goal (0.5).
    expect(goalReward(v, false)).toBeCloseTo(0.625, 9);
    expect(goalReward({ success: true, goals: [v.goals[0]!] }, false)).toBe(2);
    expect(goalReward(v, true)).toBeCloseTo(-0.375, 9);
  });

  it('judges "about" a distance or an angle between both bounds', () => {
    const goals: SkillGoal[] = [
      { kind: 'travel', direction: 'forward', minDistanceM: 0.25, maxDistanceM: 0.35 },
      { kind: 'travel', direction: 'forward', minDistanceM: 0.1, maxDistanceM: 0.2 },
      { kind: 'turn', left: true, minAngleRad: 1.3, maxAngleRad: 1.8 },
      { kind: 'turn', left: true, minAngleRad: 0.5, maxAngleRad: 1 },
    ];
    const j = new GoalJudge(goals, limbs);
    for (let i = 0; i <= 30; i++) j.add(sample(i * 0.1, { basePos: [0.01 * i, 0, 0.25], yawRad: (1.5 * i) / 30 }));
    const [about, short, turned, over] = j.verdict({ unstable: false }).goals;
    expect(about!.met).toBe(true);
    expect(about!.text).toBe('body ended 0.300 m forward of its start (needed 0.25 to 0.35 m)');
    // Past the maximum, the score falls as the overshoot grows.
    expect(short!.met).toBe(false);
    expect(short!.score).toBeCloseTo(0.2 / 0.3, 6);
    expect(turned!.met).toBe(true);
    expect(over!.met).toBe(false);
    expect(over!.score).toBeCloseTo(1 / 1.5, 6);
    expect(goalProblems({ kind: 'turn', left: true, minAngleRad: 1, maxAngleRad: 0.5 }, 'g')).toEqual(['g.maxAngleRad must not be below minAngleRad.']);
    expect(goalProblems({ kind: 'travel', direction: 'left', minDistanceM: 0.1, maxDistanceM: 9 }, 'g')).toEqual(['g.maxDistanceM must be a number from 0.01 to 5 when present.']);
    expect(goalText({ kind: 'turn', left: true, minAngleRad: Math.PI / 2 - 0.2, maxAngleRad: Math.PI / 2 + 0.2 })).toBe('the body turns 79 deg to 101 deg to the left');
  });

  it('judges a body angle by its longest stretch inside the band, and scores how close it came', () => {
    const goals: SkillGoal[] = [
      { kind: 'bodyAngle', axis: 'pitch', minRad: -0.6, maxRad: -0.2, holdS: 1 },
      { kind: 'bodyAngle', axis: 'roll', minRad: 0.1, maxRad: 0.5, holdS: 0.5 },
    ];
    const j = new GoalJudge(goals, limbs);
    // Nose up by 0.4 rad from 0.5 s to 2 s; the roll stays at 0.05 rad, just outside its band.
    for (let i = 0; i <= 300; i++) j.add(sample(i * 0.01, { pitchRad: i >= 50 && i <= 200 ? -0.4 : 0, rollRad: 0.05 }));
    const [pitch, roll] = j.verdict({ unstable: false }).goals;
    expect(pitch!.met).toBe(true);
    expect(pitch!.measured).toBeCloseTo(1.5, 6);
    expect(roll!.met).toBe(false);
    expect(roll!.score).toBeCloseTo(0.5 * (1 - 0.05 / 0.5), 9);
    expect(roll!.text).toBe('body roll between 6 deg and 29 deg for 0.00 s of 0.5 s required (it ranged 3 deg to 3 deg)');
  });

  it('judges a foot kept on the ground by its share of the run', () => {
    const j = new GoalJudge([{ kind: 'footDown', limbId: 'f1', minFraction: 0.9 }, { kind: 'footDown', limbId: 'f2', minFraction: 0.9 }], limbs);
    for (let i = 0; i < 100; i++) j.add(sample(i * 0.01, { tipContact: [false, true, i % 4 !== 0] }));
    const [one, two] = j.verdict({ unstable: false }).goals;
    expect(one!.met).toBe(true);
    expect(one!.measured).toBe(1);
    expect(two!.met).toBe(false);
    expect(two!.measured).toBeCloseTo(0.75, 9);
    expect(two!.score).toBeCloseTo(0.75 / 0.9, 9);
    expect(two!.text).toBe('foot two on the ground for 75 % of the run (needed 90 %)');
  });

  it('counts the strokes of a swinging tip along an axis of the start heading', () => {
    const goals: SkillGoal[] = [
      { kind: 'tipSwing', limbId: 'arm', axis: 'y', minAmplitudeM: 0.05, minSwings: 4 },
      { kind: 'tipSwing', limbId: 'arm', axis: 'z', minAmplitudeM: 0.05, minSwings: 1 },
      { kind: 'tipSwing', limbId: 'arm', axis: 'y', minAmplitudeM: 0.1, minSwings: 1 },
    ];
    const j = new GoalJudge(goals, limbs);
    // Facing +y, so left in the start heading is world -x. The tip swings 0.04 m to either side of its start, once a second, for 2.5 s.
    for (let i = 0; i <= 250; i++) {
      const s = 0.04 * Math.sin(2 * Math.PI * i * 0.01);
      j.add(sample(i * 0.01, { yawRad: Math.PI / 2, tipPos: [[0.3 - s, 0.1, 0.5], [0, 0, 0], [0, 0, 0]] }));
    }
    const [side, up, wide] = j.verdict({ unstable: false }).goals;
    // Strokes of 0.08 m: the first counts once the range seen spans 0.05 m, then one per reversal.
    expect(side!.met).toBe(true);
    expect(side!.measured).toBe(4);
    expect(side!.text).toBe('arm tip swung left and right 4 times by at least 0.05 m (needed 4; largest stroke 0.080 m)');
    expect(up!.met).toBe(false);
    expect(up!.measured).toBe(0);
    expect(up!.score).toBe(0);
    // Strokes shorter than required never count; the range seen still shows progress.
    expect(wide!.measured).toBe(0);
    expect(wide!.score).toBeCloseTo(0.3 * 0.8, 6);
  });

  it('judges a reached point relative to where the tip started, in the start heading', () => {
    const reach: SkillGoal = { kind: 'tipReach', limbId: 'arm', x: 0.1, y: 0, z: 0.05, toleranceM: 0.02, holdS: 0.5 };
    const j = new GoalJudge([reach, { ...reach, holdS: 2 }], limbs);
    // Facing +y: 0.1 m forward and 0.05 m up from (0.3, 0.1, 0.5) is (0.3, 0.2, 0.55); the tip stops 1 cm short of it.
    for (let i = 0; i <= 200; i++) {
      const at = i >= 100 && i <= 180;
      j.add(sample(i * 0.01, { yawRad: Math.PI / 2, tipPos: [at ? [0.3, 0.19, 0.55] : [0.3, 0.1, 0.5], [0, 0, 0], [0, 0, 0]] }));
    }
    const [held, longer] = j.verdict({ unstable: false }).goals;
    expect(held!.met).toBe(true);
    expect(held!.measured).toBeCloseTo(0.01, 9);
    expect(held!.text).toBe('arm tip came within 0.010 m of the point (tolerance 0.02 m), inside it for 0.80 s of 0.5 s required');
    expect(longer!.met).toBe(false);
    expect(longer!.score).toBeCloseTo(0.5 + 0.5 * (0.8 / 2), 6);
  });
});

describe('goal and motion validation', () => {
  it('names each problem with a goal', () => {
    expect(goalProblems({ kind: 'upright', maxTiltRad: 0.5 })).toEqual([]);
    expect(goalProblems({ kind: 'upright', maxTiltRad: 0.01 }, 'g')).toEqual(['g.maxTiltRad must be a number from 0.05 to 1.6.']);
    expect(goalProblems({ kind: 'fly' }, 'g')).toEqual(['g.kind must be one of upright, noFall, tipHeight, baseHeight, airborne, baseRise, travel, turn, standsAtEnd, bodyAngle, footDown, tipSwing, tipReach.']);
    expect(goalProblems({ kind: 'travel', direction: 'up', minDistanceM: 0.1, extra: 1 }, 'g')).toEqual(['g.direction must be forward, backward, left or right.', 'g has unknown fields: extra.']);
    expect(goalProblems({ kind: 'tipHeight', minHeightM: 0.05, holdS: Number.NaN }, 'g')).toEqual(['g.holdS must be a number from 0 to 10.', 'g.limbId must name a limb.']);
    expect(goalProblems({ kind: 'bodyAngle', axis: 'yaw', minRad: 0.2, maxRad: -0.2, holdS: 1 }, 'g')).toEqual(['g.axis must be pitch or roll.', 'g.minRad must not be above maxRad.']);
    expect(goalProblems({ kind: 'tipSwing', limbId: 'arm', axis: 'y', minAmplitudeM: 0.05, minSwings: 2.5 }, 'g')).toEqual(['g.minSwings must be a whole number.']);
    expect(goalsProblems([])).toEqual(['goals must list 1 to 12 goals.']);
  });

  it('names each problem with a motion', () => {
    expect(motionProblems({ jointIds: ['a', 'b'], keyframes: [{ t: 1, q: [0, 0] }], endHoldS: 1 })).toEqual([]);
    expect(motionProblems({ jointIds: ['a', 'b'], keyframes: [{ t: 1, q: [0, 0] }, { t: 1, q: [0, 1] }], endHoldS: 11 })).toEqual([
      'motion.keyframes[1].t must be later than the previous keyframe (1 s).',
      'motion.endHoldS must be from 0 to 10 s.',
    ]);
    expect(motionProblems({ jointIds: ['a', 'b'], keyframes: [{ t: 1, q: [0] }], endHoldS: 1 })).toEqual(['motion.keyframes[0] must hold a time and one finite target per joint.']);
    expect(motionProblems({ jointIds: ['a', 'a'], keyframes: [], endHoldS: 1 })).toEqual(['motion.jointIds must not repeat.', 'motion.keyframes must hold 1 to 24 keyframes.']);
  });

  it('only offers goals that can be measured on the body', () => {
    const arm = PRESETS.find((p) => p.id === 'arm3')!.make();
    expect(goalKindsFor(arm)).toEqual(['tipHeight', 'tipSwing', 'tipReach']);
    expect(goalKindsFor(desc)).toContain('airborne');
    expect(goalKindsFor(desc)).toContain('bodyAngle');
    expect(goalRobotProblems([{ kind: 'upright', maxTiltRad: 0.5 }], arm)).toEqual([`"Stays upright" measures the body, but ${arm.name} has a fixed base; use limb tip goals.`]);
    expect(goalRobotProblems([{ kind: 'footDown', limbId: 'arm', minFraction: 0.5 }], arm)).toEqual([`"Foot stays down" measures the body, but ${arm.name} has a fixed base; use limb tip goals.`]);
    expect(goalRobotProblems([{ kind: 'tipSwing', limbId: 'arm', axis: 'z', minAmplitudeM: 0.05, minSwings: 2 }], arm)).toEqual([]);
    expect(goalRobotProblems([{ kind: 'tipHeight', limbId: 'tail', minHeightM: 0.05, holdS: 1 }], desc)).toEqual(['The robot has no limb "tail".']);
    expect(goalRobotProblems([{ kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.05, holdS: 1 }, { kind: 'airborne', minS: 0.05 }], desc)).toEqual([]);
  });

  it('defaults a raised tip to 5 cm above where it rests in the home stance', () => {
    // A foot rests on the ground (2 mm clearance at the resting height); an arm's tip rests far above it.
    expect(homeTipClearance(desc, 0)).toBeCloseTo(0.002, 6);
    expect(defaultGoal('tipHeight', desc)).toEqual({ kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.05, holdS: 1 });
    const arm = PRESETS.find((p) => p.id === 'arm3')!.make();
    const rest = homeTipClearance(arm, 0);
    expect(rest).toBeGreaterThan(0.3);
    const g = defaultGoal('tipHeight', arm);
    expect(g.kind).toBe('tipHeight');
    expect(Math.abs((g.kind === 'tipHeight' ? g.minHeightM : 0) - (rest + 0.05))).toBeLessThanOrEqual(0.005);
  });

  it('describes goals in plain words', () => {
    const name = (id: string) => (id === 'front_left' ? 'front left leg' : id);
    expect(goalText({ kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.05, holdS: 1 }, name)).toBe('the front left leg tip is at least 0.05 m above the ground for 1 s');
    expect(goalText({ kind: 'baseHeight', below: true, heightM: 0.2, holdS: 1 })).toBe('the body is below 0.2 m for 1 s');
    expect(goalText({ kind: 'turn', left: false, minAngleRad: Math.PI / 2 })).toBe('the body turns 90 deg to the right');
  });
});

describe('commands that name a taught skill', () => {
  const names = ['wave', 'Sit down', "Rex's crouch"];
  it('match the name with polite or framing words around it', () => {
    expect(matchTaughtSkill('wave', names)).toBe('wave');
    expect(matchTaughtSkill('Please wave now!', names)).toBe('wave');
    expect(matchTaughtSkill('can you do the sit down please', names)).toBe('Sit down');
    expect(matchTaughtSkill("run Rex's crouch again", names)).toBe("Rex's crouch");
  });
  it('do not match other sentences that only contain the name', () => {
    expect(matchTaughtSkill('wave the front left leg', names)).toBeNull();
    expect(matchTaughtSkill('how did the wave go?', names)).toBeNull();
    expect(matchTaughtSkill('sit', names)).toBeNull();
  });
});

describe('a taught crouch on the quadruped (MuJoCo, 20 randomized trials)', () => {
  const crouch: KeyframeMotion = { jointIds: ids, keyframes: [{ t: 1, q: crouchQ }], endHoldS: 1.5 };
  const goals: SkillGoal[] = [
    { kind: 'baseHeight', below: true, heightM: 0.2, holdS: 1 },
    { kind: 'upright', maxTiltRad: 0.35 },
    { kind: 'noFall', minBaseHeightM: 0.1 },
  ];

  it('meets its goals, is stored bound to the morphology, and survives a file round trip', async () => {
    const e = evaluateSkill(mj, desc, -1, stance, { kind: 'keyframes', motion: crouch }, { ...DEFAULT_EVALUATION, seed: 42 }, { goals });
    expect(e.trials).toBe(20);
    expect(e.successes).toBeGreaterThanOrEqual(19);
    for (const r of e.results.filter((x) => x.success)) {
      expect(r.goals!.map((g) => g.met)).toEqual([true, true, true]);
      // Geometry: about 0.27 m standing, about 0.17 m crouched (plus the joint controller's sag).
      expect(r.minBaseHeightM).toBeGreaterThan(0.12);
      expect(r.minBaseHeightM).toBeLessThan(0.19);
      expect(r.goals![0]!.measured).toBeGreaterThan(1);
    }
    expect(e.context!.goals).toEqual(goals);
    expect(e.context!.criteria).toBeUndefined();

    const hash = await morphologyHash(desc);
    const a = createSkillArtifact({ robot: desc, morphologyHash: hash, limbId: '', spec: { kind: 'keyframes', motion: crouch }, name: 'crouch (keyframes)', evaluation: e, taught: { name: 'crouch', goals }, now: '2026-10-05T12:00:00.000Z' });
    expect(a.skillType).toBe('taught');
    expect(a.method).toBe('keyframes');
    expect(a.limbs).toEqual([]);
    expect(a.observation).toEqual([{ name: 'time', source: 'time', size: 1, units: 's' }]);
    expect(a.evaluation!.metrics['goalMet:0']).toBeGreaterThanOrEqual(19);
    expect(specFromArtifact(a)).toEqual({ kind: 'keyframes', motion: crouch });
    expect(checkSkillCompatibility(a, desc, hash)).toEqual({ ok: true });
    const stored = JSON.parse(JSON.stringify(a)) as unknown;
    const parsed = parseSkillArtifact(stored);
    expect(parsed.ok && parsed.value).toEqual(stored);

    // A record whose goals or keyframes are malformed is refused with the reason.
    const bad = { ...(stored as Record<string, unknown>), taught: { name: 'crouch', goals: [{ kind: 'upright', maxTiltRad: 9 }] }, motion: { ...crouch, endHoldS: -1 } };
    const refused = parseSkillArtifact(bad);
    expect(!refused.ok && refused.errors).toEqual(['skill.taught.goals[0].maxTiltRad must be a number from 0.05 to 1.6.', 'skill.motion.endHoldS must be from 0 to 10 s.']);
    // Who composed it survives the round trip and names the method; an unknown author is refused.
    const composed = JSON.parse(JSON.stringify(createSkillArtifact({ robot: desc, morphologyHash: hash, limbId: '', spec: { kind: 'keyframes', motion: crouch }, name: 'crouch (keyframes 2)', evaluation: e, taught: { name: 'crouch', goals, author: 'ai' } }))) as unknown;
    const back = parseSkillArtifact(composed);
    expect(back.ok && back.value.taught).toEqual({ name: 'crouch', goals, author: 'ai' });
    expect(back.ok && skillLabel(back.value)).toBe('Keyframe motion the AI composed (open loop)');
    expect(skillLabel(a)).toBe('Keyframe motion you posed (open loop)');
    const wrong = parseSkillArtifact({ ...(composed as Record<string, unknown>), taught: { name: 'crouch', goals, author: 'robot' } });
    expect(!wrong.ok && wrong.errors).toEqual(['skill.taught.author must be "ai" or "ai-edited" when present.']);
    // Another body with other joints cannot run these keyframes.
    const hexapod = PRESETS.find((p) => p.id === 'hexapod')!.make();
    const other = checkSkillCompatibility(a, hexapod, await morphologyHash(hexapod));
    expect(other.ok).toBe(false);
  });
});
