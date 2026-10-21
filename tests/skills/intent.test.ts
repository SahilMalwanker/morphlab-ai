import { describe, expect, it } from 'vitest';
import { PRESETS } from '../../src/core/model/catalog';
import { quadruped } from '../../src/core/model/presets';
import { checkIntent, namedLimb } from '../../src/skills/intent';
import { restingHeight, type SkillGoal } from '../../src/skills/taught';

const dog = quadruped();
const arm = PRESETS.find((p) => p.id === 'arm3')!.make();
const rest = restingHeight(dog)!;
const crouch: SkillGoal[] = [{ kind: 'baseHeight', below: true, heightM: rest - 0.05, holdS: 1 }, { kind: 'upright', maxTiltRad: 0.5 }];
const noseUp: SkillGoal = { kind: 'bodyAngle', axis: 'pitch', minRad: -0.6, maxRad: -0.15, holdS: 1 };
const swing = (limbId: string): SkillGoal => ({ kind: 'tipSwing', limbId, axis: 'y', minAmplitudeM: 0.05, minSwings: 4 });

describe('request check: goals against the words of a request', () => {
  it('finds no goal for a sit that is only a level crouch, and accepts a nose-up pitch', () => {
    const level = checkIntent('sit', 'sit down like a dog', crouch, dog);
    expect(level).toHaveLength(1);
    expect(level[0]).toMatchObject({ word: 'sit', goalIndex: -1 });
    expect(level[0]!.hint).toContain('bodyAngle');
    expect(checkIntent('sit', 'sit down like a dog', [...crouch, noseUp], dog)).toEqual([expect.objectContaining({ word: 'sit', goalIndex: 2 })]);
    // Nose down is a bow, not a sit.
    const down: SkillGoal = { ...noseUp, minRad: 0.1, maxRad: 0.5 };
    expect(checkIntent('sit', '', [down], dog)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('bow', '', [down], dog)[0]!.goalIndex).toBe(0);
  });

  it('reads the request only when the name says nothing, and only for a single motion', () => {
    expect(checkIntent('trick', 'wave your front right leg', crouch, dog)).toEqual([expect.objectContaining({ word: 'wave', goalIndex: -1 })]);
    // A request for several motions is not held against one of its skills.
    expect(checkIntent('trick', 'sit and then wave', crouch, dog)).toEqual([]);
    // Words without a rule are not judged.
    expect(checkIntent('zoomies', 'do the zoomies', crouch, dog)).toEqual([]);
  });

  it('ties limb words to the limb a goal measures', () => {
    expect(checkIntent('wave front right', '', [swing('front_left')], dog)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('wave front right', '', [swing('front_right')], dog)[0]!.goalIndex).toBe(0);
    expect(checkIntent('wave', 'wave with your front right leg', [swing('front_left')], dog)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('wave', '', [swing('front_left')], dog)[0]!.goalIndex).toBe(0);
    expect(namedLimb('lift the back left paw', dog)).toBe('rear_left');
    expect(namedLimb('lift a leg', dog)).toBeNull();
  });

  it('ignores negated words, and walking in place needs no travel', () => {
    expect(checkIntent('stand', "don't sit", crouch, dog)).toEqual([]);
    expect(checkIntent('march', 'march in place', [{ kind: 'upright', maxTiltRad: 0.4 }], dog)).toEqual([expect.objectContaining({ goalIndex: 0 })]);
    expect(checkIntent('walk', 'walk backward', [{ kind: 'travel', direction: 'forward', minDistanceM: 0.2 }], dog)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('walk', 'walk backward', [{ kind: 'travel', direction: 'backward', minDistanceM: 0.2 }], dog)[0]!.goalIndex).toBe(0);
  });

  it('scales body-height words to the resting height', () => {
    expect(checkIntent('crouch', '', crouch, dog)[0]!.goalIndex).toBe(0);
    expect(checkIntent('lie down', '', crouch, dog)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('lie down', '', [{ kind: 'baseHeight', below: true, heightM: 0.5 * rest, holdS: 1 }], dog)[0]!.goalIndex).toBe(0);
  });

  it('asks a jump to leave the ground and to land on its feet', () => {
    const air: SkillGoal = { kind: 'airborne', minS: 0.05 };
    const land: SkillGoal = { kind: 'standsAtEnd', maxTiltRad: 0.5, minBaseHeightM: 0.1 };
    expect(checkIntent('jump', '', [air], dog).map((f) => f.goalIndex)).toEqual([0, -1]);
    expect(checkIntent('jump', '', [air], dog)[1]!.hint).toBe('add a standsAtEnd goal, or an upright goal');
    expect(checkIntent('jump', '', [air, land], dog).map((f) => f.goalIndex)).toEqual([0, 1]);
    // From the request too: both rules belong to one motion.
    expect(checkIntent('trick', 'jump high', [air], dog).map((f) => f.goalIndex)).toEqual([0, -1]);
  });

  it('asks a stand on the hind legs or on two legs to lift the other feet together, not only to pitch the body (D30)', () => {
    const frontUp: SkillGoal = { kind: 'tipHeight', limbId: 'front_left', minHeightM: 0.05, holdS: 1 };
    const frontFeet: SkillGoal = { kind: 'airborne', limbIds: ['front_left', 'front_right'], minS: 1 };
    const diagonal: SkillGoal = { kind: 'airborne', limbIds: ['front_left', 'rear_right'], minS: 1 };
    // A nose-up pitch with the front feet down is a sit, and tip heights can be met one after the other.
    expect(checkIntent('stand on hind legs', '', [noseUp, frontUp], dog)).toEqual([
      expect.objectContaining({ word: 'on hind legs', goalIndex: -1, hint: 'add an airborne goal whose limbIds name the front feet' }),
    ]);
    expect(checkIntent('rear up', '', [noseUp, frontFeet], dog)).toEqual([expect.objectContaining({ word: 'rear up', goalIndex: 1 })]);
    // A rear foot in the air is not standing on the hind legs; lifting one hind leg is a lift.
    expect(checkIntent('stand on the back legs', '', [diagonal], dog)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('lift the left hind leg', '', [{ ...frontUp, limbId: 'rear_left' }], dog)).toEqual([expect.objectContaining({ word: 'lift', goalIndex: 0 })]);
    expect(checkIntent('two legs', 'balance on two diagonal legs', [{ kind: 'upright', maxTiltRad: 0.4 }, frontUp], dog)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('two legs', '', [{ kind: 'upright', maxTiltRad: 0.4 }, diagonal], dog)[0]!.goalIndex).toBe(1);
    // A jump takes every foot off the ground, not some of them.
    expect(checkIntent('jump', '', [diagonal], dog)[0]!.goalIndex).toBe(-1);
    // Looking up is still a pitch.
    expect(checkIntent('look up', '', [noseUp], dog)[0]!.goalIndex).toBe(0);
  });

  it('does not ask a fixed arm to sit, but does ask its wave to swing', () => {
    const lift: SkillGoal[] = [{ kind: 'tipHeight', limbId: 'arm', minHeightM: 0.3, holdS: 1 }];
    expect(checkIntent('sit', '', lift, arm)).toEqual([]);
    expect(checkIntent('wave', '', lift, arm)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('wave', '', [swing('arm')], arm)[0]!.goalIndex).toBe(0);
  });

  it('measures pointing up by tip height, and pointing elsewhere only by reaching a point', () => {
    const high: SkillGoal[] = [{ kind: 'tipHeight', limbId: 'arm', minHeightM: 0.6, holdS: 1 }];
    const reach: SkillGoal[] = [{ kind: 'tipReach', limbId: 'arm', x: 0.1, y: 0, z: 0.05, toleranceM: 0.03, holdS: 0.5 }];
    expect(checkIntent('point_straight_up', '', high, arm)[0]!.goalIndex).toBe(0);
    expect(checkIntent('point forward', '', high, arm)[0]!.goalIndex).toBe(-1);
    expect(checkIntent('point forward', '', reach, arm)[0]!.goalIndex).toBe(0);
  });
});
