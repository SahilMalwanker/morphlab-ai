import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel, forwardKinematics } from '../../src/core/kinematics/fk';
import { identity, type Quat } from '../../src/core/math/mat4';
import { PRESETS } from '../../src/core/model/catalog';
import { hexapod, quadruped, quadrupedStance } from '../../src/core/model/presets';
import { DEFAULT_EVALUATION, evaluateSkill } from '../../src/skills/evaluation';
import { DEFAULT_GAIT, GaitController, gaitGoalConflicts, gaitProblems, gaitText, patternPhases, type GaitParams } from '../../src/skills/gait';
import type { SkillGoal } from '../../src/skills/taught';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const dog = quadruped();
const stance = quadrupedStance();
const km = buildKinematicModel(dog);
const level: Quat = [1, 0, 0, 0];
const obs = (imuQuat: Quat = level) => ({ q: stance, touch: [1, 1, 1, 1], imuQuat });
/** Limb tip positions in the body frame for joint targets q. */
const tips = (q: number[]) => forwardKinematics(km, q, identity()).tips.map((T) => [T[12]!, T[13]!, T[14]!]);
const home = tips(stance);

describe('contact patterns from the body geometry', () => {
  it('groups the legs of each pattern by where the feet stand', () => {
    // Limb order: front left, front right, rear left, rear right.
    expect(patternPhases(dog, 'trot')).toEqual([0, 0.5, 0.5, 0]);
    expect(patternPhases(dog, 'pace')).toEqual([0, 0.5, 0, 0.5]);
    expect(patternPhases(dog, 'bound')).toEqual([0, 0, 0.5, 0.5]);
    expect(patternPhases(dog, 'walk')).toEqual([0.25, 0.75, 0, 0.5]);
    expect(patternPhases(dog, 'pronk')).toEqual([0, 0, 0, 0]);
    // On six legs a trot is the alternating tripod: front and rear left with middle right, and the others.
    expect(patternPhases(hexapod(), 'trot')).toEqual([0, 0.5, 0, 0.5, 0, 0.5]);
  });

  it('refuses out-of-range parameters, and bodies that cannot step', () => {
    expect(gaitProblems(DEFAULT_GAIT, dog)).toEqual([]);
    expect(gaitProblems({ ...DEFAULT_GAIT, cycleS: 0.1 }, dog)).toEqual(['gait.cycleS must be a number from 0.25 to 3.']);
    expect(gaitProblems({ ...DEFAULT_GAIT, pattern: 'gallop' }, dog)).toEqual(['gait.pattern must be one of trot, walk, pace, bound, pronk, wave, stand.']);
    expect(gaitProblems({ ...DEFAULT_GAIT, phases: [0, 0.5] }, dog)).toEqual(['gait.phases must give one value per limb (4).']);
    expect(gaitProblems({ ...DEFAULT_GAIT, heldUp: ['front_left', 'front_right'] }, dog)).toEqual(['gait.heldUp leaves fewer than three legs to stand on.']);
    expect(gaitProblems({ ...DEFAULT_GAIT, heldUp: ['tail'] }, dog)).toEqual(['gait.heldUp: Quadruped has no leg "tail".']);
    // A walk lifts one leg at a time, so on four legs each foot is down at least three quarters of the cycle.
    expect(gaitProblems({ ...DEFAULT_GAIT, pattern: 'walk', dutyFactor: 0.5 }, dog)).toEqual([
      'gait.dutyFactor must be at least 0.75 for a walk on 4 legs: it lifts one leg at a time, so each foot must be down for 75 % of the cycle. A trot is faster.',
    ]);
    expect(gaitProblems({ ...DEFAULT_GAIT, pattern: 'walk', dutyFactor: 0.8 }, dog)).toEqual([]);
    expect(gaitProblems({ ...DEFAULT_GAIT, pattern: 'wave', dutyFactor: 0.8 }, hexapod())).toHaveLength(1);
    const arm = PRESETS.find((p) => p.id === 'arm3')!.make();
    expect(gaitProblems(DEFAULT_GAIT, arm)).toEqual([`${arm.name} has fewer than two legs, so it cannot step.`]);
    expect(() => new GaitController(buildKinematicModel(arm), [0, 0, 0], DEFAULT_GAIT)).toThrow('cannot step');
  });

  it('describes a gait in words', () => {
    expect(gaitText({ ...DEFAULT_GAIT, strideM: -0.05, turnRad: 0.2 })).toBe(
      'trot pattern, 0.5 s cycle, feet down 60 % of it, steps 0.04 m high, 0.05 m backward per cycle, turning 0.2 rad to the left per cycle, body 0 m from its home height, 4 s; IMU feedback kp 0.1 m/rad, kd 0.005 m*s/rad',
    );
  });

  it('names goals a gait cannot meet because it steps the other way', () => {
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, strideM: 0, turnRad: 0.3 }, [{ kind: 'turn', left: false, minAngleRad: 1.3 }])).toEqual(['the goal turns right, so turnRad must be negative (positive turns left).']);
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, strideM: 0.06 }, [{ kind: 'travel', direction: 'backward', minDistanceM: 0.2 }])).toEqual(['the goal travels backward, so strideM must be negative.']);
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, strideM: 0, lateralM: -0.04 }, [{ kind: 'travel', direction: 'left', minDistanceM: 0.1 }])).toEqual(['the goal travels left, so lateralM must be positive.']);
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, strideM: -0.06, turnRad: -0.2 }, [{ kind: 'travel', direction: 'backward', minDistanceM: 0.2 }, { kind: 'turn', left: false, minAngleRad: 0.5 }, { kind: 'upright', maxTiltRad: 0.5 }])).toEqual([]);
    // A side step with a long stride walks diagonally while its goal sees only the sideways part (seen in the browser check).
    const right: SkillGoal = { kind: 'travel', direction: 'right', minDistanceM: 0.18, maxDistanceM: 0.22 };
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, pattern: 'wave', dutyFactor: 0.75, strideM: 0.2, lateralM: -0.04 }, [right])).toEqual([
      'the goals ask for sideways travel only, but strideM (0.2 m per cycle) moves the body further forward or backward than lateralM (-0.04 m) moves it sideways; keep strideM near 0, or add a forward or backward travel goal.',
    ]);
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, strideM: 0.02, lateralM: -0.04 }, [right])).toEqual([]);
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, strideM: 0.2, lateralM: -0.04 }, [right, { kind: 'travel', direction: 'forward', minDistanceM: 0.5 }])).toEqual([]);
    expect(gaitGoalConflicts({ ...DEFAULT_GAIT, strideM: 0.03, lateralM: 0.05 }, [{ kind: 'travel', direction: 'forward', minDistanceM: 0.2 }])).toHaveLength(1);
  });
});

describe('gait trajectories (geometry)', () => {
  it('starts at the stance, slides stance feet back and swings the others forward in an arc', () => {
    const c = new GaitController(km, stance, { ...DEFAULT_GAIT, feedbackKp: 0, feedbackKd: 0 });
    // Before the step fades in, the targets are the stance.
    c.update(0, obs()).forEach((v, i) => expect(v).toBeCloseTo(stance[i]!, 3));
    // At 2.4 s the front left foot (phase 0.8) is halfway through its swing: lifted 4 cm above where it stands.
    let p = tips(c.update(2.4, obs()));
    for (let k = 0; k < 3; k++) expect(p[0]![k]! - home[0]![k]!).toBeCloseTo(k === 2 ? 0.04 : 0, 3);
    // The front right foot (phase 0.3) is in the middle of its stance, under the hip.
    for (let k = 0; k < 3; k++) expect(p[1]![k]! - home[1]![k]!).toBeCloseTo(0, 3);
    // In stance a foot slides back relative to the body: a third of the 6 cm stride forward at phase 0.1, a third back at phase 0.5.
    p = tips(c.update(2.55, obs()));
    expect(p[0]![0]! - home[0]![0]!).toBeCloseTo(0.02, 3);
    expect(p[0]![2]! - home[0]![2]!).toBeCloseTo(0, 3);
    p = tips(c.update(2.75, obs()));
    expect(p[0]![0]! - home[0]![0]!).toBeCloseTo(-0.02, 3);
    expect(c.phase).toBe('stepping');
    expect(c.finished).toBe(false);
    c.update(4.01, obs());
    expect(c.finished).toBe(true);
  });

  it('lowers the feet on the low side and raises them on the high side (IMU feedback)', () => {
    const still: GaitParams = { ...DEFAULT_GAIT, pattern: 'stand', dutyFactor: 1, strideM: 0, feedbackKp: 0.2, feedbackKd: 0 };
    // Nose down by 0.1 rad (about +y): the front feet reach down and the rear feet draw up, levelling the body.
    let p = tips(new GaitController(km, stance, still).update(2, obs([Math.cos(0.05), 0, Math.sin(0.05), 0])));
    expect(p[0]![2]! - home[0]![2]!).toBeCloseTo(-0.02, 3);
    expect(p[2]![2]! - home[2]![2]!).toBeCloseTo(0.02, 3);
    // Left side up by 0.1 rad (about +x): the left feet draw up, the right feet reach down.
    p = tips(new GaitController(km, stance, still).update(2, obs([Math.cos(0.05), Math.sin(0.05), 0, 0])));
    expect(p[0]![2]! - home[0]![2]!).toBeCloseTo(0.02, 3);
    expect(p[1]![2]! - home[1]![2]!).toBeCloseTo(-0.02, 3);
    // Without feedback the same reading changes nothing.
    p = tips(new GaitController(km, stance, { ...still, feedbackKp: 0 }).update(2, obs([Math.cos(0.05), Math.sin(0.05), 0, 0])));
    expect(p[0]![2]! - home[0]![2]!).toBeCloseTo(0, 3);
  });
});

describe('gaits on the quadruped (MuJoCo, randomized trials)', () => {
  const upright: SkillGoal = { kind: 'upright', maxTiltRad: 0.5 };
  const run = (gait: GaitParams, goals: SkillGoal[], trials = 4) => evaluateSkill(mj, dog, -1, stance, { kind: 'gait', gait }, { ...DEFAULT_EVALUATION, trials, seed: 5 }, { goals });

  it('trots forward and stays upright', () => {
    const e = run({ ...DEFAULT_GAIT, strideM: 0.06, durationS: 4 }, [{ kind: 'travel', direction: 'forward', minDistanceM: 0.1 }, upright]);
    const seen = e.results.map((r) => `${r.goals![0]!.measured} m, tilt ${r.maxTiltRad} rad`).join('; ');
    expect(e.successes, seen).toBeGreaterThanOrEqual(3);
  });

  it('trots backward and turns in place when asked', () => {
    const back = run({ ...DEFAULT_GAIT, strideM: -0.06, durationS: 4 }, [{ kind: 'travel', direction: 'backward', minDistanceM: 0.1 }, upright]);
    expect(back.successes, back.results.map((r) => `${r.goals![0]!.measured} m`).join('; ')).toBeGreaterThanOrEqual(3);
    const turn = run({ ...DEFAULT_GAIT, strideM: 0, turnRad: 0.3, durationS: 4 }, [{ kind: 'turn', left: true, minAngleRad: 0.4 }, upright]);
    expect(turn.successes, turn.results.map((r) => `${r.goals![0]!.measured} rad`).join('; ')).toBeGreaterThanOrEqual(3);
  });
});
