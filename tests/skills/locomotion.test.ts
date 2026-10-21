import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import type { Vec3 } from '../../src/core/math/vec3';
import { gaussian, mulberry32 } from '../../src/core/math/rng';
import { PRESETS } from '../../src/core/model/catalog';
import { morphologyHash } from '../../src/core/model/hash';
import { hexapod, quadruped, quadrupedStance } from '../../src/core/model/presets';
import { DEFAULT_GAIT } from '../../src/skills/gait';
import { handleSkillJob, type SkillWorkerMessage, type SkillWorkerRequest } from '../../src/skills/jobs';
import {
  DEFAULT_LOCOMOTION, DEFAULT_WALK_RANGES, evaluationCommand, expectedPose, fitCalibration, judgeWalk, locomotionBase, locomotionController, locomotionProblem,
  nominalFor, parseLocomotionController, predictVelocity, rangeProblem, sampleCommand, TARGET_SPEED_FRACTION, trainLocomotion, walkForTarget, walkGait, walkPolicy, type LocomotionConfig,
} from '../../src/skills/locomotion';
import { defaultWorkerCount, inProcessRunner, RolloutPool } from '../../src/skills/rollouts';
import { SkillWorkerClient } from '../../src/skills/workerClient';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const dog = quadruped();
const stance = quadrupedStance();
const base = locomotionBase(dog);
const close = (a: readonly number[], b: readonly number[], digits = 9) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, digits));

// A body that walks backward 1.5 times as fast as commanded and forward 0.9 times, drifts backward by 0.6 of its side speed either way, and turns 1.6 times the command.
const TRUTH = [0.9, 1.5, -0.6, 0.6, 0, 0, 0, 0, 1.4, 1.4, 0.1, 0.1, 0, 0, 0, 0, 1.6, 1.6];

describe('walking controller: calibration, commands and judging', () => {
  it('fits the calibration from measured runs, sign-dependent gains and drift included', () => {
    const rng = mulberry32(4);
    const samples = Array.from({ length: 24 }, () => {
      const nominal: Vec3 = [0.1 * gaussian(rng), 0.06 * gaussian(rng), 0.4 * gaussian(rng)];
      return { nominal, measured: predictVelocity(TRUTH, nominal) };
    });
    close(fitCalibration(samples), TRUTH, 6);
  });

  it('inverts the calibration: a side step to the right asks the generator for a little forward stride against the drift', () => {
    const n = nominalFor(TRUTH, [0, -0.06, 0])!;
    close(n, [0.6 * (0.06 / 1.4) / 0.9, -0.06 / 1.4, 0]);
    close(predictVelocity(TRUTH, n), [0, -0.06, 0]);
    for (let k = 0; k < 20; k++) {
      const c = sampleCommand(DEFAULT_WALK_RANGES, 9, k);
      const want: Vec3 = [c.forwardMps, c.leftMps, c.turnRadps];
      close(predictVelocity(TRUTH, nominalFor(TRUTH, want)!), want);
    }
    close(nominalFor(TRUTH, [0, 0, 0])!, [0, 0, 0]);
  });

  it('turns a command into stride, side step and turn per cycle, within the generator bounds', () => {
    const plain = walkGait(base, null, { forwardMps: 0.1, leftMps: -0.04, turnRadps: 0.3 }, 4);
    expect(plain).toMatchObject({ calibrated: false, clamped: false });
    expect(plain.gait).toMatchObject({ strideM: 0.05, lateralM: -0.02, turnRad: 0.15, durationS: 4, cycleS: 0.5, pattern: 'trot' });
    const cal = walkGait(base, TRUTH, { forwardMps: 0, leftMps: -0.06, turnRadps: 0 }, 3);
    expect(cal.calibrated).toBe(true);
    expect(cal.gait.strideM).toBeGreaterThan(0);
    const fast = walkGait(base, null, { forwardMps: 2, leftMps: 0, turnRadps: 0 }, 3);
    expect(fast).toMatchObject({ clamped: true });
    expect(fast.gait.strideM).toBe(0.3);
  });

  it('predicts where a command leads, faded in and out like the gait', () => {
    const g = { cycleS: 0.5, durationS: 3 };
    // Each 0.5 s fade covers half its length, so 3 s of stepping move as far as 2.5 s at full speed.
    close(expectedPose({ forwardMps: 0.1, leftMps: 0, turnRadps: 0 }, g), [0.25, 0, 0], 6);
    close(expectedPose({ forwardMps: 0, leftMps: -0.08, turnRadps: 0 }, g), [0, -0.2, 0], 6);
    close(expectedPose({ forwardMps: 0, leftMps: 0, turnRadps: 0.4 }, g), [0, 0, 1], 6);
    // Walking while turning left curves the path to the left; turning right mirrors it.
    const left = expectedPose({ forwardMps: 0.1, leftMps: 0, turnRadps: 0.5 }, g);
    const right = expectedPose({ forwardMps: 0.1, leftMps: 0, turnRadps: -0.5 }, g);
    expect(left[1]).toBeGreaterThan(0.02);
    expect(left[0]).toBeLessThan(0.25);
    close(right, [left[0], -left[1], -left[2]], 9);
  });

  it('picks constant speeds and the shortest duration that end at a target, arcs included', () => {
    const r = DEFAULT_WALK_RANGES;
    const within = (c: { forwardMps: number; leftMps: number; turnRadps: number }) =>
      (['forwardMps', 'leftMps', 'turnRadps'] as const).every((k) => c[k] >= TARGET_SPEED_FRACTION * r[k][0] && c[k] <= TARGET_SPEED_FRACTION * r[k][1]);
    const targets: Vec3[] = [[0.3, 0, 0], [0, -0.15, 0], [0, 0, Math.PI / 2], [0.3, 0, Math.PI / 2], [-0.1, 0.1, -1], [0.05, 0.02, 3]];
    for (const target of targets) {
      const plan = walkForTarget(target, r, 0.5);
      if ('problem' in plan) throw new Error(plan.problem);
      expect(within(plan.command)).toBe(true);
      // The 4-decimal speeds still end within a millimetre and a milliradian of the target.
      const end = expectedPose(plan.command, { cycleS: 0.5, durationS: plan.durationS });
      expect(Math.hypot(end[0] - target[0], end[1] - target[1])).toBeLessThan(1e-3);
      expect(Math.abs(end[2] - target[2])).toBeLessThan(1e-3);
      // A tenth of a second less would need a speed outside the allowed share of a range (straight walks scale with the time at full speed).
      if (plan.durationS > 1 && target[2] === 0) {
        const scale = (plan.durationS - 0.5) / (plan.durationS - 0.6);
        expect(within({ forwardMps: plan.command.forwardMps * scale, leftMps: plan.command.leftMps * scale, turnRadps: 0 })).toBe(false);
      }
    }
    // 30 cm forward at up to 0.12 m/s takes 2.5 s at full speed plus the fades.
    expect(walkForTarget([0.3, 0, 0], r, 0.5)).toEqual({ command: { forwardMps: 0.12, leftMps: 0, turnRadps: 0 }, durationS: 3 });
    // Turning while walking curves the path, so reaching a point straight ahead while turning needs a side step to the right.
    const arc = walkForTarget([0.3, 0, Math.PI / 2], r, 0.5);
    if ('problem' in arc) throw new Error(arc.problem);
    expect(arc.command.leftMps).toBeLessThan(0);
    expect(arc.command.turnRadps).toBeGreaterThan(0);
    expect(walkForTarget([2, 0, 0], r, 0.5)).toEqual({ problem: 'reaching it takes longer than 10 s at the speeds this controller was trained for; walk part of the way and call locomote again from there.' });
    expect(walkForTarget([Number.NaN, 0, 0], r, 0.5)).toEqual({ problem: 'the target must be finite.' });
  });

  it('judges a walk by where the body ends, within a quarter of the commanded travel and turn', () => {
    expect(judgeWalk([0.25, 0, 0], [0.2, 0.03, 0.1], true)).toMatchObject({ success: true, toleranceM: 0.0625, toleranceRad: 0.15 });
    expect(judgeWalk([0.25, 0, 0], [0.2, 0.03, 0.1], false).success).toBe(false);
    expect(judgeWalk([0, 0, 1], [0.01, 0, 0.7], true)).toMatchObject({ success: false, toleranceRad: 0.25 });
    expect(rangeProblem({ forwardMps: 0.3, leftMps: 0, turnRadps: 0 }, DEFAULT_WALK_RANGES, 3)).toBe('forwardMps must be from -0.12 to 0.15 (the range this controller was trained for).');
    expect(rangeProblem({ forwardMps: 0.1, leftMps: 0, turnRadps: 0 }, DEFAULT_WALK_RANGES, 12)).toBe('durationS must be from 1 to 10 s.');
    expect(rangeProblem({ forwardMps: 0.1, leftMps: 0, turnRadps: 0 }, DEFAULT_WALK_RANGES, 3)).toBeNull();
  });

  it('evaluates the six single-axis commands first, then random ones within the ranges', () => {
    const q = (v: number) => Number((0.75 * v).toFixed(3));
    const r = DEFAULT_WALK_RANGES;
    expect([0, 1, 2, 3, 4, 5].map((k) => evaluationCommand(r, 1, k))).toEqual([
      { forwardMps: q(r.forwardMps[1]), leftMps: 0, turnRadps: 0 }, { forwardMps: q(r.forwardMps[0]), leftMps: 0, turnRadps: 0 }, { forwardMps: 0, leftMps: q(r.leftMps[1]), turnRadps: 0 },
      { forwardMps: 0, leftMps: q(r.leftMps[0]), turnRadps: 0 }, { forwardMps: 0, leftMps: 0, turnRadps: q(r.turnRadps[1]) }, { forwardMps: 0, leftMps: 0, turnRadps: q(r.turnRadps[0]) },
    ]);
    const c = evaluationCommand(DEFAULT_WALK_RANGES, 1, 6);
    expect(c).toEqual(sampleCommand(DEFAULT_WALK_RANGES, 1, 6));
    expect(rangeProblem(c, DEFAULT_WALK_RANGES, 3)).toBeNull();
  });

  it('walks only legged bodies, and uses one fewer worker than processors, at most four', () => {
    expect(base).toMatchObject({ ...DEFAULT_GAIT, strideM: 0, lateralM: 0, turnRad: 0, durationS: 3 });
    expect(() => locomotionBase(PRESETS.find((p) => p.id === 'arm3')!.make())).toThrow('fewer than two legs');
    expect([undefined, 1, 2, 8, 32].map(defaultWorkerCount)).toEqual([1, 1, 1, 4, 4]);
  });
});

/** Runs worker jobs in this thread with the handler the real worker uses. */
class InProcessWorker {
  onmessage: ((e: MessageEvent<SkillWorkerMessage>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  postMessage(req: SkillWorkerRequest): void {
    void handleSkillJob(() => Promise.resolve(mj), req, (m) => this.onmessage?.({ data: m } as MessageEvent<SkillWorkerMessage>));
  }
  terminate(): void {}
}

describe('training a walking controller (MuJoCo)', () => {
  const small: LocomotionConfig = { ...DEFAULT_LOCOMOTION, iterations: 2, directions: 2, top: 1, commandsPerCandidate: 1, heldOutTrials: 2, evaluationTrials: 7, episodeS: 2 };

  it('calibrates the generator, trains and judges the policy, and evaluates the result, the same on a pool of workers', async () => {
    const r = await trainLocomotion(inProcessRunner(mj), dog, stance, small);
    // 18 calibration runs, 2 iterations of 2 direction pairs on 1 command, 2 held-out commands with and without the policy, 7 evaluation commands per variant.
    expect(r.rollouts).toBe(18 + 8 + 4 + 7 * (r.policy ? 3 : 2));
    expect(r.calibrationRuns).toHaveLength(18);
    // A linear policy from 18 inputs to the 3 command changes.
    expect(r.parameters).toBe(3 * 19);
    // The measured calibration: forward, sideways and turning commands each move the body their own way.
    expect(predictVelocity(r.calibration, [0.1, 0, 0])[0]).toBeGreaterThan(0.05);
    expect(predictVelocity(r.calibration, [-0.1, 0, 0])[0]).toBeLessThan(-0.05);
    expect(predictVelocity(r.calibration, [0, 0.06, 0])[1]).toBeGreaterThan(0.02);
    expect(predictVelocity(r.calibration, [0, 0, 0.4])[2]).toBeGreaterThan(0.2);
    const e = r.evaluation;
    expect(e.trials).toBe(7);
    expect(e.final.successes).toBeLessThanOrEqual(7);
    expect(e.final.wilson95[0]).toBeLessThanOrEqual(e.final.successes / 7);
    expect(r.policy === null ? e.final.successes === e.calibrated.successes : true).toBe(true);
    // The controller's sensor-only estimate (leg odometry and IMU heading) ends near ground truth.
    expect(e.odometry!.meanPositionErrorM).toBeLessThan(0.05);
    expect(e.odometry!.meanHeadingErrorRad).toBeLessThan(0.05);
    if (r.policy) {
      expect(r.heldOut.learned.meanReturn).toBeGreaterThan(r.heldOut.generator.meanReturn);
      expect(r.heldOut.learned.meanPositionErrorM).toBeLessThanOrEqual(r.heldOut.generator.meanPositionErrorM);
      expect(r.heldOut.learned.meanHeadingErrorRad).toBeLessThanOrEqual(r.heldOut.generator.meanHeadingErrorRad);
    }

    // A pool of two workers splits each batch and joins the outcomes in order: the result is the same.
    const pool = new RolloutPool(() => new SkillWorkerClient(() => new InProcessWorker() as unknown as Worker), 2);
    const pooled = await trainLocomotion(pool, dog, stance, small);
    pool.dispose();
    expect(pooled).toMatchObject({ ...r, workers: 2 });
    expect(pooled.calibration).toEqual(r.calibration);

    // Stored and read back; another body or a policy for other joints is refused.
    const hash = await morphologyHash(dog);
    const c = locomotionController(r, { robotId: dog.id, morphologyHash: hash }, small, 12.34, '2026-10-07T12:00:00.000Z');
    const back = parseLocomotionController(JSON.parse(JSON.stringify(c)));
    if (!back.ok) throw new Error(back.errors.join(' '));
    expect(back.value).toEqual(c);
    const withGains = locomotionController(r, { robotId: dog.id, morphologyHash: hash }, { ...small, gains: { kp: 30, kd: 0.8 } }, 12.34, '2026-10-07T12:00:00.000Z');
    expect(withGains.evaluation.conditions.gains).toEqual({ kp: 30, kd: 0.8 });
    expect(parseLocomotionController(JSON.parse(JSON.stringify(withGains)))).toEqual({ ok: true, value: withGains });
    expect(parseLocomotionController({ ...JSON.parse(JSON.stringify(withGains)), evaluation: { ...withGains.evaluation, conditions: { ...withGains.evaluation.conditions, gains: { kp: -1, kd: 0.8 } } } }).ok).toBe(false);
    expect(locomotionProblem(c, dog, hash)).toBeNull();
    const six = hexapod();
    expect(locomotionProblem(c, six, await morphologyHash(six))).toMatch(/^the walking controller was trained on morphology/);
    for (const bad of [{ ...c, calibration: c.calibration.slice(1) }, { ...c, ranges: { ...c.ranges, leftMps: [0.1, -0.1] } }, { ...c, robot: { ...c.robot, morphologyHash: 'x' } }, { ...c, evaluation: { ...c.evaluation, final: { ...c.evaluation.final, wilson95: [0, 1] } } }]) {
      expect(parseLocomotionController(JSON.parse(JSON.stringify(bad))).ok).toBe(false);
    }
    expect(parseLocomotionController({ ...JSON.parse(JSON.stringify(c)), policy: { layers: [12, 12], activation: 'tanh', weights: [1], inputMean: new Array(12).fill(0), inputStd: new Array(12).fill(1) } }).ok).toBe(false);
    // A schema 1 controller (D28) still walks with its calibration, but not with a policy trained for the earlier observation.
    const v2 = walkPolicy(new Float64Array(3 * 19), 12, new Array(18).fill(0), new Array(18).fill(1));
    expect(parseLocomotionController({ ...JSON.parse(JSON.stringify(c)), schemaVersion: 1, policy: null }).ok).toBe(true);
    expect(parseLocomotionController({ ...JSON.parse(JSON.stringify(c)), schemaVersion: 2, policy: v2 }).ok).toBe(true);
    expect(parseLocomotionController({ ...JSON.parse(JSON.stringify(c)), schemaVersion: 1, policy: v2 })).toEqual({
      ok: false, errors: ['locomotion.policy of a schema 1 controller was trained before the walking policy of D29; train the walking controller again.'],
    });
  });
});
