import type { RobotDescription } from '../../src/core/model/types';
import type { PolicyJson } from '../../src/skills/controllers';
import { wilson95 } from '../../src/skills/evaluation';
import { DEFAULT_WALK_RANGES, locomotionBase, type LocomotionController, type LocomotionResult, type LocomotionScore, type WalkScore } from '../../src/skills/locomotion';

const score: WalkScore = { trials: 40, successes: 40, falls: 0, meanPositionErrorM: 0.01, meanHeadingErrorRad: 0.01 };
const runs: LocomotionScore = { meanReturn: 1, trackingErrorMps: 0, yawRateErrorRadps: 0, falls: 0, meanPositionErrorM: 0.01, meanHeadingErrorRad: 0.01 };

/**
 * A walking controller for tests, not a trained one: its calibration says
 * the generator walks `gain` times the speed it is given on every axis, and
 * its policy is the one given (or none).
 */
export function testWalkingController(robot: RobotDescription, morphologyHash: string, gain = 1, policy: PolicyJson | null = null): LocomotionController {
  return {
    kind: 'morphlab.locomotion',
    schemaVersion: 2,
    robot: { robotId: robot.id, morphologyHash },
    createdAt: '2026-10-16T12:00:00.000Z',
    base: locomotionBase(robot, 3),
    ranges: structuredClone(DEFAULT_WALK_RANGES),
    calibration: [gain, gain, 0, 0, 0, 0, 0, 0, gain, gain, 0, 0, 0, 0, 0, 0, gain, gain],
    policy,
    training: { algorithm: 'test fixture', rollouts: 0, workers: 1, iterations: 0, parameters: 0, calibrationRuns: 0, wallTimeS: 0, seed: 1, heldOut: { trials: 0, generator: runs, learned: runs } },
    evaluation: { seed: 1, trials: 40, durationS: 3, final: { ...score, wilson95: wilson95(40, 40) }, calibrated: score, uncalibrated: score, conditions: { frictionRange: [0.6, 1.2], baseMassRange: [3.5, 4.5], jointNoiseRad: 0.02 } },
  };
}

/** The training result that goes with testWalkingController, for code that reports a training. */
export function testLocomotionResult(robot: RobotDescription): LocomotionResult {
  const c = testWalkingController(robot, '0'.repeat(64));
  const e = c.evaluation;
  const evaluation = { seed: e.seed, trials: e.trials, durationS: e.durationS, final: e.final, calibrated: e.calibrated, uncalibrated: e.uncalibrated };
  return { base: c.base, ranges: c.ranges, calibration: c.calibration, calibrationRuns: [], policy: null, heldOut: c.training.heldOut, evaluation, history: [], rollouts: 0, workers: 1, parameters: 0 };
}
