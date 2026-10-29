// Shared measurement of the walking controller (D28, D29) for the per-body spikes: default settings, one thread.
// Then the trained controller walks two Navigate maps sensor-only (D38).
import type { MainModule } from '@mujoco/mujoco';
import { homeConfiguration } from '../../src/core/model/catalog';
import { morphologyHash } from '../../src/core/model/hash';
import type { RobotDescription } from '../../src/core/model/types';
import { TEST_MAPS } from '../../src/nav/maps';
import { runToEnd } from '../../src/physics/experiments';
import { runNavigation } from '../../src/physics/navigation';
import { DEFAULT_SKILL_GAINS, evaluationOptionsFor } from '../../src/skills/evaluation';
import { DEFAULT_LOCOMOTION, locomotionController, trainLocomotion, type WalkScore } from '../../src/skills/locomotion';
import { inProcessRunner } from '../../src/skills/rollouts';
import { WALK_COMMAND_OUTPUTS, WALK_POLICY_INPUTS } from '../../src/skills/walkPolicy';

export async function measureLocomotion(mj: MainModule, robot: RobotDescription, jointCorrections = false) {
  const o = evaluationOptionsFor(robot, 2026, DEFAULT_SKILL_GAINS);
  const cfg = { ...DEFAULT_LOCOMOTION, frictionRange: o.frictionRange, baseMassRange: o.baseMassRange, jointNoiseRad: o.jointNoiseRad, gains: o.gains, jointCorrections };
  const t0 = performance.now();
  const r = await trainLocomotion(inProcessRunner(mj), robot, homeConfiguration(robot), cfg);
  const wallS = (performance.now() - t0) / 1000;
  const score = (s: WalkScore) => ({ successes: s.successes, falls: s.falls, posErrM: +s.meanPositionErrorM.toFixed(3), headErrRad: +s.meanHeadingErrorRad.toFixed(3) });
  const h = r.heldOut;
  // Per-unit feedback gains of the trained policy from the error inputs to the command outputs (weight over input scale).
  let gains: Record<string, number> | null = null;
  const p = r.policy ?? null;
  const W = (out: number, inp: number, pol: { weights: number[]; inputStd: number[] }) => pol.weights[out * WALK_POLICY_INPUTS + inp]! / pol.inputStd[inp]!;
  if (p) {
    const c = p.layers[1]! - WALK_COMMAND_OUTPUTS;
    gains = {
      strideFromForwardErr: +W(c, 13, p).toFixed(2), lateralFromLeftErr: +W(c + 1, 14, p).toFixed(2), turnFromHeadingErr: +W(c + 2, 12, p).toFixed(2),
      strideFromSpeedErr: +W(c, 15, p).toFixed(2), lateralFromSideSpeedErr: +W(c + 1, 16, p).toFixed(2),
    };
  }
  const row = {
    robot: robot.name, variant: cfg.jointCorrections ? 'command and joint corrections' : 'command only', rollouts: r.rollouts, wallS: +wallS.toFixed(1), parameters: r.parameters, policyKept: r.policy !== null,
    calibration: r.calibration.map((v) => +v.toFixed(3)),
    heldOut: {
      trials: h.trials, generatorReturn: +h.generator.meanReturn.toFixed(3), learnedReturn: +h.learned.meanReturn.toFixed(3), generatorFalls: h.generator.falls, learnedFalls: h.learned.falls,
      generatorPosErrM: +h.generator.meanPositionErrorM.toFixed(3), learnedPosErrM: +h.learned.meanPositionErrorM.toFixed(3),
      generatorHeadErrRad: +h.generator.meanHeadingErrorRad.toFixed(3), learnedHeadErrRad: +h.learned.meanHeadingErrorRad.toFixed(3),
    },
    returns: r.history.filter((_, i) => i % 10 === 0 || i === r.history.length - 1).map((x) => +x.meanReturn.toFixed(3)),
    gains,
    evaluation: {
      trials: r.evaluation.trials, final: score(r.evaluation.final), wilson95: r.evaluation.final.wilson95.map((v) => +v.toFixed(3)),
      calibrated: score(r.evaluation.calibrated), uncalibrated: score(r.evaluation.uncalibrated),
      odometry: r.evaluation.odometry ? { posErrM: +r.evaluation.odometry.meanPositionErrorM.toFixed(3), headErrRad: +r.evaluation.odometry.meanHeadingErrorRad.toFixed(3) } : null,
    },
  };
  console.log('Locomotion', JSON.stringify(row));
  const controller = locomotionController(r, { robotId: robot.id, morphologyHash: await morphologyHash(robot) }, cfg, wallS);
  const navigation = ['open', 'block'].map((id) => {
    const n = runToEnd(runNavigation(mj, TEST_MAPS.find((m) => m.id === id)!, { mode: 'sensor-only', robot, walking: controller }));
    return { map: id, outcome: n.outcome, timeS: n.timeS, finalDistanceM: n.finalDistanceM, estimateErrorM: n.estimateErrorM, replans: n.replans };
  });
  console.log('Navigate with the learned walking controller', robot.name, row.variant, JSON.stringify(navigation));
  return { ...row, navigation };
}
