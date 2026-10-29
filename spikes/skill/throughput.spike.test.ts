// Throughput of MuJoCo WASM in one thread (Node in CI): physics steps per second, raw and inside gait trials.
// A measurement for docs/physical-ai-plan.md Phase 0, not a pass/fail claim.
import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, expect, it } from 'vitest';
import { pdControl } from '../../src/core/control/pd';
import { homeConfiguration } from '../../src/core/model/catalog';
import { hexapod, quadruped } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { MujocoRuntime } from '../../src/physics/runtime';
import { DEFAULT_EVALUATION, DEFAULT_SKILL_GAINS, evaluateSkill } from '../../src/skills/evaluation';
import { DEFAULT_GAIT } from '../../src/skills/gait';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

/** Physics steps per second standing under PD, sensors read every step. */
function raw(robot: RobotDescription, steps: number): number {
  const q0 = homeConfiguration(robot);
  const rt = MujocoRuntime.create(mj, robot, { initialQ: q0 });
  try {
    const gains = q0.map(() => DEFAULT_SKILL_GAINS);
    const continuous = robot.limbs.flatMap((l) => l.joints.map((j) => j.type === 'continuous'));
    const t0 = performance.now();
    for (let k = 0; k < steps; k++) rt.controlStep((s) => pdControl(gains, rt.layout.effort, { qCmd: q0, qMeas: s.q, qdMeas: s.qd, continuous }).tauApplied);
    return steps / ((performance.now() - t0) / 1000);
  } finally {
    rt.dispose();
  }
}

/** Physics steps per second inside full gait trials (controller, leg IK, rate limiter, PD, referee). */
function trials(robot: RobotDescription, n: number): { stepsPerS: number; wallS: number } {
  const durationS = 4;
  const t0 = performance.now();
  evaluateSkill(mj, robot, -1, homeConfiguration(robot), { kind: 'gait', gait: { ...DEFAULT_GAIT, durationS } }, { ...DEFAULT_EVALUATION, trials: n, seed: 1, baseMassRange: [robot.base.mass, robot.base.mass] }, { goals: [{ kind: 'upright', maxTiltRad: 0.6 }] });
  const wallS = (performance.now() - t0) / 1000;
  // Each trial runs the gait, then 0.5 s after it reports finished, at 2 ms per step.
  const steps = n * Math.ceil((durationS + 0.5) / 0.002);
  return { stepsPerS: steps / wallS, wallS };
}

it('measures physics steps per second, raw and in gait trials, on the quadruped and the hexapod', () => {
  const rows = [quadruped(), hexapod()].map((robot) => {
    raw(robot, 500);
    const t = trials(robot, 4);
    return { robot: robot.name, rawStepsPerS: Math.round(raw(robot, 10_000)), gaitTrialStepsPerS: Math.round(t.stepsPerS), gaitTrialsWallS: +t.wallS.toFixed(2) };
  });
  console.table(rows);
  console.log('Throughput', JSON.stringify(rows));
  expect(rows.every((r) => r.rawStepsPerS > 0 && r.gaitTrialStepsPerS > 0)).toBe(true);
});
