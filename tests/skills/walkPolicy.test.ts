import loadMujoco, { type MainModule } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { buildKinematicModel, clampToLimits } from '../../src/core/kinematics/fk';
import type { Quat } from '../../src/core/math/mat4';
import type { Vec3 } from '../../src/core/math/vec3';
import { quadruped, quadrupedStance } from '../../src/core/model/presets';
import { makeController } from '../../src/skills/controllers';
import { trialConditions } from '../../src/skills/evaluation';
import { GaitController, gaitEnvelope, type GaitParams } from '../../src/skills/gait';
import { DEFAULT_LOCOMOTION, locomotionBase, walkGait, walkPolicy } from '../../src/skills/locomotion';
import { RESIDUAL_SCALE_RAD } from '../../src/skills/policyGait';
import { gaitPolicyRollout } from '../../src/skills/train';
import { WALK_COMMAND_OUTPUTS, WALK_POLICY_INPUTS, walkPolicyProblem, WalkPolicyController } from '../../src/skills/walkPolicy';

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

const dog = quadruped();
const stance = quadrupedStance();
const km = buildKinematicModel(dog);
const J = stance.length;
const zeros = (n: number) => new Array<number>(n).fill(0);
const ones = (n: number) => new Array<number>(n).fill(1);
const obs = (imuQuat: Quat = [1, 0, 0, 0], gyro: Vec3 = [0, 0, 0]) => ({ q: stance, qd: zeros(J), touch: [5, 5, 5, 5], imuQuat, gyro });
/** A walking policy with joint corrections whose outputs are `joint` on every joint and `command` on the stride, side step and turn outputs, whatever it observes. */
const constant = (joint: number, command: Vec3 = [0, 0, 0]) => {
  const outputs = J + WALK_COMMAND_OUTPUTS;
  const theta = new Float64Array(outputs * (WALK_POLICY_INPUTS + 1));
  for (let i = 0; i < outputs; i++) theta[outputs * WALK_POLICY_INPUTS + i] = i < J ? joint : command[i - J]!;
  return walkPolicy(theta, J, zeros(WALK_POLICY_INPUTS), ones(WALK_POLICY_INPUTS), true);
};
/** A command-only walking policy with constant outputs. */
const commandOnly = (command: Vec3) => {
  const theta = new Float64Array(WALK_COMMAND_OUTPUTS * (WALK_POLICY_INPUTS + 1));
  command.forEach((v, k) => (theta[WALK_COMMAND_OUTPUTS * WALK_POLICY_INPUTS + k] = v));
  return walkPolicy(theta, J, zeros(WALK_POLICY_INPUTS), ones(WALK_POLICY_INPUTS));
};
const command: Vec3 = [0.1, 0, 0];
const forward: GaitParams = walkGait(locomotionBase(dog, 3), null, { forwardMps: 0.1, leftMps: 0, turnRadps: 0 }, 3).gait;

describe('walking policy (D29)', () => {
  it('observes the IMU, the phase, the command and how far the estimate is from where the command leads (18 inputs)', () => {
    const first = new WalkPolicyController(km, stance, forward, command, null);
    first.update(0, obs([1, 0, 0, 0], [0.1, -0.2, 0.3]));
    expect(Array.from(first.observation.slice(2, 5))).toEqual([0.1, -0.2, 0.3]);
    const w = new WalkPolicyController(km, stance, forward, command, null);
    const roll = 0.1;
    const tilted: Quat = [Math.cos(roll / 2), Math.sin(roll / 2), 0, 0];
    for (let k = 0; k <= 100; k++) w.update(k / 100, obs(tilted));
    const o = Array.from(w.observation);
    expect(o).toHaveLength(WALK_POLICY_INPUTS);
    expect(o[0]).toBeCloseTo(0, 12);
    expect(o[1]).toBeCloseTo(Math.sin(roll), 12);
    // 1 s is two cycles: phase 0.
    expect(o[5]).toBeCloseTo(0, 9);
    expect(o[6]).toBeCloseTo(1, 9);
    expect(o.slice(9, 12)).toEqual(command);
    // The feet never moved, so the estimate stayed at the start while the command led 0.075 m ahead (0.1 m/s for 1 s, less half the 0.5 s fade-in).
    expect(w.estimate.map(Math.abs)).toEqual([0, 0, 0]);
    expect(o[12]).toBeCloseTo(0, 12);
    expect(o[13]).toBeCloseTo(0.075, 4);
    expect(o[14]).toBeCloseTo(0, 12);
    expect(o[15]).toBeCloseTo(0.1, 12);
    expect(o[16]).toBeCloseTo(0, 12);
    expect(o[17]).toBe(1);
  });

  it('is the gait generator exactly without a policy, and with a zero policy', () => {
    const tg = [new GaitController(km, stance, forward), new GaitController(km, stance, forward)];
    const alone = new WalkPolicyController(km, stance, forward, command, null);
    const zero = new WalkPolicyController(km, stance, forward, command, constant(0));
    const tilted: Quat = [Math.cos(0.03), 0.02, Math.sin(0.03), 0];
    for (const t of [0, 0.25, 0.8, 1.3, 2.2, 3.05]) {
      const o = obs(tilted, [0.2, -0.1, 0.05]);
      expect(alone.update(t, o)).toEqual(tg[0]!.update(t, o));
      expect(zero.update(t, o)).toEqual(tg[1]!.update(t, o));
    }
    expect(zero.finished).toBe(true);
    expect(Array.from(zero.action)).toEqual(zeros(J + WALK_COMMAND_OUTPUTS));
    expect(alone.phase).toBe('done');
  });

  it('clips its outputs and moves each joint by at most 0.15 rad from the generator, faded with the steps', () => {
    const tg = new GaitController(km, stance, forward);
    const w = new WalkPolicyController(km, stance, forward, command, constant(10));
    for (const t of [0, 0.25, 1, 2.75, 3]) {
      const base = tg.update(t, obs());
      const want = clampToLimits(km, base.map((v) => v + gaitEnvelope(forward, t) * RESIDUAL_SCALE_RAD));
      w.update(t, obs()).forEach((v, i) => expect(v).toBeCloseTo(want[i]!, 12));
    }
    expect(Array.from(w.action)).toEqual([...ones(J), 0, 0, 0]);
    expect(w.phase).toBe('coming to a stop with learned feedback');
  });

  it('hands the walk to the generator for the rest of the run when the policy output is not finite', () => {
    const tg = new GaitController(km, stance, forward);
    const w = new WalkPolicyController(km, stance, forward, command, constant(0.5, [1, 1, 1]));
    const broken = obs([1, 0, 0, 0], [Number.NaN, 0, 0]);
    expect(w.update(1, broken)).toEqual(tg.update(1, broken));
    expect(w.phase).toBe('gait generator only (the policy output was not finite)');
    expect(w.update(1.1, obs())).toEqual(tg.update(1.1, obs()));
    expect(Array.from(w.action)).toEqual(zeros(J + WALK_COMMAND_OUTPUTS));
  });

  it('refuses a policy that does not fit the robot, and runs from a learned-gait spec with a walking command', () => {
    expect(walkPolicyProblem(constant(0), J)).toBeNull();
    expect(walkPolicyProblem(commandOnly([0, 0, 0]), J)).toBeNull();
    const six = walkPolicy(new Float64Array(21 * 19), 18, zeros(18), ones(18), true);
    expect(walkPolicyProblem(six, J)).toBe('the policy maps 18 inputs to 21 outputs; a walking policy on this robot needs 18 inputs and 3 or 15 outputs.');
    expect(() => new WalkPolicyController(km, stance, forward, command, six)).toThrow('The policy maps 18 inputs to 21 outputs');
    expect(walkPolicyProblem({ ...constant(0), inputStd: [...ones(17), 0] }, J)).toBe('the policy normalization does not match its inputs.');
    expect(() => walkPolicy(new Float64Array(5), J, zeros(18), ones(18), true)).toThrow('Expected 285 parameters, got 5');
    expect(() => walkPolicy(new Float64Array(5), J, zeros(18), ones(18))).toThrow('Expected 57 parameters, got 5');
    expect(makeController(km, -1, stance, { kind: 'gait-learned', gait: forward, policy: constant(0), walk: { command } })).toBeInstanceOf(WalkPolicyController);
  });

  it('leaves the joints to the generator when the policy changes the command only', () => {
    const tg = new GaitController(km, stance, forward);
    const w = new WalkPolicyController(km, stance, forward, command, commandOnly([0, 0, 0]));
    for (const t of [0, 0.5, 1.2]) expect(w.update(t, obs())).toEqual(tg.update(t, obs()));
    expect(w.action).toHaveLength(WALK_COMMAND_OUTPUTS);
    const turning = new WalkPolicyController(km, stance, forward, command, commandOnly([0, 0, 5]));
    const straight = new GaitController(km, stance, forward);
    let differs = false;
    for (let k = 0; k <= 120; k++) {
      const a = turning.update(k / 100, obs());
      const b = straight.update(k / 100, obs());
      differs ||= a.some((v, i) => Math.abs(v - b[i]!) > 1e-6);
    }
    // The turn output is clipped to 1 and reaches the generator through the filter.
    expect(Array.from(turning.action)).toEqual([0, 0, 1]);
    expect(differs).toBe(true);
  });
});

describe('walking policy and leg odometry (MuJoCo)', () => {
  const conditions = trialConditions({ ...DEFAULT_LOCOMOTION, seed: 3 }, 0);
  const input = { desc: dog, stanceQ: stance, gait: forward, conditions, noiseSeed: 5, walkCommand: command };

  it('estimates from the feet and the IMU where the body went, close to ground truth', () => {
    const alone = gaitPolicyRollout(mj, { ...input, policy: null });
    expect(alone.fell).toBe(false);
    expect(alone.forwardM).toBeGreaterThan(0.1);
    // Within a quarter of the distance plus 2 cm (the walk tolerance), and within 0.05 rad of the heading.
    const e = alone.odometry!;
    const distance = Math.hypot(alone.forwardM, alone.leftM);
    expect(Math.hypot(e[0] - alone.forwardM, e[1] - alone.leftM)).toBeLessThan(0.25 * distance + 0.02);
    expect(Math.abs(e[2] - alone.turnRad)).toBeLessThan(0.05);
    // A zero policy walks exactly like the generator alone.
    expect(gaitPolicyRollout(mj, { ...input, policy: constant(0) })).toEqual(alone);
  });

  it('walks further when the policy lengthens the stride (the command outputs reach the generator)', () => {
    const alone = gaitPolicyRollout(mj, { ...input, policy: null });
    const longer = gaitPolicyRollout(mj, { ...input, policy: commandOnly([0.5, 0, 0]) });
    expect(longer.fell).toBe(false);
    expect(longer.forwardM).toBeGreaterThan(alone.forwardM + 0.05);
  });
});
