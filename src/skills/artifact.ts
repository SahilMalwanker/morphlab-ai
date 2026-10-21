import type { ParseResult } from '../core/model/parse';
import type { RobotDescription } from '../core/model/types';
import { METHOD_LABELS, type PolicyJson, type SkillMethod, type SkillSpec, type Trajectory } from './controllers';
import { wilson95, type EvaluationContext, type EvaluationSummary } from './evaluation';
import { DEFAULT_FOOT_LIFT, footLiftParamProblems, type FootLiftParams } from './footLift';
import { gaitProblems, type GaitParams } from './gait';
import { GAIT_POLICY_INPUTS, gaitPolicyProblem } from './policyGait';
import { policyFeet, posePolicyProblem } from './posePolicy';
import { goalRobotProblems, goalsProblems, motionProblems, TAUGHT_NAME_PATTERN, type KeyframeMotion, type SkillGoal } from './taught';

/** An evaluation as stored with a skill (per-trial results are summarized into metrics). */
export interface StoredEvaluation {
  trials: number;
  successes: number;
  successRate: number;
  wilson95: [number, number];
  conditions: Record<string, [number, number]>;
  metrics: Record<string, number>;
  simulator: string;
  evaluatedAt: string;
  /** The setup needed to repeat it. Missing in skills evaluated before it was recorded. */
  context?: EvaluationContext;
}

/** Who composed a taught skill's keyframes, when it was not the user alone. */
export type TaughtAuthor = 'ai' | 'ai-edited';

export interface SkillArtifact {
  kind: 'morphlab.skill';
  schemaVersion: 1;
  id: string;
  name: string;
  /** The built-in foot lift, or a skill the user taught with keyframes and goals. */
  skillType: 'foot_lift' | 'taught';
  method: SkillMethod;
  createdAt: string;
  updatedAt: string;
  robot: { robotId: string; morphologyHash: string };
  /** Limbs the skill was evaluated on (foot lift), or the limbs its goals name (taught). */
  limbs: string[];
  parameters: Record<string, number>;
  policy: PolicyJson | null;
  /** Recorded joint targets for replay skills (decision D4 amendment). */
  trajectory: Trajectory | null;
  /** Posed keyframes of a taught skill. */
  motion?: KeyframeMotion | null;
  /** Contact-pattern gait of a taught skill. */
  gait?: GaitParams | null;
  /**
   * A taught skill's own name (commands use it) and the goals that define its
   * success. `author` is absent when the user posed it, 'ai' when the AI
   * composed it, 'ai-edited' when the user then edited the AI's draft.
   */
  taught?: { name: string; goals: SkillGoal[]; author?: TaughtAuthor };
  observation: { name: string; source: 'joint_encoder' | 'imu' | 'contact' | 'time' | 'command'; size: number; units: string }[];
  action: { name: string; type: 'joint_position_target'; units: 'rad'; size: number }[];
  training: null | { algorithm: string; config: Record<string, number | string>; seed: number; iterations: number; wallTimeS: number };
  evaluation: null | StoredEvaluation;
  /**
   * Where `evaluation` was measured: here ('local') or claimed by an imported
   * file ('imported'). Records saved before this field existed count as
   * imported when `importedAt` is set, otherwise as local.
   */
  evaluationOrigin?: 'local' | 'imported';
  /** What an imported file claimed, kept after the skill was re-evaluated here. */
  importedEvaluation?: StoredEvaluation;
  /** Set when the skill was loaded from a file. */
  importedAt?: string;
}

/** Where the skill's evaluation was measured. */
export function evaluationOrigin(a: SkillArtifact): 'local' | 'imported' {
  return a.evaluationOrigin ?? (a.importedAt ? 'imported' : 'local');
}

function observationFor(method: SkillMethod, joints: number, limbs: number, feet = 0): SkillArtifact['observation'] {
  if (method === 'replay' || method === 'keyframes') return [{ name: 'time', source: 'time', size: 1, units: 's' }];
  if (method === 'gait') return [{ name: 'base_orientation', source: 'imu', size: 4, units: 'quaternion' }, { name: 'time', source: 'time', size: 1, units: 's' }];
  if (method === 'gait-learned') {
    return [
      { name: 'gravity_direction_xy', source: 'imu', size: 2, units: '1' },
      { name: 'angular_rate', source: 'imu', size: 3, units: 'rad/s' },
      { name: 'gait_phase_sin_cos', source: 'time', size: 4, units: '1' },
      { name: 'velocity_command', source: 'command', size: 3, units: 'm/s, m/s, rad/s' },
    ];
  }
  if (method === 'keyframes-learned') {
    return [
      { name: 'gravity_direction_xy', source: 'imu', size: 2, units: '1' },
      { name: 'angular_rate', source: 'imu', size: 3, units: 'rad/s' },
      { name: 'foot_contact', source: 'contact', size: feet, units: 'binary' },
      { name: 'motion_progress', source: 'time', size: 2, units: '1' },
      { name: 'joint_position_minus_keyframe_target', source: 'joint_encoder', size: joints, units: 'rad' },
    ];
  }
  return [
    { name: 'joint_positions', source: 'joint_encoder', size: joints, units: 'rad' },
    { name: method === 'imitation-mlp' ? 'gravity_direction_xy' : 'base_orientation', source: 'imu', size: method === 'imitation-mlp' ? 2 : 4, units: method === 'imitation-mlp' ? '1' : 'quaternion' },
    { name: 'foot_contact', source: 'contact', size: limbs, units: method === 'imitation-mlp' ? 'binary' : 'N' },
    { name: 'time', source: 'time', size: 1, units: method === 'imitation-mlp' ? 'fraction' : 's' },
  ];
}

export function summarizeEvaluation(e: EvaluationSummary): StoredEvaluation {
  const ok = e.results.filter((r) => r.success);
  const mean = (f: (r: (typeof e.results)[number]) => number, rows = e.results) => (rows.length ? rows.reduce((s, r) => s + f(r), 0) / rows.length : Number.NaN);
  const failures: Record<string, number> = {};
  for (const r of e.results) for (const f of r.failures) failures[`failed:${f}`] = (failures[`failed:${f}`] ?? 0) + 1;
  // Taught skills: how many trials met each goal.
  const goalCounts: Record<string, number> = {};
  for (const r of e.results) r.goals?.forEach((g, i) => (goalCounts[`goalMet:${i}`] = (goalCounts[`goalMet:${i}`] ?? 0) + (g.met ? 1 : 0)));
  for (const r of e.results) r.checks?.forEach((c) => (goalCounts[`checkMet:${c.kind}`] = (goalCounts[`checkMet:${c.kind}`] ?? 0) + (c.met ? 1 : 0)));
  const taught = e.results.some((r) => r.goals);
  return {
    trials: e.trials,
    successes: e.successes,
    successRate: e.successRate,
    wilson95: e.wilson95,
    conditions: { friction: e.conditions.friction, baseMassKg: e.conditions.baseMassKg, jointNoiseRad: e.conditions.jointNoiseRad },
    // Means exist only when trial results do (never NaN in stored or exported data).
    metrics: {
      ...(e.results.length && taught ? { meanMaxTiltRad: +mean((r) => r.maxTiltRad).toFixed(4), meanMinBaseHeightM: +mean((r) => r.minBaseHeightM).toFixed(4), ...goalCounts } : {}),
      ...(e.results.length && !taught
        ? {
            meanClearHoldS: +mean((r) => r.longestClearHoldS).toFixed(3),
            meanMaxClearanceM: +mean((r) => r.maxClearanceM).toFixed(4),
            meanMaxTiltRad: +mean((r) => r.maxTiltRad).toFixed(4),
            meanClearHoldSuccessfulS: ok.length ? +mean((r) => r.longestClearHoldS, ok).toFixed(3) : 0,
          }
        : {}),
      ...failures,
    },
    simulator: e.simulator,
    evaluatedAt: e.evaluatedAt,
    ...(e.context ? { context: e.context } : {}),
  };
}

export interface CreateSkillInput {
  robot: RobotDescription;
  morphologyHash: string;
  limbId: string;
  spec: SkillSpec;
  name: string;
  evaluation: EvaluationSummary;
  training?: SkillArtifact['training'];
  /** Set for a taught skill: its own name and goals, and who composed it when it was not the user alone. */
  taught?: { name: string; goals: SkillGoal[]; author?: TaughtAuthor };
  now?: string;
  id?: string;
}

export function createSkillArtifact(input: CreateSkillInput): SkillArtifact {
  const now = input.now ?? new Date().toISOString();
  const joints = input.robot.limbs.reduce((n, l) => n + l.joints.length, 0);
  const spec = input.spec;
  const parameters: Record<string, number> = 'params' in spec ? { ...(spec.params as unknown as Record<string, number>) } : spec.kind === 'imitation-mlp' ? { durationS: spec.durationS } : {};
  const taught = input.taught
    ? { name: input.taught.name, goals: input.taught.goals.map((g) => ({ ...g })), ...(input.taught.author ? { author: input.taught.author } : {}) }
    : null;
  return {
    kind: 'morphlab.skill',
    schemaVersion: 1,
    id: input.id ?? `skill_${now.replace(/\D/g, '').slice(0, 17)}_${Math.floor(Math.random() * 1e6).toString(36)}`,
    name: input.name,
    skillType: taught ? 'taught' : 'foot_lift',
    method: spec.kind,
    createdAt: now,
    updatedAt: now,
    robot: { robotId: input.robot.id, morphologyHash: input.morphologyHash },
    limbs: taught ? [...new Set(taught.goals.flatMap((g) => (g.kind === 'tipHeight' ? [g.limbId] : [])))] : [input.limbId],
    parameters,
    policy: spec.kind === 'imitation-mlp' || spec.kind === 'gait-learned' || spec.kind === 'keyframes-learned' ? spec.policy : null,
    trajectory: spec.kind === 'replay' ? spec.trajectory : null,
    motion: spec.kind === 'keyframes' || spec.kind === 'keyframes-learned' ? spec.motion : null,
    gait: spec.kind === 'gait' || spec.kind === 'gait-learned' ? { ...spec.gait } : null,
    ...(taught ? { taught } : {}),
    observation: observationFor(spec.kind, joints, input.robot.limbs.length, policyFeet(input.robot).length),
    action: [{ name: 'joint_targets', type: 'joint_position_target', units: 'rad', size: joints }],
    training: input.training ?? null,
    evaluation: summarizeEvaluation(input.evaluation),
    evaluationOrigin: 'local',
  };
}

/** Rebuilds the executable controller description from a stored skill. */
export function specFromArtifact(a: SkillArtifact): SkillSpec {
  switch (a.method) {
    case 'replay':
      if (!a.trajectory) throw new Error('Replay skill has no trajectory');
      return { kind: 'replay', trajectory: a.trajectory };
    case 'imitation-mlp':
      if (!a.policy) throw new Error('Imitation skill has no policy');
      if (!(a.parameters.durationS! > 0)) throw new Error('Imitation skill has no duration');
      return { kind: 'imitation-mlp', policy: a.policy, durationS: a.parameters.durationS! };
    case 'keyframes':
      if (!a.motion) throw new Error('Keyframe skill has no keyframes');
      return { kind: 'keyframes', motion: a.motion };
    case 'keyframes-learned':
      if (!a.motion || !a.policy) throw new Error('Learned keyframe skill has no keyframes or no policy');
      return { kind: 'keyframes-learned', motion: a.motion, policy: a.policy };
    case 'gait':
      if (!a.gait) throw new Error('Gait skill has no gait');
      return { kind: 'gait', gait: a.gait };
    case 'gait-learned':
      if (!a.gait || !a.policy) throw new Error('Learned gait skill has no gait or no policy');
      return { kind: 'gait-learned', gait: a.gait, policy: a.policy };
    default:
      return { kind: a.method, params: { ...DEFAULT_FOOT_LIFT, ...(a.parameters as Partial<FootLiftParams>) } };
  }
}

export type Compatibility = { ok: true } | { ok: false; reason: string };

/** Size of the imitation policy's input for a robot (see footLiftFeatures). */
export function policyInputSize(robot: RobotDescription): number {
  const joints = robot.limbs.reduce((n, l) => n + l.joints.length, 0);
  return joints + 2 + robot.limbs.length + 1;
}

/**
 * A skill runs only on the morphology it was learned and evaluated on, only
 * on a limb it was evaluated on (decision D4), and only when its policy or
 * recording fits this robot's joints. Every refusal names the problem.
 */
export function checkSkillCompatibility(a: SkillArtifact, robot: RobotDescription, morphologyHash: string, limbId?: string): Compatibility {
  if (a.robot.morphologyHash !== morphologyHash) {
    return {
      ok: false,
      reason: `"${a.name}" was learned on morphology ${a.robot.morphologyHash.slice(0, 12)}; the current robot "${robot.name}" is ${morphologyHash.slice(0, 12)}. A skill is never reused on a different body; evaluate or train it again on this robot.`,
    };
  }
  const jointIds = robot.limbs.flatMap((l) => l.joints.map((j) => j.id));
  if (a.skillType === 'taught') {
    const m = a.motion;
    if ((a.method === 'keyframes' || a.method === 'keyframes-learned') && !m) return { ok: false, reason: `"${a.name}" is a keyframe skill without keyframes.` };
    if (m && (m.jointIds.length !== jointIds.length || m.jointIds.some((id, i) => id !== jointIds[i]))) {
      return { ok: false, reason: `The keyframes in "${a.name}" are for joints ${m.jointIds.slice(0, 3).join(', ')}..., not this robot's ${jointIds.length} joints in order; pose them again on this robot.` };
    }
    if (a.method === 'gait' || a.method === 'gait-learned') {
      const g = gaitProblems(a.gait, robot);
      if (g.length) return { ok: false, reason: `"${a.name}" cannot step on this robot: ${g[0]}` };
    }
    if (a.method === 'gait-learned') {
      const p = a.policy ? gaitPolicyProblem(a.policy, jointIds.length) : 'it has no policy.';
      if (p) return { ok: false, reason: `The learned feedback in "${a.name}" does not fit this robot: ${p} Train it again on this robot.` };
    }
    if (a.method === 'keyframes-learned') {
      const p = a.policy ? posePolicyProblem(a.policy, robot) : 'it has no policy.';
      if (p) return { ok: false, reason: `The learned feedback in "${a.name}" does not fit this robot: ${p} Train it again on this robot.` };
    }
    const problems = goalRobotProblems(a.taught?.goals ?? [], robot);
    return problems.length ? { ok: false, reason: `"${a.name}" cannot be judged on this robot: ${problems[0]}` } : { ok: true };
  }
  const limb = limbId ?? a.limbs[0];
  if (!limb || !a.limbs.includes(limb)) {
    return { ok: false, reason: `"${a.name}" was evaluated on ${a.limbs.join(', ')}, not on ${limbId}.` };
  }
  const robotLimb = robot.limbs.find((l) => l.id === limb);
  if (!robotLimb) return { ok: false, reason: `The robot has no limb "${limb}".` };
  if (robotLimb.role !== 'leg' || !(robotLimb.tip.radius > 0)) return { ok: false, reason: `"${limb}" is not a leg with a foot, so it cannot be lifted.` };
  if (a.method === 'replay') {
    const t = a.trajectory;
    if (!t) return { ok: false, reason: `"${a.name}" is a replay skill without a recording.` };
    if (t.jointIds.length !== jointIds.length || t.jointIds.some((id, i) => id !== jointIds[i])) {
      return { ok: false, reason: `The recording in "${a.name}" is for joints ${t.jointIds.slice(0, 3).join(', ')}..., not this robot's ${jointIds.length} joints in order; record it again on this robot.` };
    }
  }
  if (a.method === 'imitation-mlp') {
    const p = a.policy;
    if (!p) return { ok: false, reason: `"${a.name}" is an imitation skill without a policy.` };
    const inputs = policyInputSize(robot);
    if (p.layers[0] !== inputs || p.layers[p.layers.length - 1] !== jointIds.length) {
      return { ok: false, reason: `The policy in "${a.name}" maps ${p.layers[0]} inputs to ${p.layers[p.layers.length - 1]} outputs; this robot needs ${inputs} inputs and ${jointIds.length} outputs. Train it again on this robot.` };
    }
  }
  return { ok: true };
}

export function methodLabel(m: SkillMethod): string {
  return METHOD_LABELS[m];
}

const AUTHOR_TEXT: Record<'user' | TaughtAuthor, string> = { user: 'you set', ai: 'the AI composed', 'ai-edited': 'the AI composed and you edited' };

/**
 * The method label of a stored skill. Taught skills say who composed them,
 * whether trial and error improved them, and whether they use feedback.
 */
export function skillLabel(a: Pick<SkillArtifact, 'method' | 'taught' | 'training' | 'gait'>): string {
  if (a.method === 'gait-learned') return `Gait ${AUTHOR_TEXT[a.taught?.author ?? 'user']}, with a feedback policy learned by random search (closed loop)`;
  if (a.method === 'keyframes-learned') return `Keyframes ${AUTHOR_TEXT[a.taught?.author ?? 'user']}, with a feedback policy learned by reinforcement learning under pushes (closed loop)`;
  if (a.method !== 'keyframes' && a.method !== 'gait') return METHOD_LABELS[a.method];
  const author = a.taught?.author ?? 'user';
  const who = a.method === 'keyframes' && author === 'user' ? 'you posed' : AUTHOR_TEXT[author];
  const improved = a.training ? ', improved by trial and error (evolution strategies)' : '';
  const feedback = a.method === 'gait' && !!a.gait && (a.gait.feedbackKp > 0 || a.gait.feedbackKd > 0);
  return `${a.method === 'keyframes' ? 'Keyframe motion' : 'Gait'} ${who}${improved} (${feedback ? 'closed loop, IMU feedback' : 'open loop'})`;
}

const METHODS: SkillMethod[] = ['state-machine', 'tuned-parameters', 'replay', 'imitation-mlp', 'es-improved'];
const TAUGHT_METHODS: SkillMethod[] = ['keyframes', 'gait', 'gait-learned', 'keyframes-learned'];
const KNOWN_FIELDS = ['kind', 'schemaVersion', 'id', 'name', 'skillType', 'method', 'createdAt', 'updatedAt', 'robot', 'limbs', 'parameters', 'policy', 'trajectory', 'motion', 'gait', 'taught', 'observation', 'action', 'training', 'evaluation', 'evaluationOrigin', 'importedEvaluation', 'importedAt'] as const;
/**
 * Upper bounds that keep imported recordings and policies within what the
 * simulator handles. Demonstration recording stops at MAX_REPLAY_S, so every
 * recording made here stays valid.
 */
export const MAX_REPLAY_S = 600;
const MAX_IMITATION_S = 30;

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const finiteList = (v: unknown): v is number[] => Array.isArray(v) && v.every(finite);
const range = (v: unknown): v is [number, number] => finiteList(v) && v.length === 2 && v[0]! <= v[1]!;

/** Problems with the reproducibility record of an evaluation (absent in older skills, which is allowed). */
function contextProblems(c: unknown, where: string): string[] {
  if (c === undefined) return [];
  if (!isObject(c)) return [`${where}.context must be an object.`];
  const out: string[] = [];
  const cr = c.criteria;
  if (c.goals !== undefined) out.push(...goalsProblems(c.goals, `${where}.context.goals`));
  else if (!isObject(cr) || !['minClearanceM', 'minHoldS', 'maxTiltRad', 'minBaseHeightM', 'minSupportContactFraction'].every((k) => finite(cr[k]))) out.push(`${where}.context.criteria must list the five success thresholds.`);
  const g = c.gains;
  if (!isObject(g) || !finite(g.kp) || !finite(g.kd) || (g.kp as number) < 0 || (g.kd as number) < 0) out.push(`${where}.context.gains must hold non-negative kp and kd.`);
  if (!Number.isInteger(c.seed)) out.push(`${where}.context.seed must be an integer.`);
  if (!finiteList(c.stanceQ)) out.push(`${where}.context.stanceQ must be a list of finite joint positions.`);
  if (!range(c.frictionRange) || !(c.frictionRange[0] > 0)) out.push(`${where}.context.frictionRange must be a positive [low, high] range.`);
  if (!range(c.baseMassRange) || !(c.baseMassRange[0] > 0)) out.push(`${where}.context.baseMassRange must be a positive [low, high] range.`);
  if (!finite(c.jointNoiseRad) || c.jointNoiseRad < 0) out.push(`${where}.context.jointNoiseRad must be a non-negative number.`);
  if (!finite(c.nominalBaseMassKg) || !(c.nominalBaseMassKg > 0) || !finite(c.timestepS) || !(c.timestepS > 0) || !finite(c.controlPeriodS) || !(c.controlPeriodS > 0)) {
    out.push(`${where}.context must record the nominal base mass, the timestep and the control period.`);
  }
  if (typeof c.limbId !== 'string' || typeof c.simulator !== 'string' || typeof c.velocityLimited !== 'boolean') out.push(`${where}.context must record the limb, the simulator and whether targets were velocity limited.`);
  return out;
}

/** Problems with a stored evaluation, including internal consistency of the reported numbers. */
export function evaluationProblems(e: unknown, where = 'skill.evaluation'): string[] {
  if (!isObject(e)) return [`${where} must be an object.`];
  const out: string[] = [];
  const trials = e.trials, successes = e.successes;
  if (!Number.isInteger(trials) || (trials as number) < 1) out.push(`${where}.trials must be a positive whole number.`);
  else if (!Number.isInteger(successes) || (successes as number) < 0 || (successes as number) > (trials as number)) out.push(`${where}.successes must be a whole number from 0 to trials (${trials}).`);
  else {
    const n = trials as number, k = successes as number;
    if (!finite(e.successRate) || Math.abs(e.successRate - k / n) > 1e-9) out.push(`${where}.successRate ${String(e.successRate)} does not equal successes / trials = ${(k / n).toFixed(4)}.`);
    const w = wilson95(k, n);
    if (!finiteList(e.wilson95) || e.wilson95.length !== 2 || Math.abs(e.wilson95[0]! - w[0]) > 1e-6 || Math.abs(e.wilson95[1]! - w[1]) > 1e-6) {
      out.push(`${where}.wilson95 must be the Wilson 95 % interval of ${k} of ${n} (${w[0].toFixed(3)} to ${w[1].toFixed(3)}).`);
    }
  }
  if (!isObject(e.conditions) || !Object.values(e.conditions).every(range)) out.push(`${where}.conditions must map names to [low, high] ranges with low <= high.`);
  else {
    const c = e.conditions;
    if (c.friction && !((c.friction as number[])[0]! > 0)) out.push(`${where}.conditions.friction must be positive.`);
    if (c.baseMassKg && !((c.baseMassKg as number[])[0]! > 0)) out.push(`${where}.conditions.baseMassKg must be positive.`);
  }
  if (!isObject(e.metrics) || !Object.values(e.metrics).every(finite)) out.push(`${where}.metrics must map names to finite numbers.`);
  if (typeof e.simulator !== 'string' || typeof e.evaluatedAt !== 'string') out.push(`${where} must name the simulator and the date.`);
  return [...out, ...contextProblems(e.context, where)];
}

/**
 * Validation of an untrusted skill record (used by import and storage):
 * structure, method-specific parameters, durations, policy and recording
 * dimensions, and the consistency of reported evaluation numbers. Returns
 * a copy with only the known fields. Nothing is repaired.
 */
export function parseSkillArtifact(raw: unknown): ParseResult<SkillArtifact> {
  const errors: string[] = [];
  const o = isObject(raw) ? raw : {};
  const str = (k: string) => (typeof o[k] === 'string' && (o[k] as string).length > 0 ? (o[k] as string) : (errors.push(`skill.${k} must be a non-empty string.`), ''));
  if (o.kind !== 'morphlab.skill') errors.push('skill.kind must be "morphlab.skill".');
  if (o.schemaVersion !== 1) errors.push('skill.schemaVersion must be 1.');
  str('id');
  str('name');
  str('createdAt');
  str('updatedAt');
  if (o.importedAt !== undefined && typeof o.importedAt !== 'string') errors.push('skill.importedAt must be a string when present.');
  if (o.evaluationOrigin !== undefined && o.evaluationOrigin !== 'local' && o.evaluationOrigin !== 'imported') errors.push('skill.evaluationOrigin must be "local" or "imported".');
  const taught = o.skillType === 'taught';
  if (o.skillType !== 'foot_lift' && !taught) errors.push('skill.skillType must be "foot_lift" or "taught".');
  if (taught) {
    if (!TAUGHT_METHODS.includes(o.method as SkillMethod)) errors.push(`skill.method of a taught skill must be one of ${TAUGHT_METHODS.join(', ')}.`);
    const t = o.taught;
    if (!isObject(t) || typeof t.name !== 'string' || !TAUGHT_NAME_PATTERN.test(t.name)) errors.push('skill.taught.name must be 1 to 40 letters, digits, spaces, hyphens, underscores or apostrophes, starting with a letter or digit.');
    if (isObject(t) && t.author !== undefined && t.author !== 'ai' && t.author !== 'ai-edited') errors.push('skill.taught.author must be "ai" or "ai-edited" when present.');
    errors.push(...goalsProblems(isObject(t) ? t.goals : undefined, 'skill.taught.goals'));
    if (o.method === 'keyframes' || o.method === 'keyframes-learned') errors.push(...motionProblems(o.motion, 'skill.motion'));
    if (o.method === 'gait' || o.method === 'gait-learned') errors.push(...gaitProblems(o.gait, undefined, 'skill.gait'));
  } else if (!METHODS.includes(o.method as SkillMethod)) errors.push(`skill.method must be one of ${METHODS.join(', ')}.`);
  const robot = o.robot as Record<string, unknown> | undefined;
  if (!isObject(robot) || typeof robot.robotId !== 'string' || typeof robot.morphologyHash !== 'string' || !/^[0-9a-f]{64}$/.test(robot.morphologyHash)) {
    errors.push('skill.robot must contain robotId and a 64-character hex morphologyHash.');
  }
  if (!Array.isArray(o.limbs) || (!taught && o.limbs.length === 0) || !o.limbs.every((l) => typeof l === 'string')) errors.push(taught ? 'skill.limbs must be a list of limb ids.' : 'skill.limbs must be a non-empty list of limb ids.');
  const params = o.parameters;
  if (!isObject(params) || !Object.values(params).every(finite)) errors.push('skill.parameters must map names to finite numbers.');
  else if (o.method === 'state-machine' || o.method === 'tuned-parameters' || o.method === 'es-improved') {
    const p = footLiftParamProblems(params);
    if (p.length) errors.push(`skill.parameters for a state machine: ${p.join('; ')}.`);
  } else if (o.method === 'imitation-mlp' && !(finite(params.durationS) && params.durationS >= 0.5 && params.durationS <= MAX_IMITATION_S)) {
    errors.push(`skill.parameters.durationS must be between 0.5 and ${MAX_IMITATION_S} s for an imitation policy.`);
  }
  const actionSize = Array.isArray(o.action) && isObject(o.action[0]) && Number.isInteger(o.action[0].size) ? (o.action[0].size as number) : null;
  if (o.method === 'imitation-mlp' || o.method === 'gait-learned' || o.method === 'keyframes-learned') {
    const p = o.policy;
    const layers = isObject(p) && Array.isArray(p.layers) && p.layers.length >= 2 && p.layers.every((n) => Number.isInteger(n) && (n as number) >= 1) ? (p.layers as number[]) : null;
    const weightCount = layers ? layers.slice(1).reduce((n, out, i) => n + out * (layers[i]! + 1), 0) : -1;
    if (
      !isObject(p) || !layers || p.activation !== 'tanh' || !finiteList(p.weights) || p.weights.length !== weightCount ||
      !finiteList(p.inputMean) || !finiteList(p.inputStd) || p.inputMean.length !== layers[0] || p.inputStd.length !== layers[0] || !p.inputStd.every((s) => s > 0)
    ) {
      errors.push('skill.policy must contain integer layer sizes, tanh activation, one finite weight per connection and bias, and input normalization with positive scales.');
    } else {
      const observed = Array.isArray(o.observation) ? o.observation.reduce((n: number, x) => n + (isObject(x) && Number.isInteger(x.size) ? (x.size as number) : Number.NaN), 0) : Number.NaN;
      if (observed !== layers[0]) errors.push(`skill.policy takes ${layers[0]} inputs but skill.observation lists ${observed}.`);
      if (o.method === 'gait-learned' && layers[0] !== GAIT_POLICY_INPUTS) errors.push(`skill.policy of a learned gait must take the ${GAIT_POLICY_INPUTS} gait policy inputs, not ${layers[0]}.`);
      if (actionSize !== layers[layers.length - 1]) errors.push(`skill.policy produces ${layers[layers.length - 1]} outputs but skill.action expects ${String(actionSize)}.`);
    }
  }
  if (o.method === 'replay') {
    const t = o.trajectory;
    if (
      !isObject(t) || !(finite(t.dt) && t.dt > 0 && t.dt <= 1) || !Array.isArray(t.jointIds) || !t.jointIds.every((id) => typeof id === 'string') ||
      !Array.isArray(t.frames) || t.frames.length < 2 || !t.frames.every((f) => finiteList(f) && f.length === (t.jointIds as unknown[]).length)
    ) {
      errors.push('skill.trajectory must contain a dt between 0 and 1 s, joint ids and at least two finite frames with one value per joint.');
    } else {
      const ids = t.jointIds as string[];
      if (new Set(ids).size !== ids.length) errors.push('skill.trajectory.jointIds must not repeat.');
      if (((t.frames as unknown[]).length - 1) * (t.dt as number) > MAX_REPLAY_S) errors.push(`skill.trajectory is longer than ${MAX_REPLAY_S} s.`);
      if (actionSize !== null && actionSize !== ids.length) errors.push(`skill.trajectory has ${ids.length} joints but skill.action expects ${actionSize}.`);
    }
  }
  if (o.training !== null && o.training !== undefined) {
    const t = o.training;
    if (!isObject(t) || typeof t.algorithm !== 'string' || !finite(t.iterations) || !finite(t.wallTimeS) || !finite(t.seed) || !isObject(t.config)) {
      errors.push('skill.training must be null or contain algorithm, config, seed, iterations and wall time.');
    }
  }
  if (o.evaluation !== null && o.evaluation !== undefined) errors.push(...evaluationProblems(o.evaluation));
  if (o.importedEvaluation !== undefined) errors.push(...evaluationProblems(o.importedEvaluation, 'skill.importedEvaluation'));
  if (errors.length) return { ok: false, errors };
  const copy: Record<string, unknown> = {};
  for (const k of KNOWN_FIELDS) if (o[k] !== undefined) copy[k] = o[k];
  copy.policy ??= null;
  copy.trajectory ??= null;
  copy.motion ??= null;
  copy.gait ??= null;
  copy.training ??= null;
  copy.evaluation ??= null;
  copy.observation ??= [];
  copy.action ??= [];
  return { ok: true, value: copy as unknown as SkillArtifact };
}

/**
 * Whether a skill may be run by a language command: it needs a consistent
 * evaluation that was measured here, with at least one success. Returns the
 * reason when it may not.
 */
export function commandEligibility(a: SkillArtifact): string | null {
  const e = a.evaluation;
  if (!e) return `"${a.name}" has not been evaluated.`;
  const problems = evaluationProblems(e);
  if (problems.length) return `"${a.name}" has an inconsistent evaluation (${problems[0]}).`;
  if (evaluationOrigin(a) === 'imported') return `"${a.name}" was imported and its evaluation (${e.successes} of ${e.trials}) has not been reproduced here.`;
  if (e.successes === 0) return `"${a.name}" succeeded in none of its ${e.trials} evaluation trials.`;
  return null;
}
