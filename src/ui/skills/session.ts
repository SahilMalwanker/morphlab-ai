import { buildKinematicModel } from '../../core/kinematics/fk';
import { supportMargin } from '../../core/math/support';
import type { Vec3 } from '../../core/math/vec3';
import type { PdGains } from '../../core/control/pd';
import { homeConfiguration, PRESETS, type PresetId } from '../../core/model/catalog';
import { morphologyHash, canonicalJson } from '../../core/model/hash';
import type { RobotDescription } from '../../core/model/types';
import { validateRobot } from '../../core/model/validate';
import { exportBundle, importBundle } from '../../persistence/bundle';
import { checkSkillCompatibility, commandEligibility, createSkillArtifact, evaluationOrigin, MAX_REPLAY_S, skillLabel, specFromArtifact, summarizeEvaluation, type SkillArtifact, type StoredEvaluation, type TaughtAuthor } from '../../skills/artifact';
import { composeKeyframes, type ComposedFrame } from '../../skills/compose';
import { GAIT_BOUNDS, gaitGoalConflicts, gaitProblems, type GaitParams } from '../../skills/gait';
import { DEFAULT_TAUGHT_IMPROVE, type TaughtImproveResult } from '../../skills/improve';
import { checkIntent, type IntentFinding } from '../../skills/intent';
import { mirrorName, mirrorTaught } from '../../skills/mirror';
import { FootLiftTeleop, makeController, METHOD_LABELS, skillDuration, type PolicyJson, type SkillController, type SkillMethod, type SkillSpec, type Trajectory } from '../../skills/controllers';
import { DEFAULT_CRITERIA, DEFAULT_SKILL_GAINS, evaluationOptionsFor, FootLiftReferee, isGoals, judgeSample, supportLegs, verdictFailures, type EvaluationSummary, type Judging, type TrialFailure } from '../../skills/evaluation';
import { CommandInterpreter } from '../../language/interpreter';
import { HOLD_RANGE_S, resolveCommand } from '../../language/resolver';
import { DEFAULT_FOOT_LIFT, FOOT_LIFT_BOUNDS, type FootLiftParams } from '../../skills/footLift';
import { DEFAULT_IMITATION, DEFAULT_IMPROVE, type ImitationConfig, type ImproveConfig } from '../../skills/learning';
import { DEFAULT_GAIT_POLICY, type GaitPolicyResult } from '../../skills/train';
import type { PosePolicyConfig, PosePolicyResult } from '../../skills/trainPose';
import {
  expectedPose, judgeWalk, locomotionBase, locomotionProblem, rangeProblem, vecOf, walkGait,
  type LocomotionConfig, type LocomotionController, type LocomotionResult, type WalkCommand, type WalkJudgement,
} from '../../skills/locomotion';
import type { CancellableRunner } from '../../skills/rollouts';
import {
  defaultGoal, goalProblems, goalRobotProblems, goalsProblems, GoalJudge, matchTaughtSkill, MAX_END_HOLD_S, MAX_GOALS, MAX_KEYFRAMES, motionDuration, normalizeSkillName, restingHeight, TAUGHT_NAME_PATTERN,
  type BodyCheck, type GoalResult, type JudgeSample, type KeyframeMotion, type SkillGoal, type SkillGoalKind,
} from '../../skills/taught';
import { SkillWorkerClient, type SkillJob } from '../../skills/workerClient';
import type { TurnReport } from '../../ai/loop';
import type { LongTermMemory } from '../../ai/memory';
import type { PhysicsSession, RunAbortCause } from '../physics/session';
import { Store } from '../store';

export const SKILL_PRESETS: PresetId[] = ['quadruped', 'hexapod', 'arm3', 'arm6'];

/** The robot and leg a recording, training result or job belongs to. */
export interface WorkContext {
  robot: RobotDescription;
  limbId: string;
}

/** One pose of a skill being taught, reached `durationS` after the previous one. */
export interface DraftKeyframe {
  durationS: number;
  q: number[];
}

/** A skill being taught: poses and goals on one robot, before it is evaluated. */
export interface TaughtDraft {
  name: string;
  robot: RobotDescription;
  keyframes: DraftKeyframe[];
  endHoldS: number;
  goals: SkillGoal[];
  /** Who composed it: the user with the sliders, the AI from a request, or the AI and then the user. */
  author: 'user' | TaughtAuthor;
  /** What composing the poses could not do exactly (tips IK could not reach), for the user to see. */
  notes: string[];
  /** The command the AI composed it for; empty for the user's own drafts. */
  request: string;
  /** A contact-pattern gait instead of keyframes (locomotion); null for keyframe drafts. */
  gait: GaitParams | null;
  /** Set when trial and error improved the draft; any later edit clears it. */
  training: DraftTraining | null;
  /** A learned feedback policy for the gait (D27) or the keyframes (D30); any later edit clears it. */
  policy: DraftPolicy | null;
}

/** Held-out scores of the draft with and without its policy. */
export interface DraftPolicyScore {
  meanReturn: number;
  meanGoalReward: number;
  successes: number;
  falls: number;
  /** Gaits only: velocity error while stepping, m/s. */
  trackingErrorMps?: number;
}

/** A learned feedback policy trained for a draft, with its held-out comparison against the draft alone (pushed runs for keyframes). */
export interface DraftPolicy {
  kind: 'gait' | 'pose';
  policy: PolicyJson;
  iterations: number;
  rollouts: number;
  parameters: number;
  heldOutTrials: number;
  baseline: DraftPolicyScore;
  result: DraftPolicyScore;
  seed: number;
  wallTimeS: number;
  /** Keyframes: size of the push in every training and held-out run, N*s. */
  pushImpulseNs?: [number, number];
}

/** How trial and error improved a draft, measured on held-out randomized trials. */
export interface DraftTraining {
  iterations: number;
  rollouts: number;
  heldOutTrials: number;
  baselineReward: number;
  improvedReward: number;
  baselineSuccesses: number;
  improvedSuccesses: number;
  seed: number;
  wallTimeS: number;
}

/** A skill draft as the AI composes it: poses in words of the body (or a gait), and goals. */
export interface ComposedDraft {
  name: string;
  keyframes?: ComposedFrame[];
  gait?: GaitParams;
  endHoldS: number;
  goals: SkillGoal[];
  /** The user's command, checked against the goals (D25). */
  request: string;
}

/** The result of an evaluation of the draft: the trials, and the saved skill when it was saved. */
export interface DraftEvaluation {
  name: string;
  goals: SkillGoal[];
  evaluation: EvaluationSummary;
  artifact: SkillArtifact | null;
  wallTimeS: number;
}

export interface SkillsState {
  robot: RobotDescription;
  source: string;
  hash: string | null;
  legs: { id: string; name: string }[];
  /** Why the foot-lift skill cannot run on this robot, if it cannot. */
  unsupported: string | null;
  /** Left panel: teaching a skill of your own, the built-in foot-lift example, or walking by command. */
  panel: 'teach' | 'foot-lift' | 'walk';
  draft: TaughtDraft | null;
  limbId: string;
  params: FootLiftParams;
  demonstration: { source: 'expert' | 'you'; trajectory: Trajectory; context: WorkContext } | null;
  imitation: { policy: PolicyJson; horizonS: number; initialLoss: number; finalLoss: number; samples: number; wallTimeS: number; config: ImitationConfig; context: WorkContext } | null;
  improvement: { params: FootLiftParams; baselineReward: number; improvedReward: number; trialsRun: number; wallTimeS: number; config: ImproveConfig; context: WorkContext } | null;
  job: { label: string; fraction: number; detail: string } | null;
  library: SkillArtifact[];
  /** The walking controller of this robot's morphology, if one is trained (D28). */
  walking: LocomotionController | null;
  /** The command the Walk panel sends. */
  walkCommand: WalkCommand & { durationS: number };
  /** The last walk by command, measured from simulator ground truth. */
  lastWalk: WalkReport | null;
  message: { tone: 'info' | 'error'; text: string } | null;
  teleop: { active: boolean; recording: boolean; shift: number; lift: number; frames: number };
  playing: string | null;
  /** Result of the last completed playback, measured from simulator state. */
  lastPlay: PlayReport | null;
  commandLog: CommandOutcome[];
  /** Recorded playbacks, newest first, for side-by-side replay. */
  attempts: Attempt[];
  /** Attempt ids shown side by side (left, right), or null. */
  comparing: [string, string] | null;
}

/** Interval between recorded frames of an attempt, s. */
export const ATTEMPT_DT = 0.04;
/** Demonstration recording period: the 100 Hz controller rate, s. */
const TELEOP_DT = 0.01;
const MAX_ATTEMPTS = 6;

/** Simulator frames collected during a playback. */
interface PlayRecording {
  /** Library id, or null for a preview of a skill being taught. */
  skillId: string | null;
  name: string;
  method: SkillMethod;
  morphologyHash: string | null;
  robot: RobotDescription;
  t: number[];
  frames: Float64Array[];
  clearanceM: number[];
  plot: Omit<AttemptPlot, 'values'>;
  plotValues: number[];
  nextT: number;
}

/** What a live run plays and how it is judged. */
interface LiveRun {
  skillId: string | null;
  name: string;
  method: SkillMethod;
  morphologyHash: string | null;
  spec: SkillSpec;
  judging: Judging;
  /** The lifted leg of a foot lift; -1 for taught skills. */
  limbIndex: number;
  requiredHoldS: number;
  evaluation: StoredEvaluation | null;
  label: string;
  /** Start where the body stands when it is up (walks chain into a path); otherwise the simulation is reset to the home stance first. */
  fromCurrentPose?: boolean;
}

/** The playback in progress; `finish` settles it exactly once. */
interface ActiveRun {
  skillId: string;
  finish(end: 'ended' | 'interrupted' | 'faulted', reason: string | null, cause?: RunAbortCause): void;
}

/** A measured signal shown when two attempts are compared. */
export interface AttemptPlot {
  title: string;
  unit: string;
  values: number[];
  /** The goal threshold drawn as a dashed line, if any. */
  threshold: number | null;
}

/**
 * One playback recorded from the simulator for replay. Frames hold the base
 * transform followed by every link transform (16 numbers each, column-major,
 * global joint order). Replaying them draws what was simulated; nothing is
 * simulated again.
 */
export interface Attempt {
  id: string;
  label: string;
  skillName: string;
  method: SkillMethod;
  robot: RobotDescription;
  t: number[];
  frames: Float64Array[];
  /** Lifted-foot clearance above the ground per frame (ground truth), m; empty for taught skills. */
  clearanceM: number[];
  plot: AttemptPlot;
  holdS: number;
  maxTiltRad: number;
  outcome: PlayOutcome;
  completed: boolean;
  /** The measured result in one line. */
  resultText: string;
}

/**
 * How a live playback ended. Only 'succeeded' means the referee's criteria
 * (for a foot lift: clearance, the required hold, tilt, height and support
 * contact; for a taught skill: every goal) passed.
 */
export type PlayOutcome = 'succeeded' | 'failed' | 'aborted' | 'interrupted' | 'faulted';

/** How each playback outcome is named in the interface. */
export const OUTCOME_LABELS: Record<PlayOutcome, string> = {
  succeeded: 'succeeded',
  failed: 'failed the criteria',
  aborted: 'aborted by the skill',
  interrupted: 'interrupted',
  faulted: 'stopped by a fault',
};

export interface PlayReport {
  skillId: string;
  skillName: string;
  limbId: string;
  outcome: PlayOutcome;
  /** The controller ran to its end (succeeded, failed or aborted by its own safety check). */
  completed: boolean;
  success: boolean;
  failures: TrialFailure[];
  /** The continuous hold the referee required, s. */
  requiredHoldS: number;
  /** Longest continuous time the foot was at least 3 cm clear (ground truth), s; 0 for taught skills. */
  holdS: number;
  maxClearanceM: number;
  maxTiltRad: number;
  minBaseHeightM: number;
  supportContactFraction: number;
  /** Per-goal results of a taught skill (simulator ground truth); null for a foot lift. */
  goals: GoalResult[] | null;
  /** Whole-run checks of a free-standing body (only the feet touch the ground, no tipping over), taught skills only. */
  checks?: BodyCheck[];
  /** Why an interrupted, faulted or unfinished run ended. */
  reason: string | null;
  /** Ways this live run differs from the setup the skill was evaluated in. */
  setupNotes: string[];
  /** Where the body (and the limbs the goals name) were at each keyframe, each second of a gait and the end (taught skills). */
  checkpoints?: RunCheckpoint[];
}

/** The state of a live run at one moment, from simulator ground truth, in the start heading frame (x forward, y left, z up). */
export interface RunCheckpoint {
  label: string;
  /** Time since the start, s. */
  t: number;
  forwardM: number;
  leftM: number;
  heightM: number;
  pitchDeg: number;
  rollDeg: number;
  /** Heading change since the start, unwrapped, deg. */
  turnDeg: number;
  /** Limb tips the goals name: their movement from where they started, m. */
  tips: { limbId: string; x: number; y: number; z: number }[];
  /**
   * Free-standing bodies: the limbs whose feet touch the ground, and how far the centre of mass is inside (+) or
   * outside (-) their convex hull, with the offset to the nearest hull point in the start heading frame; null margin
   * when no foot touches.
   */
  support?: { feet: string[]; allFeet: boolean; marginM: number | null; nearestForwardM: number; nearestLeftM: number };
}

/** When a live run of a skill is checkpointed: each keyframe and the end of the hold, or each second of a gait. */
function checkpointMarks(spec: SkillSpec): { label: string; t: number }[] {
  if (spec.kind === 'keyframes' || spec.kind === 'keyframes-learned') return [...spec.motion.keyframes.map((k, i) => ({ label: `keyframe ${i + 1}`, t: k.t })), { label: 'end of the hold', t: motionDuration(spec.motion) }];
  if (spec.kind === 'gait' || spec.kind === 'gait-learned') return Array.from({ length: Math.floor(spec.gait.durationS) }, (_, i) => ({ label: `${i + 1} s`, t: i + 1 }));
  return [];
}

function checkpointOf(label: string, t: number, s: JudgeSample, s0: JudgeSample, turnRad: number, limbs: readonly number[], ids: readonly string[], allIds: readonly string[]): RunCheckpoint {
  const c = Math.cos(s0.yawRad), sn = Math.sin(s0.yawRad);
  const fwd = (x: number, y: number) => +(x * c + y * sn).toFixed(3);
  const left = (x: number, y: number) => +(-x * sn + y * c).toFixed(3);
  const deg = (r: number) => +((r * 180) / Math.PI).toFixed(1);
  const dx = s.basePos[0] - s0.basePos[0], dy = s.basePos[1] - s0.basePos[1];
  const tips = limbs.flatMap((li, k) => {
    const p = s.tipPos[li], p0 = s0.tipPos[li];
    return p && p0 ? [{ limbId: ids[k]!, x: fwd(p[0] - p0[0], p[1] - p0[1]), y: left(p[0] - p0[0], p[1] - p0[1]), z: +(p[2] - p0[2]).toFixed(3) }] : [];
  });
  return { label, t: +t.toFixed(2), forwardM: fwd(dx, dy), leftM: left(dx, dy), heightM: +s.basePos[2].toFixed(3), pitchDeg: deg(s.pitchRad), rollDeg: deg(s.rollRad), turnDeg: deg(turnRad), tips, ...supportOf(s, allIds, fwd, left) };
}

function supportOf(s: JudgeSample, allIds: readonly string[], fwd: (x: number, y: number) => number, left: (x: number, y: number) => number): Pick<RunCheckpoint, 'support'> {
  if (!s.comPos) return {};
  const down = s.tipPos.flatMap((p, i) => (s.tipContact[i] && Number.isFinite(p[0]) && Number.isFinite(p[1]) ? [i] : []));
  const feet = down.map((i) => allIds[i] ?? `limb ${i}`);
  const legs = s.tipPos.filter((p) => Number.isFinite(p[0])).length;
  const r = supportMargin([s.comPos[0], s.comPos[1]], down.map((i) => [s.tipPos[i]![0], s.tipPos[i]![1]] as const));
  return { support: { feet, allFeet: down.length === legs, marginM: r ? +r.marginM.toFixed(3) : null, nearestForwardM: r ? fwd(r.nearest[0], r.nearest[1]) : 0, nearestLeftM: r ? left(r.nearest[0], r.nearest[1]) : 0 } };
}

export interface CommandOutcome {
  command: string;
  /** 'failed': the command ran but did not achieve its goal (criteria not met, interrupted or faulted). */
  kind: 'done' | 'refusal' | 'clarify' | 'failed';
  text: string;
  suggestion?: string;
  options?: string[];
  /** Which interpreter produced the structured call. */
  source: string;
  note: string | null;
  measuredHoldS?: number;
  /** What the AI did, step by step, when the command went through the AI path. */
  ai?: TurnReport;
}

/** The tab a command was typed in, when that is not the Skills tab. */
export interface CommandTarget {
  /** Tab name, for messages. */
  tab: string;
  /** The robot that tab shows, if any. */
  robot: RobotDescription | null;
  /** That tab's live simulation for stand and reset (each returns what it did), or null. */
  simulation: { stand(): string; reset(): string } | null;
  /** Shows the Skills tab, where skills run. */
  showSkills(): void;
}

/** Optional storage hooks (IndexedDB in the app, absent in tests). */
export interface SkillsPersistence {
  saveSkill(a: SkillArtifact): Promise<void>;
  deleteSkill(id: string): Promise<void>;
  /** Saves the robot open in the Skills tab (it is restored on reload). */
  saveRobot(robot: RobotDescription): Promise<void>;
  /** Saves a robot without making it the open one. */
  storeRobot?(robot: RobotDescription): Promise<void>;
  /** A stored robot with this morphology, if any. */
  findRobot?(robotId: string, morphologyHash: string): RobotDescription | null;
  saveLocomotion?(c: LocomotionController): Promise<void>;
  deleteLocomotion?(morphologyHash: string): Promise<void>;
}

/** A walk by command and where it took the body. */
export interface WalkReport {
  command: WalkCommand;
  durationS: number;
  /** It started where the previous run left the body (otherwise from the home stance after a reset). */
  continued: boolean;
  /** The live run's outcome against its goals (stay upright, do not fall). */
  outcome: PlayOutcome;
  judgement: WalkJudgement;
  /** The learned feedback policy ran (otherwise the calibrated generator alone). */
  policy: boolean;
  calibrated: boolean;
  /** The generator's stride, side step or turn per cycle hit its bounds. */
  clamped: boolean;
  maxTiltRad: number;
  reason: string | null;
}

/** How a walk ended: whether the body stayed up (the live run's goals) and where it ended against the tolerance. */
export function walkOutcomeText(r: Pick<WalkReport, 'outcome' | 'judgement'>): string {
  if (r.outcome !== 'succeeded' && r.outcome !== 'failed') return OUTCOME_LABELS[r.outcome];
  if (!r.judgement.upright) return 'did not stay upright';
  return r.judgement.success ? 'ended within tolerance' : 'stayed upright but ended outside tolerance';
}

/** Optional test seams: the rollout pool and a smaller training. */
export interface SkillsSessionOptions {
  rolloutPool?: () => CancellableRunner;
  locomotionConfig?: Partial<LocomotionConfig>;
  posePolicyConfig?: Partial<PosePolicyConfig>;
}

/** Bounds for editable state-machine parameters. */
export const PARAM_FIELDS: { key: keyof FootLiftParams; label: string; min: number; max: number; digits: number }[] = (
  [
    ['shiftM', 'Body shift (m)', 3],
    ['liftHeightM', 'Lift height (m)', 3],
    ['holdS', 'Hold (s)', 2],
    ['shiftS', 'Shift time (s)', 2],
    ['liftS', 'Lift time (s)', 2],
    ['lowerS', 'Lower time (s)', 2],
  ] as [keyof FootLiftParams, string, number][]
).map(([key, label, digits]) => ({ key, label, min: FOOT_LIFT_BOUNDS[key][0], max: FOOT_LIFT_BOUNDS[key][1], digits }));

const FAILURE_TEXT: Record<TrialFailure, string> = {
  fell: 'the body dropped below the minimum height',
  tilted: 'the body tilted beyond the limit',
  'insufficient-clearance': 'the foot never got 3 cm clear of the ground',
  'hold-too-short': 'the foot was not held clear long enough',
  'lost-support-contact': 'a support foot lost contact while the foot was lifted',
  aborted: 'the skill aborted itself (tilt safety check)',
  'numerical-instability': 'the simulation became numerically unstable',
  'goal-not-met': 'a goal of the skill was not met',
  'body-touched-ground': 'a part other than the feet touched the ground',
  'tipped-over': 'the body tipped over',
};

/** Differences between a live playback and the setup the skill was evaluated in. */
export function liveSetupNotes(e: StoredEvaluation | null, live: { gains: PdGains; baseMassKg: number; friction: number }): string[] {
  const c = e?.context;
  if (!c) return ['This skill was evaluated before evaluation setups were recorded, so the live setup cannot be checked against it.'];
  const notes: string[] = [];
  if (c.gains.kp !== live.gains.kp || c.gains.kd !== live.gains.kd) notes.push(`Joint gains differ from the evaluation (live kp ${live.gains.kp}, kd ${live.gains.kd}; evaluated kp ${c.gains.kp}, kd ${c.gains.kd}).`);
  if (live.baseMassKg < c.baseMassRange[0] || live.baseMassKg > c.baseMassRange[1]) notes.push(`The body mass (${live.baseMassKg} kg) is outside the evaluated ${c.baseMassRange[0]}-${c.baseMassRange[1]} kg.`);
  if (live.friction < c.frictionRange[0] || live.friction > c.frictionRange[1]) notes.push(`The ground friction (${live.friction}) is outside the evaluated ${c.frictionRange[0]}-${c.frictionRange[1]}.`);
  if (!c.velocityLimited) notes.push('The evaluation did not limit joint target velocity; live runs do.');
  return notes;
}

/**
 * Command text for a finished playback. Only a run that passed the
 * referee's criteria is reported as done; everything else says what failed.
 */
export function describePlay(report: PlayReport, limb: string, requestedS: number): Pick<CommandOutcome, 'kind' | 'text' | 'suggestion'> {
  if (report.goals) return describeTaught(report);
  const measured = `Measured hold at least 3 cm above the ground: ${report.holdS.toFixed(2)} s (requested ${requestedS} s; simulator ground truth).`;
  const why = report.failures.filter((f) => f !== 'aborted').map((f) => FAILURE_TEXT[f]);
  switch (report.outcome) {
    case 'succeeded':
      return { kind: 'done', text: `Lifted the ${limb}. ${measured}` };
    case 'failed':
      return {
        kind: 'failed',
        text: `The ${limb} lift did not meet the success criteria: ${why.join('; ') || 'unknown'}${report.reason ? ` (${report.reason})` : ''}. ${measured}`,
        suggestion: report.failures.includes('hold-too-short') ? 'This skill may not hold that long; ask for a shorter hold or improve the skill.' : 'Open the attempt in the Skills tab to see what happened.',
      };
    case 'aborted':
      return { kind: 'failed', text: `The skill aborted the ${limb} lift: its tilt safety check stopped it.${why.length ? ` Also: ${why.join('; ')}.` : ''} ${measured}`, suggestion: 'Open the attempt in the Skills tab to see what happened.' };
    case 'interrupted':
      return { kind: 'failed', text: `The run was interrupted before it finished (${report.reason ?? 'stopped'}). Measured hold so far: ${report.holdS.toFixed(2)} s.`, suggestion: 'Run the command again and keep the Skills view open.' };
    case 'faulted':
      return { kind: 'failed', text: `The run stopped: ${report.reason ?? 'the simulation failed'}. Measured hold so far: ${report.holdS.toFixed(2)} s.`, suggestion: 'Reset the simulation and try again.' };
  }
}

function legsOf(robot: RobotDescription): { id: string; name: string }[] {
  return robot.limbs.filter((l) => l.role === 'leg' && l.tip.radius > 0).map((l) => ({ id: l.id, name: l.name }));
}

/** Command text for a taught skill's playback: every goal with what was measured. */
function describeTaught(r: PlayReport): Pick<CommandOutcome, 'kind' | 'text' | 'suggestion'> {
  const goals = r.goals ?? [];
  const unmet = goals.filter((g) => !g.met).length;
  const failedChecks = (r.checks ?? []).filter((c) => !c.met);
  const measured = `Measured (simulator ground truth): ${goals.map((g) => `${g.met ? 'met' : 'not met'}, ${g.text}`).join('; ')}.${failedChecks.length ? ` Body check failed: ${failedChecks.map((c) => c.text).join('; ')}.` : ''}`;
  switch (r.outcome) {
    case 'succeeded':
      // Goals that never check the landing can pass while the body ends on its back; say so.
      if (r.maxTiltRad > Math.PI / 2) {
        return {
          kind: 'done',
          text: `Did "${r.skillName}": every goal was met, but the body tipped over (largest tilt ${((r.maxTiltRad * 180) / Math.PI).toFixed(0)} deg), which none of its goals checks. ${measured}`,
          suggestion: 'Add an upright or standsAtEnd goal so a run that tips over fails, and evaluate the skill again.',
        };
      }
      return { kind: 'done', text: `Did "${r.skillName}": every goal was met. ${measured}` };
    case 'failed':
    case 'aborted':
      return {
        kind: 'failed',
        text:
          unmet === 0 && failedChecks.length
            ? `"${r.skillName}" met every goal, but not as a person watching would accept it${r.reason ? ` (${r.reason})` : ''}. ${measured}`
            : `"${r.skillName}" did not meet ${unmet} of its ${goals.length} goal${goals.length === 1 ? '' : 's'}${r.reason ? ` (${r.reason})` : ''}. ${measured}`,
        suggestion: 'Open the attempt in the Skills tab, adjust the poses, timing or goals, and evaluate again.',
      };
    case 'interrupted':
      return { kind: 'failed', text: `The run was interrupted before it finished (${r.reason ?? 'stopped'}).`, suggestion: 'Run the command again and keep the Skills view open.' };
    case 'faulted':
      return { kind: 'failed', text: `The run stopped: ${r.reason ?? 'the simulation failed'}.`, suggestion: 'Reset the simulation and try again.' };
  }
}

/** One line with the measured result of a playback. */
function attemptText(r: PlayReport): string {
  const tilt = `max tilt ${((r.maxTiltRad * 180) / Math.PI).toFixed(1)} deg`;
  if (r.goals) return `${r.goals.filter((g) => g.met).length} of ${r.goals.length} goals met, ${tilt}, ${OUTCOME_LABELS[r.outcome]}`;
  return `hold ${r.holdS.toFixed(2)} s, ${tilt}, ${OUTCOME_LABELS[r.outcome]}`;
}

const FOOT_LIFT_NEEDS = 'The foot-lift skill needs at least four legs with feet: three to stand on and one to lift.';

/** A measured signal of a live run, recorded for comparisons. */
interface PlotSpec {
  title: string;
  unit: string;
  threshold: number | null;
  value: (s: JudgeSample) => number;
}

/** The signal plotted for a taught skill: the first limb tip or body angle a goal names, otherwise the body height. */
function goalPlot(goals: readonly SkillGoal[], robot: RobotDescription): PlotSpec {
  const limbAt = (id: string) => robot.limbs.findIndex((l) => l.id === id);
  for (const g of goals) {
    if (g.kind === 'tipHeight') {
      const i = limbAt(g.limbId);
      return { title: `${robot.limbs[i]?.name ?? g.limbId} tip height`, unit: 'm', threshold: g.minHeightM, value: (s) => s.tipClearanceM[i] ?? Number.NaN };
    }
    if (g.kind === 'tipSwing' || g.kind === 'tipReach') {
      const i = limbAt(g.limbId);
      const axis = g.kind === 'tipSwing' ? g.axis : 'z';
      const k = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
      return { title: `${robot.limbs[i]?.name ?? g.limbId} tip ${axis} (world)`, unit: 'm', threshold: null, value: (s) => s.tipPos[i]?.[k] ?? Number.NaN };
    }
    if (g.kind === 'bodyAngle') return { title: `Body ${g.axis}`, unit: 'rad', threshold: g.axis === 'pitch' ? g.maxRad : g.minRad, value: (s) => (g.axis === 'pitch' ? s.pitchRad : s.rollRad) };
  }
  let threshold: number | null = null;
  for (const g of goals) {
    if (g.kind === 'baseHeight') {
      threshold = g.heightM;
      break;
    }
    if (g.kind === 'noFall' && threshold === null) threshold = g.minBaseHeightM;
  }
  return { title: 'Body height', unit: 'm', threshold, value: (s) => s.basePos[2] };
}

/** Keyframes of a draft with absolute times from the start, s. */
export function draftMotion(d: TaughtDraft): KeyframeMotion {
  let t = 0;
  const keyframes = d.keyframes.map((k) => {
    t += k.durationS;
    return { t: Number(t.toFixed(4)), q: k.q.slice() };
  });
  return { jointIds: d.robot.limbs.flatMap((l) => l.joints.map((j) => j.id)), keyframes, endHoldS: d.endHoldS };
}

/** What a draft plays: its gait or its keyframes, with the learned feedback policy trained for it, if any. */
export function draftSpec(d: TaughtDraft): SkillSpec {
  if (d.gait) return d.policy ? { kind: 'gait-learned', gait: { ...d.gait }, policy: d.policy.policy } : { kind: 'gait', gait: { ...d.gait } };
  return d.policy ? { kind: 'keyframes-learned', motion: draftMotion(d), policy: d.policy.policy } : { kind: 'keyframes', motion: draftMotion(d) };
}

/** Draft keyframes from a motion with absolute times. */
function keyframesOf(m: KeyframeMotion): DraftKeyframe[] {
  let prev = 0;
  return m.keyframes.map((k) => {
    const durationS = Number((k.t - prev).toFixed(4));
    prev = k.t;
    return { durationS, q: k.q.slice() };
  });
}

/**
 * Skills workspace: teaches skills of the user's own (posed keyframes judged
 * by the goals the user sets) and the built-in foot lift with five labeled
 * methods, runs evaluation and training in a Web Worker, keeps a library of
 * skill artifacts bound to morphology hashes, and plays skills in a live
 * simulation with sensor-only inputs.
 */
export class SkillsSession {
  readonly store: Store<SkillsState>;
  readonly interpreter: CommandInterpreter;
  private teleop: FootLiftTeleop | null = null;
  private recorded: number[][] = [];
  private evalCounter = 0;
  private jobCancelVersion = 0;
  private finishPlay: ((r: PlayReport) => void) | null = null;
  private persistence: SkillsPersistence | null = null;
  /** Walking controllers by morphology hash. */
  private readonly controllers = new Map<string, LocomotionController>();
  private pool: CancellableRunner | null = null;
  private readonly makePool: (() => CancellableRunner) | undefined;
  private readonly locomotionConfig: Partial<LocomotionConfig>;
  private readonly posePolicyConfig: Partial<PosePolicyConfig>;
  /** The last live run started where the body stood, without a reset. */
  private lastStartContinued = false;

  constructor(
    readonly physics: PhysicsSession,
    private readonly worker: SkillWorkerClient,
    robot: RobotDescription = PRESETS.find((p) => p.id === 'quadruped')!.make(),
    interpreter = new CommandInterpreter(),
    options: SkillsSessionOptions = {},
  ) {
    this.interpreter = interpreter;
    this.makePool = options.rolloutPool;
    this.locomotionConfig = options.locomotionConfig ?? {};
    this.posePolicyConfig = options.posePolicyConfig ?? {};
    const legs = legsOf(robot);
    const preset = SKILL_PRESETS.find((id) => canonicalJson(PRESETS.find((p) => p.id === id)!.make()) === canonicalJson(robot));
    this.store = new Store<SkillsState>({
      robot, source: preset ?? 'custom', hash: null, legs, unsupported: legs.length >= 4 ? null : FOOT_LIFT_NEEDS, panel: 'teach', draft: null,
      limbId: legs[0]?.id ?? '', params: { ...DEFAULT_FOOT_LIFT }, demonstration: null, imitation: null, improvement: null, job: null, library: [], message: null,
      walking: null, walkCommand: { forwardMps: 0.1, leftMps: 0, turnRadps: 0, durationS: 3 }, lastWalk: null,
      teleop: { active: false, recording: false, shift: 0, lift: 0, frames: 0 }, playing: null, lastPlay: null, commandLog: [], attempts: [], comparing: null,
    });
    this.physics.loadRobot(robot);
    void this.updateHash(robot);
  }

  /** Connects storage; later skill and robot changes are saved through it. */
  attachPersistence(p: SkillsPersistence | null): void {
    this.persistence = p;
  }

  private skillMemory: LongTermMemory | null = null;

  /** Skill memory: every finished playback is recorded there with its measured outcome. */
  attachMemory(m: LongTermMemory | null): void {
    this.skillMemory = m;
  }

  get memory(): LongTermMemory | null {
    return this.skillMemory;
  }

  private persist(what: string, task: (p: SkillsPersistence) => Promise<void>): void {
    const p = this.persistence;
    if (!p) return;
    task(p).catch((e) => this.error(`Could not save ${what}: ${e instanceof Error ? e.message : String(e)}`));
  }

  get state(): SkillsState {
    return this.store.get();
  }

  private async updateHash(robot: RobotDescription): Promise<void> {
    const h = await morphologyHash(robot);
    this.knownRobots.set(h, robot);
    if (this.state.robot === robot) this.store.update({ hash: h, walking: this.controllers.get(h) ?? null });
  }

  /** Robots seen in this session by morphology hash, for exporting skills with their robots. */
  private readonly knownRobots = new Map<string, RobotDescription>();

  private get limbIndex(): number {
    return this.state.robot.limbs.findIndex((l) => l.id === this.state.limbId);
  }

  private get stance(): number[] {
    return homeConfiguration(this.state.robot);
  }

  private info(text: string): void {
    this.store.update({ message: { tone: 'info', text } });
  }

  private error(text: string): void {
    this.store.update({ message: { tone: 'error', text } });
  }

  setPreset(id: PresetId): void {
    const p = PRESETS.find((x) => x.id === id);
    if (p) this.setRobot(p.make(), id);
  }

  setRobot(robot: RobotDescription, source = 'custom'): boolean {
    const issues = validateRobot(robot);
    if (issues.length) {
      this.error(`Refused robot: ${issues[0]!.path}: ${issues[0]!.message}.`);
      return false;
    }
    this.stopActivity('the robot was replaced');
    this.worker.cancel();
    const legs = legsOf(robot);
    this.store.update({
      robot, source, hash: null, legs, limbId: legs[0]?.id ?? '', unsupported: legs.length >= 4 ? null : FOOT_LIFT_NEEDS,
      demonstration: null, imitation: null, improvement: null, job: null, message: null, draft: null, walking: null, lastWalk: null,
    });
    this.physics.loadRobot(robot);
    void this.updateHash(robot);
    this.persist('the robot', (p) => p.saveRobot(robot));
    return true;
  }

  setLimb(id: string): void {
    if (!this.state.legs.some((l) => l.id === id) || id === this.state.limbId) return;
    this.stopActivity('another leg was selected');
    // A running job belongs to the previous leg; its result would not apply to this one.
    if (this.state.job) this.worker.cancel();
    this.store.update({ limbId: id, demonstration: null, imitation: null, improvement: null });
  }

  setParam(key: keyof FootLiftParams, value: number): boolean {
    const f = PARAM_FIELDS.find((x) => x.key === key);
    if (!f || !Number.isFinite(value) || value < f.min || value > f.max) {
      this.error(`${f?.label ?? key} must be between ${f?.min} and ${f?.max}.`);
      return false;
    }
    this.store.update({ params: { ...this.state.params, [key]: value }, message: null });
    return true;
  }

  setPanel(panel: SkillsState['panel']): void {
    if (panel === this.state.panel) return;
    this.stopActivity('another panel was opened');
    this.store.update({ panel, message: null });
  }

  /** Starts teaching a new skill on the current robot: no poses yet, and goals that keep it upright. */
  newDraft(): void {
    const robot = this.state.robot;
    this.stopActivity('a new skill was started');
    this.physics.reset();
    const goals = robot.base.kind === 'floating' ? [defaultGoal('upright', robot), defaultGoal('noFall', robot)] : [defaultGoal('tipHeight', robot)];
    this.store.update({ panel: 'teach', draft: { name: '', robot, keyframes: [], endHoldS: 1, goals, author: 'user', notes: [], request: '', gait: null, training: null, policy: null }, message: null });
  }

  discardDraft(): void {
    this.stopActivity('the skill draft was discarded');
    this.store.update({ draft: null, message: null });
  }

  private draftHere(): TaughtDraft | null {
    const d = this.state.draft;
    if (!d) this.error('Start a new skill first.');
    else if (d.robot !== this.state.robot) this.error('These poses belong to another robot; start a new skill for this one.');
    else return d;
    return null;
  }

  /** Edits by the user; an edit of the AI's draft makes it the AI's and the user's, and any edit drops an earlier improvement record and learned policy. */
  private patchDraft(patch: Partial<TaughtDraft>): void {
    const d = this.state.draft;
    if (d) this.store.update({ draft: { ...d, training: null, policy: null, ...patch, ...(d.author === 'ai' ? { author: 'ai-edited' as const } : {}) }, message: null });
  }

  setDraftName(name: string): void {
    this.patchDraft({ name: name.slice(0, 40) });
  }

  /** The pose the live robot is commanded to hold: what Add keyframe captures. */
  get poseTargets(): number[] {
    return this.physics.state.qCmd.slice();
  }

  /** Moves one joint of the pose; the live robot follows under its joint controller. */
  setPoseJoint(j: number, value: number): void {
    if (this.state.playing || this.physics.hasDriver) this.stopActivity('the pose was changed');
    this.physics.setTarget(j, value);
  }

  /** Puts the live robot back in its start state and home pose. */
  resetPose(): void {
    this.stopActivity('the pose was reset');
    this.physics.reset();
  }

  addKeyframe(durationS = 1): boolean {
    const d = this.draftHere();
    if (!d) return false;
    if (d.keyframes.length >= MAX_KEYFRAMES) {
      this.error(`A skill holds at most ${MAX_KEYFRAMES} keyframes.`);
      return false;
    }
    const q = this.poseTargets;
    if (q.length !== draftMotion(d).jointIds.length) {
      this.error('The live simulation is not ready yet.');
      return false;
    }
    this.patchDraft({ keyframes: [...d.keyframes, { durationS, q }] });
    return true;
  }

  /** Replaces a keyframe's pose with the pose the live robot holds now. */
  replaceKeyframe(i: number): void {
    const d = this.draftHere();
    const k = d?.keyframes[i];
    if (!d || !k) return;
    this.patchDraft({ keyframes: d.keyframes.map((x, j) => (j === i ? { ...x, q: this.poseTargets } : x)) });
  }

  removeKeyframe(i: number): void {
    const d = this.state.draft;
    if (d) this.patchDraft({ keyframes: d.keyframes.filter((_, j) => j !== i) });
  }

  /** Sends the live robot to a keyframe's pose, to look at it or edit it. */
  showKeyframe(i: number): void {
    const k = this.draftHere()?.keyframes[i];
    if (!k) return;
    if (this.state.playing || this.physics.hasDriver) this.stopActivity('a keyframe was shown');
    this.physics.setTargets(k.q);
  }

  setKeyframeDuration(i: number, durationS: number): boolean {
    const d = this.state.draft;
    if (!d || !d.keyframes[i]) return false;
    if (!Number.isFinite(durationS) || durationS < 0.1 || durationS > 10) {
      this.error('The time to reach a keyframe must be between 0.1 and 10 s.');
      return false;
    }
    this.patchDraft({ keyframes: d.keyframes.map((x, j) => (j === i ? { ...x, durationS } : x)) });
    return true;
  }

  setEndHold(s: number): boolean {
    if (!Number.isFinite(s) || s < 0 || s > MAX_END_HOLD_S) {
      this.error(`The final hold must be between 0 and ${MAX_END_HOLD_S} s.`);
      return false;
    }
    this.patchDraft({ endHoldS: s });
    return true;
  }

  addGoal(kind: SkillGoalKind): boolean {
    const d = this.draftHere();
    if (!d) return false;
    if (d.goals.length >= MAX_GOALS) {
      this.error(`A skill has at most ${MAX_GOALS} goals.`);
      return false;
    }
    const goal = defaultGoal(kind, d.robot);
    const problems = goalRobotProblems([goal], d.robot);
    if (problems.length) {
      this.error(problems[0]!);
      return false;
    }
    this.patchDraft({ goals: [...d.goals, goal] });
    return true;
  }

  /** Replaces a goal; refused with the reason when a value is out of range or names a missing limb. */
  setGoal(i: number, goal: SkillGoal): boolean {
    const d = this.state.draft;
    if (!d || !d.goals[i]) return false;
    const problems = [...goalProblems(goal, `Goal ${i + 1}`), ...goalRobotProblems([goal], d.robot)];
    if (problems.length) {
      this.error(problems[0]!);
      return false;
    }
    this.patchDraft({ goals: d.goals.map((g, j) => (j === i ? goal : g)) });
    return true;
  }

  removeGoal(i: number): void {
    const d = this.state.draft;
    if (d) this.patchDraft({ goals: d.goals.filter((_, j) => j !== i) });
  }

  /** Why the draft cannot be previewed or evaluated yet, or null. */
  private draftProblem(d: TaughtDraft): string | null {
    if (d.gait) {
      const g = gaitProblems(d.gait, d.robot);
      if (g.length) return g[0]!;
    } else if (d.keyframes.length === 0) return 'Add at least one keyframe: pose the robot with the joint sliders, then press Add keyframe.';
    const problems = [...goalsProblems(d.goals, 'Goals'), ...goalRobotProblems(d.goals, d.robot)];
    if (d.goals.length === 0) return 'Add at least one goal: it says what success means.';
    return problems[0] ?? null;
  }

  /** Changes one parameter of a gait draft; refused with the reason when out of range. */
  setGaitParam(key: keyof typeof GAIT_BOUNDS, value: number): boolean {
    const d = this.draftHere();
    if (!d?.gait) return false;
    const [lo, hi] = GAIT_BOUNDS[key];
    if (!Number.isFinite(value) || value < lo || value > hi) {
      this.error(`The gait's ${key} must be from ${lo} to ${hi}.`);
      return false;
    }
    this.patchDraft({ gait: { ...d.gait, [key]: value } });
    return true;
  }

  /** Plays the draft once in the live simulation and judges it by its goals; nothing is saved. */
  async previewDraft(): Promise<boolean> {
    const d = this.draftHere();
    if (!d) return false;
    const problem = this.draftProblem(d);
    if (problem) {
      this.error(problem);
      return false;
    }
    const name = `${d.name.trim() || 'Untitled skill'} (preview)`;
    const spec = draftSpec(d);
    return this.startLive({
      skillId: null, name, method: spec.kind, morphologyHash: null, spec, judging: { goals: d.goals },
      limbIndex: -1, requiredHoldS: 0, evaluation: null, label: `Preview of "${name}": ${d.gait ? (d.policy ? 'the gait steps with its learned feedback policy' : 'the gait steps') : d.policy ? 'the keyframes play with their learned feedback policy (closed loop)' : 'the keyframes play open loop'} and is judged by the goals. Not evaluated.`,
    });
  }

  /** Evaluates the draft over 20 randomized trials in the worker and adds it to the library under its name. */
  async evaluateDraft(): Promise<SkillArtifact | null> {
    return (await this.runDraftEvaluation(true))?.artifact ?? null;
  }

  /**
   * Evaluates the draft over 20 randomized trials in the worker. `saveAlways`
   * saves it whatever the result (the Teach panel); otherwise it is saved only
   * when at least one trial met every goal (the AI's evaluations).
   */
  private async runDraftEvaluation(saveAlways: boolean): Promise<DraftEvaluation | null> {
    const d = this.draftHere();
    if (!d || !this.ready(false)) return null;
    const name = d.name.trim().replace(/\s+/g, ' ');
    if (!TAUGHT_NAME_PATTERN.test(name)) {
      this.error('Name the skill first: 1 to 40 letters, digits, spaces, hyphens, underscores or apostrophes, starting with a letter or digit. Commands use this name.');
      return null;
    }
    const problem = this.draftProblem(d);
    if (problem) {
      this.error(problem);
      return null;
    }
    this.stopActivity('the skill is being evaluated');
    const ctx = this.context();
    const s = this.state;
    const hash = s.hash ?? (await morphologyHash(s.robot));
    const options = evaluationOptionsFor(s.robot, 2026 + this.evalCounter++, await this.liveGains());
    const spec = draftSpec(d);
    const goals = d.goals.map((g) => ({ ...g }));
    const r = await this.job<EvaluationSummary>(`Evaluating "${name}"`, { type: 'evaluate', robot: s.robot, limbIndex: -1, stanceQ: this.stance, spec, options, goals }, ctx);
    if (!r) return null;
    const e = r.result;
    if (!saveAlways && e.successes === 0) {
      this.info(`"${name}": none of ${e.trials} trials met every goal (${r.wallTimeS.toFixed(1)} s), so it was not saved. The draft stays in the Teach panel.`);
      return { name, goals, evaluation: e, artifact: null, wallTimeS: r.wallTimeS };
    }
    const taken = new Set(this.state.library.map((a) => a.name));
    const method = d.gait ? (d.policy ? 'learned gait' : 'gait') : d.policy ? 'learned keyframes' : 'keyframes';
    let label = `${name} (${method})`;
    for (let n = 2; taken.has(label); n++) label = `${name} (${method} ${n})`;
    const author = d.author === 'user' ? undefined : d.author;
    const tr = d.training;
    const lp = d.policy;
    const training: SkillArtifact['training'] = lp
      ? {
          algorithm: lp.kind === 'pose'
            ? 'augmented random search on the goals in randomized, pushed runs (learned feedback policy for keyframes)'
            : 'augmented random search on speed tracking, stability and the goal reward (learned feedback policy)',
          config: {
            parameters: lp.parameters, rollouts: lp.rollouts, heldOutTrials: lp.heldOutTrials,
            baselineReturn: +lp.baseline.meanReturn.toFixed(4), learnedReturn: +lp.result.meanReturn.toFixed(4),
            baselineGoalReward: +lp.baseline.meanGoalReward.toFixed(4), learnedGoalReward: +lp.result.meanGoalReward.toFixed(4),
            ...(lp.baseline.trackingErrorMps !== undefined && lp.result.trackingErrorMps !== undefined
              ? { baselineTrackingErrorMps: +lp.baseline.trackingErrorMps.toFixed(4), learnedTrackingErrorMps: +lp.result.trackingErrorMps.toFixed(4) }
              : {}),
            ...(lp.pushImpulseNs ? { pushImpulseMinNs: lp.pushImpulseNs[0], pushImpulseMaxNs: lp.pushImpulseNs[1] } : {}),
            baselineSuccesses: lp.baseline.successes, learnedSuccesses: lp.result.successes, baselineFalls: lp.baseline.falls, learnedFalls: lp.result.falls,
            ...(tr ? { gaitImprovedBefore: 'evolution strategies on the goal reward' } : {}),
          },
          seed: lp.seed,
          iterations: lp.iterations,
          wallTimeS: lp.wallTimeS,
        }
      : tr
        ? {
            algorithm: 'evolution strategies on the goal reward',
            config: { heldOutTrials: tr.heldOutTrials, rollouts: tr.rollouts, baselineReward: tr.baselineReward, improvedReward: tr.improvedReward, baselineSuccesses: tr.baselineSuccesses, improvedSuccesses: tr.improvedSuccesses },
            seed: tr.seed,
            iterations: tr.iterations,
            wallTimeS: tr.wallTimeS,
          }
        : null;
    const artifact = createSkillArtifact({ robot: s.robot, morphologyHash: hash, limbId: '', spec, name: label, evaluation: e, training, taught: { name, goals, ...(author ? { author } : {}) } });
    this.store.update({ library: [artifact, ...this.state.library] });
    this.persist('the skill', (p) => p.saveSkill(artifact));
    this.info(`"${name}": ${e.successes} of ${e.trials} trials met every goal (${r.wallTimeS.toFixed(1)} s). Added to the library as "${label}"${e.successes > 0 ? `; say "${name}" to run it` : '; it runs from commands once some trials succeed'}.`);
    return { name, goals, evaluation: e, artifact, wallTimeS: r.wallTimeS };
  }

  /** Why a draft composed by the AI may not replace the one in the Teach panel: unsaved work of the user's. */
  composedDraftRefusal(): string | null {
    const d = this.state.draft;
    if (!d || d.robot !== this.state.robot || d.author === 'ai' || (d.keyframes.length === 0 && !d.gait && d.name.trim() === '')) return null;
    const n = d.keyframes.length;
    return `The Teach panel holds an unsaved draft${d.name.trim() ? ` "${d.name.trim()}"` : ''} with ${d.gait ? 'a gait' : `${n} keyframe${n === 1 ? '' : 's'}`}; the AI does not replace it.`;
  }

  /**
   * Loads a skill the AI composed into the Teach panel, where the user sees
   * and can edit it: the poses become joint targets through the kinematics
   * core, and the goals are checked against this robot. Nothing runs.
   */
  composeDraft(c: ComposedDraft): { ok: true; notes: string[]; intent: IntentFinding[] } | { ok: false; reason: string } {
    const robot = this.state.robot;
    const refusal = this.composedDraftRefusal();
    if (refusal) return { ok: false, reason: refusal };
    const name = c.name.trim().replace(/\s+/g, ' ');
    const problems: string[] = [];
    if (!TAUGHT_NAME_PATTERN.test(name)) problems.push('name must be 1 to 40 letters, digits, spaces, hyphens, underscores or apostrophes, starting with a letter or digit.');
    if (!(c.endHoldS >= 0 && c.endHoldS <= MAX_END_HOLD_S)) problems.push(`the final hold must be from 0 to ${MAX_END_HOLD_S} s.`);
    problems.push(...goalsProblems(c.goals, 'goals'));
    if (problems.length === 0) problems.push(...goalRobotProblems(c.goals, robot));
    const intent = problems.length === 0 ? checkIntent(name, c.request, c.goals, robot) : [];
    for (const f of intent) if (f.goalIndex < 0) problems.push(`The goals do not measure what was asked: "${f.word}" means ${f.meaning}; ${f.hint}.`);
    if ((c.keyframes === undefined) === (c.gait === undefined)) problems.push('Give either keyframes or a gait.');
    let keyframes: DraftKeyframe[] = [];
    let notes: string[] = [];
    if (c.gait) problems.push(...gaitProblems(c.gait, robot), ...gaitGoalConflicts(c.gait, c.goals).map((p) => `The gait does not match its goals: ${p}`));
    else if (c.keyframes) {
      const composed = composeKeyframes(robot, c.keyframes, c.goals);
      if (composed.ok) {
        keyframes = composed.keyframes;
        notes = composed.notes;
      } else problems.push(...composed.problems);
    }
    if (problems.length) return { ok: false, reason: problems.slice(0, 4).join(' ') };
    const draft: TaughtDraft = {
      name, robot, keyframes, endHoldS: c.endHoldS, goals: c.goals.map((g) => ({ ...g })), author: 'ai', notes: notes.slice(0, 8), request: c.request.slice(0, 300),
      gait: c.gait ? { ...c.gait } : null, training: null, policy: null,
    };
    const problem = this.draftProblem(draft);
    if (problem) return { ok: false, reason: problem };
    this.stopActivity('the AI composed a skill');
    this.store.update({ panel: 'teach', draft, message: null });
    return { ok: true, notes: draft.notes, intent };
  }

  /**
   * Loads the left-right mirror image of a saved taught skill into the Teach
   * panel as the AI's draft (D26): its keyframes or gait and its goals for the
   * other side, checked against this robot and the request. Nothing runs.
   */
  mirrorSkill(id: string, request: string, name?: string): { ok: true; name: string; intent: IntentFinding[] } | { ok: false; reason: string } {
    const robot = this.state.robot;
    const a = this.state.library.find((x) => x.id === id);
    if (!a?.taught || (!a.motion && !a.gait)) return { ok: false, reason: 'only skills taught with keyframes or a gait can be mirrored.' };
    if (this.state.hash && a.robot.morphologyHash !== this.state.hash) return { ok: false, reason: `"${a.name}" was evaluated on another body.` };
    const refusal = this.composedDraftRefusal();
    if (refusal) return { ok: false, reason: refusal };
    const newName = (name ?? mirrorName(a.taught.name) ?? '').trim().replace(/\s+/g, ' ');
    if (!newName) return { ok: false, reason: `"${a.taught.name}" names neither left nor right; give the mirrored skill a name.` };
    if (!TAUGHT_NAME_PATTERN.test(newName)) return { ok: false, reason: 'name must be 1 to 40 letters, digits, spaces, hyphens, underscores or apostrophes, starting with a letter or digit.' };
    const m = mirrorTaught(robot, { goals: a.taught.goals, motion: a.motion ?? null, gait: a.gait ?? null });
    if (!m.ok) return m;
    const problems = [...goalsProblems(m.goals, 'goals'), ...goalRobotProblems(m.goals, robot)];
    const intent = problems.length === 0 ? checkIntent(newName, request, m.goals, robot) : [];
    for (const f of intent) if (f.goalIndex < 0) problems.push(`The goals do not measure what was asked: "${f.word}" means ${f.meaning}; ${f.hint}.`);
    if (m.gait) problems.push(...gaitProblems(m.gait, robot), ...gaitGoalConflicts(m.gait, m.goals).map((p) => `The gait does not match its goals: ${p}`));
    if (problems.length) return { ok: false, reason: problems.slice(0, 4).join(' ') };
    const draft: TaughtDraft = {
      name: newName, robot, keyframes: m.motion ? keyframesOf(m.motion) : [], endHoldS: m.motion?.endHoldS ?? 1, goals: m.goals, author: 'ai',
      notes: [`Left-right mirror image of "${a.name}".`, ...(a.method === 'gait-learned' || a.method === 'keyframes-learned' ? ['Its learned feedback policy is not mirrored; train one for this side.'] : [])], request: request.slice(0, 300), gait: m.gait, training: null, policy: null,
    };
    const problem = this.draftProblem(draft);
    if (problem) return { ok: false, reason: problem };
    this.stopActivity('the AI mirrored a skill');
    this.store.update({ panel: 'teach', draft, message: null });
    return { ok: true, name: newName, intent };
  }

  /** Replaces the draft by its left-right mirror image (D26): limbs, sideways motion, turns and goals change sides. */
  mirrorDraft(): boolean {
    const d = this.draftHere();
    if (!d) return false;
    if (!d.gait && d.keyframes.length === 0) {
      this.error('Add keyframes or a gait first.');
      return false;
    }
    const m = mirrorTaught(d.robot, { goals: d.goals, motion: d.gait ? null : draftMotion(d), gait: d.gait });
    if (!m.ok) {
      this.error(`Cannot mirror this skill: ${m.reason}`);
      return false;
    }
    this.stopActivity('the skill was mirrored');
    const name = mirrorName(d.name);
    const renamed = name !== null && name.length <= 40;
    this.patchDraft({ name: renamed ? name : d.name, keyframes: m.motion ? keyframesOf(m.motion) : d.keyframes, gait: m.gait, goals: m.goals });
    this.info(`Mirrored left and right${renamed ? ` as "${name}"` : '; rename it for the other side'}. Try it and evaluate it: the mirror image has not run yet.`);
    return true;
  }

  /** Plays the draft once in the live simulation and resolves with the referee's report; aborting stops it. */
  async tryDraft(signal: AbortSignal): Promise<{ ok: false; reason: string } | { ok: true; report: PlayReport }> {
    if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.' };
    if (!(await this.previewDraft())) return { ok: false, reason: this.state.message?.text ?? 'The draft could not be tried.' };
    // Registered in the same task as previewDraft() returns; physics steps run in later tasks.
    const finished = this.waitForPlay();
    const stop = (): void => this.stopActivity('the AI task was cancelled');
    signal.addEventListener('abort', stop, { once: true });
    const report = await finished;
    signal.removeEventListener('abort', stop);
    return { ok: true, report };
  }

  /** The AI's evaluation of the draft: saved only when at least one trial met every goal; aborting cancels it. */
  async evaluateDraftForAi(signal: AbortSignal): Promise<{ ok: false; reason: string; cancelled?: boolean } | ({ ok: true } & DraftEvaluation)> {
    if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.' };
    if (!this.state.draft || this.state.draft.robot !== this.state.robot) return { ok: false, reason: 'There is no skill draft for this robot in the Teach panel; compose one first.' };
    const cancelVersion = this.jobCancelVersion;
    const cancel = (): void => this.worker.cancel();
    signal.addEventListener('abort', cancel, { once: true });
    try {
      const r = await this.runDraftEvaluation(false);
      return r ? { ok: true, ...r } : { ok: false, reason: this.state.message?.text ?? 'The evaluation did not finish.', cancelled: signal.aborted || cancelVersion !== this.jobCancelVersion };
    } finally {
      signal.removeEventListener('abort', cancel);
    }
  }

  /** Improves the draft by trial and error (the Teach panel). */
  async improveDraft(): Promise<boolean> {
    return (await this.runDraftImprovement()) !== null;
  }

  /** The AI's improvement of the draft; aborting cancels it. */
  async improveDraftForAi(signal: AbortSignal): Promise<{ ok: false; reason: string; cancelled?: boolean } | { ok: true; name: string; result: TaughtImproveResult; wallTimeS: number }> {
    if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.' };
    const d = this.state.draft;
    if (!d || d.robot !== this.state.robot) return { ok: false, reason: 'There is no skill draft for this robot in the Teach panel; compose one first.' };
    const cancelVersion = this.jobCancelVersion;
    const cancel = (): void => this.worker.cancel();
    signal.addEventListener('abort', cancel, { once: true });
    try {
      const r = await this.runDraftImprovement();
      return r ? { ok: true, name: d.name.trim(), ...r } : { ok: false, reason: this.state.message?.text ?? 'The improvement did not finish.', cancelled: signal.aborted || cancelVersion !== this.jobCancelVersion };
    } finally {
      signal.removeEventListener('abort', cancel);
    }
  }

  /**
   * Evolution strategies in the worker adjust the draft's poses and timing
   * (or its gait parameters and feedback gains) to raise the goal reward in
   * randomized trials. The draft changes only when the result scored better
   * than the original on held-out trials.
   */
  private async runDraftImprovement(): Promise<{ result: TaughtImproveResult; wallTimeS: number } | null> {
    const d = this.draftHere();
    if (!d || !this.ready(false)) return null;
    const problem = this.draftProblem(d);
    if (problem) {
      this.error(problem);
      return null;
    }
    this.stopActivity('the skill is being improved');
    const name = d.name.trim() || 'the draft';
    const config = { ...DEFAULT_TAUGHT_IMPROVE, ...(await this.conditionRanges()) };
    // Trial and error tunes the gait itself; a learned policy trained for the old gait does not carry over.
    const r = await this.job<TaughtImproveResult>(`Improving "${name}" by trial and error`, { type: 'improve-taught', robot: d.robot, limbIndex: -1, stanceQ: this.stance, spec: draftSpec({ ...d, policy: null }), goals: d.goals.map((g) => ({ ...g })), config }, this.context());
    if (!r) return null;
    const x = r.result;
    const n = x.heldOutTrials;
    const now = this.state.draft;
    const dropped = x.improved && now === d && d.policy !== null;
    if (x.improved && now === d) {
      const training: DraftTraining = {
        iterations: config.iterations, rollouts: x.rollouts, heldOutTrials: n, baselineReward: +x.baseline.reward.toFixed(4), improvedReward: +x.result.reward.toFixed(4),
        baselineSuccesses: x.baseline.successes, improvedSuccesses: x.result.successes, seed: config.seed, wallTimeS: +r.wallTimeS.toFixed(1),
      };
      const changed = x.spec.kind === 'gait' ? { gait: { ...x.spec.gait } } : x.spec.kind === 'keyframes' ? { keyframes: keyframesOf(x.spec.motion) } : {};
      this.store.update({ draft: { ...d, ...changed, training, policy: null } });
    }
    const b = x.baseline;
    const a = x.result;
    const cost = `${x.rollouts} simulated runs, ${r.wallTimeS.toFixed(1)} s`;
    this.info(
      x.improved
        ? `"${name}": trial and error raised the mean goal reward on ${n} held-out trials from ${b.reward.toFixed(3)} to ${a.reward.toFixed(3)} (trials meeting every goal: ${b.successes} to ${a.successes} of ${n}; ${cost}). The draft now uses the improved version${dropped ? ` without its learned feedback policy, which was trained for the earlier ${d.gait ? 'gait' : 'keyframes'}; train it again` : ''}; evaluate it to save it.`
        : `"${name}": trial and error found nothing better than the draft on ${n} held-out trials (mean goal reward ${b.reward.toFixed(3)}, ${b.successes} of ${n} met every goal; ${cost}). The draft is unchanged.`,
    );
    return r;
  }

  /** Trains a learned feedback policy for the draft (the Teach panel): for its gait (D27) or its keyframes (D30). */
  async trainDraftPolicy(architecture: 'linear' | 'mlp' = 'linear'): Promise<boolean> {
    const d = this.state.draft;
    if (d?.gait && architecture === 'mlp') {
      this.error('Nonlinear feedback training currently supports keyframe drafts only.');
      return false;
    }
    return (d && !d.gait ? await this.runPosePolicyTraining(architecture) : await this.runDraftPolicyTraining()) !== null;
  }

  /** The AI's training of a learned feedback policy for the draft; aborting cancels it. */
  async trainDraftPolicyForAi(
    signal: AbortSignal,
    architecture: 'linear' | 'mlp' = 'linear',
  ): Promise<{ ok: false; reason: string; cancelled?: boolean } | { ok: true; kind: 'gait'; name: string; result: GaitPolicyResult; wallTimeS: number } | { ok: true; kind: 'pose'; name: string; result: PosePolicyResult; wallTimeS: number }> {
    if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.' };
    const d = this.state.draft;
    if (!d || d.robot !== this.state.robot) return { ok: false, reason: 'There is no skill draft for this robot in the Teach panel; compose one first.' };
    if (d.gait && architecture === 'mlp') return { ok: false, reason: 'Nonlinear feedback training currently supports keyframe drafts only.' };
    const cancelVersion = this.jobCancelVersion;
    const cancel = (): void => this.cancelJob();
    signal.addEventListener('abort', cancel, { once: true });
    try {
      const failed = () => ({ ok: false as const, reason: this.state.message?.text ?? 'The training did not finish.', cancelled: signal.aborted || cancelVersion !== this.jobCancelVersion });
      if (d.gait) {
        const r = await this.runDraftPolicyTraining();
        return r ? { ok: true, kind: 'gait', name: d.name.trim(), ...r } : failed();
      }
      const r = await this.runPosePolicyTraining(architecture);
      return r ? { ok: true, kind: 'pose', name: d.name.trim(), ...r } : failed();
    } finally {
      signal.removeEventListener('abort', cancel);
    }
  }

  /**
   * Reinforcement learning of a feedback policy for the draft's keyframes
   * (D30), in the worker pool: augmented random search on randomized runs in
   * which the body is pushed, rewarded by the goals. The draft keeps the
   * policy only when, on held-out pushed runs, it meets every goal more often
   * than the keyframes alone without more falls.
   */
  private async acquirePool(): Promise<CancellableRunner> {
    if (this.pool) return this.pool;
    if (this.makePool) return (this.pool = this.makePool());
    const { defaultWorkerCount, RolloutPool } = await import('../../skills/rollouts');
    return (this.pool = new RolloutPool(() => new SkillWorkerClient(), defaultWorkerCount(globalThis.navigator?.hardwareConcurrency)));
  }

  private async runPosePolicyTraining(architecture: 'linear' | 'mlp'): Promise<{ result: PosePolicyResult; wallTimeS: number } | null> {
    const d = this.draftHere();
    if (!d || !this.ready(false)) return null;
    if (d.robot.base.kind !== 'floating') {
      this.error('A learned feedback policy for keyframes balances a free-standing body from its IMU; this robot is fixed to the ground.');
      return null;
    }
    const problem = this.draftProblem(d);
    if (problem) {
      this.error(problem);
      return null;
    }
    this.stopActivity('a feedback policy is being trained');
    const name = d.name.trim() || 'the draft';
    const ctx = this.context();
    const { DEFAULT_POSE_POLICY, trainPosePolicy } = await import('../../skills/trainPose');
    const cfg: PosePolicyConfig = { ...DEFAULT_POSE_POLICY, ...(await this.conditionRanges()), ...this.posePolicyConfig, architecture, ...(d.policy?.kind === 'pose' ? { incumbentPolicy: d.policy.policy } : {}) };
    const pool = await this.acquirePool();
    const label = `Training a learned feedback policy for "${name}"`;
    this.store.update({ job: { label, fraction: 0, detail: `Starting ${pool.workers} worker${pool.workers === 1 ? '' : 's'}` }, message: null });
    const t0 = performance.now();
    try {
      const result = await trainPosePolicy(pool, d.robot, this.stance, draftMotion(d), d.goals.map((g) => ({ ...g })), cfg, (fraction, detail) => {
        if (this.isCurrent(ctx)) this.store.update({ job: { label, fraction, detail } });
      });
      const wallTimeS = (performance.now() - t0) / 1000;
      if (!this.isCurrent(ctx)) {
        this.info(`${label}: the result was discarded because the robot changed while it ran.`);
        return null;
      }
      if (result.improved && this.state.draft === d) {
        const policy: DraftPolicy = {
          kind: 'pose', policy: result.policy, iterations: result.history.length, rollouts: result.rollouts, parameters: result.parameters, heldOutTrials: result.heldOutTrials,
          baseline: result.baseline, result: result.result, seed: cfg.seed, wallTimeS: +wallTimeS.toFixed(1), pushImpulseNs: result.pushImpulseNs,
        };
        this.store.update({ draft: { ...d, policy } });
      }
      const b = result.baseline;
      const a = result.result;
      const n = result.heldOutTrials;
      const pushes = `${result.pushImpulseNs[0]} to ${result.pushImpulseNs[1]} N*s pushes`;
      const cost = `${result.parameters} parameters, ${result.rollouts} simulated runs on ${result.workers} worker${result.workers === 1 ? '' : 's'}, ${wallTimeS.toFixed(1)} s`;
      const compared = `every goal met in ${a.successes} and ${b.successes}, goal reward ${a.meanGoalReward.toFixed(3)} and ${b.meanGoalReward.toFixed(3)}, falls ${a.falls} and ${b.falls}`;
      const current = result.incumbent ? ` Current policy on the same runs: ${result.incumbent.successes}/${n} successes, ${result.incumbent.falls} falls, goal reward ${result.incumbent.meanGoalReward.toFixed(3)}.` : '';
      const ablation = result.feedbackAblation ? ` Sensor ablation (inputs fixed at training means, phase preserved): ${result.feedbackAblation.successes}/${n} successes, ${result.feedbackAblation.falls} falls, goal reward ${result.feedbackAblation.meanGoalReward.toFixed(3)}.` : '';
      this.info(
        (result.improved
          ? `"${name}": on ${n} held-out runs with ${pushes}, with the learned feedback policy and with the keyframes alone: ${compared} (${cost}). The draft now uses it${a.successes === 0 ? ', closer to the goals but not yet meeting them all' : ''}; evaluate it to save it.`
          : `"${name}": the learned feedback policy did not beat ${result.incumbent ? 'both the keyframes and the current policy' : 'the keyframes alone'} on ${n} held-out runs with ${pushes} (with and without it: ${compared}; ${cost}).${current} The draft is unchanged.`) + ablation,
      );
      return { result, wallTimeS };
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err);
      if (text === 'Cancelled.') this.info(`${label} was cancelled; the draft is unchanged.`);
      else this.error(`${label} failed: ${text}`);
      return null;
    } finally {
      pool.dispose();
      if (this.pool === pool) this.pool = null;
      this.store.update({ job: null });
    }
  }

  /**
   * Augmented random search in the worker trains a linear feedback policy
   * that corrects the gait's joint targets every control step from the IMU
   * and the gait phase (D27). The draft keeps the policy only when it beat
   * the gait alone on held-out randomized trials without more falls or fewer
   * goal successes.
   */
  private async runDraftPolicyTraining(): Promise<{ result: GaitPolicyResult; wallTimeS: number } | null> {
    const d = this.draftHere();
    if (!d || !this.ready(false)) return null;
    if (!d.gait) {
      this.error('A learned feedback policy corrects a gait; this draft is a keyframe motion.');
      return null;
    }
    const problem = this.draftProblem(d);
    if (problem) {
      this.error(problem);
      return null;
    }
    this.stopActivity('a feedback policy is being trained');
    const name = d.name.trim() || 'the draft';
    const config = { ...DEFAULT_GAIT_POLICY, ...(await this.conditionRanges()) };
    const r = await this.job<GaitPolicyResult>(`Training a learned feedback policy for "${name}"`, { type: 'train-gait-policy', robot: d.robot, limbIndex: -1, stanceQ: this.stance, gait: { ...d.gait }, goals: d.goals.map((g) => ({ ...g })), config }, this.context());
    if (!r) return null;
    const x = r.result;
    const n = x.heldOutTrials;
    if (x.improved && this.state.draft === d) {
      const policy: DraftPolicy = { kind: 'gait', policy: x.policy, iterations: x.history.length, rollouts: x.rollouts, parameters: x.parameters, heldOutTrials: n, baseline: x.baseline, result: x.result, seed: config.seed, wallTimeS: +r.wallTimeS.toFixed(1) };
      this.store.update({ draft: { ...d, policy } });
    }
    const b = x.baseline;
    const a = x.result;
    const cms = (v: number) => `${(v * 100).toFixed(1)} cm/s`;
    const cost = `${x.parameters} parameters, ${x.rollouts} simulated runs, ${r.wallTimeS.toFixed(1)} s`;
    this.info(
      x.improved
        ? `"${name}": the learned feedback policy raised the training return on ${n} held-out trials from ${b.meanReturn.toFixed(3)} to ${a.meanReturn.toFixed(3)} (goal reward ${b.meanGoalReward.toFixed(3)} to ${a.meanGoalReward.toFixed(3)}; velocity error ${cms(b.trackingErrorMps)} to ${cms(a.trackingErrorMps)}; trials meeting every goal ${b.successes} to ${a.successes} of ${n}; falls ${b.falls} to ${a.falls}; ${cost}). The draft now uses it; evaluate it to save it.`
        : `"${name}": the learned feedback policy did not beat the gait alone on ${n} held-out trials (training return ${b.meanReturn.toFixed(3)} without it, ${a.meanReturn.toFixed(3)} with it; goal reward ${b.meanGoalReward.toFixed(3)} and ${a.meanGoalReward.toFixed(3)}; goals met ${b.successes} and ${a.successes} of ${n}; falls ${b.falls} and ${a.falls}; ${cost}). The draft is unchanged.`,
    );
    return r;
  }

  /** Names of taught skills that can run on this robot by command. */
  private taughtNamesHere(hash: string): string[] {
    const s = this.state;
    return [...new Set(s.library.filter((a) => a.skillType === 'taught' && a.taught && checkSkillCompatibility(a, s.robot, hash).ok && commandEligibility(a) === null).map((a) => a.taught!.name))];
  }

  /** The taught skill a command names (its name, with words like "please" or "do the" around it), if any. */
  taughtSkillFor(text: string): string | null {
    return matchTaughtSkill(text, this.taughtNames);
  }

  /** A taught skill whose name appears as whole words in a command that says more than the name, if any. */
  taughtSkillMentioned(text: string): string | null {
    const t = ` ${normalizeSkillName(text)} `;
    return this.taughtNames.find((n) => t.includes(` ${normalizeSkillName(n)} `)) ?? null;
  }

  private get taughtNames(): string[] {
    return [...new Set(this.state.library.flatMap((a) => (a.skillType === 'taught' && a.taught ? [a.taught.name] : [])))];
  }

  private ready(footLift = true): boolean {
    const s = this.state;
    if (footLift && s.unsupported) {
      this.error(s.unsupported);
      return false;
    }
    if (s.job || this.trainingElsewhere) {
      this.error(s.job ? 'Another job is running. Wait for it or cancel it first.' : 'A walking controller is being trained for Navigate. Wait for it or cancel it there first.');
      return false;
    }
    return true;
  }

  /** The robot and leg the workspace is on now. */
  private context(): WorkContext {
    return { robot: this.state.robot, limbId: this.state.limbId };
  }

  private isCurrent(c: WorkContext): boolean {
    return c.robot === this.state.robot && c.limbId === this.state.limbId;
  }

  /**
   * Runs a worker job bound to the current robot and leg. A result that
   * arrives after either changed is discarded instead of overwriting newer state.
   */
  private async job<T>(label: string, job: SkillJob, ctx: WorkContext = this.context()): Promise<{ result: T; wallTimeS: number } | null> {
    this.store.update({ job: { label, fraction: 0, detail: 'Starting' }, message: null });
    try {
      const r = await this.worker.run<T>(job, (fraction, detail) => {
        if (this.isCurrent(ctx)) this.store.update({ job: { label, fraction, detail } });
      });
      if (!this.isCurrent(ctx)) {
        this.info(`${label}: the result was discarded because the robot or leg changed while it ran.`);
        return null;
      }
      return r;
    } catch (e) {
      const text = e instanceof Error ? e.message : String(e);
      if (text === 'Cancelled.') this.info(`${label} was cancelled.`);
      else this.error(`${label} failed: ${text}`);
      return null;
    } finally {
      this.store.update({ job: null });
    }
  }

  private common() {
    return { robot: this.state.robot, limbIndex: this.limbIndex, stanceQ: this.stance };
  }

  /** Joint gains of the live simulation for this robot; evaluations, training and playback all use them. */
  private async liveGains(): Promise<PdGains> {
    await this.physics.init();
    return this.physics.state.description === this.state.robot ? { ...this.physics.state.gains } : { ...DEFAULT_SKILL_GAINS };
  }

  /** Robot-relative randomization (mass around this robot's own) with the live gains. */
  private async conditionRanges() {
    const o = evaluationOptionsFor(this.state.robot, 0, await this.liveGains());
    return { frictionRange: o.frictionRange, baseMassRange: o.baseMassRange, jointNoiseRad: o.jointNoiseRad, gains: o.gains };
  }

  cancelJob(): void {
    this.jobCancelVersion++;
    this.worker.cancel();
    this.pool?.cancel();
  }

  async recordExpertDemonstration(): Promise<void> {
    if (!this.ready()) return;
    const ctx = this.context();
    const gains = await this.liveGains();
    const r = await this.job<Trajectory>('Recording the expert demonstration', { type: 'demonstrate', ...this.common(), params: this.state.params, gains }, ctx);
    if (r) {
      this.store.update({ demonstration: { source: 'expert', trajectory: r.result, context: ctx } });
      this.info(`Recorded the state machine as a demonstration: ${((r.result.frames.length - 1) * r.result.dt).toFixed(1)} s of joint targets.`);
    }
  }

  async trainImitation(): Promise<void> {
    if (!this.ready()) return;
    const ctx = this.context();
    const config: ImitationConfig = { ...DEFAULT_IMITATION, ...(await this.conditionRanges()) };
    const r = await this.job<{ policy: PolicyJson; horizonS: number; initialLoss: number; finalLoss: number; samples: number }>('Imitation learning', { type: 'imitation', ...this.common(), expert: this.state.params, config }, ctx);
    if (r) {
      this.store.update({ imitation: { ...r.result, wallTimeS: r.wallTimeS, config, context: ctx } });
      this.info(`Trained a neural policy on ${r.result.samples} expert samples in ${r.wallTimeS.toFixed(1)} s (loss ${r.result.initialLoss.toFixed(3)} to ${r.result.finalLoss.toExponential(2)}). Evaluate it before trusting it.`);
    }
  }

  async improve(): Promise<void> {
    if (!this.ready()) return;
    const ctx = this.context();
    const config: ImproveConfig = { ...DEFAULT_IMPROVE, ...(await this.conditionRanges()) };
    const r = await this.job<{ params: FootLiftParams; baselineReward: number; improvedReward: number; trialsRun: number }>('Reward-based improvement', { type: 'improve', ...this.common(), base: this.state.params, config }, ctx);
    if (r) {
      this.store.update({ improvement: { ...r.result, wallTimeS: r.wallTimeS, config, context: ctx } });
      const better = r.result.improvedReward > r.result.baselineReward;
      this.info(`Evolution strategies ran ${r.result.trialsRun} trials in ${r.wallTimeS.toFixed(1)} s. Held-out reward: baseline ${r.result.baselineReward.toFixed(3)}, result ${r.result.improvedReward.toFixed(3)}${better ? '' : ' (no improvement)'}.`);
    }
  }

  /** Spec for a method from the current workspace, or a reason it is not available yet. */
  specFor(method: SkillMethod): SkillSpec | string {
    const s = this.state;
    const stale = (c: WorkContext) => !this.isCurrent(c);
    switch (method) {
      case 'state-machine': return { kind: 'state-machine', params: { ...DEFAULT_FOOT_LIFT } };
      case 'tuned-parameters': return { kind: 'tuned-parameters', params: { ...s.params } };
      case 'replay':
        if (!s.demonstration) return 'Record a demonstration first.';
        return stale(s.demonstration.context) ? 'The demonstration was recorded on another robot or leg; record it again.' : { kind: 'replay', trajectory: s.demonstration.trajectory };
      case 'imitation-mlp':
        if (!s.imitation) return 'Train the imitation policy first.';
        return stale(s.imitation.context) ? 'The policy was trained on another robot or leg; train it again.' : { kind: 'imitation-mlp', policy: s.imitation.policy, durationS: s.imitation.horizonS };
      case 'es-improved':
        if (!s.improvement) return 'Run the reward-based improvement first.';
        return stale(s.improvement.context) ? 'The improvement ran on another robot or leg; run it again.' : { kind: 'es-improved', params: s.improvement.params };
      case 'keyframes':
      case 'keyframes-learned':
      case 'gait':
      case 'gait-learned':
        return 'Keyframe and gait skills are taught and evaluated in the Teach a skill panel.';
    }
  }

  /** Evaluates a method over 20 randomized trials in the worker and adds the result to the library. */
  async evaluate(method: SkillMethod): Promise<SkillArtifact | null> {
    if (!this.ready()) return null;
    const spec = this.specFor(method);
    if (typeof spec === 'string') {
      this.error(spec);
      return null;
    }
    const ctx = this.context();
    const s = this.state;
    const hash = s.hash ?? (await morphologyHash(s.robot));
    const options = evaluationOptionsFor(s.robot, 2026 + this.evalCounter++, await this.liveGains());
    const r = await this.job<EvaluationSummary>(`Evaluating: ${METHOD_LABELS[method]}`, { type: 'evaluate', ...this.common(), spec, options }, ctx);
    if (!r) return null;
    const limb = s.robot.limbs.find((l) => l.id === s.limbId)!;
    const training: SkillArtifact['training'] =
      method === 'imitation-mlp' && s.imitation
        ? { algorithm: 'behavior cloning (MSE, Adam)', config: { episodes: s.imitation.config.episodes, hidden: s.imitation.config.hidden.join('x'), batchSize: s.imitation.config.batchSize, learningRate: s.imitation.config.learningRate }, seed: s.imitation.config.seed, iterations: s.imitation.config.iterations, wallTimeS: +s.imitation.wallTimeS.toFixed(2) }
        : method === 'es-improved' && s.improvement
          ? { algorithm: 'evolution strategies on state-machine parameters', config: { pairs: s.improvement.config.pairs, sigma: s.improvement.config.sigma, baselineReward: +s.improvement.baselineReward.toFixed(3), improvedReward: +s.improvement.improvedReward.toFixed(3) }, seed: s.improvement.config.seed, iterations: s.improvement.config.iterations, wallTimeS: +s.improvement.wallTimeS.toFixed(2) }
          : null;
    const artifact = createSkillArtifact({ robot: s.robot, morphologyHash: hash, limbId: s.limbId, spec, name: `Lift ${limb.name.toLowerCase()} (${method})`, evaluation: r.result, training });
    this.store.update({ library: [artifact, ...this.state.library] });
    this.persist('the skill', (p) => p.saveSkill(artifact));
    this.info(`${METHOD_LABELS[method]}: ${r.result.successes} of ${r.result.trials} trials succeeded in ${r.wallTimeS.toFixed(1)} s. Added to the library.`);
    return artifact;
  }

  /**
   * Runs a stored skill's own controller through a fresh randomized
   * evaluation on this robot and replaces its evaluation with the local
   * result. For an imported skill, the file's claim is kept for comparison.
   */
  async reevaluate(id: string): Promise<SkillArtifact | null> {
    const a = this.state.library.find((x) => x.id === id);
    if (!a || !this.ready(a.skillType !== 'taught')) return null;
    const s = this.state;
    const hash = s.hash ?? (await morphologyHash(s.robot));
    const compat = checkSkillCompatibility(a, s.robot, hash);
    if (!compat.ok) {
      this.error(`Cannot re-evaluate here: ${compat.reason}`);
      return null;
    }
    let spec: SkillSpec;
    try {
      spec = specFromArtifact(a);
    } catch (e) {
      this.error(`Cannot re-evaluate: ${e instanceof Error ? e.message : String(e)}`);
      return null;
    }
    const ctx = this.context();
    const limbIndex = s.robot.limbs.findIndex((l) => l.id === a.limbs[0]);
    const options = evaluationOptionsFor(s.robot, 2026 + this.evalCounter++, await this.liveGains());
    const r = await this.job<EvaluationSummary>(`Re-evaluating ${a.name}`, { type: 'evaluate', robot: s.robot, limbIndex, stanceQ: this.stance, spec, options, ...(a.skillType === 'taught' ? { goals: a.taught?.goals ?? [] } : {}) }, ctx);
    if (!r) return null;
    const current = this.state.library.find((x) => x.id === id);
    if (!current) return null;
    const claimed = evaluationOrigin(current) === 'imported' && current.evaluation ? current.evaluation : current.importedEvaluation;
    const updated: SkillArtifact = {
      ...current,
      evaluation: summarizeEvaluation(r.result),
      evaluationOrigin: 'local',
      updatedAt: new Date().toISOString(),
      ...(claimed ? { importedEvaluation: claimed } : {}),
    };
    this.store.update({ library: this.state.library.map((x) => (x.id === id ? updated : x)) });
    this.persist('the re-evaluated skill', (p) => p.saveSkill(updated));
    this.info(`Re-evaluated ${a.name} here: ${r.result.successes} of ${r.result.trials} trials succeeded${claimed ? ` (the file claimed ${claimed.successes} of ${claimed.trials})` : ''}.`);
    return updated;
  }

  deleteSkill(id: string): void {
    if (this.state.playing === id) this.stopActivity('the skill was deleted');
    this.store.update({ library: this.state.library.filter((a) => a.id !== id) });
    this.persist('the deletion', (p) => p.deleteSkill(id));
    this.skillMemory?.forgetSkillRuns(id);
  }

  /** Adds stored skills (for example from persistence) to the library. */
  addSkills(skills: SkillArtifact[]): void {
    const known = new Set(this.state.library.map((a) => a.id));
    this.store.update({ library: [...this.state.library, ...skills.filter((a) => !known.has(a.id))] });
  }

  /** Adds stored walking controllers (one per morphology). */
  addLocomotion(controllers: LocomotionController[]): void {
    for (const c of controllers) this.controllers.set(c.robot.morphologyHash, c);
    const h = this.state.hash;
    if (h && this.controllers.has(h)) this.store.update({ walking: this.controllers.get(h)! });
  }

  /** The walking controller trained for this morphology (in any view), or null. */
  walkingController(morphologyHash: string): LocomotionController | null {
    return this.controllers.get(morphologyHash) ?? null;
  }

  private trainingElsewhere = false;

  /**
   * Trains a walking controller for another view's robot (Navigate) without
   * switching this tab's robot: the same calibration, policy search,
   * held-out comparison and evaluation as trainLocomotion, with the joint
   * gains Navigate walks with, saved by morphology for every view.
   */
  async trainWalkingFor(robot: RobotDescription, onProgress: (fraction: number, detail: string) => void, signal: AbortSignal): Promise<{ ok: false; reason: string } | { ok: true; controller: LocomotionController; result: LocomotionResult; wallTimeS: number }> {
    if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.' };
    if (this.state.job || this.trainingElsewhere) return { ok: false, reason: 'The AI Playground is running another job; wait for it or cancel it first.' };
    try {
      locomotionBase(robot);
    } catch (e) {
      return { ok: false, reason: `${robot.name} cannot walk with a gait: ${e instanceof Error ? e.message : String(e)}` };
    }
    this.trainingElsewhere = true;
    const pool = await this.acquirePool();
    const cancel = (): void => pool.cancel();
    signal.addEventListener('abort', cancel, { once: true });
    const t0 = performance.now();
    try {
      const [{ DEFAULT_LOCOMOTION, locomotionController, trainLocomotion }, hash] = await Promise.all([
        import('../../skills/locomotion'),
        morphologyHash(robot),
      ]);
      const o = evaluationOptionsFor(robot, 0, { ...DEFAULT_SKILL_GAINS });
      const cfg: LocomotionConfig = { ...DEFAULT_LOCOMOTION, frictionRange: o.frictionRange, baseMassRange: o.baseMassRange, jointNoiseRad: o.jointNoiseRad, gains: o.gains, ...this.locomotionConfig };
      const result = await trainLocomotion(pool, robot, homeConfiguration(robot), cfg, onProgress);
      const wallTimeS = (performance.now() - t0) / 1000;
      const controller = locomotionController(result, { robotId: robot.id, morphologyHash: hash }, cfg, wallTimeS);
      this.controllers.set(hash, controller);
      if (this.state.hash === hash) this.store.update({ walking: controller });
      this.persist('the walking controller', (p) => p.saveLocomotion?.(controller) ?? Promise.resolve());
      return { ok: true, controller, result, wallTimeS };
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err);
      return { ok: false, reason: text === 'Cancelled.' ? 'The training was cancelled; nothing was saved.' : `The training failed: ${text}` };
    } finally {
      signal.removeEventListener('abort', cancel);
      // Idle workers would hold a MuJoCo instance each; the next training starts fresh ones.
      pool.dispose();
      if (this.pool === pool) this.pool = null;
      this.trainingElsewhere = false;
    }
  }

  /** Trains this body's walking controller (the Walk panel). */
  async trainLocomotion(): Promise<boolean> {
    return (await this.runLocomotionTraining()) !== null;
  }

  /** The AI's training of the walking controller; aborting cancels it. */
  async trainLocomotionForAi(signal: AbortSignal): Promise<{ ok: false; reason: string; cancelled?: boolean } | { ok: true; controller: LocomotionController; result: LocomotionResult; wallTimeS: number }> {
    if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.', cancelled: true };
    const cancelVersion = this.jobCancelVersion;
    const cancel = (): void => this.cancelJob();
    signal.addEventListener('abort', cancel, { once: true });
    try {
      const r = await this.runLocomotionTraining();
      return r ? { ok: true, ...r } : { ok: false, reason: this.state.message?.text ?? 'The training did not finish.', cancelled: signal.aborted || cancelVersion !== this.jobCancelVersion };
    } finally {
      signal.removeEventListener('abort', cancel);
    }
  }

  /**
   * Learns a walking controller for this body in a pool of workers (D28):
   * calibration of the gait generator, a feedback policy by random search
   * over random commands, a held-out comparison and an evaluation. It replaces
   * an earlier controller of the same morphology.
   */
  private async runLocomotionTraining(): Promise<{ controller: LocomotionController; result: LocomotionResult; wallTimeS: number } | null> {
    if (!this.ready(false)) return null;
    const robot = this.state.robot;
    try {
      locomotionBase(robot);
    } catch (e) {
      this.error(`${robot.name} cannot walk with a gait: ${e instanceof Error ? e.message : String(e)}`);
      return null;
    }
    this.stopActivity('a walking controller is being trained');
    const ctx = this.context();
    const [{ DEFAULT_LOCOMOTION, locomotionController, trainLocomotion }, hash] = await Promise.all([
      import('../../skills/locomotion'),
      this.state.hash ? Promise.resolve(this.state.hash) : morphologyHash(robot),
    ]);
    const cfg: LocomotionConfig = { ...DEFAULT_LOCOMOTION, ...(await this.conditionRanges()), ...this.locomotionConfig };
    const pool = await this.acquirePool();
    const label = `Training a walking controller for ${robot.name}`;
    this.store.update({ job: { label, fraction: 0, detail: `Starting ${pool.workers} worker${pool.workers === 1 ? '' : 's'}` }, message: null });
    const t0 = performance.now();
    try {
      const result = await trainLocomotion(pool, robot, this.stance, cfg, (fraction, detail) => {
        if (this.isCurrent(ctx)) this.store.update({ job: { label, fraction, detail } });
      });
      const wallTimeS = (performance.now() - t0) / 1000;
      if (!this.isCurrent(ctx)) {
        this.info(`${label}: the result was discarded because the robot changed while it ran.`);
        return null;
      }
      const controller = locomotionController(result, { robotId: robot.id, morphologyHash: hash }, cfg, wallTimeS);
      this.controllers.set(hash, controller);
      this.store.update({ walking: controller });
      this.persist('the walking controller', (p) => p.saveLocomotion?.(controller) ?? Promise.resolve());
      const e = controller.evaluation;
      const cm = (m: number) => `${(m * 100).toFixed(1)} cm`;
      this.info(
        `Walking controller for ${robot.name}: ${e.final.successes} of ${e.trials} randomized commands ended within tolerance (mean position error ${cm(e.final.meanPositionErrorM)}); the generator without the learned calibration managed ${e.uncalibrated.successes} (${cm(e.uncalibrated.meanPositionErrorM)}). ` +
          `${result.policy ? 'The learned feedback policy beat the calibrated generator on held-out commands and walks with it.' : 'The feedback policy did not beat the calibrated generator on held-out commands, so the calibrated generator walks alone.'} ${result.rollouts} simulated runs on ${result.workers} worker${result.workers === 1 ? '' : 's'}, ${wallTimeS.toFixed(1)} s.`,
      );
      return { controller, result, wallTimeS };
    } catch (err) {
      const text = err instanceof Error ? err.message : String(err);
      if (text === 'Cancelled.') this.info(`${label} was cancelled; nothing was saved.`);
      else this.error(`${label} failed: ${text}`);
      return null;
    } finally {
      // Idle workers would hold a MuJoCo instance each; the next training starts fresh ones.
      pool.dispose();
      if (this.pool === pool) this.pool = null;
      this.store.update({ job: null });
    }
  }

  deleteLocomotion(): void {
    const h = this.state.hash;
    if (!h || !this.controllers.has(h)) return;
    this.controllers.delete(h);
    this.store.update({ walking: null, lastWalk: null, message: null });
    this.persist('the deletion', (p) => p.deleteLocomotion?.(h) ?? Promise.resolve());
  }

  setWalkCommand(patch: Partial<WalkCommand & { durationS: number }>): void {
    if (!Object.values(patch).every((v) => Number.isFinite(v))) return;
    this.store.update({ walkCommand: { ...this.state.walkCommand, ...patch } });
  }

  /** Walks the Walk panel's command. */
  async walkFromPanel(): Promise<WalkReport | null> {
    const { durationS, ...cmd } = this.state.walkCommand;
    const r = await this.walk(cmd, durationS);
    if (!r.ok) this.error(r.reason);
    return r.ok ? r.report : null;
  }

  /**
   * Walks a command with this body's controller in the live simulation and
   * reports where the body went (ground truth) against where the command
   * leads. The controller sees sensors only. Aborting stops the walk.
   */
  async walk(cmd: WalkCommand, durationS: number, signal?: AbortSignal): Promise<{ ok: false; reason: string } | { ok: true; report: WalkReport }> {
    const s = this.state;
    const robot = s.robot;
    const c = s.walking;
    if (!c) return { ok: false, reason: `No walking controller is trained for ${robot.name} yet; train one first.` };
    const hash = s.hash ?? (await morphologyHash(robot));
    const problem = locomotionProblem(c, robot, hash) ?? rangeProblem(cmd, c.ranges, durationS);
    if (problem) return { ok: false, reason: problem.charAt(0).toUpperCase() + problem.slice(1) };
    const wg = walkGait(c.base, c.calibration, cmd, durationS);
    const spec: SkillSpec = c.policy ? { kind: 'gait-learned', gait: wg.gait, policy: c.policy, walk: { command: vecOf(cmd) } } : { kind: 'gait', gait: wg.gait };
    const goals = [defaultGoal('upright', robot), defaultGoal('noFall', robot)];
    const text = `${cmd.forwardMps} m/s forward, ${cmd.leftMps} m/s left, ${cmd.turnRadps} rad/s turning for ${durationS} s`;
    const started = await this.startLive({
      skillId: null, name: 'walk by command', method: spec.kind, morphologyHash: hash, spec, judging: { goals }, limbIndex: -1, requiredHoldS: 0, evaluation: null, fromCurrentPose: true,
      label: `Walking ${text} with the learned walking controller (${c.policy ? 'calibrated gait generator and feedback policy' : 'calibrated gait generator'}). The controller sees sensors only.`,
    });
    if (!started) return { ok: false, reason: this.state.message?.text ?? 'The walk could not start.' };
    const continued = this.lastStartContinued;
    const finished = this.waitForPlay();
    const stop = (): void => this.stopActivity('the walk was cancelled');
    signal?.addEventListener('abort', stop, { once: true });
    const rep = await finished;
    signal?.removeEventListener('abort', stop);
    const end = rep.checkpoints?.find((x) => x.label === 'end of the run');
    const measured: Vec3 = end ? [end.forwardM, end.leftM, (end.turnDeg * Math.PI) / 180] : [Number.NaN, Number.NaN, Number.NaN];
    const report: WalkReport = {
      command: { ...cmd }, durationS, continued, outcome: rep.outcome, judgement: judgeWalk(expectedPose(cmd, wg.gait), measured, rep.outcome === 'succeeded'),
      policy: c.policy !== null, calibrated: wg.calibrated, clamped: wg.clamped, maxTiltRad: rep.maxTiltRad, reason: rep.reason,
    };
    this.store.update({ lastWalk: report });
    return { ok: true, report };
  }

  /**
   * Robots to export with the library: the open robot and every robot a skill
   * was evaluated on, and the skills' morphologies whose robot is not available.
   */
  exportPlan(): { robots: RobotDescription[]; missing: string[] } {
    const robots = [this.state.robot];
    const seen = new Set<string>(this.state.hash ? [this.state.hash] : []);
    const missing: string[] = [];
    for (const a of this.state.library) {
      const h = a.robot.morphologyHash;
      if (seen.has(h)) continue;
      seen.add(h);
      const r = this.knownRobots.get(h) ?? this.persistence?.findRobot?.(a.robot.robotId, h) ?? null;
      if (!r) missing.push(`${a.robot.robotId} (morphology ${h.slice(0, 12)})`);
      else if (r !== this.state.robot) robots.push(r);
    }
    return { robots, missing };
  }

  /** Bundle text with every skill in the library and the robots they were evaluated on. */
  exportText(exportedAt: string = new Date().toISOString()): string {
    const { robots, missing } = this.exportPlan();
    if (missing.length) this.info(`Exported ${this.state.library.length} skills and ${robots.length} robots. Not stored here, so not in the file: ${missing.join(', ')}.`);
    return exportBundle(robots, exportedAt, this.state.library);
  }

  /** Reports an import refused before reading the file (for example, too large). */
  refuseImport(message: string): void {
    this.error(`Import refused: ${message}`);
  }

  /**
   * Adds the valid skills of a bundle to the library and saves them, with the
   * robots they were evaluated on. They are marked as imported: their
   * evaluation comes from the file and was not re-run here. Returns the
   * number of skills added.
   */
  async importText(text: string, now: string = new Date().toISOString()): Promise<number> {
    const res = importBundle(text);
    if (res.errors.length) {
      this.refuseImport(res.errors[0]!);
      return 0;
    }
    const known = new Set(this.state.library.map((a) => a.id));
    // Whatever the file says, here its evaluation is an unverified claim until re-run.
    const fresh = res.skills.filter((a) => !known.has(a.id)).map((a): SkillArtifact => ({ ...a, importedAt: now, evaluationOrigin: 'imported' }));
    if (fresh.length) {
      this.store.update({ library: [...fresh, ...this.state.library] });
      for (const a of fresh) this.persist('the imported skill', (p) => p.saveSkill(a));
    }
    // Robots the skills were evaluated on are kept, so the skills can be opened, re-evaluated and exported again.
    const wanted = new Set(res.skills.map((a) => a.robot.morphologyHash));
    let saved = 0, others = 0;
    for (const r of res.robots) {
      const h = await morphologyHash(r);
      if (!wanted.has(h)) {
        others++;
        continue;
      }
      if (!this.knownRobots.has(h)) this.knownRobots.set(h, r);
      if (this.persistence?.storeRobot) {
        saved++;
        this.persist('a robot from the file', (p) => p.storeRobot!(r));
      }
    }
    const rejected = res.rejected.filter((r) => r.kind === 'skill');
    const parts = [res.skills.length + rejected.length === 0 ? 'The file contains no skills.' : `Imported ${fresh.length} skill${fresh.length === 1 ? '' : 's'}.`];
    const duplicates = res.skills.length - fresh.length;
    if (duplicates) parts.push(`${duplicates} ${duplicates === 1 ? 'was' : 'were'} already in the library.`);
    if (fresh.length) parts.push('Their evaluation results come from the file and were not re-run here; text commands will not use them until you re-evaluate them here.');
    if (rejected.length) parts.push(`Rejected: ${rejected.map((r) => `skill ${r.index + 1}: ${r.reasons.join(' ')}`).join('; ')}`);
    if (saved) parts.push(`Saved ${saved} robot${saved === 1 ? '' : 's'} the skills were evaluated on.`);
    if (others) parts.push(`${others} other robot${others === 1 ? '' : 's'} in the file can be imported in the Build tab.`);
    if (fresh.length || (duplicates && !rejected.length)) this.info(parts.join(' '));
    else this.error(parts.join(' '));
    return fresh.length;
  }

  /**
   * Plays a skill once in the live simulation. The controller sees sensors
   * only; the evaluation's referee judges the run from simulator ground truth
   * (a foot lift: clearance, the required continuous hold, tilt, height and
   * support contact; a taught skill: its goals). Refuses incompatible
   * morphologies and controllers that cannot be built. `spec` overrides the
   * stored controller (for example, a different hold time); `requiredHoldS`
   * is the hold the referee demands (default: the evaluation's).
   */
  async play(id: string, spec?: SkillSpec, opts: { requiredHoldS?: number } = {}): Promise<boolean> {
    const a = this.state.library.find((x) => x.id === id);
    if (!a) return false;
    const s = this.state;
    const hash = s.hash ?? (await morphologyHash(s.robot));
    const compat = checkSkillCompatibility(a, s.robot, hash);
    if (!compat.ok) {
      this.error(`Refused: ${compat.reason}`);
      return false;
    }
    let runSpec: SkillSpec;
    try {
      runSpec = spec ?? specFromArtifact(a);
    } catch (e) {
      this.error(`Refused: the stored controller cannot run on this robot (${e instanceof Error ? e.message : String(e)}).`);
      return false;
    }
    const taught = a.skillType === 'taught';
    const criteria = { ...(a.evaluation?.context?.criteria ?? DEFAULT_CRITERIA) };
    return this.startLive({
      skillId: id,
      name: a.name,
      method: a.method,
      morphologyHash: a.robot.morphologyHash,
      spec: runSpec,
      judging: taught ? { goals: a.taught?.goals ?? [] } : criteria,
      limbIndex: taught ? -1 : s.robot.limbs.findIndex((l) => l.id === a.limbs[0]),
      requiredHoldS: taught ? 0 : (opts.requiredHoldS ?? criteria.minHoldS),
      evaluation: a.evaluation,
      label: `Playing "${a.name}" (${skillLabel(a)}). The controller sees sensors only.`,
      fromCurrentPose: runSpec.kind === 'gait' || runSpec.kind === 'gait-learned',
    });
  }

  /** The live body is up (tilt under 0.35 rad, body above 70 % of its resting height): a walk may go on from here. Ground truth, used only to decide on a reset. */
  private standingUp(robot: RobotDescription): boolean {
    const rt = this.physics.runtime;
    const rest = restingHeight(robot);
    if (!rt || rest === null) return false;
    const b = rt.groundTruthBaseState();
    return b.tilt < 0.35 && b.pos[2] > 0.7 * rest;
  }

  /** Runs a controller live with its judge, records the attempt, and reports when the run settles. */
  private async startLive(run: LiveRun): Promise<boolean> {
    const robot = this.state.robot;
    const km = buildKinematicModel(robot);
    // Build the controller before touching the live simulation, so a bad record is refused cleanly.
    let ctl: SkillController;
    try {
      ctl = makeController(km, run.limbIndex, homeConfiguration(robot), run.spec);
    } catch (e) {
      this.error(`Refused: the stored controller cannot run on this robot (${e instanceof Error ? e.message : String(e)}).`);
      return false;
    }
    // The live view may still be loading the engine (for example right after a reload).
    await this.physics.init();
    if (!this.physics.runtime || this.physics.state.description !== robot || this.state.robot !== robot) {
      this.error(`The live simulation is not available${this.physics.state.error ? `: ${this.physics.state.error}` : '.'}`);
      return false;
    }
    this.stopActivity('another playback started');
    if (this.state.comparing) this.store.update({ comparing: null });
    const continued = run.fromCurrentPose === true && this.standingUp(robot);
    if (!continued) this.physics.reset();
    this.lastStartContinued = continued;
    const t0 = this.physics.runtime.time;
    const judging = run.judging;
    const judge = isGoals(judging) ? GoalJudge.forRobot(judging.goals, robot) : null;
    const referee = isGoals(judging) ? null : new FootLiftReferee(judging, run.requiredHoldS);
    const limbIndex = run.limbIndex;
    const support = referee ? supportLegs(robot, limbIndex) : [];
    const tipRadius = referee ? robot.limbs[limbIndex]!.tip.radius : 0;
    // The evaluation stops a trial this long after it starts; the live run uses the same limit.
    const limitS = skillDuration(run.spec) + 2;
    const setupNotes: string[] = run.skillId === null ? [] : liveSetupNotes(run.evaluation, { gains: this.physics.state.gains, baseMassKg: robot.base.mass, friction: 1 });
    if (continued && run.skillId !== null) setupNotes.push('It started where the previous run left the body, not from a reset (gaits chain into a path).');
    const plotOf: PlotSpec = isGoals(judging) ? goalPlot(judging.goals, robot) : { title: 'Foot clearance', unit: 'm', threshold: judging.minClearanceM, value: () => 0 };
    const rec: PlayRecording = {
      skillId: run.skillId, name: run.name, method: run.method, morphologyHash: run.morphologyHash, robot, t: [], frames: [], clearanceM: [],
      plot: { title: plotOf.title, unit: plotOf.unit, threshold: plotOf.threshold }, plotValues: [], nextT: t0,
    };
    const linkCount = km.joints.length;
    let finishedAt = -1;
    // Checkpoints of a taught skill's run, and the limbs its goals name.
    const marks = judge ? checkpointMarks(run.spec) : [];
    const checkpoints: RunCheckpoint[] = [];
    const goalLimbIds = isGoals(judging) ? [...new Set(judging.goals.flatMap((g) => ('limbId' in g ? [g.limbId] : [])))] : [];
    const goalLimbs = goalLimbIds.map((id) => robot.limbs.findIndex((l) => l.id === id));
    const allLimbIds = robot.limbs.map((l) => l.id);
    let first: JudgeSample | null = null;
    let last: JudgeSample | null = null;
    let turned = 0;
    const active: ActiveRun = {
      skillId: run.skillId ?? 'preview',
      finish: (end, reason, cause) => {
        if (this.activeRun !== active) return;
        this.activeRun = null;
        const unstable = cause === 'unstable';
        const completed = end === 'ended' && ctl.finished;
        if (judge) {
          const v = judge.verdict({ unstable });
          const outcome: PlayOutcome = end === 'interrupted' ? 'interrupted' : end === 'faulted' ? 'faulted' : v.success ? 'succeeded' : 'failed';
          const ending = first && last ? [checkpointOf('end of the run', last.t - t0, last, first, turned, goalLimbs, goalLimbIds, allLimbIds)] : [];
          this.endPlay({
            skillId: run.skillId ?? '', skillName: run.name, limbId: '', outcome, completed, success: outcome === 'succeeded',
            failures: end === 'ended' ? (v.success ? [] : verdictFailures(v)) : unstable ? ['numerical-instability'] : [],
            requiredHoldS: 0, holdS: 0, maxClearanceM: 0, maxTiltRad: v.maxTiltRad, minBaseHeightM: v.minBaseHeightM, supportContactFraction: 0,
            goals: v.goals, checks: v.checks, reason, setupNotes, checkpoints: [...checkpoints, ...ending],
          }, rec);
          return;
        }
        const v = referee!.verdict({ unstable, aborted: ctl.phase === 'aborted' });
        const outcome: PlayOutcome =
          end === 'interrupted' ? 'interrupted'
          : end === 'faulted' ? 'faulted'
          : ctl.phase === 'aborted' ? 'aborted'
          : v.success ? 'succeeded' : 'failed';
        this.endPlay({
          skillId: run.skillId ?? '', skillName: run.name, limbId: robot.limbs[limbIndex]?.id ?? '', outcome, completed, success: outcome === 'succeeded',
          failures: end === 'ended' ? v.failures : unstable ? ['numerical-instability'] : [],
          requiredHoldS: run.requiredHoldS, holdS: v.longestClearHoldS, maxClearanceM: v.maxClearanceM, maxTiltRad: v.maxTiltRad,
          minBaseHeightM: v.minBaseHeightM, supportContactFraction: v.supportContactFraction, goals: null, reason, setupNotes,
        }, rec);
      },
    };
    this.activeRun = active;
    this.physics.setStepObserver({
      step: (rt, sensors) => {
        const t = rt.time;
        let clearance = 0;
        let plotValue: number;
        if (judge) {
          const sample = judgeSample(rt, t, sensors.touch);
          judge.add(sample);
          if (last) turned += Math.atan2(Math.sin(sample.yawRad - last.yawRad), Math.cos(sample.yawRad - last.yawRad));
          first ??= sample;
          last = sample;
          while (marks.length > 0 && t - t0 >= marks[0]!.t - 1e-9) checkpoints.push(checkpointOf(marks.shift()!.label, t - t0, sample, first, turned, goalLimbs, goalLimbIds, allLimbIds));
          plotValue = plotOf.value(sample);
        } else {
          clearance = rt.groundTruthTipPositions()[limbIndex]![2] - tipRadius;
          const b = rt.groundTruthBaseState();
          referee!.add({ t, clearanceM: clearance, tiltRad: b.tilt, baseHeightM: b.pos[2], supportContact: support.every((i) => (sensors.touch[i] ?? 0) > 0) });
          plotValue = clearance;
        }
        if (t >= rec.nextT - 1e-9) {
          rec.nextT += ATTEMPT_DT;
          const f = new Float64Array(16 * (1 + linkCount));
          f.set(rt.groundTruthBaseTransform(), 0);
          rt.groundTruthLinkTransforms().forEach((T, i) => f.set(T, 16 * (i + 1)));
          rec.t.push(t);
          rec.frames.push(f);
          if (!judge) rec.clearanceM.push(clearance);
          rec.plotValues.push(plotValue);
        }
        if (ctl.finished && finishedAt < 0) finishedAt = t;
        if (finishedAt >= 0 && t - finishedAt > 0.5) active.finish('ended', null);
        else if (t - t0 > limitS) active.finish('ended', `the controller had not finished after ${limitS.toFixed(1)} s`);
      },
      abort: (reason, cause) => active.finish(cause === 'interrupted' ? 'interrupted' : 'faulted', reason, cause),
    });
    this.physics.setDriver((t, sensors) => (ctl.finished ? null : ctl.update(t, { q: sensors.q, qd: sensors.qd, touch: sensors.touch, imuQuat: sensors.imuQuat, gyro: sensors.gyro })), run.label);
    this.store.update({ playing: active.skillId, message: null });
    return true;
  }

  private activeRun: ActiveRun | null = null;
  private attemptCounter = 0;

  private endPlay(report: PlayReport, rec: PlayRecording): void {
    this.physics.setStepObserver(null);
    if (this.physics.hasDriver) this.physics.setDriver(null);
    const attempts = this.state.attempts;
    if (rec.frames.length >= 2) {
      const n = ++this.attemptCounter;
      const attempt: Attempt = {
        id: `attempt_${n}`, label: `Attempt ${n}: ${rec.name}`, skillName: rec.name, method: rec.method, robot: rec.robot,
        t: rec.t, frames: rec.frames, clearanceM: rec.clearanceM, plot: { ...rec.plot, values: rec.plotValues }, holdS: report.holdS, maxTiltRad: report.maxTiltRad,
        outcome: report.outcome, completed: report.completed, resultText: attemptText(report),
      };
      this.store.update({ attempts: [attempt, ...attempts].slice(0, MAX_ATTEMPTS) });
    }
    this.store.update({ lastPlay: report, playing: null });
    // Skill memory counts runs of library skills; previews of a skill being taught are not recorded.
    if (rec.skillId && rec.morphologyHash) {
      this.skillMemory?.recordSkillRun(rec.skillId, rec.morphologyHash, report.goals ? { outcome: report.outcome, holdS: Number.NaN, requiredHoldS: 0, failures: report.failures } : report);
    }
    const done = this.finishPlay;
    this.finishPlay = null;
    done?.(report);
  }

  /** Shows two recorded attempts side by side in place of the live view. */
  openComparison(left: string, right: string): boolean {
    const ids = new Set(this.state.attempts.map((a) => a.id));
    if (!ids.has(left) || !ids.has(right)) {
      this.error('Choose two recorded attempts to compare.');
      return false;
    }
    this.stopActivity('a comparison was opened');
    this.store.update({ comparing: [left, right], message: null });
    return true;
  }

  closeComparison(): void {
    this.store.update({ comparing: null });
  }

  /** Resolves when the current playback completes or is interrupted. */
  private waitForPlay(): Promise<PlayReport> {
    return new Promise((resolve) => {
      this.finishPlay = resolve;
    });
  }

  /**
   * Runs a text command: interpret with the local rules, resolve against the
   * skill library for this exact morphology, then execute and report the
   * result measured from simulator state.
   * `target` is the tab the command was typed in when that is not the Skills
   * tab: stand and reset act on that tab's own simulation, and a foot lift
   * runs only if that tab shows the Skills robot's morphology.
   * `escalate` hands a lift that no evaluated skill can serve to the AI (which
   * can compose one); it returns null when the AI is not available.
   */
  async runCommand(text: string, target?: CommandTarget, escalate?: () => Promise<CommandOutcome | null>): Promise<CommandOutcome> {
    const s = this.state;
    const taughtName = this.taughtSkillFor(text);
    if (taughtName) {
      const o = await this.runTaughtCommand(taughtName, text, target);
      this.store.update({ commandLog: [o, ...this.state.commandLog].slice(0, 20) });
      return o;
    }
    const shown = target ? target.robot : s.robot;
    const ctxRobot = shown ?? s.robot;
    const ctx = { robotName: ctxRobot.name, limbs: ctxRobot.limbs.map((l) => ({ id: l.id, name: l.name, role: l.role })) };
    const interp = await this.interpreter.interpret(text, ctx);
    const hash = this.state.hash ?? (await morphologyHash(s.robot));
    const base = { command: text, source: interp.source, note: interp.note };
    const parsed = interp.result;
    const log = (o: CommandOutcome) => {
      this.store.update({ commandLog: [o, ...this.state.commandLog].slice(0, 20) });
      return o;
    };
    if (target && parsed.kind === 'skill' && parsed.skillType === 'foot_lift') {
      const shownHash = shown ? await morphologyHash(shown) : null;
      if (shownHash !== hash) {
        const skillsRobot = `${s.robot.name} (morphology ${hash.slice(0, 12)})`;
        return log({
          ...base,
          kind: 'refusal',
          text: shown
            ? `The ${target.tab} tab shows ${shown.name} (morphology ${shownHash!.slice(0, 12)}), but foot-lift skills run on the Skills robot, ${skillsRobot}. A skill is never reused on a different body.`
            : `The ${target.tab} tab shows no robot; foot-lift skills run on the Skills robot, ${skillsRobot}.`,
          suggestion: shown ? `Open the Skills tab to use ${s.robot.name}, or load ${shown.name} there and evaluate a skill for it.` : 'Open the Skills tab.',
        });
      }
    }
    if (target && parsed.kind === 'skill' && parsed.skillType !== 'foot_lift') {
      if (!target.simulation) {
        return log({ ...base, kind: 'refusal', text: `The ${target.tab} tab has no live simulation to ${parsed.skillType}.`, suggestion: 'Give the command in the Physics, Skills or Navigate tab.' });
      }
      return log({ ...base, kind: 'done', text: parsed.skillType === 'stand' ? target.simulation.stand() : target.simulation.reset() });
    }
    const r = resolveCommand(parsed, { robot: s.robot, hash, library: this.state.library });
    if (r.kind === 'refusal' && parsed.kind === 'skill' && escalate) {
      const ai = await escalate();
      if (ai) return ai;
    }
    let outcome: CommandOutcome;
    switch (r.kind) {
      case 'refusal': {
        const taught = this.taughtNamesHere(hash);
        const named = this.taughtSkillMentioned(text);
        if (named && taught.includes(named)) {
          outcome = {
            ...base,
            kind: 'refusal',
            text: `"${named}" runs when the command is its name alone, such as "${named}" or "please ${named}"; the local rules cannot follow the rest of this request.`,
            suggestion: `Type "${named}", or ask the AI with the local AI server running: it can run the skill and report what was measured.`,
          };
          break;
        }
        outcome = { ...base, kind: 'refusal', text: r.reason, suggestion: taught.length ? `${r.suggestion} Skills taught on this robot run by their name: ${taught.map((n) => `"${n}"`).join(', ')}.` : r.suggestion };
        break;
      }
      case 'clarify':
        outcome = { ...base, kind: 'clarify', text: r.question, options: r.options };
        break;
      case 'stand':
      case 'reset':
        await this.physics.init();
        this.stopActivity(r.kind === 'stand' ? 'a stand command was given' : 'the simulation was reset');
        this.physics.reset();
        outcome = { ...base, kind: 'done', text: r.kind === 'stand' ? 'Standing in the home stance under the joint controller.' : 'Reset the simulation to its start state.' };
        break;
      case 'run-skill': {
        target?.showSkills();
        const limb = s.robot.limbs.find((l) => l.id === r.limbId)!.name.toLowerCase();
        if (!(await this.play(r.skill.id, r.spec, { requiredHoldS: r.holdS }))) {
          outcome = { ...base, kind: 'refusal', text: this.state.message?.text ?? 'The skill could not be started.', suggestion: 'Check the skill library for this robot.' };
          break;
        }
        // Registered in the same task as play() returns; physics steps run in later tasks.
        const report = await this.waitForPlay();
        const note = [r.note, interp.note, ...report.setupNotes].filter(Boolean).join(' ') || null;
        outcome = { ...base, ...describePlay(report, limb, r.holdS), note, measuredHoldS: report.holdS };
        break;
      }
    }
    this.store.update({ commandLog: [outcome, ...this.state.commandLog].slice(0, 20) });
    return outcome;
  }

  /** Runs a taught skill named in a command: the best evaluated version for this exact body. */
  private async runTaughtCommand(name: string, text: string, target?: CommandTarget): Promise<CommandOutcome> {
    const s = this.state;
    const hash = s.hash ?? (await morphologyHash(s.robot));
    const base = { command: text, source: 'local rules', note: null as string | null };
    if (target) {
      const shownHash = target.robot ? await morphologyHash(target.robot) : null;
      if (shownHash !== hash) {
        return {
          ...base,
          kind: 'refusal',
          text: `"${name}" is a skill of the Skills robot, ${s.robot.name} (morphology ${hash.slice(0, 12)}); the ${target.tab} tab shows ${target.robot ? `${target.robot.name} (morphology ${shownHash!.slice(0, 12)})` : 'no robot'}. A skill is never reused on a different body.`,
          suggestion: 'Open the Skills tab to run it.',
        };
      }
    }
    const named = s.library.filter((a) => a.skillType === 'taught' && a.taught?.name === name);
    const compatible = named.filter((a) => checkSkillCompatibility(a, s.robot, hash).ok);
    if (compatible.length === 0) {
      return { ...base, kind: 'refusal', text: `"${name}" was taught on another body (morphology ${named[0]?.robot.morphologyHash.slice(0, 12) ?? 'unknown'}); a skill is never reused on a different body.`, suggestion: `Teach "${name}" on ${s.robot.name} in the Skills tab.` };
    }
    const eligible = compatible.filter((a) => commandEligibility(a) === null);
    if (eligible.length === 0) {
      const imported = compatible.some((a) => evaluationOrigin(a) === 'imported');
      return { ...base, kind: 'refusal', text: commandEligibility(compatible[0]!)!, suggestion: imported ? 'Open the Skills tab and press Re-evaluate here on the imported skill card.' : 'Adjust its poses or goals in the Skills tab and evaluate it again until some trials succeed.' };
    }
    // Versions whose goals measure what the name asks for come first (D25); then the lower end of the 95 % interval, so 19 of 20 beats 2 of 2.
    const fits = (a: SkillArtifact) => checkIntent(a.taught!.name, '', a.taught!.goals, s.robot).every((f) => f.goalIndex >= 0);
    const best = [...eligible].sort((a, b) => Number(fits(b)) - Number(fits(a)) || b.evaluation!.wilson95[0] - a.evaluation!.wilson95[0] || b.createdAt.localeCompare(a.createdAt))[0]!;
    target?.showSkills();
    if (!(await this.play(best.id))) {
      return { ...base, kind: 'refusal', text: this.state.message?.text ?? 'The skill could not be started.', suggestion: 'Check the skill library for this robot.' };
    }
    // Registered in the same task as play() returns; physics steps run in later tasks.
    const report = await this.waitForPlay();
    const e = best.evaluation!;
    const note = [`Using "${best.name}" (${skillLabel(best)}, ${e.successes}/${e.trials} in evaluation).`, ...report.setupNotes].join(' ');
    return { ...base, ...describePlay({ ...report, skillName: name }, name, 0), note };
  }

  /**
   * Plays a saved skill by its library name for an AI tool call and returns
   * the referee's report. Only skills evaluated here on this exact body may
   * run; aborting the signal interrupts the playback.
   */
  async runSkillByName(name: string, holdS: number | undefined, signal: AbortSignal): Promise<{ ok: false; reason: string; fallback: string | null } | { ok: true; report: PlayReport; limb: string; note: string }> {
    const s = this.state;
    const a = s.library.find((x) => x.name === name);
    if (!a) return { ok: false, reason: `No skill named "${name}" is in the library.`, fallback: 'List the saved skills first.' };
    const hash = s.hash ?? (await morphologyHash(s.robot));
    const compat = checkSkillCompatibility(a, s.robot, hash);
    if (!compat.ok) return { ok: false, reason: compat.reason, fallback: 'Skills are never reused on a different body; evaluate one for this robot.' };
    const notEligible = commandEligibility(a);
    if (notEligible) return { ok: false, reason: notEligible, fallback: 'Evaluate the skill here first (Re-evaluate here for imported skills).' };
    if (a.skillType === 'taught') {
      if (holdS !== undefined) return { ok: false, reason: `"${a.name}" is a taught skill; holdSeconds applies to foot lifts only.`, fallback: 'Call it without holdSeconds.' };
      if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.', fallback: null };
      if (!(await this.play(a.id))) return { ok: false, reason: this.state.message?.text ?? 'The skill could not be started.', fallback: null };
      const finished = this.waitForPlay();
      const stop = (): void => this.stopActivity('the AI task was cancelled');
      signal.addEventListener('abort', stop, { once: true });
      const report = await finished;
      signal.removeEventListener('abort', stop);
      const e = a.evaluation!;
      return { ok: true, report, limb: '', note: `Skill "${a.name}" (${e.successes}/${e.trials} in its evaluation).` };
    }
    const evaluated = a.parameters.holdS ?? DEFAULT_FOOT_LIFT.holdS;
    const hold = holdS ?? evaluated;
    if (!(hold >= HOLD_RANGE_S[0] && hold <= HOLD_RANGE_S[1])) {
      return { ok: false, reason: `A ${hold} s hold is outside the supported range of ${HOLD_RANGE_S[0]} to ${HOLD_RANGE_S[1]} s.`, fallback: null };
    }
    let spec = specFromArtifact(a);
    if ('params' in spec) spec = { ...spec, params: { ...spec.params, holdS: hold } };
    else if (Math.abs(hold - evaluated) > 1e-9) {
      return { ok: false, reason: `"${a.name}" replays a fixed recording or a learned timing and cannot hold for ${hold} s.`, fallback: `Run it with its evaluated ${evaluated} s hold.` };
    }
    if (signal.aborted) return { ok: false, reason: 'Cancelled before it started.', fallback: null };
    if (!(await this.play(a.id, spec, { requiredHoldS: hold }))) {
      return { ok: false, reason: this.state.message?.text ?? 'The skill could not be started.', fallback: null };
    }
    // Registered in the same task as play() returns; physics steps run in later tasks.
    const finished = this.waitForPlay();
    const onAbort = (): void => this.stopActivity('the AI task was cancelled');
    signal.addEventListener('abort', onAbort, { once: true });
    const report = await finished;
    signal.removeEventListener('abort', onAbort);
    const limb = s.robot.limbs.find((l) => l.id === a.limbs[0])?.name.toLowerCase() ?? a.limbs[0] ?? 'leg';
    const e = a.evaluation;
    const record = e ? `${e.successes}/${e.trials} in its evaluation` : 'no evaluation record';
    const differs = Math.abs(hold - evaluated) > 1e-9 ? ` It was evaluated with a ${evaluated} s hold; this ${hold} s hold was not part of that evaluation.` : '';
    return { ok: true, report, limb, note: `Skill "${a.name}" (${record}).${differs}` };
  }

  /**
   * Runs saved skills one after another, each from the home stance and judged
   * on its own, and stops at the first run that does not succeed. The plan's
   * chance of success is estimated beforehand as the product of each run's
   * evaluated success rate (an estimate, not a measurement).
   */
  async runSequence(steps: readonly { skill: string; times: number }[], signal: AbortSignal): Promise<
    { ok: false; reason: string } | { ok: true; predicted: number; planned: number; runs: { skill: string; report: PlayReport; note: string }[]; stopped: string | null }
  > {
    const s = this.state;
    const hash = s.hash ?? (await morphologyHash(s.robot));
    const plan: string[] = [];
    let predicted = 1;
    for (const step of steps) {
      const a = s.library.find((x) => x.name === step.skill);
      if (!a) return { ok: false, reason: `No skill named "${step.skill}" is in the library.` };
      const compat = checkSkillCompatibility(a, s.robot, hash);
      if (!compat.ok) return { ok: false, reason: compat.reason };
      const notEligible = commandEligibility(a);
      if (notEligible) return { ok: false, reason: notEligible };
      for (let k = 0; k < step.times; k++) {
        plan.push(a.name);
        predicted *= a.evaluation!.successRate;
      }
    }
    if (plan.length === 0 || plan.length > 12) return { ok: false, reason: 'A sequence runs 1 to 12 skills.' };
    const runs: { skill: string; report: PlayReport; note: string }[] = [];
    for (const name of plan) {
      if (signal.aborted) return { ok: true, predicted, planned: plan.length, runs, stopped: 'the sequence was cancelled' };
      const r = await this.runSkillByName(name, undefined, signal);
      if (!r.ok) return { ok: true, predicted, planned: plan.length, runs, stopped: `"${name}" could not start: ${r.reason}` };
      runs.push({ skill: name, report: r.report, note: r.note });
      if (r.report.outcome !== 'succeeded') return { ok: true, predicted, planned: plan.length, runs, stopped: `"${name}" ${OUTCOME_LABELS[r.report.outcome]}` };
    }
    return { ok: true, predicted, planned: plan.length, runs, stopped: null };
  }

  startTeleop(): void {
    if (!this.ready()) return;
    this.stopActivity('demonstration control started');
    try {
      this.teleop = new FootLiftTeleop(this.state.robot, this.limbIndex, this.stance);
    } catch (e) {
      this.error(e instanceof Error ? e.message : String(e));
      return;
    }
    this.physics.reset();
    this.recorded = [];
    const tele = this.teleop;
    this.physics.setDriver(() => {
      const t = this.state.teleop;
      const q = tele.targetsFor(t.shift, t.lift);
      if (t.recording) {
        // Recordings stop at the longest replay a skill file may hold.
        if (this.recorded.length * TELEOP_DT >= MAX_REPLAY_S) queueMicrotask(() => this.stopRecording());
        else this.recorded.push(q);
      }
      return q;
    }, 'Demonstration control: you set the body shift and foot height; leg IK turns them into joint targets.');
    this.store.update({ teleop: { active: true, recording: false, shift: 0, lift: 0, frames: 0 }, playing: null, message: null });
  }

  setTeleop(shift: number, lift: number): void {
    const t = this.state.teleop;
    if (!t.active || !Number.isFinite(shift) || !Number.isFinite(lift)) return;
    this.store.update({ teleop: { ...t, shift, lift, frames: this.recorded.length } });
  }

  startRecording(): void {
    const t = this.state.teleop;
    if (!t.active) return;
    this.recorded = [];
    this.store.update({ teleop: { ...t, recording: true, frames: 0 } });
  }

  stopRecording(): void {
    const t = this.state.teleop;
    if (!t.recording) return;
    this.store.update({ teleop: { ...t, recording: false, frames: this.recorded.length } });
    if (this.recorded.length < 50) {
      this.error('The demonstration is shorter than 0.5 s and was discarded.');
      return;
    }
    const km = buildKinematicModel(this.state.robot);
    const trajectory: Trajectory = { dt: TELEOP_DT, jointIds: km.joints.map((j) => j.id), frames: this.recorded.slice() };
    this.store.update({ demonstration: { source: 'you', trajectory, context: this.context() } });
    this.info(`Recorded your demonstration: ${((trajectory.frames.length - 1) * trajectory.dt).toFixed(1)} s of joint targets. Evaluate the replay to see how well it holds up.`);
  }

  /** Stops playback and teleoperation; fixed arms hold their measured pose. */
  stopActivity(reason = 'the playback was stopped'): void {
    const runtime = this.physics.runtime;
    const heldPose = this.state.robot.base.kind === 'fixed' && this.physics.hasDriver && runtime
      ? Array.from(runtime.readSensors().q)
      : null;
    this.activeRun?.finish('interrupted', reason);
    if (this.physics.hasDriver) this.physics.setDriver(null);
    if (heldPose) this.physics.setTargets(heldPose);
    this.teleop = null;
    this.recorded = [];
    const t = this.state.teleop;
    if (t.active || this.state.playing) this.store.update({ teleop: { active: false, recording: false, shift: 0, lift: 0, frames: 0 }, playing: null });
  }

  dispose(): void {
    this.stopActivity('the Skills view was closed');
    this.worker.dispose();
    this.pool?.dispose();
    this.physics.dispose();
  }
}
