import type { RobotDescription } from '../core/model/types';
import { checkSkillCompatibility, commandEligibility, evaluationOrigin, methodLabel, specFromArtifact, type SkillArtifact } from '../skills/artifact';
import type { SkillSpec } from '../skills/controllers';
import { DEFAULT_FOOT_LIFT, FOOT_LIFT_BOUNDS } from '../skills/footLift';
import type { ParseResult } from './parser';

export const HOLD_RANGE_S: [number, number] = FOOT_LIFT_BOUNDS.holdS;

export type Resolution =
  | { kind: 'run-skill'; skill: SkillArtifact; spec: SkillSpec; limbId: string; holdS: number; note: string | null }
  | { kind: 'stand' }
  | { kind: 'reset' }
  | { kind: 'refusal'; reason: string; suggestion: string }
  | { kind: 'clarify'; question: string; options: string[] };

const ADJUSTABLE = new Set(['state-machine', 'tuned-parameters', 'es-improved']);

/**
 * Turns a parsed command into an executable decision: picks an evaluated
 * skill for this exact morphology and leg, or refuses with a reason and a
 * suggestion. Language never produces joint targets; it only selects skills.
 */
export function resolveCommand(parsed: ParseResult, ctx: { robot: RobotDescription; hash: string; library: SkillArtifact[] }): Resolution {
  if (parsed.kind === 'refusal' || parsed.kind === 'clarify') return parsed;
  if (parsed.skillType === 'stand') return { kind: 'stand' };
  if (parsed.skillType === 'reset') return { kind: 'reset' };
  const { limbId, holdS } = parsed;
  const limb = ctx.robot.limbs.find((l) => l.id === limbId);
  const limbName = limb?.name.toLowerCase() ?? limbId;
  if (!(holdS >= HOLD_RANGE_S[0] && holdS <= HOLD_RANGE_S[1])) {
    return { kind: 'refusal', reason: `A ${holdS} s hold is outside the supported range of ${HOLD_RANGE_S[0]} to ${HOLD_RANGE_S[1]} s.`, suggestion: 'Ask for a hold between half a second and five seconds.' };
  }
  const forLimb = ctx.library.filter((a) => a.skillType === 'foot_lift' && a.limbs.includes(limbId));
  const compatible = forLimb.filter((a) => checkSkillCompatibility(a, ctx.robot, ctx.hash, limbId).ok);
  if (compatible.length === 0) {
    const others = forLimb.length;
    return {
      kind: 'refusal',
      reason: `There is no evaluated foot-lift skill for the ${limbName} of this robot (morphology ${ctx.hash.slice(0, 12)}).${others ? ` ${others} skill${others > 1 ? 's' : ''} for that leg belong to other morphologies and are never reused on a different body.` : ''}`,
      suggestion: `Open the Skills tab, choose Foot lift example and the ${limbName}, and evaluate a method; the state machine takes a few seconds.`,
    };
  }
  const evaluatedHold = (a: SkillArtifact) => a.parameters.holdS ?? DEFAULT_FOOT_LIFT.holdS;
  const eligible = compatible.filter((a) => commandEligibility(a) === null);
  if (eligible.length === 0) {
    const imported = compatible.some((a) => evaluationOrigin(a) === 'imported');
    return {
      kind: 'refusal',
      reason: `No skill for the ${limbName} has an evaluation verified here: ${compatible.slice(0, 2).map((a) => commandEligibility(a)).join(' ')}`,
      suggestion: imported ? 'Open the Skills tab and press Re-evaluate here on the imported skill card.' : `Open the Skills tab, choose Foot lift example and the ${limbName}, and evaluate a method.`,
    };
  }
  const usable = Math.abs(holdS - DEFAULT_FOOT_LIFT.holdS) < 1e-9 ? eligible : eligible.filter((a) => ADJUSTABLE.has(a.method) || Math.abs(evaluatedHold(a) - holdS) < 1e-9);
  if (usable.length === 0) {
    return {
      kind: 'refusal',
      reason: `The skills for the ${limbName} replay a fixed recording or a learned timing and cannot hold for ${holdS} s.`,
      suggestion: `Ask for a ${DEFAULT_FOOT_LIFT.holdS} s hold, or evaluate a state-machine skill for this leg.`,
    };
  }
  // Rank by the lower end of the 95 % interval, so 19 of 20 beats 2 of 2.
  const best = [...usable].sort((a, b) => b.evaluation!.wilson95[0] - a.evaluation!.wilson95[0] || b.evaluation!.successRate - a.evaluation!.successRate || b.createdAt.localeCompare(a.createdAt))[0]!;
  let spec = specFromArtifact(best);
  if ('params' in spec) spec = { ...spec, params: { ...spec.params, holdS } };
  const evaluated = evaluatedHold(best);
  const e = best.evaluation!;
  const record = `${e.successes}/${e.trials} in evaluation`;
  const note = Math.abs(evaluated - holdS) > 1e-9
    ? `Using "${best.name}" (${methodLabel(best.method)}, ${record}). It was evaluated with a ${evaluated} s hold; this ${holdS} s hold was not part of that evaluation.`
    : `Using "${best.name}" (${methodLabel(best.method)}, ${record}).`;
  return { kind: 'run-skill', skill: best, spec, limbId, holdS, note };
}
