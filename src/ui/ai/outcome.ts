import type { AiCommandResult } from '../../ai/controller';
import { ACTION_TOOLS, MEMORY_TOOLS } from '../../ai/loop';
import type { CommandOutcome } from '../skills/session';

export const AI_SOURCE = 'AI through the local proxy';

/**
 * Command-bar text for an AI command. The status line comes from the last
 * measured action result (a motion first, otherwise a memory change); the
 * model's own words are shown separately and labeled, never as the outcome.
 */
export function aiOutcome(command: string, r: AiCommandResult): CommandOutcome {
  const source = AI_SOURCE;
  switch (r.kind) {
    case 'unavailable':
      return { command, kind: 'refusal', text: `AI is not available: ${r.reason}`, source, note: null };
    case 'unsupported':
      return { command, kind: 'refusal', text: r.reason, source, note: null };
    case 'busy':
      return { command, kind: 'refusal', text: 'An AI command is still running. Wait for it, or press Stop.', source, note: null };
    case 'turn': {
      const rep = r.report;
      const model = rep.text ? `Model reply (not a measurement): ${rep.text}` : null;
      if (rep.end === 'error' || rep.end === 'filtered') return { command, kind: 'failed', text: `The AI request failed: ${rep.error ?? 'unknown error'}`, source, note: model, ai: rep };
      if (rep.end === 'cancelled') return { command, kind: 'failed', text: 'The AI command was cancelled.', source, note: null, ai: rep };
      const ran = [...rep.steps].reverse().filter((s) => s.accepted);
      const action = ran.find((s) => ACTION_TOOLS.has(s.tool)) ?? ran.find((s) => MEMORY_TOOLS.has(s.tool));
      if (action) {
        const res = action.result;
        return {
          command,
          kind: res.status === 'succeeded' ? 'done' : 'failed',
          text: `${action.tool} ${res.status}: ${res.reason}`,
          ...(res.fallback ? { suggestion: res.fallback } : {}),
          source,
          note: model,
          ai: rep,
        };
      }
      if (rep.end === 'step-limit') return { command, kind: 'failed', text: rep.error ?? 'The AI used its step budget without finishing.', source, note: model, ai: rep };
      return { command, kind: 'done', text: rep.text ?? 'The AI gave no answer.', source, note: 'No action was taken; this is the model reply, not a measurement.', ai: rep };
    }
  }
}
