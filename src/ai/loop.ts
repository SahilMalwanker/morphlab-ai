import type { RobotCapabilities } from './capabilities.ts';
import { MAX_ARGUMENT_CHARS, MAX_MESSAGES, MAX_TOTAL_CHARS, requestChars, type ChatMessage, type ChatRequest, type ChatResult } from './protocol.ts';
import { rejected, type ToolResult } from './result.ts';
import { validateToolCall, type ModelToolSpec } from './tools.ts';

/** Tools that start or change robot activity; at most one runs per model answer. */
export const ACTION_TOOLS: ReadonlySet<string> = new Set([
  'pointAt',
  'navigateTo',
  'navigateToPlace',
  'executeSkill',
  'draftSkill',
  'mirrorSkill',
  'evaluateSkillDraft',
  'improveSkillDraft',
  'trainGaitPolicy',
  'trainLocomotion',
  'locomote',
  'runSkillSequence',
  'selectRobot',
  'pushRobot',
  'resetSimulation',
  'moveTipTo',
  'setJointAngles',
  'buildRobot',
  'modifyTask',
  'cancelTask',
  'saveSkillFromDemonstration',
  'evaluateSkill',
  'runControllerExperiment',
  'saveSkill',
]);

/** Tools that change stored memory without moving the robot; not limited to one per answer, but reported like actions. */
export const MEMORY_TOOLS: ReadonlySet<string> = new Set(['rememberPlace', 'forgetPlace']);

export interface ToolStep {
  tool: string;
  /** Arguments as the model sent them. */
  arguments: string;
  /** Passed schema and availability validation. */
  accepted: boolean;
  result: ToolResult;
}

export interface TurnReport {
  /** The model's closing text: not a measurement. */
  text: string | null;
  steps: ToolStep[];
  end: 'answered' | 'step-limit' | 'cancelled' | 'error' | 'filtered';
  error: string | null;
  modelCalls: number;
  model: string | null;
  tokens: { prompt: number; completion: number };
}

export interface TurnOptions {
  chat: (request: ChatRequest, signal: AbortSignal) => Promise<ChatResult>;
  system: string;
  history: readonly ChatMessage[];
  userText: string;
  tools: ModelToolSpec[];
  caps: RobotCapabilities;
  /** Runs a validated call and returns its measured result. */
  execute: (name: string, args: Record<string, unknown>, signal: AbortSignal) => Promise<ToolResult>;
  signal: AbortSignal;
  maxModelCalls?: number;
  onStep?: (step: ToolStep) => void;
  /**
   * Capabilities, tools and the system text after an action changed them (a
   * skill saved during the turn, another robot); null keeps the current ones.
   */
  refresh?: () => Promise<{ caps: RobotCapabilities; tools: ModelToolSpec[]; system?: string } | null>;
  /**
   * When the model answers in text, a reminder to continue instead (null to
   * accept the answer); see `skillFollowUp`. Used at most MAX_FOLLOW_UPS
   * times per command and only while model calls remain.
   */
  followUp?: (steps: readonly ToolStep[]) => string | null;
}

const MAX_FOLLOW_UPS = 2;

function declinedActions(request: string): Set<string> {
  const declined = new Set<string>();
  if (/\b(?:do not|don't|never)\s+substitute\b/i.test(request)) {
    for (const action of ['draft', 'train', 'improve']) declined.add(action);
  }
  if (/\b(?:preview|try)\s+(?:it\s+)?only\b|\bonly\s+(?:preview|try)\b/i.test(request)) {
    for (const action of ['train', 'evaluate', 'save', 'improve']) declined.add(action);
  }
  for (const match of request.matchAll(/\b(?:do not|don't|never)\s+((?:train|retrain|learn|evaluate|save|improve|compose|teach|create)\b(?:\s*(?:,|and|or)\s*(?:train|retrain|learn|evaluate|save|improve|compose|teach|create)\b)*)/gi)) {
    for (const action of match[1]!.toLowerCase().match(/\b(?:train|retrain|learn|evaluate|save|improve|compose|teach|create)\b/g) ?? []) {
      declined.add(action === 'retrain' || action === 'learn' ? 'train' : ['compose', 'teach', 'create'].includes(action) ? 'draft' : action);
    }
  }
  return declined;
}

function requestRefusesTool(tool: string, request: string): boolean {
  const declined = declinedActions(request);
  if (tool === 'draftSkill' || tool === 'mirrorSkill') return declined.has('draft');
  if (tool === 'trainGaitPolicy') return declined.has('train');
  if (tool === 'trainLocomotion') return declined.has('train') || declined.has('evaluate') || declined.has('save');
  if (tool === 'evaluateSkillDraft') return declined.has('evaluate') || declined.has('save');
  if (tool === 'evaluateSkill') return declined.has('evaluate');
  if (tool === 'saveSkill' || tool === 'saveSkillFromDemonstration') return declined.has('save');
  if (tool === 'improveSkillDraft') return declined.has('improve');
  return false;
}

/** A request that asks for learning (a learned policy, controller or network, training): a draft then gets trainGaitPolicy before its evaluation (D27, D30). */
export function asksForLearning(request: string): boolean {
  return !declinedActions(request).has('train') && /\b(learn|learns|learned|learnt|learning|policy|policies|neural|network|train|trains|trained|training|reinforcement)\b/i.test(request);
}

/** Whether the draft the steps last composed is a gait (its draftSkill arguments give one). */
function composedGait(acted: readonly ToolStep[]): boolean {
  const d = [...acted].reverse().find((s) => s.tool === 'draftSkill');
  if (!d) return false;
  try {
    const args: unknown = JSON.parse(d.arguments);
    return typeof args === 'object' && args !== null && 'gait' in args;
  } catch {
    return false;
  }
}

/** Whether a learned feedback policy was trained after the draft the steps last composed. */
function trainedSinceDraft(acted: readonly ToolStep[]): boolean {
  const i = acted.map((s) => s.tool).lastIndexOf('draftSkill');
  return acted.slice(i + 1).some((s) => s.tool === 'trainGaitPolicy');
}

/** The result's own next step names a learned feedback policy (the bridge decides by the body and the request, D30). */
const namesTraining = (r: ToolResult): boolean => /\btrainGaitPolicy\b/.test(r.fallback ?? '');

/**
 * The follow-up for skill drafts (D25): a model often answers "I will revise
 * it" after a try and stops. After a try that missed a goal it is reminded to
 * revise (up to three drafts) or improve; after a try or an improvement that
 * met every goal, to evaluate and save it, or first to train a learned
 * feedback policy for a gait when the request asked for learning (D27).
 */
export function skillFollowUp(steps: readonly ToolStep[], request = ''): string | null {
  const final = steps[steps.length - 1];
  if (requestRefusesTool('evaluateSkillDraft', request) || (final && requestRefusesTool(final.tool, request))) return null;
  if (final && ACTION_TOOLS.has(final.tool) && final.result.status === 'rejected') {
    return `The ${final.tool} call was refused before it ran: ${final.result.reason.slice(0, 300)} Correct the call and make it again instead of answering, unless the refusal says it cannot be done here.`;
  }
  const acted = steps.filter((s) => s.accepted && ACTION_TOOLS.has(s.tool));
  const last = acted[acted.length - 1];
  if (!last) return null;
  // A mirrored skill is a draft too: it is tried once and then revised, improved or evaluated.
  const tried = (s: ToolStep): boolean => s.tool === 'draftSkill' || s.tool === 'mirrorSkill';
  const drafts = acted.filter(tried).length;
  if (tried(last) && last.result.status === 'failed') {
    const train = !requestRefusesTool('trainGaitPolicy', request) && namesTraining(last.result) ? ', or call trainGaitPolicy if the poses are right but the body cannot hold them open loop' : '';
    if (drafts < 3) return `The try missed a goal. Do not answer yet: change what the measured goals and checkpoints point to and call draftSkill again (draft ${drafts + 1} of 3), or call improveSkillDraft if it came close${train}. Answer only when the measurements show the body cannot do it.`;
    if (!acted.some((s) => s.tool === 'improveSkillDraft')) return 'Three drafts missed a goal. Call improveSkillDraft once on the current draft before answering, unless the measurements show the body cannot do it.';
    return null;
  }
  if ((tried(last) || last.tool === 'improveSkillDraft') && last.result.status === 'succeeded' && !trainedSinceDraft(acted)) {
    if (asksForLearning(request) && composedGait(acted)) {
      return 'The draft met every goal, and the user asked for learning. Call trainGaitPolicy now, so a learned feedback policy corrects the gait, then evaluateSkillDraft; answer after the evaluation.';
    }
    if (!requestRefusesTool('trainGaitPolicy', request) && namesTraining(last.result)) {
      return 'The draft met every goal open loop, and its result asks for a learned feedback policy. Call trainGaitPolicy now, so a policy learned under pushes corrects the keyframes from the IMU, the foot contacts and the joint encoders, then evaluateSkillDraft; answer after the evaluation.';
    }
  }
  if ((tried(last) || last.tool === 'improveSkillDraft') && last.result.status === 'succeeded') {
    return 'The draft met every goal. Call evaluateSkillDraft now, so the skill is saved and runs by name next time; answer after the evaluation.';
  }
  if (last.tool === 'trainGaitPolicy' && last.result.status === 'succeeded') {
    return 'The learned feedback policy was kept in the draft. Call evaluateSkillDraft now, so the skill is evaluated with it and saved; answer after the evaluation.';
  }
  // Seen in the browser: after a learned jump failed only its landing goal, the model gave up instead of revising.
  if (last.tool === 'evaluateSkillDraft' && last.result.status === 'failed' && drafts > 0 && drafts < 3) {
    return `No evaluation trial met every goal. Do not answer yet: revise the draft with draftSkill (draft ${drafts + 1} of 3), starting with the goal met least often (a threshold you chose stricter than the request may be corrected, the request itself not), then train and evaluate it again.`;
  }
  return null;
}

const MAX_CALLS_PER_ANSWER = 4;
/** Model calls per command: enough to compose a skill, try it, revise it, improve or train it, evaluate it and run it. */
export const DEFAULT_MAX_MODEL_CALLS = 12;
/** Actions that may follow each other in one answer, in order, while each succeeds: a path of walks and saved skills. */
export const CHAINABLE_ACTIONS: ReadonlySet<string> = new Set(['locomote', 'executeSkill']);
const MAX_TOOL_CONTENT = 2500;
/** A tool result shortened to stay within the request limits keeps about this many characters. */
const SHORT_TOOL_CONTENT = 400;

/** A tool result already read by the model, cut to its status and the start of its reason. */
function shortened(content: string): string {
  try {
    const r = JSON.parse(content) as { status?: unknown; reason?: unknown };
    if (typeof r.status === 'string' && typeof r.reason === 'string') {
      return JSON.stringify({ status: r.status, reason: `${r.reason.slice(0, SHORT_TOOL_CONTENT - 120)} (shortened after it was read)` });
    }
  } catch {
    // Not one of our results: cut the text.
  }
  return `${content.slice(0, SHORT_TOOL_CONTENT - 40)} (shortened after it was read)`;
}
const MAX_OUTPUT_TOKENS = 1600;
const SAFE_ID = /^[A-Za-z0-9_.:-]{1,64}$/;

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * A tool result as the model reads it, within MAX_TOOL_CONTENT and still
 * valid JSON: measurements are left out first, then the end of the reason.
 */
function toolContent(r: ToolResult): string {
  let s = JSON.stringify(r);
  if (s.length <= MAX_TOOL_CONTENT) return s;
  const lean = Object.keys(r.measurements).length ? { ...r, measurements: 'left out to fit the request limit' } : r;
  s = JSON.stringify(lean);
  let reason = r.reason;
  while (s.length > MAX_TOOL_CONTENT && reason.length > 0) {
    reason = reason.slice(0, Math.max(0, reason.length - (s.length - MAX_TOOL_CONTENT) - 8));
    s = JSON.stringify({ ...lean, reason: `${reason} [cut]` });
  }
  return s.slice(0, MAX_TOOL_CONTENT);
}

const LATEX_WORDS: Readonly<Record<string, string>> = {
  cdot: '*', times: '*', approx: '≈', le: '≤', leq: '≤', ge: '≥', geq: '≥', ne: '≠', neq: '≠', pm: '±',
  to: '→', rightarrow: '→', infty: 'infinity', left: '', right: '', quad: ' ', qquad: ' ',
};
const LATEX_STRUCTURE = new Set(['frac', 'dfrac', 'tfrac', 'sqrt', 'text', 'mathrm', 'mathit', 'mathbf', 'operatorname']);
const fractionPart = (x: string): string => (/^[\w.^]+$/.test(x.trim()) ? x.trim() : `(${x.trim()})`);

/** LaTeX math as plain text: k_d/(2 sqrt(k_p I)) rather than \frac{k_d}{2\sqrt{k_p I}}. */
function plainMath(src: string): string {
  let s = src
    .replace(/\\[,;:! ]/g, ' ')
    .replace(/\\%/g, '%')
    .replace(/\^\{?\\circ\}?/g, ' deg')
    .replace(/\\([A-Za-z]+)/g, (all: string, name: string) => (LATEX_STRUCTURE.has(name) ? all : (LATEX_WORDS[name] ?? name)));
  // Innermost groups first, so nested fractions and roots unwrap from the inside out.
  for (let i = 0; i < 8; i++) {
    const next = s
      .replace(/([_^])\{([^{}]*)\}/g, '$1$2')
      .replace(/\\(?:text|mathrm|mathit|mathbf|operatorname)\{([^{}]*)\}/g, '$1')
      .replace(/\\sqrt\{([^{}]*)\}/g, ' sqrt($1)')
      .replace(/\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}/g, (_: string, a: string, b: string) => `${fractionPart(a)}/${fractionPart(b)}`);
    if (next === s) break;
    s = next;
  }
  return s.replace(/\\([A-Za-z]+)/g, '$1').replace(/[{}]/g, '').replace(/\s+/g, ' ').trim();
}

/**
 * The reply as plain text for the panel and for speech: LaTeX math delimiters
 * are unwrapped and their content rewritten, markdown emphasis, code marks and
 * headings are removed. Models add these despite the plain-text instruction.
 */
export function plainReply(text: string): string {
  return text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_: string, m: string) => plainMath(m))
    .replace(/\\\(([\s\S]*?)\\\)/g, (_: string, m: string) => plainMath(m))
    .replace(/\*\*([^*\n]+)\*\*/g, '$1')
    .replace(/__([^_\n]+)__/g, '$1')
    .replace(/`([^`\n]+)`/g, '$1')
    .replace(/^#{1,6}[ \t]+/gm, '')
    .trim();
}

/**
 * One closed-loop turn: the model selects tools, every call is validated and
 * run by the caller, measured results go back to the model, until it answers
 * in text or the call budget is used up. The model never acts directly.
 */
export async function runAiTurn(o: TurnOptions): Promise<TurnReport> {
  const messages: ChatMessage[] = [{ role: 'system', content: o.system }, ...o.history, { role: 'user', content: o.userText }];
  const report: TurnReport = { text: null, steps: [], end: 'step-limit', error: null, modelCalls: 0, model: null, tokens: { prompt: 0, completion: 0 } };
  const maxCalls = o.maxModelCalls ?? DEFAULT_MAX_MODEL_CALLS;
  let caps = o.caps;
  let tools = o.tools;
  // Remembered exchanges follow the system message; a long turn drops the oldest first to stay within the request limits.
  let history = o.history.length;
  let seq = 0;
  let followUps = 0;
  while (report.modelCalls < maxCalls) {
    if (o.signal.aborted) {
      report.end = 'cancelled';
      return report;
    }
    const over = (): boolean => messages.length > MAX_MESSAGES || requestChars({ messages, tools }) > MAX_TOTAL_CHARS;
    while (over() && history > 0) {
      const n = Math.min(3, history);
      messages.splice(1, n);
      history -= n;
    }
    // Then the oldest tool results of this command, already read, keep only their status and the start of their reason.
    const unread = messages.map((m) => m.role).lastIndexOf('assistant');
    for (let i = 1; over() && i < unread; i++) {
      const m = messages[i]!;
      if (m.role === 'tool' && m.content.length > SHORT_TOOL_CONTENT) messages[i] = { ...m, content: shortened(m.content) };
    }
    if (over()) {
      report.error = 'The command grew past the request limits; ask for the rest in a new command.';
      return report;
    }
    let res: ChatResult;
    try {
      res = await o.chat({ messages, tools, toolChoice: 'auto', maxOutputTokens: MAX_OUTPUT_TOKENS }, o.signal);
    } catch (e) {
      report.end = o.signal.aborted ? 'cancelled' : 'error';
      report.error = messageOf(e);
      return report;
    }
    report.modelCalls++;
    report.model = res.model ?? report.model;
    if (res.usage) {
      report.tokens.prompt += res.usage.promptTokens;
      report.tokens.completion += res.usage.completionTokens;
    }
    if (res.finishReason === 'content_filter') {
      report.end = 'filtered';
      report.error = 'The provider content filter withheld the answer.';
      return report;
    }
    if (res.toolCalls.length === 0) {
      const nudge = followUps < MAX_FOLLOW_UPS && report.modelCalls < maxCalls ? (o.followUp?.(report.steps) ?? null) : null;
      if (nudge) {
        followUps++;
        if (res.content) messages.push({ role: 'assistant', content: res.content.slice(0, 8000) });
        messages.push({ role: 'system', content: nudge });
        continue;
      }
      report.text = plainReply(res.content ?? '') || null;
      report.end = 'answered';
      return report;
    }
    // IDs and names go back to the proxy, so they must fit its schema.
    const calls = res.toolCalls.slice(0, MAX_CALLS_PER_ANSWER).map((c) => ({
      id: SAFE_ID.test(c.id) ? c.id : `call_${++seq}`,
      name: c.name.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 64) || 'unknown',
      arguments: c.arguments.slice(0, MAX_ARGUMENT_CHARS),
      tooLong: c.arguments.length > MAX_ARGUMENT_CHARS ? c.arguments.length : 0,
    }));
    messages.push({ role: 'assistant', content: res.content ? res.content.slice(0, 8000) : null, toolCalls: calls.map(({ id, name, arguments: args }) => ({ id, name, arguments: args })) });
    let acted = false;
    // While every action of this answer is chainable and succeeded, the next chainable one runs too.
    let chainOk = true;
    let chainFailed = false;
    let changed = false;
    for (const call of calls) {
      let step: ToolStep;
      if (o.signal.aborted) {
        step = { tool: call.name, arguments: call.arguments, accepted: false, result: rejected('Cancelled before it ran.') };
      } else if (requestRefusesTool(call.name, o.userText)) {
        step = { tool: call.name, arguments: call.arguments, accepted: false, result: rejected(`Not permitted by this request: ${call.name}. The user explicitly restricted authoring, substitution, training, evaluation, improvement or saving; respect that restriction and report only what actually ran.`) };
      } else if (call.tooLong) {
        step = { tool: call.name, arguments: call.arguments, accepted: false, result: rejected(`The arguments are ${call.tooLong} characters long; at most ${MAX_ARGUMENT_CHARS} fit. Send fewer or simpler poses.`) };
      } else if (acted && ACTION_TOOLS.has(call.name) && !(chainOk && CHAINABLE_ACTIONS.has(call.name))) {
        const reason = chainFailed ? 'Not run: an earlier action in this answer did not succeed; read its result first.' : 'One action at a time: request it again after reading the previous result.';
        step = { tool: call.name, arguments: call.arguments, accepted: false, result: rejected(reason) };
      } else {
        const v = validateToolCall(call.name, call.arguments, caps);
        if (!v.ok) {
          step = { tool: call.name, arguments: call.arguments, accepted: false, result: v.result };
        } else {
          let result: ToolResult;
          try {
            result = await o.execute(v.name, v.args, o.signal);
          } catch (e) {
            result = { taskId: null, status: 'failed', reason: `The tool failed: ${messageOf(e)}`, measurements: {}, fallback: null };
          }
          if (result.fallback && [...ACTION_TOOLS].some((tool) => result.fallback!.includes(tool) && requestRefusesTool(tool, o.userText))) {
            result = { ...result, fallback: null };
          }
          step = { tool: v.name, arguments: call.arguments, accepted: true, result };
          if (ACTION_TOOLS.has(v.name)) {
            acted = true;
            if (!CHAINABLE_ACTIONS.has(v.name) || result.status !== 'succeeded') chainOk = false;
            if (CHAINABLE_ACTIONS.has(v.name) && result.status !== 'succeeded') chainFailed = true;
          }
          if (ACTION_TOOLS.has(v.name) || MEMORY_TOOLS.has(v.name)) changed = true;
        }
      }
      report.steps.push(step);
      o.onStep?.(step);
      if (step.result.status === 'cancelled') {
        report.end = 'cancelled';
        return report;
      }
      messages.push({ role: 'tool', toolCallId: call.id, content: toolContent(step.result) });
    }
    if (changed && o.refresh && !o.signal.aborted) {
      const next = await o.refresh();
      if (next) {
        caps = next.caps;
        tools = next.tools;
        if (next.system) messages[0] = { role: 'system', content: next.system };
      }
    }
  }
  return report;
}
