/**
 * Local rule-based command parser. Converts text into a structured skill call,
 * a clarification question, or a refusal with a suggestion. It never produces
 * joint targets or torques: execution is decided later by the skill resolver.
 */

export interface LimbInfo {
  id: string;
  name: string;
  role: 'arm' | 'leg' | 'wheel';
}

export interface CommandContext {
  robotName: string;
  limbs: LimbInfo[];
}

export type SkillCall =
  | { kind: 'skill'; skillType: 'foot_lift'; limbId: string; holdS: number }
  | { kind: 'skill'; skillType: 'stand' }
  | { kind: 'skill'; skillType: 'reset' };

export interface Refusal {
  kind: 'refusal';
  reason: string;
  suggestion: string;
  /** The local rules do not cover this request; the AI path (proxy and tools) may. */
  escalate?: true;
}

export interface Clarification {
  kind: 'clarify';
  question: string;
  options: string[];
}

export type ParseResult = SkillCall | Refusal | Clarification;

export const DEFAULT_HOLD_S = 2;

const NUMBER_WORDS: Record<string, number> = {
  zero: 0, one: 1, a: 1, an: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, fifteen: 15, twenty: 20, thirty: 30,
};

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[-_/]/g, ' ')
    .replace(/[^a-z0-9.\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Extracts a duration in seconds, or null if none is mentioned; NaN if malformed. */
export function parseDuration(t: string): number | null {
  const half = /\b(?:half a|half an) (second|minute)s?\b/.exec(t);
  if (half) return half[1] === 'minute' ? 30 : 0.5;
  const m = /\b(\d+(?:\.\d+)?|[a-z]+)(?: and a half)? (seconds?|secs?|s|minutes?|mins?|milliseconds?|ms)\b/.exec(t);
  if (!m) return null;
  const raw = m[1]!;
  let v = /^\d/.test(raw) ? Number(raw) : NUMBER_WORDS[raw];
  if (v === undefined || !Number.isFinite(v)) return Number.NaN;
  if (/ and a half /.test(m[0] + ' ')) v += 0.5;
  const unit = m[2]!;
  if (unit.startsWith('min')) return v * 60;
  if (unit.startsWith('ms') || unit.startsWith('milli')) return v / 1000;
  return v;
}

const SIDE_WORDS = ['left', 'right'];
const POSITION_WORDS = ['front', 'middle', 'rear', 'back', 'hind'];

function limbWords(t: string): string[] {
  const words = t.split(' ');
  const out: string[] = [];
  for (const w of words) {
    if (w === 'back' || w === 'hind') out.push('rear');
    else if (w === 'fore' || w === 'foreleg') out.push('front');
    else if (SIDE_WORDS.includes(w) || POSITION_WORDS.includes(w)) out.push(w);
  }
  const abbrev = /\b(fl|fr|rl|rr|bl|br|ml|mr|lf|rf|lr|rh|lh)\b/.exec(t);
  if (abbrev) {
    const map: Record<string, string[]> = {
      fl: ['front', 'left'], lf: ['front', 'left'], fr: ['front', 'right'], rf: ['front', 'right'],
      rl: ['rear', 'left'], bl: ['rear', 'left'], lh: ['rear', 'left'], lr: ['rear', 'left'],
      rr: ['rear', 'right'], br: ['rear', 'right'], rh: ['rear', 'right'], ml: ['middle', 'left'], mr: ['middle', 'right'],
    };
    out.push(...map[abbrev[1]!]!);
  }
  return [...new Set(out)];
}

function limbLabel(l: LimbInfo): string {
  return l.name.toLowerCase();
}

/** Resolves limb words against the robot's own limbs. */
function resolveLimb(t: string, ctx: CommandContext): { limb?: LimbInfo; refusal?: Refusal; clarify?: Clarification } {
  const legs = ctx.limbs.filter((l) => l.role === 'leg');
  const words = limbWords(t);
  if (legs.length === 0) {
    return { refusal: { kind: 'refusal', reason: `${ctx.robotName} has no legs.`, suggestion: 'Build or load a legged robot, such as the quadruped preset.', escalate: true } };
  }
  const available = legs.map(limbLabel).join(', ');
  if (words.length === 0) {
    return { clarify: { kind: 'clarify', question: 'Which leg should be lifted?', options: legs.map(limbLabel) } };
  }
  const matches = legs.filter((l) => words.every((w) => l.id.split('_').includes(w)));
  if (matches.length === 1) return { limb: matches[0]! };
  if (matches.length > 1) {
    return { clarify: { kind: 'clarify', question: `Which ${words.join(' ')} leg?`, options: matches.map(limbLabel) } };
  }
  return { refusal: { kind: 'refusal', reason: `${ctx.robotName} has no ${words.join(' ')} leg.`, suggestion: `Available legs: ${available}.` } };
}

interface UnsupportedRule {
  pattern: RegExp;
  reason: string;
  suggestion: string;
  escalate?: true;
  /** The rule refuses commands only; a question that matches it goes on to the question check. */
  commandsOnly?: true;
}

const TEACH_IT = 'Teach it in the Skills tab (Teach a skill: pose the robot, keep each pose as a keyframe, set the goals that mean success, evaluate it), then say its name. With the local AI server running, the AI composes, tries and evaluates such a skill itself.';

// Motions nobody taught are escalated: the AI can compose them as a skill draft. Torque and several-leg refusals stay local.
const UNSUPPORTED: UnsupportedRule[] = [
  // "run" alone or as locomotion only: "the live run record" or "run the evaluation" is not a walking request.
  { pattern: /\b(walk|trot|gallop|crawl|step forward|go forward|move forward|run forward|run around|run fast|run away)\b|^run$/, reason: 'Walking is not a skill this robot has learned yet.', suggestion: `${TEACH_IT} The AI composes walking as a gait: feet stepping in a contact pattern, with IMU feedback against body tilt; the goal "Body moves" measures how far it gets.`, escalate: true },
  { pattern: /\b(jump|hop|leap|flip|backflip)\b/, reason: 'No jumping skill has been taught to this robot yet.', suggestion: `${TEACH_IT} For a jump, pose a crouch and then a fast extension, and add the goal "Feet leave the ground"; the physics decides whether the motors are strong enough.`, escalate: true },
  { pattern: /\b(dance|wave|shake)\b/, reason: 'That motion has not been taught to this robot yet.', suggestion: TEACH_IT, escalate: true },
  { pattern: /\b(go to|navigate|drive to|move to|come here|fetch|bring)\b/, reason: 'The local rules do not handle navigation commands.', suggestion: 'In the Navigate tab, set a goal and use Drive to goal. With the AI proxy running, the rover also takes navigation requests in words.', escalate: true },
  { pattern: /\b(sit|lie down|crouch|roll over|stand tall|stand up tall|stretch|tiptoes?)\b/, reason: 'That posture has not been taught to this robot yet.', suggestion: TEACH_IT, escalate: true },
  // Seen in the browser: "teach it to stand on its hind legs" ran the home stance, because the stand rule matched "stand" anywhere.
  {
    pattern: /\b(hind legs|back legs|rear legs|front legs|two (?:legs|feet)|2 (?:legs|feet)|rear up|rears up|rearing up|handstand|stand on|balance on)\b/,
    reason: 'Balancing on fewer feet has not been taught to this robot yet.',
    suggestion: `${TEACH_IT} A pose on fewer feet is unstable: after composing it, train a learned feedback policy so the body balances from its own sensors.`,
    escalate: true,
  },
  // Asking why torque saturates is a question for the experiment coach; setting a torque is refused. An impulse in newton seconds (a push) is not a torque.
  { pattern: /\b(?:torque|newtons?(?! (?:seconds?|s)\b)|apply \d+ ?n(?! s\b)|set joint|motor current)\b/, reason: 'Language commands select evaluated skills; they never set joint torques or targets directly.', suggestion: 'Use the control panel to set joint targets and gains manually.', commandsOnly: true },
];

export const SUPPORTED_COMMANDS = ['lift the front left leg and hold for three seconds', 'stand', 'reset'];

// Questions are answered, never carried out: "how did the lift go?" must not lift a leg, and
// "why did it reset?" must not reset. Polite requests ("can you lift ...?") stay commands.
const QUESTION_START =
  /^(how|what|which|why|when|where|who|whose|tell me|explain|describe)\b|^(did|does|is|are|was|were|has|have|had) (you|it|the|this|that|your|my|a|an|there|we|they|i|he|she)\b|^do (you|i|we|they)\b/;
const POLITE_REQUEST = /^(please|can you|could you|would you|will you)\b/;

function isQuestion(raw: string, t: string): boolean {
  return QUESTION_START.test(t) || (raw.trim().endsWith('?') && !POLITE_REQUEST.test(t));
}

/** Whether a command is a question (answered, never carried out). */
export function isQuestionText(text: string): boolean {
  return isQuestion(text, normalize(text));
}

export function parseCommand(text: string, ctx: CommandContext): ParseResult {
  if (typeof text !== 'string' || text.trim().length === 0) {
    return { kind: 'refusal', reason: 'The command is empty.', suggestion: `Try: "${SUPPORTED_COMMANDS[0]}".` };
  }
  if (text.length > 300) {
    return { kind: 'refusal', reason: 'The command is too long.', suggestion: 'Use one short instruction at a time.' };
  }
  const t = normalize(text);
  const question = isQuestion(text, t);
  for (const rule of UNSUPPORTED) {
    if (rule.commandsOnly && question) continue;
    if (rule.pattern.test(t)) return { kind: 'refusal', reason: rule.reason, suggestion: rule.suggestion, ...(rule.escalate ? { escalate: true as const } : {}) };
  }
  if (question) {
    return {
      kind: 'refusal',
      reason: 'This is a question; the local rules only carry out commands.',
      suggestion: 'With the AI proxy running, questions go to the AI, which answers from measured results and stored records.',
      escalate: true,
    };
  }
  if (/\b(reset|start over|restart)\b/.test(t)) return { kind: 'skill', skillType: 'reset' };
  const lift = /\b(lift|raise|pick up|pickup|hold up|put up)\b/.test(t);
  if (lift) {
    if (/\b(all|every|both|four|all four)\b/.test(t) && /\b(legs|feet)\b/.test(t)) {
      return { kind: 'refusal', reason: 'Lifting several legs at once would remove the support the robot needs to stay upright.', suggestion: 'Lift one leg at a time.' };
    }
    const r = resolveLimb(t, ctx);
    if (r.refusal) return r.refusal;
    if (r.clarify) return r.clarify;
    const d = parseDuration(t);
    if (d !== null && !Number.isFinite(d)) {
      return { kind: 'refusal', reason: 'The hold duration could not be understood.', suggestion: 'Say a duration such as "for three seconds".' };
    }
    return { kind: 'skill', skillType: 'foot_lift', limbId: r.limb!.id, holdS: d ?? DEFAULT_HOLD_S };
  }
  // Only a short stand command means the home stance; a longer request that mentions standing goes to the AI.
  if (/^(?:please )?(?:stand|stand up|stand still|hold still|stance|balance)(?: please| now)?$/.test(t)) return { kind: 'skill', skillType: 'stand' };
  return { kind: 'refusal', reason: 'This command is not recognized.', suggestion: `Supported commands include: ${SUPPORTED_COMMANDS.map((c) => `"${c}"`).join(', ')}.`, escalate: true };
}
