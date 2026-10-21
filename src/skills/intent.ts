import type { RobotDescription } from '../core/model/types';
import { normalizeSkillName, restingHeight, type SkillGoal, type TravelDirection } from './taught';

/**
 * Checks that a skill's goals measure what its name or request asks for
 * (D25). Each rule ties motion words to a property of the body that can be
 * measured, so a "sit" that is only a level crouch, or a "wave" with no
 * swinging limb, is caught before it is tried. Rules are deliberately few
 * and literal; words without a rule are not judged.
 */

export interface IntentFinding {
  /** The word or phrase that triggered the rule. */
  word: string;
  /** What the request means in measurable terms. */
  meaning: string;
  /** Index of the first goal that measures it, or -1. */
  goalIndex: number;
  /** How to measure it when no goal does. */
  hint: string;
}

interface Rule {
  pattern: RegExp;
  meaning: string;
  hint: string;
  /** The rule only applies to bodies that can do it (a fixed base cannot sit). */
  floatingOnly?: true;
  measures: (g: SkillGoal, ctx: Context) => boolean;
}

interface Context {
  text: string;
  rest: number;
  limb: string | null;
}

const NEGATION = /\b(?:not|no|don't|dont|never|without|stop)\s+(?:\w+\s+)?$/;

const direction = (t: string): TravelDirection => (/\b(back|backward|backwards|reverse)\b/.test(t) ? 'backward' : /\bleft\b/.test(t) && /\b(side|sideways|strafe|step left)\b/.test(t) ? 'left' : /\bright\b/.test(t) && /\b(side|sideways|strafe|step right)\b/.test(t) ? 'right' : 'forward');
const turnLeft = (t: string): boolean | null => (/\bleft\b/.test(t) ? true : /\bright\b/.test(t) ? false : null);
const onLimb = (g: SkillGoal, ctx: Context): boolean => ctx.limb === null || ('limbId' in g && g.limbId === ctx.limb);

const RULES: Rule[] = [
  {
    pattern: /\b(sit|sits|sitting|sit down)\b/,
    meaning: 'a sit lowers the rear below the front, so the body pitches nose up',
    hint: 'add a bodyAngle goal on pitch with maxRad at most -0.1 (nose up), and keep the rear feet down',
    floatingOnly: true,
    measures: (g) => g.kind === 'bodyAngle' && g.axis === 'pitch' && g.maxRad <= -0.1,
  },
  {
    pattern: /\b(bow|bows|bowing|nod|look down|bend forward|bend down)\b/,
    meaning: 'a bow lowers the front, so the body pitches nose down',
    hint: 'add a bodyAngle goal on pitch with minRad at least 0.1 (front down)',
    floatingOnly: true,
    measures: (g) => g.kind === 'bodyAngle' && g.axis === 'pitch' && g.minRad >= 0.1,
  },
  {
    pattern: /\b(look up|nose up|beg)\b/,
    meaning: 'looking up raises the front, so the body pitches nose up',
    hint: 'add a bodyAngle goal on pitch with maxRad at most -0.1 (nose up)',
    floatingOnly: true,
    measures: (g) => g.kind === 'bodyAngle' && g.axis === 'pitch' && g.maxRad <= -0.1,
  },
  {
    // A nose-up pitch alone is also a sit, and two tip heights can be met one after the other: the front feet must be up together.
    pattern: /\b(on (?:its |the |your |his |her |their )?(?:two )?(?:hind|back|rear) legs|rear up|rears up|rearing up)\b/,
    meaning: 'standing on the hind legs lifts the front feet off the ground together',
    hint: 'add an airborne goal whose limbIds name the front feet',
    floatingOnly: true,
    measures: (g) => g.kind === 'airborne' && !!g.limbIds && g.limbIds.length >= 2 && g.limbIds.every((id) => !/(rear|back|hind)/.test(id)),
  },
  {
    pattern: /\b(two legs|2 legs|two feet|2 feet|handstand)\b/,
    meaning: 'standing on two legs lifts the other feet off the ground together',
    hint: 'add an airborne goal whose limbIds name the lifted feet',
    floatingOnly: true,
    measures: (g) => g.kind === 'airborne' && !!g.limbIds && g.limbIds.length >= 2,
  },
  {
    pattern: /\b(lean|leans|leaning|tilt|tilts)\s+(left|right)\b/,
    meaning: 'a lean lowers one side, so the body rolls',
    hint: 'add a bodyAngle goal on roll: maxRad at most -0.1 to lean left (left side down), minRad at least 0.1 to lean right',
    floatingOnly: true,
    measures: (g, ctx) => g.kind === 'bodyAngle' && g.axis === 'roll' && (/\bleft\b/.test(ctx.text) ? g.maxRad <= -0.1 : g.minRad >= 0.1),
  },
  {
    pattern: /\b(lie down|lay down|lie flat|lay flat|play dead|belly)\b/,
    meaning: 'lying down brings the body close to the ground',
    hint: 'add a baseHeight goal with below true and heightM at most 60 % of the resting height',
    floatingOnly: true,
    measures: (g, ctx) => g.kind === 'baseHeight' && g.below && g.heightM <= 0.6 * ctx.rest + 1e-9,
  },
  {
    pattern: /\b(crouch|crouches|crouching|squat|squats|duck|get low|go low|lower yourself)\b/,
    meaning: 'a crouch lowers the body below its resting height',
    hint: 'add a baseHeight goal with below true and heightM a few cm below the resting height',
    floatingOnly: true,
    measures: (g, ctx) => g.kind === 'baseHeight' && g.below && g.heightM < ctx.rest - 0.02,
  },
  {
    pattern: /\b(stand tall|stand up tall|stretch up|tiptoe|tiptoes|rise up|get tall)\b/,
    meaning: 'standing tall raises the body above its resting height',
    hint: 'add a baseHeight goal with below false and heightM above the resting height, or a baseRise goal',
    floatingOnly: true,
    measures: (g, ctx) => (g.kind === 'baseHeight' && !g.below && g.heightM > ctx.rest + 0.01) || g.kind === 'baseRise',
  },
  {
    pattern: /\b(jump|jumps|jumping|hop|hops|hopping|leap|leaps|bounce|pronk)\b/,
    meaning: 'a jump takes every foot off the ground at once',
    hint: 'add an airborne goal without limbIds (every foot)',
    floatingOnly: true,
    measures: (g) => g.kind === 'airborne' && !g.limbIds,
  },
  {
    pattern: /\b(jump|jumps|jumping|hop|hops|hopping|leap|leaps|bounce|pronk)\b/,
    meaning: 'a jump lands on its feet (a body that tips over also has every foot in the air)',
    hint: 'add a standsAtEnd goal, or an upright goal',
    floatingOnly: true,
    measures: (g) => g.kind === 'standsAtEnd' || g.kind === 'upright',
  },
  {
    pattern: /\b(walk|walks|walking|trot|trots|trotting|crawl|crawls|crawling|march|marches|marching|step forward|go forward|move forward|run forward|back up|step back|go back|move back|strafe|sidestep|side step)\b/,
    meaning: 'walking moves the body over the ground',
    hint: 'add a travel goal in the direction asked (forward unless the request says back or sideways)',
    floatingOnly: true,
    measures: (g, ctx) => /\bin place\b/.test(ctx.text) || (g.kind === 'travel' && g.direction === direction(ctx.text)),
  },
  {
    pattern: /\b(turn|turns|turning|rotate|rotates|spin|spins|pivot|pivots)\b/,
    meaning: 'turning changes the heading of the body',
    hint: 'add a turn goal in the direction asked',
    floatingOnly: true,
    measures: (g, ctx) => g.kind === 'turn' && (turnLeft(ctx.text) === null || g.left === turnLeft(ctx.text)),
  },
  {
    pattern: /\b(wave|waves|waving|shake hands|shake a paw|give a paw|high five)\b/,
    meaning: 'a wave swings a limb tip back and forth',
    hint: 'add a tipSwing goal on the waving limb',
    measures: (g, ctx) => g.kind === 'tipSwing' && onLimb(g, ctx),
  },
  {
    pattern: /\b(point|points|pointing|reach|reaches|reaching|touch|touches)\b/,
    meaning: 'reaching moves a limb tip to a point',
    hint: 'add a tipReach goal on that limb (pointing up may instead be measured by a tipHeight goal)',
    measures: (g, ctx) => (g.kind === 'tipReach' || (g.kind === 'tipHeight' && /\bup\b/.test(ctx.text))) && onLimb(g, ctx),
  },
  {
    pattern: /\b(lift|lifts|lifting|raise|raises|raising)\b/,
    meaning: 'lifting raises a limb tip, or the body when the request says so',
    hint: 'add a tipHeight goal on the limb named in the request (or a baseRise goal to raise the body)',
    measures: (g, ctx) =>
      ((g.kind === 'tipHeight' || g.kind === 'tipReach' || g.kind === 'tipSwing') && onLimb(g, ctx)) ||
      (/\b(body|yourself)\b/.test(ctx.text) && (g.kind === 'baseRise' || (g.kind === 'baseHeight' && !g.below))),
  },
];

const SYNONYMS: Record<string, string> = { back: 'rear', hind: 'rear', fore: 'front' };

/** The one limb a text names by its id words (front left, rear right, arm), if exactly one matches. */
export function namedLimb(text: string, robot: RobotDescription): string | null {
  const words = new Set(text.split(' ').map((w) => SYNONYMS[w] ?? w));
  const hits = robot.limbs.filter((l) => l.role !== 'wheel' && l.id.split('_').every((w) => words.has(w)));
  return hits.length === 1 ? hits[0]!.id : null;
}

/** The rules a text triggers, with the words that triggered them. */
function triggered(text: string, robot: RobotDescription): { rule: Rule; word: string }[] {
  const out: { rule: Rule; word: string }[] = [];
  for (const rule of RULES) {
    if (rule.floatingOnly && robot.base.kind !== 'floating') continue;
    const m = rule.pattern.exec(text);
    if (!m || NEGATION.test(text.slice(0, m.index))) continue;
    out.push({ rule, word: m[0] });
  }
  return out;
}

/**
 * Checks goals against what a skill's name asks for; when the name triggers
 * no rule, against the request if it asks for exactly one kind of motion.
 * A request for several skills is not held against each one of them. The
 * request's words still qualify the name's rules (a direction, a limb, "in
 * place").
 */
export function checkIntent(name: string, request: string, goals: readonly SkillGoal[], robot: RobotDescription): IntentFinding[] {
  const n = normalizeSkillName(name);
  const r = normalizeSkillName(request);
  let text = r ? `${n} ${r}` : n;
  let rules = triggered(n, robot);
  if (rules.length === 0) {
    const fromRequest = triggered(r, robot);
    // Several rules of one motion (a jump leaves the ground and lands) still make a single motion.
    if (new Set(fromRequest.map((x) => x.rule.pattern.source)).size !== 1) return [];
    text = r;
    rules = fromRequest;
  }
  const ctx: Context = { text, rest: restingHeight(robot) ?? robot.base.pose.xyz[2], limb: namedLimb(text, robot) };
  return rules.map(({ rule, word }) => ({ word, meaning: rule.meaning, goalIndex: goals.findIndex((g) => rule.measures(g, ctx)), hint: rule.hint }));
}
