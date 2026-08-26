/**
 * Short commands that change or stop the running task. They are handled
 * locally, without AI, so they work at once while a task runs.
 */

export type LiveChange = 'slower' | 'faster' | 'higher' | 'lower' | 'closer' | 'farther';

export type LiveCommand = { kind: 'stop' } | { kind: 'modify'; change: LiveChange; amount: 'little' | 'normal' } | { kind: 'unknown' };

const MAX_WORDS = 6;

const CHANGES: [LiveChange, RegExp][] = [
  ['slower', /\b(slower|slow down|slow it down|less fast|not so fast|reduce (the )?speed)\b/],
  ['faster', /\b(faster|speed up|quicker|hurry( up)?|increase (the )?speed)\b/],
  ['higher', /\bhigher\b/],
  ['lower', /\blower\b/],
  ['closer', /\b(closer|nearer)\b/],
  ['farther', /\b(farther|further)\b/],
];

export function parseLiveCommand(text: string): LiveCommand {
  const t = text
    .toLowerCase()
    .replace(/[^a-z\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!t || t.split(' ').length > MAX_WORDS) return { kind: 'unknown' };
  if (/\b(don t|do not|never)\b/.test(t)) return { kind: 'unknown' };
  if (/^(please )?(stop|halt|cancel|abort|freeze)\b/.test(t)) return { kind: 'stop' };
  // "go to the dock faster" names a new goal; it is not a change to the running one.
  if (/\b(to|towards|toward|into|at)\b/.test(t)) return { kind: 'unknown' };
  const found = CHANGES.filter(([, re]) => re.test(t));
  if (found.length !== 1) return { kind: 'unknown' };
  const amount = /\b(a little|a bit|slightly|a tad|bit)\b/.test(t) ? 'little' : 'normal';
  return { kind: 'modify', change: found[0]![0], amount };
}
