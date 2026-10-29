// Spike S6: rule-based mapping of example prompts to skill calls, clarifications or refusals.
import { expect, it } from 'vitest';
import { hexapod, quadruped, rover } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { parseCommand, type CommandContext, type ParseResult } from '../../src/language/parser';

const ctx = (d: RobotDescription): CommandContext => ({ robotName: d.name, limbs: d.limbs.map((l) => ({ id: l.id, name: l.name, role: l.role })) });
const quad = ctx(quadruped());

interface Case {
  prompt: string;
  context?: CommandContext;
  expect: Partial<ParseResult> & { kind: ParseResult['kind'] };
}

const cases: Case[] = [
  { prompt: 'lift your front-left leg and hold for three seconds', expect: { kind: 'skill', skillType: 'foot_lift', limbId: 'front_left', holdS: 3 } },
  { prompt: 'Raise the rear right foot for 2.5 s', expect: { kind: 'skill', skillType: 'foot_lift', limbId: 'rear_right', holdS: 2.5 } },
  { prompt: 'pick up FR', expect: { kind: 'skill', skillType: 'foot_lift', limbId: 'front_right', holdS: 2 } },
  { prompt: 'lift the left hind leg and hold it for half a second', expect: { kind: 'skill', skillType: 'foot_lift', limbId: 'rear_left', holdS: 0.5 } },
  { prompt: 'lift your left leg', expect: { kind: 'clarify', options: ['front left leg', 'rear left leg'] } },
  { prompt: 'lift a leg', expect: { kind: 'clarify' } },
  { prompt: 'lift the middle left leg', expect: { kind: 'refusal' } },
  { prompt: 'lift the middle left leg', context: ctx(hexapod()), expect: { kind: 'skill', skillType: 'foot_lift', limbId: 'middle_left', holdS: 2 } },
  { prompt: 'lift all four legs', expect: { kind: 'refusal' } },
  { prompt: 'walk forward two meters', expect: { kind: 'refusal' } },
  { prompt: 'do a backflip', expect: { kind: 'refusal' } },
  { prompt: 'apply 5 N*m to the knee', expect: { kind: 'refusal' } },
  { prompt: 'lift the front left leg', context: ctx(rover()), expect: { kind: 'refusal' } },
  { prompt: 'stand up', expect: { kind: 'skill', skillType: 'stand' } },
  { prompt: 'reset', expect: { kind: 'skill', skillType: 'reset' } },
  { prompt: 'make me a sandwich', expect: { kind: 'refusal' } },
  { prompt: '   ', expect: { kind: 'refusal' } },
];

it('S6: example prompts map to the expected structured results', () => {
  const rows = cases.map((c) => {
    const r = parseCommand(c.prompt, c.context ?? quad);
    const ok = Object.entries(c.expect).every(([k, v]) => JSON.stringify((r as unknown as Record<string, unknown>)[k]) === JSON.stringify(v));
    const detail = r.kind === 'skill' ? JSON.stringify(r) : r.kind === 'refusal' ? `${r.reason} | ${r.suggestion}` : `${r.question} ${r.options.join(' / ')}`;
    return { prompt: c.prompt, robot: (c.context ?? quad).robotName, kind: r.kind, ok, detail };
  });
  console.table(rows.map((r) => ({ prompt: r.prompt, robot: r.robot, kind: r.kind, ok: r.ok })));
  rows.forEach((r) => console.log(`${r.ok ? 'PASS' : 'FAIL'} "${r.prompt}" -> ${r.detail}`));
  expect(rows.length).toBeGreaterThanOrEqual(10);
  expect(rows.filter((r) => !r.ok)).toEqual([]);
});
