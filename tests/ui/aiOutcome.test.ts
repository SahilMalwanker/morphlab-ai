import { describe, expect, it } from 'vitest';
import type { TurnReport } from '../../src/ai/loop.ts';
import { AI_SOURCE, aiOutcome } from '../../src/ui/ai/outcome.ts';

function report(over: Partial<TurnReport>): TurnReport {
  return { text: null, steps: [], end: 'answered', error: null, modelCalls: 1, model: 'fake-model', tokens: { prompt: 0, completion: 0 }, ...over };
}

describe('aiOutcome', () => {
  it('takes the status line from the measured action result, not from the model text', () => {
    const rep = report({
      text: 'Done, the rover is there.',
      steps: [
        { tool: 'getRobotState', arguments: '{}', accepted: true, result: { taskId: null, status: 'succeeded', reason: 'Idle.', measurements: {}, fallback: null } },
        { tool: 'navigateTo', arguments: '{"x":1,"y":0}', accepted: true, result: { taskId: 'task-2', status: 'failed', reason: 'Collision with an obstacle.', measurements: {}, fallback: 'Reset the rover.' } },
      ],
    });
    expect(aiOutcome('drive', { kind: 'turn', report: rep })).toEqual({
      command: 'drive',
      kind: 'failed',
      text: 'navigateTo failed: Collision with an obstacle.',
      suggestion: 'Reset the rover.',
      source: AI_SOURCE,
      note: 'Model reply (not a measurement): Done, the rover is there.',
      ai: rep,
    });
  });

  it('labels a reply without any action as such', () => {
    const rep = report({ text: 'The rover can drive to goals.' });
    expect(aiOutcome('what can you do', { kind: 'turn', report: rep })).toMatchObject({ kind: 'done', text: 'The rover can drive to goals.', note: 'No action was taken; this is the model reply, not a measurement.' });
  });

  it('reports a memory change as the outcome when nothing moved, and a motion result first when both ran', () => {
    const remember = { tool: 'rememberPlace', arguments: '{"name":"dock"}', accepted: true, result: { taskId: null, status: 'succeeded' as const, reason: 'Remembered "dock" (0.00, 0.00) m.', measurements: {}, fallback: null } };
    const rep = report({ text: 'I saved the dock.', steps: [remember] });
    expect(aiOutcome('remember', { kind: 'turn', report: rep })).toMatchObject({ kind: 'done', text: 'rememberPlace succeeded: Remembered "dock" (0.00, 0.00) m.', note: 'Model reply (not a measurement): I saved the dock.' });
    const refused = report({ steps: [{ ...remember, result: { ...remember.result, status: 'rejected' as const, reason: 'That position is outside the arena or too close to an obstacle.' } }] });
    expect(aiOutcome('remember', { kind: 'turn', report: refused })).toMatchObject({ kind: 'failed', text: 'rememberPlace rejected: That position is outside the arena or too close to an obstacle.' });
    const drive = { tool: 'navigateTo', arguments: '{"x":1,"y":0}', accepted: true, result: { taskId: 'task-1', status: 'succeeded' as const, reason: 'Reached the goal.', measurements: {}, fallback: null } };
    expect(aiOutcome('both', { kind: 'turn', report: report({ steps: [drive, remember] }) })).toMatchObject({ kind: 'done', text: 'navigateTo succeeded: Reached the goal.' });
  });

  it('reports unavailability, refusals before running, errors and cancellation as such', () => {
    expect(aiOutcome('x', { kind: 'unavailable', reason: 'The AI proxy is not running here.' })).toMatchObject({ kind: 'refusal', text: 'AI is not available: The AI proxy is not running here.' });
    expect(aiOutcome('x', { kind: 'busy' }).kind).toBe('refusal');
    const refusedOnly = report({ steps: [{ tool: 'navigateTo', arguments: '{"x":9}', accepted: false, result: { taskId: null, status: 'rejected', reason: 'Invalid arguments.', measurements: {}, fallback: null } }], text: 'That is outside the arena.' });
    expect(aiOutcome('x', { kind: 'turn', report: refusedOnly })).toMatchObject({ kind: 'done', text: 'That is outside the arena.' });
    expect(aiOutcome('x', { kind: 'turn', report: report({ end: 'error', error: 'The AI proxy is not running here.' }) })).toMatchObject({ kind: 'failed', text: 'The AI request failed: The AI proxy is not running here.' });
    expect(aiOutcome('x', { kind: 'turn', report: report({ end: 'cancelled' }) })).toMatchObject({ kind: 'failed', text: 'The AI command was cancelled.' });
  });
});
