import { describe, expect, it } from 'vitest';
import { parseLiveCommand } from '../../src/language/live';

describe('live commands while a task runs', () => {
  it('recognizes stop, slower and faster with their size', () => {
    for (const text of ['stop', 'Stop!', 'please stop', 'halt', 'cancel that', 'stop the rover']) expect(parseLiveCommand(text), text).toEqual({ kind: 'stop' });
    expect(parseLiveCommand('slower')).toEqual({ kind: 'modify', change: 'slower', amount: 'normal' });
    expect(parseLiveCommand('slow down a bit')).toEqual({ kind: 'modify', change: 'slower', amount: 'little' });
    expect(parseLiveCommand('please go a little faster')).toEqual({ kind: 'modify', change: 'faster', amount: 'little' });
    expect(parseLiveCommand('speed up')).toEqual({ kind: 'modify', change: 'faster', amount: 'normal' });
    expect(parseLiveCommand('a bit higher')).toEqual({ kind: 'modify', change: 'higher', amount: 'little' });
    expect(parseLiveCommand('lower')).toEqual({ kind: 'modify', change: 'lower', amount: 'normal' });
  });

  it('leaves longer requests, new goals, negations and mixed changes to the other paths', () => {
    for (const text of [
      'go to the dock faster',
      'drive slower towards x 2',
      "don't go faster",
      'faster and higher',
      'lift the front left leg and hold it for three seconds please',
      'what is the speed limit',
      '',
    ]) {
      expect(parseLiveCommand(text), text).toEqual({ kind: 'unknown' });
    }
  });
});
