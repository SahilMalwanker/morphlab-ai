import { describe, expect, it, vi } from 'vitest';
import { listenOnce, Speaker, spokenText, type RecognitionLike, type SynthLike, type UtteranceLike, type VoiceLike } from '../../src/ui/voice';

function fakeSynth(voices: VoiceLike[]) {
  const spoken: UtteranceLike[] = [];
  let cancels = 0;
  const synth: SynthLike = {
    getVoices: () => voices,
    speak: (u) => {
      spoken.push(u);
    },
    cancel: () => {
      cancels++;
    },
  };
  return { synth, spoken, cancels: () => cancels };
}

const utter = (text: string): UtteranceLike => ({ text, lang: '', voice: null, onend: null, onerror: null });

describe('spoken replies', () => {
  it('use an installed local English voice only, cancel earlier speech and cap the text', async () => {
    const remote = { name: 'Online English', lang: 'en-US', localService: false };
    const german = { name: 'Katja', lang: 'de-DE', localService: true };
    expect(new Speaker(fakeSynth([remote, german]).synth, utter).voice()).toBeNull();
    expect(await new Speaker(fakeSynth([remote]).synth, utter).speak('Hello')).toBe('unavailable');
    expect(await new Speaker(null, utter).speak('Hello')).toBe('unavailable');
    const david = { name: 'Microsoft David', lang: 'en-US', localService: true };
    const f = fakeSynth([remote, german, david]);
    const s = new Speaker(f.synth, utter);
    const p = s.speak('x'.repeat(1000));
    expect(f.spoken).toHaveLength(1);
    expect(f.spoken[0]!.voice).toBe(david);
    expect(f.spoken[0]!.text).toHaveLength(400);
    expect(f.cancels()).toBe(1);
    f.spoken[0]!.onend?.();
    expect(await p).toBe('done');
    const q = s.speak('again');
    f.spoken[1]!.onerror?.({ error: 'interrupted' });
    expect(await q).toBe('cancelled');
  });

  it("announce the kind of outcome and label the model's own words", () => {
    const base = { command: 'x', source: 's', note: null };
    expect(spokenText({ ...base, kind: 'refusal', text: 'No skill.' })).toBe('Refused. No skill.');
    expect(spokenText({ ...base, kind: 'failed', text: 'Collision.' })).toBe('Failed. Collision.');
    expect(spokenText({ ...base, kind: 'done', text: 'navigateTo succeeded: Reached the goal.' })).toBe('navigateTo succeeded: Reached the goal.');
    expect(spokenText({ ...base, kind: 'done', text: 'A grey block is ahead.', note: 'No action was taken; this is the model reply, not a measurement.' })).toBe('The AI says: A grey block is ahead.');
  });
});

class FakeRecognition implements RecognitionLike {
  lang = '';
  interimResults = true;
  maxAlternatives = 5;
  continuous = true;
  onstart: RecognitionLike['onstart'] = null;
  onresult: RecognitionLike['onresult'] = null;
  onerror: RecognitionLike['onerror'] = null;
  onend: RecognitionLike['onend'] = null;
  started = 0;
  aborted = 0;
  start(): void {
    this.started++;
  }
  abort(): void {
    this.aborted++;
  }
}

describe('voice input', () => {
  it('listens once for English and returns the trimmed transcript', async () => {
    const r = new FakeRecognition();
    const p = listenOnce(() => r);
    expect(r).toMatchObject({ lang: 'en-US', interimResults: false, continuous: false, maxAlternatives: 1, started: 1 });
    r.onresult?.({ results: [[{ transcript: '  slower ', confidence: 0.92 }]] });
    expect(await p).toEqual({ ok: true, text: 'slower', confidence: 0.92 });
    expect(r.aborted).toBe(1);
  });

  it('explains failures, gives up when the service never answers, and stops on request', async () => {
    const denied = new FakeRecognition();
    const p1 = listenOnce(() => denied);
    denied.onerror?.({ error: 'not-allowed' });
    expect(await p1).toEqual({ ok: false, reason: 'Microphone or speech service access was refused.' });
    const silent = new FakeRecognition();
    const p2 = listenOnce(() => silent);
    silent.onend?.();
    expect(await p2).toEqual({ ok: false, reason: 'No speech was recognized.' });
    vi.useFakeTimers();
    try {
      const p3 = listenOnce(() => new FakeRecognition(), { timeoutMs: 5000 });
      vi.advanceTimersByTime(5000);
      expect(await p3).toEqual({ ok: false, reason: 'The browser speech service did not answer.' });
    } finally {
      vi.useRealTimers();
    }
    const stop = new AbortController();
    const p4 = listenOnce(() => new FakeRecognition(), { signal: stop.signal });
    stop.abort();
    expect(await p4).toEqual({ ok: false, reason: 'Listening was stopped.' });
    const broken = new FakeRecognition();
    broken.start = () => {
      throw new Error('not supported');
    };
    expect(await listenOnce(() => broken)).toEqual({ ok: false, reason: 'Speech recognition could not start.' });
  });
});
