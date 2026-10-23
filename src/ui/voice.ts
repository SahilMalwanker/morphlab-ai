import type { CommandOutcome } from './skills/session';

/**
 * Browser speech for the command bar. Output uses installed local voices
 * only, so reply text never leaves the machine. Input uses the browser's
 * speech recognition, which in Chrome and Edge sends audio to the browser
 * vendor's service; it is used only after the user agrees.
 */

export interface VoiceLike {
  name: string;
  lang: string;
  localService: boolean;
}

export interface UtteranceLike {
  text: string;
  lang: string;
  voice: unknown;
  onend: (() => void) | null;
  onerror: ((e: { error: string }) => void) | null;
}

export interface SynthLike {
  getVoices(): VoiceLike[];
  speak(u: UtteranceLike): void;
  cancel(): void;
}

const MAX_SPOKEN_CHARS = 400;

export class Speaker {
  constructor(
    private readonly synth: SynthLike | null,
    private readonly makeUtterance: (text: string) => UtteranceLike,
  ) {}

  /** An installed English voice that works without a network service, or null. */
  voice(): VoiceLike | null {
    const local = this.synth?.getVoices().filter((v) => v.localService) ?? [];
    return local.find((v) => v.lang.toLowerCase().startsWith('en')) ?? null;
  }

  speak(text: string): Promise<'done' | 'cancelled' | 'error' | 'unavailable'> {
    const voice = this.voice();
    const synth = this.synth;
    if (!synth || !voice || !text.trim()) return Promise.resolve('unavailable');
    synth.cancel();
    return new Promise((resolve) => {
      const u = this.makeUtterance(text.slice(0, MAX_SPOKEN_CHARS));
      u.voice = voice;
      u.lang = voice.lang;
      u.onend = () => resolve('done');
      u.onerror = (e) => resolve(e.error === 'canceled' || e.error === 'interrupted' ? 'cancelled' : 'error');
      synth.speak(u);
    });
  }

  cancel(): void {
    this.synth?.cancel();
  }
}

/** What to read aloud for a command outcome; the model's own words are announced as such. */
export function spokenText(o: CommandOutcome): string {
  const noAction = o.note === 'No action was taken; this is the model reply, not a measurement.';
  switch (o.kind) {
    case 'refusal':
      return `Refused. ${o.text}`;
    case 'failed':
      return `Failed. ${o.text}`;
    case 'clarify':
      return `Question. ${o.text}`;
    case 'done':
      return noAction ? `The AI says: ${o.text}` : o.text;
  }
}

export interface RecognitionResultLike {
  readonly length: number;
  readonly isFinal?: boolean;
  [index: number]: { transcript: string; confidence: number };
}

export interface RecognitionLike {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  continuous: boolean;
  onstart: (() => void) | null;
  onresult: ((e: { results: ArrayLike<RecognitionResultLike> }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  abort(): void;
}

export type ListenResult = { ok: true; text: string; confidence: number } | { ok: false; reason: string };

const RECOGNITION_ERRORS: Record<string, string> = {
  'not-allowed': 'Microphone or speech service access was refused.',
  'service-not-allowed': 'Microphone or speech service access was refused.',
  'no-speech': 'No speech was heard.',
  'audio-capture': 'No microphone was found.',
  network: 'The browser speech service could not be reached.',
  aborted: 'Listening was stopped.',
  'language-not-supported': 'English speech recognition is not available in this browser.',
};

/** Listens for one English utterance; the local command rules understand English only. */
export function listenOnce(make: () => RecognitionLike, opts: { timeoutMs?: number; signal?: AbortSignal } = {}): Promise<ListenResult> {
  return new Promise((resolve) => {
    let settled = false;
    const r = make();
    const finish = (x: ListenResult): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      opts.signal?.removeEventListener('abort', onAbort);
      try {
        r.abort();
      } catch {
        // Already ended.
      }
      resolve(x);
    };
    const onAbort = (): void => finish({ ok: false, reason: 'Listening was stopped.' });
    // Some browsers never answer without their speech service; do not wait forever.
    const timer = setTimeout(() => finish({ ok: false, reason: 'The browser speech service did not answer.' }), opts.timeoutMs ?? 10_000);
    r.lang = 'en-US';
    r.interimResults = false;
    r.continuous = false;
    r.maxAlternatives = 1;
    r.onresult = (e) => {
      const best = e.results[0]?.[0];
      const text = best?.transcript.trim() ?? '';
      if (text) finish({ ok: true, text: text.slice(0, 300), confidence: Number.isFinite(best?.confidence) ? best!.confidence : 0 });
    };
    r.onerror = (e) => finish({ ok: false, reason: RECOGNITION_ERRORS[e.error] ?? 'Speech recognition failed.' });
    r.onend = () => finish({ ok: false, reason: 'No speech was recognized.' });
    if (opts.signal?.aborted) return onAbort();
    opts.signal?.addEventListener('abort', onAbort, { once: true });
    try {
      r.start();
    } catch {
      finish({ ok: false, reason: 'Speech recognition could not start.' });
    }
  });
}

/** The browser's speech objects, or null where they do not exist. */
export function browserSpeech(): { speaker: Speaker; recognition: (() => RecognitionLike) | null } {
  const w = typeof window === 'undefined' ? null : (window as unknown as Record<string, unknown>);
  const synth = w && 'speechSynthesis' in w ? (w['speechSynthesis'] as SynthLike) : null;
  const Utterance = w?.['SpeechSynthesisUtterance'] as (new (text: string) => UtteranceLike) | undefined;
  const Recognition = (w?.['SpeechRecognition'] ?? w?.['webkitSpeechRecognition']) as (new () => RecognitionLike) | undefined;
  return {
    speaker: new Speaker(synth && Utterance ? synth : null, (text) => new Utterance!(text)),
    recognition: Recognition ? () => new Recognition() : null,
  };
}
