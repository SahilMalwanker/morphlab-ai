import { useEffect, useId, useMemo, useRef, useState, type FormEvent } from 'react';
import type { AiAvailability } from '../ai/client';
import type { AiController } from '../ai/controller';
import type { ToolStep, TurnReport } from '../ai/loop';
import type { CommandOutcome } from './skills/session';
import { browserSpeech, listenOnce, spokenText } from './voice';

export interface VoicePrefs {
  speakReplies: boolean;
  voiceInputConsent: boolean;
}

interface CommandBarProps {
  ai: AiController;
  onRun: (text: string, onStep: (step: ToolStep) => void) => Promise<CommandOutcome>;
  /** While a command runs: applies slower, faster or stop to it; null when the text is not such a command. */
  onLive: (text: string) => string | null;
  /** The Stop control; works without any AI service and returns what it did. */
  onStop: () => string;
  hint: string;
  voice: VoicePrefs;
  onVoice: (patch: Partial<VoicePrefs>) => void;
}

export function CommandBar({ ai, onRun, onLive, onStop, hint, voice, onVoice }: CommandBarProps) {
  const inputId = useId();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<CommandOutcome | null>(null);
  const [steps, setSteps] = useState<ToolStep[]>([]);
  const [stopped, setStopped] = useState<string | null>(null);
  const [forgot, setForgot] = useState(false);
  const [live, setLive] = useState<string | null>(null);
  const [aiStatus, setAiStatus] = useState<AiAvailability>(ai.status);
  const speech = useMemo(() => browserSpeech(), []);
  const busyRef = useRef(false);
  const contextRevision = useRef(0);
  const listenStop = useRef<AbortController | null>(null);
  const [listening, setListening] = useState(false);
  const [askConsent, setAskConsent] = useState(false);
  const [voiceNote, setVoiceNote] = useState<string | null>(null);
  const [moreOpen, setMoreOpen] = useState(false);
  const moreId = useId();
  // Read at speaking time: a command that finishes after Read replies was turned off stays silent.
  const voiceRef = useRef(voice);
  voiceRef.current = voice;
  const say = (t: string) => {
    // Nothing is spoken while the microphone listens, so the app's own voice is never taken as a command.
    if (voiceRef.current.speakReplies && !listenStop.current) void speech.speaker.speak(t);
  };
  useEffect(() => () => {
    listenStop.current?.abort();
    speech.speaker.cancel();
  }, [speech]);
  const check = async (force: boolean) => setAiStatus(await ai.refreshStatus(force));
  useEffect(() => {
    let alive = true;
    void ai.refreshStatus().then((s) => {
      if (alive) setAiStatus(s);
    });
    return () => {
      alive = false;
    };
  }, [ai]);
  useEffect(() => {
    contextRevision.current++;
    setOutcome(null);
    setSteps([]);
    setStopped(null);
    setForgot(false);
    setLive(null);
  }, [hint]);
  const run = async (command: string) => {
    if (!command.trim() || busyRef.current) return;
    const revision = contextRevision.current;
    busyRef.current = true;
    setBusy(true);
    setOutcome(null);
    setSteps([]);
    setStopped(null);
    setForgot(false);
    setLive(null);
    let result: CommandOutcome;
    try {
      result = await onRun(command, (step) => {
        if (revision === contextRevision.current) setSteps((prev) => [...prev, step]);
      });
    } catch (e) {
      // A command that throws is reported, never left without an answer.
      result = { command, kind: 'failed', text: `The command could not be run: ${e instanceof Error ? e.message : String(e)}`, source: 'command bar', note: null };
    } finally {
      busyRef.current = false;
      setBusy(false);
      setAiStatus(ai.status);
    }
    if (revision === contextRevision.current) {
      setOutcome(result);
      say(spokenText(result));
    }
  };
  const liveCommand = (t: string) => {
    const reply = onLive(t) ?? `Not taken while the command runs: "${t.trim().slice(0, 60)}". Say slower, faster or stop, or press Stop.`;
    setLive(reply);
    say(reply);
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!busy) {
      void run(text);
      return;
    }
    if (!text.trim()) return;
    liveCommand(text);
    setText('');
  };
  const listen = async () => {
    const make = speech.recognition;
    if (!make) return;
    if (listenStop.current) {
      listenStop.current.abort();
      return;
    }
    if (!voice.voiceInputConsent) {
      setAskConsent(true);
      return;
    }
    const stop = new AbortController();
    listenStop.current = stop;
    speech.speaker.cancel();
    setListening(true);
    setVoiceNote('Listening (English)...');
    const r = await listenOnce(make, { signal: stop.signal });
    listenStop.current = null;
    setListening(false);
    if (!r.ok) {
      setVoiceNote(`Voice: ${r.reason}`);
      return;
    }
    setVoiceNote(`Heard: "${r.text}"`);
    // Spoken commands follow the same paths as typed ones: live changes while busy, otherwise a full command.
    if (busyRef.current) liveCommand(r.text);
    else {
      setText(r.text);
      void run(r.text);
    }
  };
  const last = steps[steps.length - 1];
  // The options stay open while the microphone listens, so Listening can always be pressed to cancel.
  const showMore = moreOpen || listening;
  return (
    <footer className="commandbar" aria-label="Command bar">
      <form className="command-row" onSubmit={submit} aria-label="Command">
        <label htmlFor={inputId} className="command-label">Command</label>
        <input id={inputId} className="command-input" type="text" value={text} maxLength={300} autoComplete="off" placeholder="lift the front left leg for three seconds" onChange={(e) => setText(e.target.value)} data-testid="command-input" />
        <button className="btn primary" type="submit" disabled={!text.trim()} title={busy ? 'While a command runs: slower, faster or stop' : undefined} data-testid="command-run">{busy ? 'Send' : 'Run'}</button>
        <button
          className="btn stop"
          type="button"
          onClick={() => {
            speech.speaker.cancel();
            const s = onStop();
            setStopped(s);
            say(s);
          }}
          title="Stops the running command or AI task here, without any AI service. It does not pause the simulation or cancel training and evaluation jobs."
          data-testid="command-stop"
        >
          Stop task
        </button>
        {busy && <span className="status idle">Running</span>}
        <AiStatusBadge status={aiStatus} />
        <button className="btn" type="button" aria-expanded={showMore} aria-controls={moreId} onClick={() => setMoreOpen(!moreOpen)} data-testid="command-more">
          Voice and AI
        </button>
      </form>
      <div id={moreId} className="command-more" role="group" aria-label="Voice and AI options" hidden={!showMore}>
        {speech.recognition ? (
          <button className="btn" type="button" aria-pressed={listening} onClick={() => void listen()} title="Say one command in English" data-testid="voice-speak">
            {listening ? 'Listening' : 'Speak'}
          </button>
        ) : (
          <span className="muted">Voice input: this browser has no speech recognition.</span>
        )}
        <button className="btn" type="button" onClick={() => void check(true)} data-testid="ai-check">
          Check AI
        </button>
        {aiStatus.state === 'unavailable' && <span className="muted">AI unavailable: {aiStatus.reason}</span>}
        <button
          className="btn"
          type="button"
          aria-pressed={voice.speakReplies}
          onClick={() => {
            if (voice.speakReplies) {
              speech.speaker.cancel();
              onVoice({ speakReplies: false });
            } else if (speech.speaker.voice()) {
              onVoice({ speakReplies: true });
              setVoiceNote(null);
            } else {
              setVoiceNote('No local English voice is installed in this browser, so replies are not read aloud.');
            }
          }}
          title="Read results aloud with a voice installed on this computer"
          data-testid="voice-read"
        >
          Read replies
        </button>
        {ai.history.length > 0 && !busy && (
          <button
            className="btn"
            type="button"
            onClick={() => {
              ai.forget();
              setForgot(true);
            }}
            title={`Conversation memory: the last ${ai.history.filter((m) => m.role === 'user').length} requests with their replies for this robot and scene, kept in this page only.`}
            data-testid="ai-forget"
          >
            Forget conversation
          </button>
        )}
      </div>
      <div role="status" className="command-output" data-testid="command-output">
        {askConsent && (
          <div className="note" style={{ margin: 0 }} data-testid="voice-consent">
            Voice input uses the browser&apos;s speech recognition. In Chrome and Edge the audio goes to the browser maker&apos;s speech service (Google or Microsoft); MorphLab neither receives nor stores it. Spoken commands run like typed ones.{' '}
            <button
              className="btn"
              type="button"
              onClick={() => {
                setAskConsent(false);
                onVoice({ voiceInputConsent: true });
                setVoiceNote('Voice input is on. Press Speak and say one command.');
              }}
              data-testid="voice-consent-accept"
            >
              Use voice input
            </button>{' '}
            <button className="btn" type="button" onClick={() => setAskConsent(false)}>Not now</button>
          </div>
        )}
        {voiceNote && (
          <div className="note" style={{ margin: 0 }} data-testid="voice-message">
            {voiceNote}
            {voice.voiceInputConsent && !listening && (
              <>
                {' '}
                <button
                  className="btn"
                  type="button"
                  onClick={() => {
                    onVoice({ voiceInputConsent: false });
                    setVoiceNote('Voice input is off; Speak asks again before it listens.');
                  }}
                  data-testid="voice-input-off"
                >
                  Turn off voice input
                </button>
              </>
            )}
          </div>
        )}
        {forgot && <div className="note" style={{ margin: 0 }} data-testid="forget-message">Conversation memory cleared; the next AI command starts without earlier messages.</div>}
        {live && <div className="note" style={{ margin: 0 }} data-testid="live-message">Live: {live}</div>}
        {stopped && <div className="note" style={{ margin: 0 }} data-testid="stop-message">Stop: {stopped}</div>}
        {outcome && (
          <>
            <span className={outcome.kind === 'refusal' || outcome.kind === 'failed' ? 'note error' : 'note'} style={{ margin: 0 }} data-testid="command-kind" data-kind={outcome.kind}>
              {outcome.kind === 'refusal' ? 'Refused: ' : outcome.kind === 'failed' ? 'Failed: ' : outcome.kind === 'clarify' ? 'Question: ' : ''}
              {outcome.text}
            </span>
            {outcome.suggestion && <span className="muted"> Suggestion: {outcome.suggestion}</span>}
            {outcome.options && (
              <span style={{ marginLeft: 8 }}>
                {outcome.options.map((o) => (
                  <button key={o} className="btn" type="button" style={{ marginRight: 4 }} onClick={() => void run(`${text} ${o}`)}>
                    {o}
                  </button>
                ))}
              </span>
            )}
            <span className="muted"> ({outcome.source})</span>
            {outcome.note && <div className="muted">{outcome.note}</div>}
            {outcome.ai && <AiTrace report={outcome.ai} />}
          </>
        )}
        {!outcome && !busy && !stopped && !forgot && (
          <span className="muted">
            {hint}
            {aiStatus.state === 'unavailable' && ` AI commands are off: ${aiStatus.reason}`}
          </span>
        )}
        {busy && <span className="muted">{last ? `Running: ${last.tool} ${last.result.status}. ` : 'Interpreting and running. '}The active tab shows the robot. While it runs you can send slower, faster or stop; Stop ends it.</span>}
      </div>
    </footer>
  );
}

function AiStatusBadge({ status }: { status: AiAvailability }) {
  const label = status.state === 'available' ? `AI: ${status.provider}` : status.state === 'unavailable' ? 'AI: unavailable' : 'AI: checking';
  return (
    <span className={`status ${status.state === 'available' ? 'ok' : 'idle'}`} data-testid="ai-status" data-state={status.state}>
      {label}
    </span>
  );
}

function formatArgs(args: string): string {
  const t = args.trim() || '{}';
  return t.length > 160 ? `${t.slice(0, 160)}...` : t;
}

/** Explainability: every tool call, its validation, the measured result and the model's own words, labeled. */
function AiTrace({ report }: { report: TurnReport }) {
  const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const tokens = report.tokens.prompt + report.tokens.completion;
  return (
    <details data-testid="ai-trace" style={{ marginTop: 4 }}>
      <summary>
        What the AI did: {plural(report.steps.length, 'tool call')}, {plural(report.modelCalls, 'model call')}
        {report.model ? `, model ${report.model}` : ''}
        {tokens ? `, ${tokens} tokens` : ''}
      </summary>
      <ol style={{ margin: '4px 0 0', paddingLeft: 20 }}>
        {report.steps.map((s, i) => (
          <li key={i}>
            <span className="mono">{s.tool} {formatArgs(s.arguments)}</span>
            {' '}
            {s.accepted ? '' : 'refused before running, '}
            <strong>{s.result.status}</strong>
            {s.result.taskId ? ` (${s.result.taskId})` : ''}: {s.result.reason}
            {Object.keys(s.result.measurements).length > 0 && (
              <span className="muted"> Measured: {Object.entries(s.result.measurements).map(([k, v]) => `${k} ${v.value} ${v.unit}`).join(', ')}.</span>
            )}
            {s.result.fallback && <span className="muted"> Fallback: {s.result.fallback}</span>}
          </li>
        ))}
      </ol>
      {report.text && <p className="note">Model reply (not a measurement): {report.text}</p>}
    </details>
  );
}
