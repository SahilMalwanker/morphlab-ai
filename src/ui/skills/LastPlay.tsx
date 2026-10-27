import { RAD2DEG } from '../components';
import { useStore } from '../store';
import { OUTCOME_LABELS, type SkillsSession } from './session';

export function pct(v: number): string {
  return `${(v * 100).toFixed(0)} %`;
}

/** The latest live playback (a try, Play or a command), measured from simulator state; a record, not an evaluation. */
export function LastPlay({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  const r = s.lastPlay;
  if (!r) return null;
  const failures = r.failures.map((f) => f.replace(/-/g, ' '));
  if (r.goals) {
    const met = r.goals.filter((g) => g.met).length;
    const missed = r.goals.filter((g) => !g.met).map((g) => g.text);
    return (
      <div className="note" data-testid="last-play" style={{ marginBottom: 8 }}>
        <span className={r.success ? '' : 'error'}>Last playback of {r.skillName} {OUTCOME_LABELS[r.outcome]}</span>
        : {met} of {r.goals.length} goals met, max tilt {(r.maxTiltRad * RAD2DEG).toFixed(1)} deg (simulator ground truth).
        {missed.length > 0 && ` Missed: ${missed.join('; ')}.`}
        {r.reason && ` Reason: ${r.reason}.`}
        {r.setupNotes.length > 0 && <span className="muted"> {r.setupNotes.join(' ')}</span>}
      </div>
    );
  }
  return (
    <div className="note" data-testid="last-play" style={{ marginBottom: 8 }}>
      <span className={r.success ? '' : 'error'}>Last playback {OUTCOME_LABELS[r.outcome]}</span>
      : hold {r.holdS.toFixed(2)} s of {r.requiredHoldS} s required, max tilt {(r.maxTiltRad * RAD2DEG).toFixed(1)} deg, support contact {pct(r.supportContactFraction)} while lifted (simulator ground truth).
      {failures.length > 0 && ` Failed: ${failures.join(', ')}.`}
      {r.reason && ` Reason: ${r.reason}.`}
      {r.setupNotes.length > 0 && <span className="muted"> {r.setupNotes.join(' ')}</span>}
    </div>
  );
}
