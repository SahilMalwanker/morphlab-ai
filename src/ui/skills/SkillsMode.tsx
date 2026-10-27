import { lazy, Suspense, useEffect, useId, useState } from 'react';
import { createPortal } from 'react-dom';
import { PRESETS, type PresetId } from '../../core/model/catalog';
import { MAX_BUNDLE_CHARS } from '../../persistence/bundle';
import { evaluationOrigin, skillLabel, type SkillArtifact } from '../../skills/artifact';
import type { SkillMethod } from '../../skills/controllers';
import { GOAL_LABELS, goalText } from '../../skills/taught';
import { downloadText, FileButton, NumberField, Section, StatusBadge } from '../components';
import { Explain } from '../learn';
import { PhysicsViewport } from '../physics/PhysicsViewport';
import { useStore, useVersion } from '../store';
import { LastPlay, pct } from './LastPlay';
import { OUTCOME_LABELS, PARAM_FIELDS, SKILL_PRESETS, type SkillsSession } from './session';
import { TeachPanel } from './TeachPanel';

const CompareView = lazy(() => import('./CompareView').then((m) => ({ default: m.CompareView })));
const WalkPanel = lazy(() => import('./WalkPanel').then((m) => ({ default: m.WalkPanel })));

type SideView = 'teach' | 'library' | 'attempts';

const SIDE_TITLES: Record<SideView, string> = { teach: 'Teach', library: 'Skill library', attempts: 'Attempts' };

function SelectField({ label, value, options, onChange, testId }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void; testId?: string }) {
  const id = useId();
  return (
    <div className="field grow">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)} data-testid={testId}>
        {options.map(([v, l]) => (
          <option key={v} value={v}>{l}</option>
        ))}
      </select>
    </div>
  );
}

function Range({ label, value, min, max, step, onChange, testId }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; testId?: string }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} data-testid={testId} />
    </div>
  );
}

function EvaluateButton({ session, method, label = 'Evaluate 20 trials' }: { session: SkillsSession; method: SkillMethod; label?: string }) {
  const s = useStore(session.store);
  return (
    <button className="btn" disabled={s.job !== null || s.unsupported !== null} onClick={() => void session.evaluate(method)} data-testid={`evaluate-${method}`}>
      {label}
    </button>
  );
}

/** The playground context: robot, body facts, engine state and the side panel controls. */
function PlaygroundHeader({ session, view, collapsed, sideId, onView, onCollapsed }: { session: SkillsSession; view: SideView; collapsed: boolean; sideId: string; onView: (v: SideView) => void; onCollapsed: (collapsed: boolean) => void }) {
  const s = useStore(session.store);
  const p = useStore(session.physics.store);
  const robotOptions: [string, string][] = [
    ...(s.source === 'custom' ? ([['custom', `Builder robot: ${s.robot.name}`]] as [string, string][]) : []),
    ...SKILL_PRESETS.map((id) => [id, PRESETS.find((x) => x.id === id)!.label] as [string, string]),
  ];
  const joints = s.robot.limbs.reduce((n, l) => n + l.joints.length, 0);
  const counts: Record<SideView, string> = { teach: '', library: ` (${s.library.length})`, attempts: ` (${s.attempts.length})` };
  return (
    <header className="pg-head">
      <div className="pg-title">
        <h2>AI Playground</h2>
        <span className="muted">Skills</span>
      </div>
      <div className="pg-robot">
        <SelectField label="Robot" value={s.source} options={robotOptions} onChange={(v) => v !== 'custom' && session.setPreset(v as PresetId)} testId="skills-robot" />
        <p className="pg-facts">
          <span>
            {s.robot.base.kind === 'floating' ? 'Free-standing' : 'Fixed base'}, {s.robot.limbs.length} limbs, {joints} joints
          </span>
          <span className="mono" title={s.hash ?? ''} data-testid="pg-morphology">
            morphology {s.hash ? s.hash.slice(0, 12) : 'computing'}
          </span>
        </p>
      </div>
      {p.status === 'ready' ? (
        <StatusBadge tone="ok" testId="pg-engine">Live simulation</StatusBadge>
      ) : p.status === 'loading' ? (
        <StatusBadge tone="idle" testId="pg-engine">Loading physics engine</StatusBadge>
      ) : (
        <StatusBadge tone="error" testId="pg-engine">Simulation unavailable</StatusBadge>
      )}
      <div className="seg pg-views" role="group" aria-label="Side panel">
        {(['teach', 'library', 'attempts'] as const).map((v) => (
          <button key={v} type="button" className="btn" aria-pressed={view === v && !collapsed} onClick={() => onView(v)} data-testid={`pg-view-${v}`}>
            {v === 'teach' ? 'Teach' : v === 'library' ? 'Library' : 'Attempts'}
            {counts[v]}
          </button>
        ))}
        <button type="button" className="btn" aria-expanded={!collapsed} aria-controls={sideId} onClick={() => onCollapsed(!collapsed)} data-testid="pg-panel-toggle">
          {collapsed ? 'Show panel' : 'Hide panel'}
        </button>
      </div>
    </header>
  );
}

/** Job progress with its Cancel, and the latest message; never hidden with the side panel. */
function PlaygroundStatus({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  if (!s.job && !s.message) return null;
  return (
    <div className="pg-status">
      {s.job && (
        <div className="job" role="status" data-testid="skills-job">
          <span>
            {s.job.label}: {(s.job.fraction * 100).toFixed(0)} %
          </span>
          <progress max={1} value={s.job.fraction} aria-label={`${s.job.label} progress`} />
          <span className="muted job-detail">{s.job.detail}</span>
          <button className="btn" type="button" onClick={() => session.cancelJob()} data-testid="skills-job-cancel">
            Cancel
          </button>
        </div>
      )}
      {s.message && (
        <p className={`note ${s.message.tone === 'error' ? 'error' : ''}`} role={s.message.tone === 'error' ? 'alert' : 'status'} data-testid="skills-message">
          {s.message.text}
        </p>
      )}
    </div>
  );
}

/** Shown over the robot view until the simulation is ready, or when it cannot start. */
function EngineNotice({ session }: { session: SkillsSession }) {
  const p = useStore(session.physics.store);
  if (p.status === 'ready') return null;
  return (
    <div className="view-notice" role={p.status === 'error' ? 'alert' : 'status'} data-testid="pg-engine-notice">
      {p.status === 'loading'
        ? 'Loading the physics engine for the live robot. The lab tools stay available in the navigation; Build and Kinematics do not need the engine.'
        : `${p.error ?? 'The simulation could not start.'} Trying, evaluating and playing skills need it; reload the page to try again. Build and Kinematics work without it.`}
    </div>
  );
}

function TeachView({ session, onOpenLibrary }: { session: SkillsSession; onOpenLibrary: () => void }) {
  const s = useStore(session.store);
  return (
    <>
      <div className="seg method-row" role="group" aria-label="Workflow">
        <button type="button" className="btn" aria-pressed={s.panel === 'teach'} onClick={() => session.setPanel('teach')} data-testid="skills-panel-teach">
          Teach a skill
        </button>
        <button type="button" className="btn" aria-pressed={s.panel === 'foot-lift'} onClick={() => session.setPanel('foot-lift')} data-testid="skills-panel-foot-lift">
          Foot lift example
        </button>
        <button type="button" className="btn" aria-pressed={s.panel === 'walk'} onClick={() => session.setPanel('walk')} data-testid="skills-panel-walk">
          Walk by command
        </button>
      </div>
      {s.panel === 'teach' ? (
        <TeachPanel session={session} onOpenLibrary={onOpenLibrary} />
      ) : s.panel === 'walk' ? (
        <Suspense fallback={<p className="note" role="status">Loading walking controls.</p>}>
          <WalkPanel session={session} />
        </Suspense>
      ) : (
        <FootLiftPanel session={session} />
      )}
    </>
  );
}

/** The built-in example skill, taught five ways: hand design, tuning, demonstration, imitation and reward. */
function FootLiftPanel({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  const busy = s.job !== null || s.unsupported !== null;
  const t = s.teleop;
  return (
    <>
      <Section title="Example skill: lift one foot">
        {s.unsupported ? (
          <p className="note warn">{s.unsupported}</p>
        ) : (
          <SelectField label="Leg to lift" value={s.limbId} options={s.legs.map((l) => [l.id, l.name])} onChange={(v) => session.setLimb(v)} testId="skills-leg" />
        )}
        <p className="note muted">Success: the foot stays at least 3 cm clear for 2 s while the other feet keep contact, tilt stays below 0.35 rad and the body does not drop. Each evaluation runs 20 randomized trials (friction 0.6-1.2, body mass within 12.5 % of the robot's own, joint offsets up to 0.02 rad) in a Web Worker with the live joint gains. Playback is judged by the same rules.</p>
        <Explain>
          <p>The methods differ in where the behavior comes from. Replay plays recorded joint targets open loop: it cannot react if the body moves differently. The state machine is designed by hand and closes the loop on sensors (it lowers the foot until the touch sensor reports contact). Tuned and improved parameters change that design's numbers. The imitation policy is a small neural network fitted to the state machine's demonstrations, so at best it copies the expert, including its limits.</p>
          <p>With 20 trials the 95 % interval stays wide: 20 of 20 successes only shows a success rate of about 84 % or more. The referee uses simulator ground truth; the controllers see encoders, the IMU and foot touch only.</p>
          <p>Reward-based improvement scores each trial as 0.5 x min(clearance / 3 cm, 1.5) + the fraction of the 2 s hold achieved + 1 for success - 2 x (tilt beyond 0.1 rad) - 1 for a fall or abort, and moves the parameters toward higher reward.</p>
        </Explain>
      </Section>
      <Section title="1. State machine (hand-designed)">
        <p className="note muted">Shift the body over the other feet, lift with leg IK, hold, lower until the touch sensor reports contact, shift back.</p>
        <EvaluateButton session={session} method="state-machine" />
      </Section>
      <Section title="2. Tuned parameters">
        <div className="row" style={{ flexWrap: 'wrap' }}>
          {PARAM_FIELDS.map((f) => (
            <div key={f.key} style={{ width: 'calc(50% - 4px)' }}>
              <NumberField label={f.label} value={s.params[f.key]} digits={f.digits} step={f.digits === 3 ? 0.005 : 0.1} min={f.min} max={f.max} onCommit={(v) => session.setParam(f.key, v)} testId={`param-${f.key}`} />
            </div>
          ))}
        </div>
        <EvaluateButton session={session} method="tuned-parameters" />
      </Section>
      <Section title="3. Demonstration and replay">
        <div className="row">
          <button className="btn" disabled={busy} onClick={() => void session.recordExpertDemonstration()} data-testid="record-expert">Record expert</button>
          {!t.active ? (
            <button className="btn" disabled={busy} onClick={() => session.startTeleop()} data-testid="teleop-start">Demonstrate yourself</button>
          ) : (
            <button className="btn" onClick={() => session.stopActivity()}>End demonstration</button>
          )}
        </div>
        {t.active && (
          <div style={{ marginTop: 8 }}>
            <Range label={`Body shift: ${(t.shift * 100).toFixed(0)} %`} value={t.shift} min={0} max={1} step={0.01} onChange={(v) => session.setTeleop(v, t.lift)} testId="teleop-shift" />
            <Range label={`Foot height: ${(t.lift * 100).toFixed(1)} cm`} value={t.lift} min={0} max={0.1} step={0.002} onChange={(v) => session.setTeleop(t.shift, v)} testId="teleop-lift" />
            {!t.recording ? (
              <button className="btn primary" onClick={() => session.startRecording()} data-testid="teleop-record">Start recording</button>
            ) : (
              <button className="btn primary" onClick={() => session.stopRecording()} data-testid="teleop-stop">Stop recording</button>
            )}
            {t.recording && <span className="note" style={{ marginLeft: 8 }}>Recording</span>}
          </div>
        )}
        <p className="note muted" data-testid="demonstration-status">
          {s.demonstration
            ? `Demonstration: ${s.demonstration.source === 'expert' ? 'state-machine expert' : 'yours'}, ${((s.demonstration.trajectory.frames.length - 1) * s.demonstration.trajectory.dt).toFixed(1)} s. Replay plays the joint targets open loop, without feedback.`
            : 'No demonstration yet.'}
        </p>
        <EvaluateButton session={session} method="replay" label="Evaluate replay" />
      </Section>
      <Section title="4. Imitation learning">
        <p className="note muted">Records 12 expert episodes under random conditions and trains a 64x64 network (behavior cloning) to map sensor readings to joint targets.</p>
        <div className="row">
          <button className="btn" disabled={busy} onClick={() => void session.trainImitation()} data-testid="train-imitation">Train policy</button>
          <EvaluateButton session={session} method="imitation-mlp" label="Evaluate policy" />
        </div>
        {s.imitation && <p className="note muted">Policy trained on {s.imitation.samples} samples; loss {s.imitation.initialLoss.toFixed(3)} to {s.imitation.finalLoss.toExponential(2)}.</p>}
      </Section>
      <Section title="5. Reward-based improvement">
        <p className="note muted">Evolution strategies adjust shift, lift height and timings, starting from the tuned parameters above (never from scratch).</p>
        <div className="row">
          <button className="btn" disabled={busy} onClick={() => void session.improve()} data-testid="improve">Improve parameters</button>
          <EvaluateButton session={session} method="es-improved" label="Evaluate result" />
        </div>
        {s.improvement && (
          <p className="note muted" data-testid="improvement">
            Held-out reward: baseline {s.improvement.baselineReward.toFixed(3)}, result {s.improvement.improvedReward.toFixed(3)}. Lift {(s.improvement.params.liftHeightM * 100).toFixed(1)} cm, shift {(s.improvement.params.shiftM * 100).toFixed(1)} cm.
          </p>
        )}
      </Section>
    </>
  );
}

/** One library record: a scannable summary, and the full evaluation and actions when expanded. */
function SkillRecord({ session, a, hash, expanded, onToggle }: { session: SkillsSession; a: SkillArtifact; hash: string | null; expanded: boolean; onToggle: () => void }) {
  const s = useStore(session.store);
  const memory = session.memory;
  useVersion(memory);
  const bodyId = useId();
  const runs = memory?.skillRuns(a.id) ?? null;
  const e = a.evaluation;
  const matches = hash !== null && a.robot.morphologyHash === hash;
  const origin = evaluationOrigin(a);
  const claimed = a.importedEvaluation;
  const taught = a.skillType === 'taught' ? a.taught : undefined;
  const limbName = (id: string) => (matches ? s.robot.limbs.find((l) => l.id === id)?.name.toLowerCase() : undefined) ?? id;
  const summary = `${e ? `${e.successes} / ${e.trials} trials${origin === 'imported' ? ' claimed by the file' : ''}` : 'not evaluated'}, ${matches ? 'this robot' : 'other robot'}`;
  return (
    <li>
      <article className={`record${expanded ? ' open' : ''}`} data-testid="skill-card">
        <div className="record-top">
          <button type="button" className="record-head" aria-expanded={expanded} aria-controls={expanded ? bodyId : undefined} onClick={onToggle}>
            <span className="record-name">{a.name}</span>
            <span className="record-sub" data-testid="skill-method">{skillLabel(a)}</span>
            <span className="record-sub">{summary}</span>
          </button>
          {(s.playing === a.id || origin === 'imported') && (
            <div className="record-badges">
              {s.playing === a.id && <StatusBadge tone="ok">Playing</StatusBadge>}
              {origin === 'imported' && <StatusBadge tone="warn">Unverified here</StatusBadge>}
            </div>
          )}
        </div>
        {expanded && (
          <div id={bodyId} className="record-body">
      <dl className="readout">
        <dt>Success</dt>
        <dd data-testid="skill-success">{e ? `${e.successes} / ${e.trials} (${pct(e.successRate)}; 95 % CI ${pct(e.wilson95[0])}-${pct(e.wilson95[1])})${origin === 'imported' ? ', claimed by the file' : ''}` : 'not evaluated'}</dd>
        <dt>Live runs</dt>
        <dd data-testid="skill-live-runs" title="Skill memory: every playback of this skill, from Play, commands and AI, judged by the referee. A record, not an evaluation.">
          {runs
            ? `${runs.completed} completed, ${runs.succeeded} succeeded${runs.interrupted ? `, ${runs.interrupted} stopped early` : ''}; last ${OUTCOME_LABELS[runs.last.outcome]}${runs.last.holdS !== null ? `, hold ${runs.last.holdS.toFixed(2)} s` : ''}`
            : 'none recorded'}
          {runs && (
            <button className="btn" type="button" style={{ marginLeft: 6 }} onClick={() => memory?.forgetSkillRuns(a.id)} data-testid="skill-live-runs-clear">
              Clear
            </button>
          )}
        </dd>
        <dt>Morphology</dt>
        <dd title={a.robot.morphologyHash}>{a.robot.morphologyHash.slice(0, 12)} {matches ? '(this robot)' : '(other robot)'}</dd>
        {taught ? (
          <>
            <dt>Goals</dt>
            <dd data-testid="skill-goals">{taught.goals.map((g) => goalText(g, limbName)).join('; ')}</dd>
            <dt>Command</dt>
            <dd>{`say "${taught.name}"`}</dd>
          </>
        ) : (
          <>
            <dt>Leg</dt>
            <dd>{a.limbs.join(', ')}</dd>
          </>
        )}
        {e && (
          <>
            <dt>Tested conditions</dt>
            <dd>friction {e.conditions.friction?.join('-')}, mass {e.conditions.baseMassKg?.join('-')} kg, noise +-{e.conditions.jointNoiseRad?.[1]} rad</dd>
            {e.context && (
              <>
                <dt>Setup</dt>
                <dd data-testid="skill-setup">kp {e.context.gains.kp} N*m/rad, kd {e.context.gains.kd} N*m*s/rad, seed {e.context.seed}, {e.context.velocityLimited ? 'target velocity limited' : 'no velocity limit'}</dd>
              </>
            )}
            {taught ? (
              <>
                <dt>Goals met</dt>
                <dd data-testid="skill-goals-met">{taught.goals.map((g, i) => `${GOAL_LABELS[g.kind]} ${e.metrics[`goalMet:${i}`] ?? 0} / ${e.trials}`).join(', ')}</dd>
              </>
            ) : (
              <>
                <dt>Mean clear hold</dt>
                <dd>{e.metrics.meanClearHoldS?.toFixed(2)} s</dd>
              </>
            )}
            {Object.keys(e.metrics).some((k) => k.startsWith('failed:')) && (
              <>
                <dt>Failures</dt>
                <dd data-testid="skill-failures">
                  {Object.entries(e.metrics)
                    .filter(([k]) => k.startsWith('failed:'))
                    .map(([k, n]) => `${k.slice(7).replace(/-/g, ' ')} ${n}`)
                    .join(', ')}
                </dd>
              </>
            )}
          </>
        )}
        {a.training && (
          <>
            <dt>Training</dt>
            <dd>{a.training.algorithm}, {a.training.iterations} iterations, {a.training.wallTimeS} s</dd>
          </>
        )}
        <dt>Evaluated</dt>
        <dd>{e ? `${e.simulator}, ${new Date(e.evaluatedAt).toLocaleString()}` : ''}</dd>
        {(a.importedAt || origin === 'imported') && (
          <>
            <dt>Origin</dt>
            <dd data-testid="skill-origin">
              {origin === 'imported'
                ? `Imported from a file${a.importedAt ? ` ${new Date(a.importedAt).toLocaleString()}` : ''}; the evaluation was not re-run here. Text commands will not use it until it is re-evaluated here.`
                : `Imported from a file${a.importedAt ? ` ${new Date(a.importedAt).toLocaleString()}` : ''}; re-evaluated here${claimed ? ` (the file claimed ${claimed.successes} / ${claimed.trials})` : ''}.`}
            </dd>
          </>
        )}
      </dl>
            <div className="row wrap" style={{ marginTop: 6 }}>
              <button className="btn" onClick={() => void session.play(a.id)} disabled={s.job !== null} data-testid="skill-play">Play</button>
              <button className="btn" onClick={() => void session.reevaluate(a.id)} disabled={s.job !== null || !matches} data-testid="skill-reevaluate">Re-evaluate here</button>
              <button className="btn danger" onClick={() => session.deleteSkill(a.id)}>Delete</button>
            </div>
            {hash !== null && !matches && (
              <p className="note muted">Evaluated on another body (morphology {a.robot.morphologyHash.slice(0, 12)}): Play refuses to run it on this robot, and re-evaluating needs that robot.</p>
            )}
          </div>
        )}
      </article>
    </li>
  );
}

function Attempts({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  const [pick, setPick] = useState<{ left: string; right: string } | null>(null);
  const ids = s.attempts.map((a) => a.id);
  // Default: the previous attempt on the left, the newest on the right.
  const left = pick && ids.includes(pick.left) ? pick.left : (s.attempts[1]?.id ?? '');
  const right = pick && ids.includes(pick.right) ? pick.right : (s.attempts[0]?.id ?? '');
  const options: [string, string][] = s.attempts.map((a) => [a.id, a.label]);
  return (
    <Section title={`Attempts (${s.attempts.length})`}>
      <LastPlay session={session} />
      {s.attempts.length === 0 && <p className="note muted">Each Play records the simulated motion. Play skills at least twice to compare attempts side by side.</p>}
      {s.attempts.length > 0 && (
        <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12 }} data-testid="attempts">
          {s.attempts.map((a) => (
            <li key={a.id}>
              {a.label}: {a.resultText}
            </li>
          ))}
        </ul>
      )}
      {s.attempts.length >= 2 && (
        <>
          <div className="row" style={{ marginTop: 8 }}>
            <SelectField label="A (left)" value={left} options={options} onChange={(v) => setPick({ left: v, right })} testId="compare-a" />
            <SelectField label="B (right)" value={right} options={options} onChange={(v) => setPick({ left, right: v })} testId="compare-b" />
          </div>
          <button className="btn" type="button" onClick={() => session.openComparison(left, right)} disabled={s.comparing !== null} data-testid="compare-open">
            Compare side by side
          </button>
        </>
      )}
      <Explain>
        <p>The comparison replays recorded simulator states of two playbacks on one timeline, so differences in timing and motion are easy to see. It is a recording, not a new simulation.</p>
      </Explain>
    </Section>
  );
}

function Library({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  // null: the newest record is open; '': none; otherwise the record the user opened.
  const [picked, setPicked] = useState<string | null>(null);
  const newest = s.library[0]?.id ?? '';
  useEffect(() => {
    setPicked(null);
  }, [newest]);
  const open = picked !== null && (picked === '' || s.library.some((a) => a.id === picked)) ? picked : newest;
  return (
    <Section title={`Skill library (${s.library.length})`}>
      <div className="row wrap" style={{ marginBottom: 8 }}>
        <button className="btn" onClick={() => downloadText(session.exportText(), `morphlab-skills-${new Date().toISOString().slice(0, 10)}.morphlab.json`)} disabled={s.library.length === 0} data-testid="skills-export">
          Export library
        </button>
        <FileButton label="Import skills" maxBytes={MAX_BUNDLE_CHARS} onText={(t) => session.importText(t)} onRefuse={(m) => session.refuseImport(m)} testId="skills-import" />
      </div>
      {s.library.length === 0 ? (
        <p className="note muted">No saved skills yet. Teach a skill or evaluate the foot lift example; evaluated skills appear here with their method, success rate and the body they were evaluated on. A skill only plays on that body.</p>
      ) : (
        <ul className="record-list">
          {s.library.map((a) => (
            <SkillRecord key={a.id} session={session} a={a} hash={s.hash} expanded={open === a.id} onToggle={() => setPicked(open === a.id ? '' : a.id)} />
          ))}
        </ul>
      )}
    </Section>
  );
}

/**
 * The AI Playground. The side panel is rendered into `sideSlot`, an element the
 * App places after the command bar, so that in a single column the command field
 * and Stop task come before a long teaching form in both reading and Tab order.
 */
export function SkillsMode({ session, sideSlot, panelCollapsed, onPanelCollapsed }: { session: SkillsSession; sideSlot: HTMLElement | null; panelCollapsed: boolean; onPanelCollapsed: (collapsed: boolean) => void }) {
  const s = useStore(session.store);
  const sideId = useId();
  const [view, setView] = useState<SideView>('teach');
  useEffect(() => {
    void session.physics.init();
    return () => {
      session.stopActivity('the Skills view was closed');
    };
  }, [session]);
  // A draft the AI composes or revises is shown where it can be inspected and edited.
  const draft = s.draft;
  useEffect(() => {
    if (draft && draft.author !== 'user') setView('teach');
  }, [draft]);
  const show = (v: SideView) => {
    setView(v);
    if (panelCollapsed) onPanelCollapsed(false);
  };
  const left = s.comparing && s.attempts.find((a) => a.id === s.comparing![0]);
  const right = s.comparing && s.attempts.find((a) => a.id === s.comparing![1]);
  const comparing = Boolean(left && right);
  return (
    <>
      <PlaygroundHeader session={session} view={view} collapsed={panelCollapsed} sideId={sideId} onView={show} onCollapsed={onPanelCollapsed} />
      <PlaygroundStatus session={session} />
      {left && right ? (
        <Suspense fallback={<p className="note" role="status">Loading comparison.</p>}>
          <CompareView left={left} right={right} onClose={() => session.closeComparison()} />
        </Suspense>
      ) : (
        <PhysicsViewport session={session.physics} label="Skills: live MuJoCo simulation. Controllers see joint encoders, IMU and foot touch only; rendering uses simulator state." />
      )}
      {!comparing && <EngineNotice session={session} />}
      {sideSlot &&
        createPortal(
          <aside id={sideId} className="panel pg-side" aria-label={SIDE_TITLES[view]} hidden={panelCollapsed}>
            {view === 'teach' && <TeachView session={session} onOpenLibrary={() => show('library')} />}
            {view === 'library' && <Library session={session} />}
            {view === 'attempts' && <Attempts session={session} />}
          </aside>,
          sideSlot,
        )}
    </>
  );
}
