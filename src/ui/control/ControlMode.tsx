import { useEffect, useId, useMemo } from 'react';
import { restsOffTarget, type StepMetrics } from '../../core/control/stepMetrics';
import { PRESETS, type PresetId } from '../../core/model/catalog';
import { jointUnits, sharedGainUnits } from '../../core/model/units';
import { CHALLENGES, type ChallengeId, type ChallengeResult, type ControllerSpec, type Criterion, type StepCommand, type StepTestResult } from '../../physics/experiments';
import { NumberField, Section, StatusBadge, Toggle } from '../components';
import { Explain } from '../learn';
import { StaticPlot, type StaticSeries } from '../plot';
import { useStore } from '../store';
import { CONTROL_PRESETS, type ControlSession, type ControlState, type GainSet, type StepRunSetup } from './session';

const COLOR = { cmd: '#8592a3', a: '#58a6ff', b: '#e3b341' };

const COMMAND_OPTIONS: [StepCommand, string][] = [
  ['rate-limited', 'Rate-limited at the joint velocity limit'],
  ['ideal-step', 'Ideal step (analytical reference)'],
];

function sameSpec(x: ControllerSpec | null, y: ControllerSpec | null): boolean {
  return x === y || (!!x && !!y && x.gains.kp === y.gains.kp && x.gains.kd === y.gains.kd && x.gravityCompensation === y.gravityCompensation);
}

/** Whether the shown step results were produced with the current settings. */
function setupChanged(s: ControlState): boolean {
  const r = s.stepSetup;
  if (!r) return false;
  return r.joint !== s.joint || r.step !== s.step || r.durationS !== s.durationS || r.command !== s.command || !sameSpec(r.a, s.a) || (r.b !== null) !== s.compare || (r.b !== null && !sameSpec(r.b, s.b));
}

function challengeSetupChanged(s: ControlState, setup: { a: ControllerSpec; b: ControllerSpec | null }): boolean {
  return !sameSpec(setup.a, s.a) || (setup.b !== null) !== s.compare || (setup.b !== null && !sameSpec(setup.b, s.b));
}

function describeSpec(c: ControllerSpec): string {
  return `kp ${c.gains.kp}, kd ${c.gains.kd}${c.gravityCompensation ? ', gravity compensation' : ''}`;
}

function fmtNum(v: number | null | undefined, digits: number, unit = ''): string {
  return v === null || v === undefined || !Number.isFinite(v) ? 'n/a' : `${v.toFixed(digits)}${unit}`;
}

/** Explains a run that came to rest outside the settling band, which more damping cannot fix. */
function RestNote({ a, b, setup, unit, scale }: { a: StepTestResult; b: StepTestResult | null; setup: StepRunSetup | null; unit: string; scale: number }) {
  if (!setup) return null;
  const runs: [string, StepTestResult | null, ControllerSpec | null][] = [['A', a, setup.a], ['B', b, setup.b]];
  const facts: string[] = [];
  const why = new Set<string>();
  for (const [label, r, spec] of runs) {
    if (!r || !spec || r.unstable || !restsOffTarget(r.qMeas, r.y0, r.target)) continue;
    facts.push(`${label} came to rest ${fmtNum(Math.abs(r.metrics.steadyStateError) * scale, 2, ` ${unit}`)} from the target, outside the 2 % band, so it never settles.`);
    why.add(
      spec.gravityCompensation
        ? 'An error the joint keeps at rest does not depend on kd; a larger kp shrinks it while the actuator stays below its effort limit.'
        : 'Under PD alone the joint rests where kp times the error balances gravity: gravity compensation removes that error, a larger kp shrinks it, and kd does not change it.',
    );
  }
  if (facts.length === 0) return null;
  return (
    <p className="note" data-testid="rest-note">
      {[...facts, ...why].join(' ')}
    </p>
  );
}

function GainEditor({ session, which }: { session: ControlSession; which: GainSet }) {
  const s = useStore(session.store);
  const spec = s[which];
  const st = s.stability?.[which];
  const label = which === 'a' ? 'Gain set A' : 'Gain set B';
  const units = sharedGainUnits(s.robot.limbs.flatMap((l) => l.joints.map((j) => j.type)));
  return (
    <Section title={label}>
      <div className="row">
        <NumberField label={`kp (${units.stiffness})`} value={spec.gains.kp} digits={1} step={1} min={0} onCommit={(v) => session.setController(which, { kp: v })} testId={`kp-${which}`} />
        <NumberField label={`kd (${units.damping})`} value={spec.gains.kd} digits={2} step={0.1} min={0} onCommit={(v) => session.setController(which, { kd: v })} testId={`kd-${which}`} />
      </div>
      <Toggle label="Gravity compensation (model-based feedforward)" checked={spec.gravityCompensation} onChange={(v) => session.setController(which, { gravityCompensation: v })} />
      {st && (
        <p className={`note ${st.ok ? 'muted' : 'warn'}`} data-testid={`stability-${which}`}>
          {st.ok ? 'Within' : 'Outside'} the explicit stability bound for the lightest joint (wn*h {st.wnH.toFixed(3)}, limit 0.2; kd*h/I {st.dampingNumber.toFixed(2)}, limit 1).
          {!st.ok && ' The run may go unstable; the result will say so.'}
        </p>
      )}
    </Section>
  );
}

function MetricsTable({ a, b, linearA, linearB, unit, scale }: { a: StepMetrics | null; b: StepMetrics | null; linearA: StepMetrics | null; linearB: StepMetrics | null; unit: string; scale: number }) {
  const rows: [string, (m: StepMetrics) => string][] = [
    ['Rise time 10-90 %', (m) => fmtNum(m.riseTimeS, 3, ' s')],
    ['Overshoot', (m) => fmtNum(m.overshootPct, 1, ' %')],
    ['Settling time (2 %)', (m) => (m.settlingTimeS === null ? 'not settled' : fmtNum(m.settlingTimeS, 3, ' s'))],
    ['Steady-state error', (m) => fmtNum(m.steadyStateError * scale, 3, ` ${unit}`)],
  ];
  const cols: [string, StepMetrics | null][] = [['A measured', a], ['A linear', linearA], ['B measured', b], ['B linear', linearB]];
  const shown = cols.filter(([, m]) => m !== null);
  return (
    <table className="chain" data-testid="step-metrics">
      <thead>
        <tr>
          <th style={{ width: 'auto' }}>Metric</th>
          {shown.map(([h]) => (
            <th key={h} style={{ width: 'auto' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, f]) => (
          <tr key={label}>
            <td>{label}</td>
            {shown.map(([h, m]) => (
              <td key={h} className="mono">{f(m!)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function CriteriaList({ result, label }: { result: ChallengeResult; label: string }) {
  const fmt = (c: Criterion) => (c.measured === Number.POSITIVE_INFINITY ? 'never' : fmtNum(c.measured, ['rad', 'rad/s', 'm', 'm/s'].includes(c.unit) ? 4 : 2, ` ${c.unit}`));
  return (
    <div style={{ marginTop: 6 }}>
      <div className="row">
        <span className="muted" style={{ fontSize: 12 }}>{label}</span>
        <StatusBadge tone={result.passed ? 'ok' : 'error'}>{result.unstable ? 'Unstable' : result.passed ? 'Passed' : 'Failed'}</StatusBadge>
      </div>
      <ul style={{ margin: '4px 0 0', paddingLeft: 16, fontSize: 12 }}>
        {result.criteria.map((c) => (
          <li key={c.label} style={{ color: c.passed ? 'var(--text-1)' : 'var(--error)' }}>
            {c.label}: <span className="mono">{fmt(c)}</span> ({c.pass === 'below' ? 'must be below' : 'must be above'} {c.threshold} {c.unit}; {c.source})
          </li>
        ))}
      </ul>
    </div>
  );
}

function Challenges({ session }: { session: ControlSession }) {
  const s = useStore(session.store);
  return (
    <Section title="Challenges">
      {CHALLENGES.map((c) => {
        const r = s.challenges[c.id];
        return (
          <div key={c.id} style={{ marginBottom: 10 }} data-testid={`challenge-${c.id}`}>
            <div className="row">
              <strong style={{ fontSize: 13 }} className="grow">{c.title}</strong>
              <button className="btn" disabled={s.status !== 'ready' || s.running !== null} onClick={() => void session.runChallenge(c.id as ChallengeId)}>Run</button>
            </div>
            <p className="note muted" style={{ marginTop: 2 }}>{c.goal}</p>
            {r && !r.a.applicable && <p className="note warn">{r.a.reason}</p>}
            {r && r.a.applicable && challengeSetupChanged(s, r.setup) && (
              <p className="note warn" data-testid={`challenge-stale-${c.id}`}>
                Run with A ({describeSpec(r.setup.a)}){r.setup.b ? ` and B (${describeSpec(r.setup.b)})` : ' only'}. The gain settings have changed since; run again to compare.
              </p>
            )}
            {r && r.a.applicable && <CriteriaList result={r.a} label="Gain set A" />}
            {r?.b && s.compare && <CriteriaList result={r.b} label="Gain set B" />}
            {r?.a.onStand && <p className="note muted">Run with the base fixed on a test stand.</p>}
          </div>
        );
      })}
    </Section>
  );
}

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

export function ControlMode({ session }: { session: ControlSession }) {
  const s = useStore(session.store);
  useEffect(() => {
    void session.init();
    return () => session.cancel();
  }, [session]);
  const joints = useMemo(() => s.robot.limbs.flatMap((l) => l.joints.map((j) => ({ limb: l.name, joint: j }))), [s.robot]);
  const sel = joints[s.joint]?.joint;
  const units = jointUnits(sel?.type ?? 'revolute');
  const angular = units.position === 'rad';
  const unit = units.display.unit;
  const scale = units.display.scale;
  const a = s.stepA, b = s.compare ? s.stepB : null;
  const stale = setupChanged(s);
  const position = useMemo<StaticSeries[]>(() => {
    if (!a) return [];
    const out: StaticSeries[] = [{ label: a.command === 'ideal-step' ? 'commanded (ideal step)' : 'commanded (rate-limited)', color: COLOR.cmd, dashed: true, t: a.t, y: a.qCmd.map((v) => v * scale) }];
    out.push({ label: 'A measured (encoder)', color: COLOR.a, t: a.t, y: a.qMeas.map((v) => v * scale) });
    if (a.linearQ) out.push({ label: 'A linear model', color: COLOR.a, dashed: true, t: a.t, y: a.linearQ.map((v) => v * scale) });
    if (b) out.push({ label: 'B measured (encoder)', color: COLOR.b, t: b.t, y: b.qMeas.map((v) => v * scale) });
    if (b?.linearQ) out.push({ label: 'B linear model', color: COLOR.b, dashed: true, t: b.t, y: b.linearQ.map((v) => v * scale) });
    return out;
  }, [a, b, scale]);
  const torque = useMemo<StaticSeries[]>(() => {
    if (!a) return [];
    const out: StaticSeries[] = [
      { label: 'A commanded', color: COLOR.a, dashed: true, t: a.t, y: a.tauCmd },
      { label: 'A applied', color: COLOR.a, t: a.t, y: a.tauApplied },
    ];
    if (b) out.push({ label: 'B commanded', color: COLOR.b, dashed: true, t: b.t, y: b.tauCmd }, { label: 'B applied', color: COLOR.b, t: b.t, y: b.tauApplied });
    return out;
  }, [a, b]);
  const presetOptions: [string, string][] = [
    ...(s.source === 'custom' ? ([['custom', `Builder robot: ${s.robot.name}`]] as [string, string][]) : []),
    ...CONTROL_PRESETS.map((id) => [id, PRESETS.find((p) => p.id === id)!.label] as [string, string]),
  ];
  return (
    <>
      <aside className="panel left" aria-label="Experiment setup">
        <Section title="Experiment">
          <SelectField label="Robot" value={s.source} options={presetOptions} onChange={(v) => v !== 'custom' && session.setPreset(v as PresetId)} testId="control-robot" />
          {s.onStand && <p className="note muted">Floating base: experiments other than the push run with the base fixed on a test stand, legs clear of the ground.</p>}
          <SelectField label="Joint" value={String(s.joint)} options={joints.map((j, i) => [String(i), `${j.limb}: ${j.joint.name}`])} onChange={(v) => session.setJoint(Number(v))} testId="control-joint" />
          <div className="row">
            <NumberField label={`Step (${unit})`} value={s.step} digits={angular ? 1 : 3} step={angular ? 1 : 0.01} onCommit={(v) => session.setStep(v)} testId="control-step" />
            <NumberField label="Record (s)" value={s.durationS} digits={1} step={0.5} min={0.5} max={10} onCommit={(v) => session.setDuration(v)} />
          </div>
          <SelectField label="Command" value={s.command} options={COMMAND_OPTIONS} onChange={(v) => session.setCommand(v as StepCommand)} testId="control-command" />
          <Toggle label="Compare with gain set B" checked={s.compare} onChange={(v) => session.setCompare(v)} />
          <button className="btn primary" disabled={s.status !== 'ready' || s.running !== null} onClick={() => void session.runStepTest()} data-testid="run-step">Run step test</button>
          <p className="note muted">
            The joint holds its home position for 0.5 s, then steps. Every other joint holds its home position with the same controller.
            {s.command === 'rate-limited'
              ? ` The commanded target moves at the joint's velocity limit (${sel?.limit.velocity ?? '-'} ${units.velocity}), as in live control.`
              : ' The ideal step jumps instantly: a reference for the analytical second-order model that real actuators cannot follow.'}
          </p>
        </Section>
        <GainEditor session={session} which="a" />
        <GainEditor session={session} which="b" />
      </aside>
      <section className="viewport" aria-label="Step response plots" tabIndex={0} style={{ overflowY: 'auto', padding: 12 }}>
        <div className="viewport-label" style={{ position: 'static', marginBottom: 8, display: 'inline-block' }}>
          Recorded MuJoCo runs, plotted after each run completes. Positions come from joint encoders.
        </div>
        {s.status === 'loading' && <p className="note">Loading physics engine.</p>}
        {s.status === 'error' && <p className="note error" role="alert">{s.error}</p>}
        {s.running && (
          <p className="note" role="status" data-testid="control-running">
            {s.running.label}: {(s.running.progress * 100).toFixed(0)} %
          </p>
        )}
        {s.message && <p className="note error" role="alert">{s.message}</p>}
        <StaticPlot title={`Joint position, ${joints[s.joint]?.joint.name ?? ''}`} unit={unit} series={position} height={260} testId="lab-position" />
        <div style={{ height: 10 }} />
        <StaticPlot title={`Joint ${units.effortName.toLowerCase()}`} unit={units.effort} series={torque} limit={a?.effort} height={220} testId="lab-torque" />
        {a && (
          <div style={{ marginTop: 10 }}>
            {stale && s.stepSetup && (
              <p className="note warn" data-testid="step-stale">
                These results were run with: step {s.stepSetup.step} {unit}, {s.stepSetup.durationS} s, {s.stepSetup.command === 'ideal-step' ? 'ideal step' : 'rate-limited command'}; A {describeSpec(s.stepSetup.a)}{s.stepSetup.b ? `; B ${describeSpec(s.stepSetup.b)}` : ''}. The settings have changed since; run the step test again to compare.
              </p>
            )}
            <MetricsTable a={a.unstable ? null : a.metrics} b={b && !b.unstable ? b.metrics : null} linearA={a.linear} linearB={b?.linear ?? null} unit={unit} scale={scale} />
            <p className="note muted" data-testid="saturation-fraction">
              Time at the effort limit after the step: A {(a.saturationFraction * 100).toFixed(1)} %{b ? `, B ${(b.saturationFraction * 100).toFixed(1)} %` : ''}.
              {a.command === 'rate-limited' && ` The command was held back by the ${a.velocityLimit} ${units.velocity} velocity limit for ${a.rateLimitedS.toFixed(3)} s.`}
              {a.clamped && ' The requested step went past a joint limit and was clamped to the limit.'}
            </p>
            <RestNote a={a} b={b} setup={s.stepSetup} unit={unit} scale={scale} />
            <p className="note muted">
              Linear columns: {a.command === 'ideal-step' ? 'ideal second-order step response' : 'second-order model driven by the same rate-limited command'} with the joint inertia from MuJoCo at the home pose (I = {a.inertia.toFixed(4)} {units.inertia}; wn = {a.wn.toFixed(1)} rad/s, zeta = {a.zeta.toFixed(2)} for A). The linear model ignores gravity, coupling between joints and saturation, so differences show their effect.
            </p>
            {(a.unstable || b?.unstable) && <p className="note error">A run became numerically unstable and was stopped; its metrics are not shown.</p>}
          </div>
        )}
        <Explain>
          <p>Rise time runs from 10 % to 90 % of the step. Overshoot is the largest excursion past the target as a percentage of the step. After the settling time the response stays within 2 % of the step size around the target, so a joint that sags under gravity may never settle. Steady-state error is the target minus the mean of the last 10 % of the record.</p>
          <p>The linear columns predict I q'' + kd q' + kp q = kp q_cmd with the joint inertia I from MuJoCo, driven by the same commanded profile as the run (or the ideal jump in the reference mode). Gravity, the other joints and the effort limit are not in that model: under PD alone the joint sags until kp times the error equals the gravity torque, and at the effort limit it cannot accelerate faster than effort / I, so the rise slows.</p>
          <p>Gravity compensation adds the torque that holds the links against gravity, computed from the robot description with the mass model the simulator uses, so the PD part only corrects what is left.</p>
        </Explain>
      </section>
      <aside className="panel right" aria-label="Challenges">
        <Challenges session={session} />
      </aside>
    </>
  );
}
