import { useCallback, useEffect, useMemo, useState } from 'react';
import { PRESETS, type PresetId } from '../../core/model/catalog';
import { fmt, RAD2DEG, Readout, Section, StatusBadge } from '../components';
import { jointUnits, sharedGainUnits } from '../../core/model/units';
import { Explain } from '../learn';
import { TimePlot } from '../plot';
import { useStore } from '../store';
import { PhysicsViewport } from './PhysicsViewport';
import { PHYSICS_PRESETS, type PhysicsSession } from './session';

export { PhysicsViewport };

function GainControls({ session }: { session: PhysicsSession }) {
  const s = useStore(session.store);
  const [kp, setKp] = useState(String(s.gains.kp));
  const [kd, setKd] = useState(String(s.gains.kd));
  useEffect(() => {
    setKp(String(s.gains.kp));
    setKd(String(s.gains.kd));
  }, [s.gains.kp, s.gains.kd]);
  const apply = (p: string, d: string) => {
    if (!session.setGains({ kp: Number(p), kd: Number(d) })) {
      setKp(String(session.state.gains.kp));
      setKd(String(session.state.gains.kd));
    }
  };
  const st = s.stability;
  const units = sharedGainUnits(s.description ? s.description.limbs.flatMap((l) => l.joints.map((j) => j.type)) : ['revolute']);
  return (
    <div>
      <div className="field">
        <label htmlFor="kp">Proportional gain kp ({units.stiffness}): <span className="mono">{s.gains.kp}</span></label>
        <input id="kp" type="range" min={0} max={150} step={1} value={kp} onChange={(e) => { setKp(e.target.value); apply(e.target.value, kd); }} />
      </div>
      <div className="field">
        <label htmlFor="kd">Derivative gain kd ({units.damping}): <span className="mono">{s.gains.kd}</span></label>
        <input id="kd" type="range" min={0} max={8} step={0.05} value={kd} onChange={(e) => { setKd(e.target.value); apply(kp, e.target.value); }} />
      </div>
      {st && (
        <Readout
          rows={[
            ['Damping ratio (selected joint)', fmt(st.zetaSelected, 2)],
            ['wn*h (limit 0.2)', fmt(st.wnH, 3)],
            ['kd*h/I (limit 1)', fmt(st.dampingNumber, 3)],
          ]}
        />
      )}
      {s.gainError && <p className="note error" role="alert" data-testid="gain-error">{s.gainError}</p>}
      <p className="note muted">Gains apply to every joint immediately; the model is not rebuilt. Pure PD sags under gravity by roughly load torque / kp.</p>
    </div>
  );
}

function PushTool({ session }: { session: PhysicsSession }) {
  const [impulse, setImpulse] = useState(2);
  const dirs: [string, [number, number, number]][] = [['+X', [1, 0, 0]], ['-X', [-1, 0, 0]], ['+Y', [0, 1, 0]], ['-Y', [0, -1, 0]]];
  return (
    <div>
      <div className="field">
        <label htmlFor="impulse">Impulse (N*s): <span className="mono">{impulse.toFixed(1)}</span></label>
        <input id="impulse" type="range" min={0.5} max={8} step={0.5} value={impulse} onChange={(e) => setImpulse(Number(e.target.value))} />
      </div>
      <div className="row">
        {dirs.map(([label, d]) => (
          <button key={label} className="btn" onClick={() => session.push(d, impulse)} aria-label={`Push ${label}`}>{label}</button>
        ))}
      </div>
      <p className="note muted">Applied as a world-frame force at the base (or the last link of a fixed arm) for 50 ms.</p>
    </div>
  );
}

export function PhysicsMode({ session }: { session: PhysicsSession }) {
  const s = useStore(session.store);
  useEffect(() => {
    if (s.status === 'loading' && !session.runtime) void session.init();
  }, [session, s.status]);
  const getTelemetry = useCallback(() => session.telemetry, [session]);
  const desc = s.description;
  const joints = useMemo(() => (desc ? desc.limbs.flatMap((l) => l.joints.map((j) => ({ limb: l.name, joint: j }))) : []), [desc]);
  const sel = s.selectedJoint;
  const selJoint = joints[sel]?.joint;
  const ch = 4 * sel;
  const u = jointUnits(selJoint?.type ?? 'revolute');
  return (
    <>
      <aside className="panel left" aria-label="Simulation setup">
        <Section title="Robot">
          <div className="field">
            <label htmlFor="physics-preset">Preset</label>
            <select id="physics-preset" value={s.presetId} onChange={(e) => session.setPreset(e.target.value as PresetId)}>
              {s.presetId === 'custom' && <option value="custom">Builder robot: {s.description?.name}</option>}
              {PHYSICS_PRESETS.map((id) => (
                <option key={id} value={id}>{PRESETS.find((p) => p.id === id)!.label}</option>
              ))}
            </select>
          </div>
        </Section>
        <Section title="Joint PD controller">
          <GainControls session={session} />
          <Explain>
            <p>Each joint gets the torque tau = kp (q_cmd - q) - kd dq from its encoder readings, 500 times per second, clamped to the joint's effort limit before it reaches the simulated motor. The commanded position q_cmd moves toward the slider target no faster than the joint's velocity limit, so a large slider jump becomes a ramp the joint can follow.</p>
            <p>For a joint with inertia I the damping ratio is zeta = kd / (2 sqrt(kp I)): below 1 the joint overshoots, near 1 it settles without overshoot. Under gravity a PD joint must sag until kp times the error equals the gravity torque, so stiffer gains reduce the sag but never remove it; the Control lab adds gravity compensation.</p>
            <p>The controller runs at a fixed 2 ms step, so gains that are stable for the continuous system can still make the discrete loop unstable. The panel refuses gains past that bound for the lightest joint.</p>
          </Explain>
        </Section>
        <Section title="Push tool">
          <PushTool session={session} />
        </Section>
      </aside>
      <PhysicsViewport session={session} />
      <aside className="panel right" aria-label="Simulation inspector">
        <Section title="Simulation">
          {s.status === 'loading' && <StatusBadge tone="idle">Loading physics engine</StatusBadge>}
          {s.status === 'error' && <p className="note error" role="alert">{s.error}</p>}
          {s.status === 'ready' && (
            <>
              <div className="row" style={{ marginBottom: 6 }}>
                <button className="btn primary" aria-label={s.running ? undefined : 'Run simulation'} onClick={() => session.setRunning(!s.running)} disabled={s.unstable} data-testid="sim-toggle">{s.running ? 'Pause' : 'Run'}</button>
                <button className="btn" onClick={() => session.reset()}>Reset</button>
                <StatusBadge tone={s.unstable ? 'error' : s.running ? 'ok' : 'idle'} testId="sim-status">{s.unstable ? 'Unstable' : s.running ? 'Running' : 'Paused'}</StatusBadge>
              </div>
              <Readout
                rows={[
                  ['Simulated time', <span data-testid="sim-time">{s.simTime.toFixed(2)} s</span>],
                  ['Real-time factor', fmt(s.realTimeFactor, 2)],
                  ['Physics + control per step', `${s.stepMs.toFixed(3)} ms`],
                  ...(s.base ? ([['Base height (ground truth)', <span data-testid="base-height">{s.base.height.toFixed(4)} m</span>], ['Base tilt (ground truth)', `${(s.base.tilt * RAD2DEG).toFixed(2)} deg`]] as [string, React.ReactNode][]) : []),
                ]}
              />
              {Object.entries(s.warnings).some(([, v]) => v > 0) && (
                <p className="note warn">MuJoCo warnings: {Object.entries(s.warnings).filter(([, v]) => v > 0).map(([k, v]) => `${k} ${v}`).join(', ')}</p>
              )}
            </>
          )}
        </Section>
        {desc && selJoint && (
          <Section title="Joint">
            <div className="field">
              <label htmlFor="physics-joint">Selected joint</label>
              <select id="physics-joint" value={sel} onChange={(e) => session.selectJoint(Number(e.target.value))}>
                {joints.map((j, i) => (
                  <option key={j.joint.id} value={i}>{j.limb}: {j.joint.name}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="physics-target">Commanded position: <span className="mono">{((s.qCmd[sel] ?? 0) * u.display.scale).toFixed(u.display.digits)} {u.display.unit}</span></label>
              <input
                id="physics-target"
                type="range"
                min={selJoint.type === 'continuous' ? -Math.PI : selJoint.limit.lower}
                max={selJoint.type === 'continuous' ? Math.PI : selJoint.limit.upper}
                step={0.005}
                value={s.qCmd[sel] ?? 0}
                onChange={(e) => session.setTarget(sel, Number(e.target.value))}
              />
            </div>
            <TimePlot
              getTelemetry={getTelemetry}
              spec={{ title: 'Position', unit: u.display.unit, scale: u.display.scale, series: [{ channel: ch, label: 'commanded', color: '#4ea1ff', dashed: true }, { channel: ch + 1, label: 'measured (encoder)', color: '#e8edf3' }] }}
              testId="plot-position"
            />
            <div style={{ height: 8 }} />
            <TimePlot
              getTelemetry={getTelemetry}
              spec={{ title: u.effortName, unit: u.effort, limit: selJoint.limit.effort, series: [{ channel: ch + 2, label: 'commanded', color: '#e3b341', dashed: true }, { channel: ch + 3, label: 'applied', color: '#3fb68b' }] }}
              testId="plot-torque"
            />
            <p className="note muted">Dotted red lines: effort limit. Saturated over the last second: <span className="mono" data-testid="saturation">{(s.saturationFraction * 100).toFixed(0)} %</span>.</p>
            <p className={`note ${s.overspeedFraction > 0 ? 'warn' : 'muted'}`} data-testid="overspeed">
              Velocity limit {selJoint.limit.velocity} {u.velocity}. Over it during the last second: <span className="mono">{(s.overspeedFraction * 100).toFixed(0)} %</span>; fastest joint at <span className="mono">{(s.maxSpeedRatio * 100).toFixed(0)} %</span> of its limit (encoders).
            </p>
          </Section>
        )}
        <Section title="Events">
          <ol className="events" style={{ margin: 0, paddingLeft: 18, fontSize: 12 }} data-testid="events">
            {s.events.slice(0, 8).map((e, i) => (
              <li key={`${e.t}-${i}`}><span className="mono muted">{e.t.toFixed(2)} s</span> {e.text}</li>
            ))}
          </ol>
        </Section>
      </aside>
    </>
  );
}
