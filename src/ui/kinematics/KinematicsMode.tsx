import { useEffect, useMemo, useRef } from 'react';
import { baseTransform } from '../../core/kinematics/fk';
import type { Vec3 } from '../../core/math/vec3';
import { PRESETS } from '../../core/model/catalog';
import type { RobotDescription } from '../../core/model/types';
import { KinematicsOverlay, type TargetStatus } from '../../render/kinematicsOverlay';
import { RobotView } from '../../render/robotView';
import { Viewport } from '../../render/viewport';
import { DEG2RAD, fmt, NumberField, RAD2DEG, Readout, Section, StatusBadge, Toggle } from '../components';
import { Explain } from '../learn';
import { useStore } from '../store';
import { ChainEditor, isEditableChain } from './ChainEditor';
import { KinematicsSession, targetableLimbs, type KinematicsState } from './session';

function tipOf(s: KinematicsState): Vec3 {
  if (s.limbIndex < 0) return [0, 0, 0];
  const T = s.fk.tips[s.limbIndex]!;
  return [T[12]!, T[13]!, T[14]!];
}

function statusOf(s: KinematicsState): TargetStatus {
  return s.ik ? s.ik.status : 'idle';
}

/** Owns the Three.js scene for kinematics mode and mirrors session state into it. */
function KinematicsViewport({ session }: { session: KinematicsSession }) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = host.current!;
    const viewport = new Viewport(el);
    let view: RobotView | null = null;
    let viewDesc: RobotDescription | null = null;
    const overlay = new KinematicsOverlay(viewport, (p) => session.setTarget(p));
    let lastWorkspace: Float32Array | null = null;
    const apply = () => {
      const s = session.state;
      if (s.description !== viewDesc) {
        view?.dispose();
        view = new RobotView(s.description);
        viewDesc = s.description;
        viewport.scene.add(view.root);
      }
      view!.setOverlays({ jointAxes: s.overlays.axes, frames: s.overlays.frames });
      view!.applyTransforms(baseTransform(s.model), s.fk.links);
      const tip = tipOf(s);
      overlay.setTargetVisible(s.limbIndex >= 0);
      overlay.setTarget(s.target, statusOf(s));
      const gap = Math.hypot(tip[0] - s.target[0], tip[1] - s.target[1], tip[2] - s.target[2]);
      overlay.setResidual(tip, s.target, s.limbIndex >= 0 && gap > 1e-3);
      overlay.setEllipsoid(tip, s.manip, s.overlays.ellipsoid);
      if (s.workspace !== lastWorkspace) {
        overlay.setWorkspace(s.overlays.workspace ? s.workspace : null);
        lastWorkspace = s.workspace;
      }
    };
    apply();
    const unsubscribe = session.store.subscribe(apply);
    viewport.start(() => {});
    return () => {
      unsubscribe();
      overlay.dispose();
      view?.dispose();
      viewport.dispose();
    };
  }, [session]);
  return (
    <div className="viewport" ref={host} data-testid="viewport">
      <div className="viewport-label">Kinematics mode: poses computed by forward kinematics. No physics.</div>
    </div>
  );
}

function IkStatus({ s }: { s: KinematicsState }) {
  if (s.limbIndex < 0) return <p className="note muted">This robot has no limb that can follow a position target.</p>;
  const ik = s.ik;
  const tone = !ik ? 'idle' : ik.status === 'converged' ? 'ok' : ik.status === 'invalid-input' ? 'error' : 'warn';
  const label = !ik ? 'Not solved' : ik.status === 'converged' ? 'Converged' : ik.status === 'unreachable' ? 'Unreachable' : ik.status === 'stalled' ? 'Stalled' : ik.status === 'max-iterations' ? 'Not converged' : 'Invalid input';
  const limitNames = s.limits.map((l) => `${s.description.limbs[s.limbIndex]!.joints[l.jointIndex - s.model.limbs[s.limbIndex]!.start]!.name} (${l.side})`);
  return (
    <div>
      <StatusBadge tone={tone} testId="ik-status">{label}</StatusBadge>
      {ik && (
        <Readout
          rows={[
            ['Residual', <span data-testid="ik-residual">{(ik.residual * 1000).toFixed(3)} mm</span>],
            ['Iterations', `${ik.iterations}${ik.restartsUsed ? ` (${ik.restartsUsed} restarts)` : ''}`],
          ]}
        />
      )}
      {ik && ik.status !== 'converged' && <p className="note warn" data-testid="ik-message">{ik.message}</p>}
      {ik && ik.status !== 'converged' && !ik.beyondReach && (
        <p className="note warn" data-testid="ik-diagnosis">
          {limitNames.length > 0
            ? `Joints at a limit: ${limitNames.join(', ')}. The target may need motion beyond these limits; that is likely but not proven.`
            : s.manip && s.manip.inverseCondition < 0.02
              ? 'At the closest point found the limb is fully extended or folded (singular). The target may be outside its reach; that is not proven.'
              : 'The solver stalled away from the target after retrying from other starting poses. Moving the target gradually may help.'}
        </p>
      )}
    </div>
  );
}

function JointSliders({ session, s }: { session: KinematicsSession; s: KinematicsState }) {
  const limbs = s.limbIndex >= 0 ? [s.limbIndex] : s.description.limbs.map((_, i) => i);
  return (
    <>
      {limbs.map((li) => {
        const kl = s.model.limbs[li]!;
        const limb = s.description.limbs[li]!;
        return limb.joints.map((j, k) => {
          const gi = kl.start + k;
          const kj = s.model.joints[gi]!;
          const v = s.q[gi]!;
          const lo = Number.isFinite(kj.lower) ? kj.lower : -Math.PI;
          const hi = Number.isFinite(kj.upper) ? kj.upper : Math.PI;
          const atLimit = v <= lo + 1e-6 || v >= hi - 1e-6;
          const prismatic = kj.type === 'prismatic';
          const shown = prismatic ? `${v.toFixed(3)} m` : `${(v * RAD2DEG).toFixed(1)} deg`;
          const label = limbs.length > 1 ? `${limb.name}: ${j.name}` : j.name;
          return (
            <div className="joint" key={j.id}>
              <div className="joint-head">
                <label className="name" htmlFor={`joint-${j.id}`}>{label}</label>
                <span className={`value${atLimit ? ' at-limit' : ''}`} data-testid={`joint-value-${j.id}`}>{shown}</span>
              </div>
              <input
                id={`joint-${j.id}`}
                type="range"
                min={lo}
                max={hi}
                step={0.001}
                value={v}
                onChange={(e) => session.setJoint(gi, Number(e.target.value))}
              />
              <div className="joint-range">
                <span>{prismatic ? lo.toFixed(2) : (lo * RAD2DEG).toFixed(0)}</span>
                <span>{prismatic ? hi.toFixed(2) : (hi * RAD2DEG).toFixed(0)}</span>
              </div>
            </div>
          );
        });
      })}
    </>
  );
}

export function KinematicsMode({ session }: { session: KinematicsSession }) {
  const s = useStore(session.store);
  const tip = tipOf(s);
  const limbs = useMemo(() => targetableLimbs(s.description), [s.description]);
  const nearSingular = s.manip !== null && s.manip.inverseCondition < 0.02;
  return (
    <>
      <aside className="panel left" aria-label="Robot">
        <Section title="Robot">
          <div className="field">
            <label htmlFor="preset">Preset</label>
            <select
              id="preset"
              value={s.source}
              onChange={(e) => {
                const p = PRESETS.find((x) => x.id === e.target.value);
                if (p) session.setRobot(p.make(), p.id);
              }}
            >
              {s.source === 'custom' && <option value="custom">Custom: {s.description.name}</option>}
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>{p.label}</option>
              ))}
            </select>
          </div>
          {limbs.length > 1 && (
            <div className="field">
              <label htmlFor="limb">Target limb</label>
              <select id="limb" value={s.limbIndex} onChange={(e) => session.selectLimb(Number(e.target.value))}>
                {limbs.map((i) => (
                  <option key={i} value={i}>{s.description.limbs[i]!.name}</option>
                ))}
              </select>
            </div>
          )}
          <p className="note muted">
            {s.description.base.kind === 'floating' ? 'Floating base held fixed for kinematics.' : 'Fixed base.'} {s.model.dof} joints.
          </p>
        </Section>
        {isEditableChain(s.description) && (
          <Section title="Serial chain">
            <ChainEditor description={s.description} onApply={(d) => session.setRobot(d)} />
          </Section>
        )}
        <Section title="Overlays">
          <Toggle label="Coordinate frames" checked={s.overlays.frames} onChange={(v) => session.setOverlay('frames', v)} />
          <Toggle label="Joint axes" checked={s.overlays.axes} onChange={(v) => session.setOverlay('axes', v)} />
          <Toggle label="Manipulability ellipsoid" checked={s.overlays.ellipsoid} disabled={s.limbIndex < 0} onChange={(v) => session.setOverlay('ellipsoid', v)} />
          <Toggle label="Reachable workspace samples" checked={s.overlays.workspace} disabled={s.limbIndex < 0} onChange={(v) => session.setOverlay('workspace', v)} />
          <p className="note muted">Frames: X red, Y green, Z blue. Ellipsoid scaled by 0.2. Workspace: 4000 random joint samples within limits.</p>
        </Section>
      </aside>
      <KinematicsViewport session={session} />
      <aside className="panel right" aria-label="Inspector">
        <Section title="Inverse kinematics">
          <div className="row">
            <NumberField label="Target X (m)" value={s.target[0]} disabled={s.limbIndex < 0} onCommit={(v) => session.setTarget([v, s.target[1], s.target[2]])} testId="target-x" />
            <NumberField label="Y (m)" value={s.target[1]} disabled={s.limbIndex < 0} onCommit={(v) => session.setTarget([s.target[0], v, s.target[2]])} testId="target-y" />
            <NumberField label="Z (m)" value={s.target[2]} disabled={s.limbIndex < 0} onCommit={(v) => session.setTarget([s.target[0], s.target[1], v])} testId="target-z" />
          </div>
          <Toggle label="Solve while the target moves" checked={s.followTarget} disabled={s.limbIndex < 0} onChange={(v) => session.setFollowTarget(v)} />
          <div className="row" style={{ margin: '6px 0' }}>
            <button className="btn primary" onClick={() => session.solve()} disabled={s.limbIndex < 0}>Solve</button>
            <button className="btn" onClick={() => session.reset()}>Reset pose</button>
          </div>
          <IkStatus s={s} />
          {s.inputError && <p className="note error" role="alert">{s.inputError}</p>}
          <Explain>
            <p>Inverse kinematics searches for joint angles that put the tip on the target. Each step solves the damped least-squares update dq = J^T (J J^T + lambda^2 I)^-1 e, where e is the tip error and J the position Jacobian. The damping lambda keeps steps bounded near singular poses: it shrinks after a step that reduces the error and grows after one that does not.</p>
            <p>Joints that reach a limit are held there and left out of the next step. Unreachable is shown only when the target lies beyond an upper bound on the limb's reach (the sum of its link lengths), which proves no joint angles can reach it. Stalled means the solver stopped improving, for example against a joint limit or in a local minimum; the target might still be reachable. The residual is the distance left at the closest pose found.</p>
          </Explain>
        </Section>
        <Section title="Joints">
          {s.description.limbs.length > 1 && s.limbIndex >= 0 && (
            <p className="note muted" style={{ marginTop: 0 }}>Showing the joints of the {s.description.limbs[s.limbIndex]!.name.toLowerCase()}.</p>
          )}
          <JointSliders session={session} s={s} />
          <Explain>
            <p>Forward kinematics: each joint adds a fixed transform from its parent link and a motion about its axis (a rotation, or a slide for prismatic joints). Multiplying them from the base outward gives every link pose, T_W_tip = T_W_B T_B_J1 R1(q1) T_L1_J2 R2(q2) ... The tip readout is the translation part of that product.</p>
          </Explain>
        </Section>
        {s.limbIndex >= 0 && (
          <Section title="Tip position (world frame)">
            <Readout
              rows={[
                ['X', <span data-testid="tip-x">{fmt(tip[0], 4)} m</span>],
                ['Y', <span data-testid="tip-y">{fmt(tip[1], 4)} m</span>],
                ['Z', <span data-testid="tip-z">{fmt(tip[2], 4)} m</span>],
              ]}
            />
          </Section>
        )}
        {s.manip && (
          <Section title="Jacobian">
            <Readout
              rows={[
                ['Singular values', s.manip.singularValues.map((v) => v.toFixed(3)).join(' / ')],
                ['Manipulability', s.manip.measure.toExponential(2)],
                ['Inverse condition', s.manip.inverseCondition.toFixed(3)],
              ]}
            />
            {nearSingular && (
              <p className="note warn" data-testid="singular-note">
                Near a singular configuration: the tip cannot move along at least one direction, and small target changes may need large joint motion.
              </p>
            )}
            <Explain>
              <p>The position Jacobian J maps joint speeds to tip velocity, v = J dq/dt. Its singular values are the tip speeds per unit joint speed along the best and worst directions. Manipulability is their product (proportional to the volume of the ellipsoid overlay) and the inverse condition is the smallest divided by the largest; values near zero mean the arm has lost a direction of motion.</p>
            </Explain>
          </Section>
        )}
      </aside>
    </>
  );
}

export { DEG2RAD };
