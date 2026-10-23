import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { buildKinematicModel, forwardKinematics } from '../../core/kinematics/fk';
import type { Vec3 } from '../../core/math/vec3';
import type { LeggedSpec } from '../../core/model/builder';
import { homeConfiguration, PRESETS, type PresetId } from '../../core/model/catalog';
import type { JointDescription, RobotDescription } from '../../core/model/types';
import { MAX_JOINTS_PER_LIMB } from '../../core/model/validate';
import { driveGeometry } from '../../nav/drive';
import { MAX_BUNDLE_CHARS } from '../../persistence/bundle';
import { RobotView } from '../../render/robotView';
import { Viewport } from '../../render/viewport';
import { DEG2RAD, downloadText, FileButton, NumberField, RAD2DEG, Readout, Section, Toggle } from '../components';
import { Explain } from '../learn';
import { useStore } from '../store';
import type { AppStorage } from '../storage';
import type { BuilderSession } from './session';

function BuilderViewport({ robot }: { robot: RobotDescription }) {
  const host = useRef<HTMLDivElement>(null);
  const viewport = useRef<Viewport | null>(null);
  useEffect(() => {
    const vp = new Viewport(host.current!);
    viewport.current = vp;
    vp.start(() => {});
    return () => {
      viewport.current = null;
      vp.dispose();
    };
  }, []);
  useEffect(() => {
    const vp = viewport.current;
    if (!vp) return;
    const view = new RobotView(robot);
    vp.scene.add(view.root);
    const fk = forwardKinematics(buildKinematicModel(robot), homeConfiguration(robot));
    view.applyTransforms(fk.base, fk.links);
    return () => view.dispose();
  }, [robot]);
  return (
    <div className="viewport" ref={host} data-testid="viewport">
      <div className="viewport-label">Builder preview: home configuration computed by forward kinematics. No physics.</div>
    </div>
  );
}

function Select<T extends string | number>({ label, value, options, onChange, testId }: { label: string; value: T; options: [T, string][]; onChange: (v: T) => void; testId?: string }) {
  const id = useId();
  return (
    <div className="field grow">
      <label htmlFor={id}>{label}</label>
      <select id={id} data-testid={testId} value={String(value)} onChange={(e) => onChange(options.find(([v]) => String(v) === e.target.value)![0])}>
        {options.map(([v, l]) => (
          <option key={String(v)} value={String(v)}>{l}</option>
        ))}
      </select>
    </div>
  );
}

function TextField({ label, value, onCommit, testId }: { label: string; value: string; onCommit: (v: string) => void; testId?: string }) {
  const id = useId();
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const commit = () => {
    if (text !== value) onCommit(text);
  };
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} type="text" value={text} data-testid={testId} onChange={(e) => setText(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && commit()} />
    </div>
  );
}

const AXES: [string, Vec3][] = [
  ['+X', [1, 0, 0]], ['-X', [-1, 0, 0]], ['+Y', [0, 1, 0]], ['-Y', [0, -1, 0]], ['+Z', [0, 0, 1]], ['-Z', [0, 0, -1]],
];

function axisLabel(a: Vec3): string {
  const n = Math.hypot(...a);
  const hit = AXES.find(([, v]) => v.every((c, i) => Math.abs(c - a[i]! / n) < 1e-9));
  return hit ? hit[0] : 'custom';
}

function Generators({ session }: { session: BuilderSession }) {
  const s = useStore(session.store);
  const [preset, setPreset] = useState<PresetId>('quadruped');
  const [spec, setSpec] = useState<LeggedSpec>(s.legged);
  const [arm, setArm] = useState(s.arm);
  const set = (patch: Partial<LeggedSpec>) => setSpec((p) => ({ ...p, ...patch }));
  return (
    <>
      <Section title="Start from a preset">
        <div className="row">
          <Select label="Preset" value={preset} options={PRESETS.map((p) => [p.id, p.label])} onChange={setPreset} testId="builder-preset" />
        </div>
        <button className="btn" onClick={() => session.startFromPreset(preset)}>Load preset</button>
      </Section>
      <Section title="Legged generator">
        <div className="row">
          <Select label="Legs" value={spec.pairs} options={[[1, '2'], [2, '4'], [3, '6'], [4, '8']]} onChange={(v) => set({ pairs: v })} testId="gen-legs" />
          <Select label="Joints per leg" value={spec.dofPerLeg} options={[[1, '1'], [2, '2'], [3, '3'], [4, '4']]} onChange={(v) => set({ dofPerLeg: v })} testId="gen-dof" />
        </div>
        <div className="row">
          <NumberField label="Body length (m)" value={spec.bodyLength} onCommit={(v) => set({ bodyLength: v })} />
          <NumberField label="Body width (m)" value={spec.bodyWidth} onCommit={(v) => set({ bodyWidth: v })} />
        </div>
        <div className="row">
          <NumberField label="Body height (m)" value={spec.bodyHeight} onCommit={(v) => set({ bodyHeight: v })} />
          <NumberField label="Body mass (kg)" value={spec.bodyMass} digits={2} onCommit={(v) => set({ bodyMass: v })} />
        </div>
        <div className="row">
          <NumberField label="Upper leg (m)" value={spec.thigh} onCommit={(v) => set({ thigh: v })} testId="gen-thigh" />
          <NumberField label="Lower leg (m)" value={spec.shank} onCommit={(v) => set({ shank: v })} />
        </div>
        <NumberField label="Joint effort limit (N*m)" value={spec.effort} digits={1} onCommit={(v) => set({ effort: v })} />
        <button className="btn" onClick={() => session.generateLegged(spec)}>Generate legged robot</button>
        <p className="note muted">Legs come in mirrored left and right pairs. The body is placed so the feet start on the ground.</p>
      </Section>
      <Section title="Arm generator">
        <div className="row">
          <Select label="Joints" value={arm.dof} options={[1, 2, 3, 4, 5, 6, 7].map((n) => [n, String(n)] as [number, string])} onChange={(v) => setArm((a) => ({ ...a, dof: v }))} />
          <NumberField label="Total link length (m)" value={arm.reach} onCommit={(v) => setArm((a) => ({ ...a, reach: v }))} />
        </div>
        <button className="btn" onClick={() => session.generateArm(arm.dof, arm.reach)}>Generate arm</button>
      </Section>
    </>
  );
}

function BodyEditor({ session }: { session: BuilderSession }) {
  const s = useStore(session.store);
  const r = s.robot;
  const size = r.base.size;
  const setSize = (i: number, v: number) => session.setBase({ size: size.map((c, k) => (k === i ? v : c)) as Vec3 });
  return (
    <Section title="Body" key={s.attempt}>
      <TextField label="Robot name" value={r.name} onCommit={(v) => session.rename(v)} testId="robot-name" />
      <Select label="Base" value={r.base.kind} options={[['floating', 'Floating (free body)'], ['fixed', 'Fixed to the ground']]} onChange={(v) => session.setBase({ kind: v })} testId="base-kind" />
      <div className="row">
        <NumberField label="Length (m)" value={size[0]} onCommit={(v) => setSize(0, v)} />
        <NumberField label="Width (m)" value={size[1]} onCommit={(v) => setSize(1, v)} />
        <NumberField label="Height (m)" value={size[2]} onCommit={(v) => setSize(2, v)} />
      </div>
      <NumberField label="Base mass (kg)" value={r.base.mass} digits={2} onCommit={(v) => session.setBase({ mass: v })} testId="base-mass" />
    </Section>
  );
}

function JointEditor({ session, joint }: { session: BuilderSession; joint: JointDescription }) {
  const xyz = joint.origin.xyz;
  const angular = joint.type !== 'prismatic';
  const unit = angular ? 'deg' : 'm';
  const toUi = (v: number) => (angular ? v * RAD2DEG : v);
  const fromUi = (v: number) => (angular ? v * DEG2RAD : v);
  const axis = axisLabel(joint.axis);
  return (
    <>
      <div className="row">
        <Select
          label="Axis (joint frame)"
          value={axis}
          options={[...AXES.map(([l]) => [l, l] as [string, string]), ...(axis === 'custom' ? [['custom', `Custom (${joint.axis.map((c) => c.toFixed(2)).join(', ')})`] as [string, string]] : [])]}
          onChange={(v) => {
            const hit = AXES.find(([l]) => l === v);
            if (hit) session.updateJoint({ axis: hit[1] });
          }}
          testId="joint-axis"
        />
        <Select label="Type" value={joint.type} options={[['revolute', 'Revolute'], ['prismatic', 'Prismatic'], ['continuous', 'Continuous']]} onChange={(v) => session.updateJoint({ type: v })} />
      </div>
      <div className="field-label note muted">Offset from the parent joint (m)</div>
      <div className="row">
        <NumberField label="x" value={xyz[0]} onCommit={(v) => session.updateJoint({ xyz: [v, xyz[1], xyz[2]] })} />
        <NumberField label="y" value={xyz[1]} onCommit={(v) => session.updateJoint({ xyz: [xyz[0], v, xyz[2]] })} />
        <NumberField label="z" value={xyz[2]} onCommit={(v) => session.updateJoint({ xyz: [xyz[0], xyz[1], v] })} testId="joint-z" />
      </div>
      {joint.type !== 'continuous' && (
        <div className="row">
          <NumberField label={`Lower limit (${unit})`} value={toUi(joint.limit.lower)} digits={1} step={1} onCommit={(v) => session.updateJoint({ lower: fromUi(v) })} testId="joint-lower" />
          <NumberField label={`Upper limit (${unit})`} value={toUi(joint.limit.upper)} digits={1} step={1} onCommit={(v) => session.updateJoint({ upper: fromUi(v) })} testId="joint-upper" />
        </div>
      )}
      <div className="row">
        <NumberField label={angular ? 'Effort (N*m)' : 'Effort (N)'} value={joint.limit.effort} digits={1} onCommit={(v) => session.updateJoint({ effort: v })} />
        <NumberField label="Link radius (m)" value={joint.link.radius} onCommit={(v) => session.updateJoint({ radius: v })} />
        <NumberField label="Link mass (kg)" value={joint.link.mass} onCommit={(v) => session.updateJoint({ mass: v })} testId="link-mass" />
      </div>
    </>
  );
}

function SavedRobots({ session, storage }: { session: BuilderSession; storage: AppStorage }) {
  const st = useStore(storage.store);
  const current = useStore(session.store).hash;
  if (st.status !== 'ready') return null;
  return (
    <Section title={`Saved robot versions (${st.robots.length})`}>
      {st.robots.length === 0 && <p className="note muted">Accepted edits are saved here automatically.</p>}
      <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none', fontSize: 12 }} data-testid="saved-robots">
        {st.robots.slice(0, 12).map((r) => (
          <li key={`${r.id}:${r.morphologyHash}`} className="row" style={{ marginBottom: 4 }}>
            <span className="grow">
              {r.name} <span className="mono muted">{r.morphologyHash.slice(0, 8)}</span>
              {r.morphologyHash === current && <span className="muted"> (current)</span>}
            </span>
            <button className="btn" onClick={() => session.loadRobot(r.robot)} disabled={r.morphologyHash === current}>Load</button>
            <button className="btn" onClick={() => void storage.deleteRobot(r.id, r.morphologyHash)}>Delete</button>
          </li>
        ))}
      </ul>
    </Section>
  );
}

export interface RobotHandoffs {
  onOpenKinematics: (d: RobotDescription) => void;
  onSimulate: (d: RobotDescription) => void;
  onOpenControl: (d: RobotDescription) => void;
  onOpenSkills: (d: RobotDescription) => void;
  onOpenNavigate: (d: RobotDescription) => void;
}

function Inspector({ session, storage, onOpenKinematics, onSimulate, onOpenControl, onOpenSkills, onOpenNavigate }: { session: BuilderSession; storage?: AppStorage } & RobotHandoffs) {
  const s = useStore(session.store);
  const limb = session.limb;
  const partner = session.partner;
  const joint = limb?.joints[s.selectedJoint];
  const joints = s.robot.limbs.reduce((n, l) => n + l.joints.length, 0);
  const drive = useMemo(() => driveGeometry(s.robot), [s.robot]);
  const driveNoteId = useId();
  return (
    <aside className="panel right" aria-label="Limb inspector">
      {s.issues.length > 0 && (
        <div className="note error" role="alert" data-testid="builder-issues">
          {s.issues.map((m, i) => (
            <div key={i}>{m}</div>
          ))}
          <div className="muted">The change was not applied.</div>
        </div>
      )}
      {s.notice && <p className="note" data-testid="builder-notice">{s.notice}</p>}
      <Section title="Limbs">
        <Select label="Limb" value={s.selectedLimb} options={s.robot.limbs.map((l) => [l.id, `${l.name} (${l.joints.length})`])} onChange={(v) => session.selectLimb(v)} testId="limb-select" />
        {partner ? (
          <Toggle label={`Mirror edits to ${partner.name}`} checked={s.mirror} onChange={(v) => session.setMirror(v)} />
        ) : (
          <p className="note muted">No mirror partner: no limb is mounted at the mirrored position.</p>
        )}
        <button className="btn" onClick={() => session.removeSelectedLimb()} disabled={s.robot.limbs.length <= 1}>
          {s.mirror && partner ? 'Remove limb and its mirror' : 'Remove limb'}
        </button>
      </Section>
      {limb && joint && (
        <Section title="Joints">
          <div className="row">
            {limb.role !== 'wheel' && (
              <Select
                label="Joints in limb"
                value={limb.joints.length}
                options={Array.from({ length: MAX_JOINTS_PER_LIMB }, (_, i) => [i + 1, String(i + 1)] as [number, string])}
                onChange={(v) => session.setJointCount(v)}
                testId="joint-count"
              />
            )}
            <Select label="Joint" value={s.selectedJoint} options={limb.joints.map((j, i) => [i, `${i + 1}. ${j.name}`])} onChange={(v) => session.selectJoint(v)} testId="joint-select" />
          </div>
          <JointEditor key={`${limb.id}:${s.selectedJoint}:${s.attempt}`} session={session} joint={joint} />
          <div className="field-label note muted">Tip in the last link frame (m) and contact radius</div>
          <div className="row" key={`tip:${limb.id}:${s.attempt}`}>
            {[0, 1, 2].map((i) => (
              <NumberField key={i} label={'xyz'[i]!} value={limb.tip.xyz[i]!} onCommit={(v) => session.updateTip({ xyz: limb.tip.xyz.map((c, k) => (k === i ? v : c)) as Vec3 })} />
            ))}
            <NumberField label="r" value={limb.tip.radius} onCommit={(v) => session.updateTip({ radius: v })} />
          </div>
        </Section>
      )}
      <Section title="Morphology version">
        <Readout
          rows={[
            ['Version', <span data-testid="morph-version">{s.version}</span>],
            ['Hash (SHA-256)', <span data-testid="morph-hash" title={s.hash ?? ''}>{s.hash ? s.hash.slice(0, 12) : 'computing'}</span>],
            ['Limbs / joints', `${s.robot.limbs.length} / ${joints}`],
          ]}
        />
        <ol className="events" style={{ margin: 'var(--space-2) 0 0', paddingLeft: 18, fontSize: 12 }} data-testid="morph-history">
          {s.history.slice(0, 8).map((h) => (
            <li key={h.version}>
              <span className="mono muted">v{h.version} {h.hash ? h.hash.slice(0, 8) : '........'}</span> {h.change}
            </li>
          ))}
        </ol>
        <p className="note muted">Learned skills are bound to a hash; any physical change creates a new one. Names, the start pose and the stance do not count.</p>
        <Explain>
          <p>The morphology hash is a SHA-256 digest of the physical description: structure, joint origins and axes, limits, masses, radii, damping and the body. A controller tuned or learned on one body can fail on another, so skills record the hash they were evaluated on and refuse to run on any other.</p>
          <p>Mirrored edits reflect the limb across the body's left-right plane. Rotation axes reflect as pseudovectors, so the same joint angle on both sides gives mirror-image poses.</p>
        </Explain>
      </Section>
      <Section title="Export and import">
        <div className="row">
          <button className="btn" onClick={() => downloadText(session.exportText(), `${s.robot.id}-${(s.hash ?? 'robot').slice(0, 8)}.morphlab.json`)}>Export JSON</button>
          <FileButton label="Import JSON" maxBytes={MAX_BUNDLE_CHARS} onText={(t) => session.importText(t)} onRefuse={(m) => session.refuseImport(m)} testId="import-file" />
        </div>
      </Section>
      <Section title="Use this robot">
        <div className="row">
          <button className="btn" onClick={() => onOpenKinematics(s.robot)}>Open in Kinematics</button>
          <button className="btn primary" onClick={() => onSimulate(s.robot)}>Simulate in Physics</button>
        </div>
        <div className="row" style={{ marginTop: 6 }}>
          <button className="btn" onClick={() => onOpenControl(s.robot)}>Open in Control lab</button>
          <button className="btn" onClick={() => onOpenSkills(s.robot)}>Teach skills</button>
        </div>
        <div className="row" style={{ marginTop: 6 }}>
          <button className="btn" onClick={() => onOpenNavigate(s.robot)} disabled={!drive.ok} aria-describedby={drive.ok ? undefined : driveNoteId} data-testid="builder-navigate">Drive in Navigate</button>
        </div>
        {!drive.ok && <p className="note muted" id={driveNoteId} data-testid="builder-navigate-reason">Navigate cannot drive this robot: {drive.reason}. Navigate drives wheeled robots and walks robots with four or more legs.</p>}
      </Section>
      {storage && <SavedRobots session={session} storage={storage} />}
    </aside>
  );
}

export function BuilderMode({ session, storage, ...handoffs }: { session: BuilderSession; storage?: AppStorage } & RobotHandoffs) {
  const robot = useStore(session.store).robot;
  return (
    <>
      <aside className="panel left" aria-label="Robot builder">
        <Generators session={session} />
        <BodyEditor session={session} />
      </aside>
      <BuilderViewport robot={robot} />
      <Inspector session={session} storage={storage} {...handoffs} />
    </>
  );
}
