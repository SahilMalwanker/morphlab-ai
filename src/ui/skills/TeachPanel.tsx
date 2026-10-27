import { useId, useState, type ReactNode } from 'react';
import { GAIT_BOUNDS, gaitText, type GaitParams } from '../../skills/gait';
import { checkIntent } from '../../skills/intent';
import { RESIDUAL_SCALE_RAD } from '../../skills/policyGait';
import { POSE_RESIDUAL_RAD } from '../../skills/posePolicy';
import { GOAL_BOUNDS, GOAL_LABELS, goalKindsFor, goalText, hasFoot, motionDuration, type SkillGoal, type SkillGoalKind, type TipAxis, type TravelDirection } from '../../skills/taught';
import { DEG2RAD, NumberField, RAD2DEG, Section } from '../components';
import { Explain } from '../learn';
import { useStore } from '../store';
import { LastPlay } from './LastPlay';
import { draftMotion, type SkillsSession } from './session';

function Choice({ label, value, options, onChange, testId, disabled = false }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void; testId?: string; disabled?: boolean }) {
  const id = useId();
  return (
    <div className="field grow">
      <label htmlFor={id}>{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)} data-testid={testId} disabled={disabled}>
        {options.map(([v, l]) => (
          <option key={v} value={v}>{l}</option>
        ))}
      </select>
    </div>
  );
}

function JointSlider({ label, value, min, max, angular, onChange, testId }: { label: string; value: number; min: number; max: number; angular: boolean; onChange: (v: number) => void; testId: string }) {
  const id = useId();
  const shown = angular ? `${(value * RAD2DEG).toFixed(0)} deg` : `${(value * 100).toFixed(1)} cm`;
  return (
    <div className="field">
      <label htmlFor={id}>{label}: {shown}</label>
      <input id={id} type="range" min={min} max={max} step={angular ? 0.01 : 0.001} value={value} onChange={(e) => onChange(Number(e.target.value))} data-testid={testId} />
    </div>
  );
}

/** One slider per joint, grouped by limb; each sets the target the live robot holds. */
function PoseSliders({ session }: { session: SkillsSession }) {
  const p = useStore(session.physics.store);
  const robot = useStore(session.store).robot;
  let start = 0;
  return (
    <div data-testid="pose-sliders">
      {robot.limbs.map((limb) => {
        const first = start;
        start += limb.joints.length;
        return (
          <fieldset key={limb.id} className="pose-limb">
            <legend>{limb.name}</legend>
            {limb.joints.map((joint, k) => {
              const j = first + k;
              const continuous = joint.type === 'continuous';
              return (
                <JointSlider
                  key={joint.id}
                  label={joint.name}
                  value={p.qCmd[j] ?? 0}
                  min={continuous ? -Math.PI : joint.limit.lower}
                  max={continuous ? Math.PI : joint.limit.upper}
                  angular={joint.type !== 'prismatic'}
                  onChange={(v) => session.setPoseJoint(j, v)}
                  testId={`pose-joint-${j}`}
                />
              );
            })}
          </fieldset>
        );
      })}
    </div>
  );
}

function Keyframes({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  const d = s.draft!;
  const busy = s.job !== null;
  return (
    <>
      <div className="row">
        <button className="btn primary" onClick={() => session.addKeyframe()} disabled={busy} data-testid="teach-add-keyframe">Add keyframe from this pose</button>
        <button className="btn" onClick={() => session.resetPose()} disabled={busy} data-testid="teach-home">Back to home pose</button>
      </div>
      {d.keyframes.length === 0 ? (
        <p className="note muted">No keyframes yet. The skill starts from the home pose and moves through each keyframe in turn.</p>
      ) : (
        <ol style={{ margin: '8px 0 0', paddingLeft: 18 }} data-testid="teach-keyframes">
          {d.keyframes.map((k, i) => (
            <li key={i} style={{ marginBottom: 6 }}>
              <NumberField label={`Seconds to reach it from ${i === 0 ? 'the home pose' : `keyframe ${i}`}`} value={k.durationS} digits={2} step={0.1} min={0.1} max={10} onCommit={(v) => session.setKeyframeDuration(i, v)} testId={`teach-keyframe-time-${i}`} />
              <div className="row">
                <button className="btn" onClick={() => session.showKeyframe(i)} disabled={busy}>Show</button>
                <button className="btn" onClick={() => session.replaceKeyframe(i)} disabled={busy}>Use current pose</button>
                <button className="btn" onClick={() => session.removeKeyframe(i)} disabled={busy}>Delete</button>
              </div>
            </li>
          ))}
        </ol>
      )}
      <NumberField label="Hold the last pose for (s)" value={d.endHoldS} digits={2} step={0.1} min={0} max={10} onCommit={(v) => session.setEndHold(v)} testId="teach-end-hold" />
    </>
  );
}

const half = (key: string, node: ReactNode) => (
  <div key={key} style={{ width: 'calc(50% - 4px)' }}>
    {node}
  </div>
);

const GAIT_FIELDS: { key: keyof typeof GAIT_BOUNDS; label: string; digits: number }[] = [
  { key: 'cycleS', label: 'Step cycle (s)', digits: 2 },
  { key: 'dutyFactor', label: 'Share of the cycle a foot is down', digits: 2 },
  { key: 'strideM', label: 'Forward per cycle (m)', digits: 3 },
  { key: 'lateralM', label: 'Left per cycle (m)', digits: 3 },
  { key: 'turnRad', label: 'Turn left per cycle (rad)', digits: 2 },
  { key: 'stepHeightM', label: 'Step height (m)', digits: 3 },
  { key: 'bodyHeightM', label: 'Body height change (m)', digits: 3 },
  { key: 'durationS', label: 'Duration (s)', digits: 1 },
  { key: 'feedbackKp', label: 'IMU feedback kp (m/rad)', digits: 3 },
  { key: 'feedbackKd', label: 'IMU feedback kd (m*s/rad)', digits: 4 },
];

/** The parameters of a gait draft; out-of-range entries are flagged and not applied. */
function GaitEditor({ session, gait }: { session: SkillsSession; gait: GaitParams }) {
  return (
    <>
      <p className="note muted" data-testid="teach-gait">{gaitText(gait)}.</p>
      <div className="row" style={{ flexWrap: 'wrap' }}>
        {GAIT_FIELDS.map((f) =>
          half(f.key, <NumberField label={f.label} value={gait[f.key]} digits={f.digits} step={10 ** -f.digits} min={GAIT_BOUNDS[f.key][0]} max={GAIT_BOUNDS[f.key][1]} onCommit={(v) => session.setGaitParam(f.key, v)} testId={`teach-gait-${f.key}`} />),
        )}
      </div>
    </>
  );
}

/** Editors for one goal's parameters; angles are shown in degrees and stored in radians. Out-of-range entries are flagged and not applied. */
function goalFields(g: SkillGoal, set: (g: SkillGoal) => void, limbs: [string, string][], feet: [string, string][]): ReactNode[] {
  const num = (field: string, label: string, value: number, digits: number, onCommit: (v: number) => void) => {
    const [lo, hi] = GOAL_BOUNDS[field]!;
    return half(field, <NumberField label={label} value={value} digits={digits} step={digits === 0 ? 1 : 10 ** -digits} min={lo} max={hi} onCommit={onCommit} />);
  };
  const angle = (field: string, label: string, rad: number, onCommit: (rad: number) => void) => {
    const [lo, hi] = GOAL_BOUNDS[field]!;
    return half(field, <NumberField label={label} value={rad * RAD2DEG} digits={0} step={1} min={Math.ceil(lo * RAD2DEG)} max={Math.floor(hi * RAD2DEG)} onCommit={(v) => onCommit(v * DEG2RAD)} />);
  };
  switch (g.kind) {
    case 'upright':
      return [angle('maxTiltRad', 'Largest tilt (deg)', g.maxTiltRad, (v) => set({ ...g, maxTiltRad: v }))];
    case 'noFall':
      return [num('minBaseHeightM', 'Lowest body height (m)', g.minBaseHeightM, 3, (v) => set({ ...g, minBaseHeightM: v }))];
    case 'tipHeight':
      return [
        half('limb', <Choice label="Limb" value={g.limbId} options={limbs} onChange={(v) => set({ ...g, limbId: v })} />),
        num('minHeightM', 'Tip height above ground (m)', g.minHeightM, 3, (v) => set({ ...g, minHeightM: v })),
        num('holdS', 'For at least (s)', g.holdS, 2, (v) => set({ ...g, holdS: v })),
      ];
    case 'baseHeight':
      return [
        half('dir', <Choice label="Body" value={g.below ? 'below' : 'above'} options={[['below', 'below'], ['above', 'above']]} onChange={(v) => set({ ...g, below: v === 'below' })} />),
        num('heightM', 'Height (m)', g.heightM, 3, (v) => set({ ...g, heightM: v })),
        num('holdS', 'For at least (s)', g.holdS, 2, (v) => set({ ...g, holdS: v })),
      ];
    case 'airborne': {
      const toggle = (id: string, on: boolean) => {
        const next = on ? [...(g.limbIds ?? []), id] : (g.limbIds ?? []).filter((x) => x !== id);
        set(next.length ? { ...g, limbIds: next } : { kind: 'airborne', minS: g.minS });
      };
      return [
        num('minS', 'For at least (s)', g.minS, 2, (v) => set({ ...g, minS: v })),
        <fieldset key="feet" className="field" style={{ width: '100%' }}>
          <legend>Feet that leave the ground together (none ticked: every foot)</legend>
          {feet.map(([id, name]) => (
            <label key={id} style={{ marginRight: 12 }}>
              <input type="checkbox" checked={g.limbIds?.includes(id) ?? false} onChange={(e) => toggle(id, e.target.checked)} /> {name}
            </label>
          ))}
        </fieldset>,
      ];
    }
    case 'baseRise':
      return [num('minRiseM', 'Rise above resting height (m)', g.minRiseM, 3, (v) => set({ ...g, minRiseM: v }))];
    case 'travel':
      return [
        half('dir', <Choice label="Direction" value={g.direction} options={[['forward', 'forward'], ['backward', 'backward'], ['left', 'left'], ['right', 'right']]} onChange={(v) => set({ ...g, direction: v as TravelDirection })} />),
        num('minDistanceM', 'Distance (m)', g.minDistanceM, 3, (v) => set({ ...g, minDistanceM: v })),
        g.maxDistanceM !== undefined
          ? num('maxDistanceM', 'At most (m)', g.maxDistanceM, 3, (v) => set({ ...g, maxDistanceM: v }))
          : half('maxDistanceM', <button className="btn" onClick={() => set({ ...g, maxDistanceM: Number(Math.min(5, 1.5 * g.minDistanceM).toFixed(3)) })}>Add an upper limit</button>),
        ...(g.maxDistanceM !== undefined ? [half('noMax', <button className="btn" onClick={() => set({ kind: 'travel', direction: g.direction, minDistanceM: g.minDistanceM })}>No upper limit</button>)] : []),
      ];
    case 'turn':
      return [
        half('dir', <Choice label="Direction" value={g.left ? 'left' : 'right'} options={[['left', 'left'], ['right', 'right']]} onChange={(v) => set({ ...g, left: v === 'left' })} />),
        angle('minAngleRad', 'Angle (deg)', g.minAngleRad, (v) => set({ ...g, minAngleRad: v })),
        g.maxAngleRad !== undefined
          ? angle('maxAngleRad', 'At most (deg)', g.maxAngleRad, (v) => set({ ...g, maxAngleRad: v }))
          : half('maxAngleRad', <button className="btn" onClick={() => set({ ...g, maxAngleRad: Number(Math.min(6.3, 1.5 * g.minAngleRad).toFixed(3)) })}>Add an upper limit</button>),
        ...(g.maxAngleRad !== undefined ? [half('noMax', <button className="btn" onClick={() => set({ kind: 'turn', left: g.left, minAngleRad: g.minAngleRad })}>No upper limit</button>)] : []),
      ];
    case 'standsAtEnd':
      return [
        angle('maxTiltRad', 'Largest tilt (deg)', g.maxTiltRad, (v) => set({ ...g, maxTiltRad: v })),
        num('minBaseHeightM', 'Lowest body height (m)', g.minBaseHeightM, 3, (v) => set({ ...g, minBaseHeightM: v })),
      ];
    case 'bodyAngle':
      return [
        half('axis', <Choice label="Axis" value={g.axis} options={[['pitch', 'pitch (positive lowers the front)'], ['roll', 'roll (positive raises the left side)']]} onChange={(v) => set({ ...g, axis: v === 'roll' ? 'roll' : 'pitch' })} />),
        angle('minRad', 'From (deg)', g.minRad, (v) => set({ ...g, minRad: v })),
        angle('maxRad', 'To (deg)', g.maxRad, (v) => set({ ...g, maxRad: v })),
        num('holdS', 'For at least (s)', g.holdS, 2, (v) => set({ ...g, holdS: v })),
      ];
    case 'footDown':
      return [
        half('limb', <Choice label="Foot" value={g.limbId} options={limbs} onChange={(v) => set({ ...g, limbId: v })} />),
        num('minFraction', 'Share of the run on the ground (0 to 1)', g.minFraction, 2, (v) => set({ ...g, minFraction: v })),
      ];
    case 'tipSwing':
      return [
        half('limb', <Choice label="Limb" value={g.limbId} options={limbs} onChange={(v) => set({ ...g, limbId: v })} />),
        half('axis', <Choice label="Swings" value={g.axis} options={[['x', 'forward and back'], ['y', 'left and right'], ['z', 'up and down']]} onChange={(v) => set({ ...g, axis: v as TipAxis })} />),
        num('minAmplitudeM', 'Each swing at least (m)', g.minAmplitudeM, 3, (v) => set({ ...g, minAmplitudeM: v })),
        num('minSwings', 'Swings', g.minSwings, 0, (v) => set({ ...g, minSwings: Math.round(v) })),
      ];
    case 'tipReach':
      return [
        half('limb', <Choice label="Limb" value={g.limbId} options={limbs} onChange={(v) => set({ ...g, limbId: v })} />),
        num('x', 'Forward of its start (m)', g.x, 3, (v) => set({ ...g, x: v })),
        num('y', 'Left of its start (m)', g.y, 3, (v) => set({ ...g, y: v })),
        num('z', 'Above its start (m)', g.z, 3, (v) => set({ ...g, z: v })),
        num('toleranceM', 'Within (m)', g.toleranceM, 3, (v) => set({ ...g, toleranceM: v })),
        num('holdS', 'For at least (s)', g.holdS, 2, (v) => set({ ...g, holdS: v })),
      ];
  }
}

function Goals({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  const d = s.draft!;
  const [picked, setKind] = useState<SkillGoalKind>('tipHeight');
  const kinds = goalKindsFor(s.robot);
  const kind = kinds.includes(picked) ? picked : kinds[0]!;
  const limbs: [string, string][] = s.robot.limbs.filter((l) => l.role !== 'wheel').map((l) => [l.id, l.name]);
  const feet: [string, string][] = s.robot.limbs.filter(hasFoot).map((l) => [l.id, l.name]);
  const limbName = (id: string) => s.robot.limbs.find((l) => l.id === id)?.name.toLowerCase() ?? id;
  return (
    <>
      {d.goals.length === 0 && <p className="note warn">Add at least one goal: without one, nothing says whether the skill worked.</p>}
      <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
        {d.goals.map((g, i) => (
          <li key={i} data-testid="teach-goal" style={{ marginBottom: 8 }}>
            <div className="row">
              <strong className="grow" style={{ fontSize: 12 }}>{GOAL_LABELS[g.kind]}</strong>
              <button className="btn" onClick={() => session.removeGoal(i)}>Remove</button>
            </div>
            <p className="note muted" style={{ margin: '2px 0' }}>Success needs: {goalText(g, limbName)}.</p>
            <div className="row" style={{ flexWrap: 'wrap' }}>{goalFields(g, (x) => session.setGoal(i, x), limbs, feet)}</div>
          </li>
        ))}
      </ul>
      <div className="row">
        <Choice label="Add a goal" value={kind} options={kinds.map((k) => [k, GOAL_LABELS[k]])} onChange={(v) => setKind(v as SkillGoalKind)} testId="teach-goal-kind" />
        <button className="btn" style={{ alignSelf: 'flex-end' }} onClick={() => session.addGoal(kind)} data-testid="teach-goal-add">Add</button>
      </div>
    </>
  );
}

/**
 * Teaching a skill of the user's own: name it, pose the live robot and keep
 * each pose as a keyframe, say what success means, try it, then evaluate it.
 */
export function TeachPanel({ session, onOpenLibrary }: { session: SkillsSession; onOpenLibrary?: () => void }) {
  const s = useStore(session.store);
  const nameId = useId();
  const [architecture, setArchitecture] = useState<'linear' | 'mlp'>('linear');
  const busy = s.job !== null;
  const d = s.draft && s.draft.robot === s.robot ? s.draft : null;
  if (!d) {
    return (
      <Section title="Teach a skill">
        <p className="note muted">
          Teach {s.robot.name} a motion of your own: wave, crouch, sit, stretch, hop, or anything its body can do. Pose it in the live simulation, keep each pose as a keyframe, say what success means, and evaluate it over 20 randomized trials. Saved skills run by their name from the command bar and from the AI. With the local AI server running, you can also ask for a motion in the command bar: the AI composes a draft here, tries it and evaluates it.
        </p>
        <div className="row wrap">
          <button className="btn primary" onClick={() => session.newDraft()} disabled={busy} data-testid="teach-new">New skill</button>
          <button className="btn" onClick={() => session.setPanel('foot-lift')} data-testid="teach-example">Explore the foot lift example</button>
          {onOpenLibrary && (
            <button className="btn" onClick={onOpenLibrary} data-testid="teach-open-library">
              Open the skill library ({s.library.length})
            </button>
          )}
        </div>
        <Explain>
          <p>A keyframe skill is teaching by demonstration in its simplest form: you show the poses, the robot blends from one to the next with its joint controller. The motion is open loop; it does not react if the body slips or tips, so evaluate it under random conditions before trusting it.</p>
          <p>For a free-standing robot, training a learned feedback policy makes a keyframe skill closed loop: reinforcement learning in randomized runs in which the body is pushed teaches a small network to correct every joint target from the IMU, the foot contacts and the joint encoders. The keyframes only sketch the motion; the network learns the balance. That is what an unstable pose (balancing on two legs) or a landing needs; the draft keeps the policy only if it beats the keyframes alone on held-out pushed runs.</p>
          <p>Goals are measured from simulator ground truth (body position and tilt, limb tip heights, foot contacts). Whether the body can do the motion at all is up to physics: its joints have torque and speed limits, and a pose that looks fine on its own may fall over in motion.</p>
        </Explain>
      </Section>
    );
  }
  const motion = draftMotion(d);
  const gait = d.gait;
  const playable = gait !== null || d.keyframes.length > 0;
  const missing = d.name.trim() ? checkIntent(d.name, d.request, d.goals, s.robot).filter((f) => f.goalIndex < 0) : [];
  const tr = d.training;
  const lp = d.policy;
  const cms = (v: number) => `${(v * 100).toFixed(1)} cm/s`;
  return (
    <>
      <Section title="Teach a skill">
        {d.author !== 'user' && (
          <div className="note" data-testid="teach-author">
            <p style={{ margin: 0 }}>
              {d.author === 'ai' ? 'Composed by the AI from your request' : 'Composed by the AI, then edited by you'}{d.request ? ` ("${d.request}")` : ''}. Its {gait ? 'gait' : 'poses'} and goals are below: try it, change it, improve it or evaluate it like any skill.
            </p>
            {d.notes.length > 0 && (
              <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {d.notes.map((x, i) => (
                  <li key={i}>{x}</li>
                ))}
              </ul>
            )}
          </div>
        )}
        <div className="field">
          <label htmlFor={nameId}>Name (commands use it)</label>
          <input id={nameId} type="text" value={d.name} maxLength={40} placeholder="for example: wave, crouch, hop" onChange={(e) => session.setDraftName(e.target.value)} data-testid="teach-name" />
        </div>
        {missing.map((f) => (
          <p key={f.word} className="note warn" data-testid="teach-intent">
            "{f.word}": {f.meaning}, and no goal measures that yet ({f.hint}).
          </p>
        ))}
        <p className="note muted" data-testid="teach-draft-state">Draft, not saved: evaluate it to add it to the library. Choosing another robot or reloading the page discards it.</p>
        <div className="row wrap">
          <button className="btn" onClick={() => session.newDraft()} disabled={busy}>Start over</button>
          <button className="btn" onClick={() => session.mirrorDraft()} disabled={busy || !playable} title="Swap left and right: limbs, sideways steps, turns and goals change sides" data-testid="teach-mirror">Mirror left and right</button>
          <button className="btn danger" onClick={() => session.discardDraft()} disabled={busy}>Discard</button>
        </div>
      </Section>
      <Section title="1. Motion">
        {gait ? (
          <>
            <p className="note muted">A gait: each foot is on the ground for part of every cycle and swings forward in an arc for the rest; the pattern sets which feet move together. Leg inverse kinematics turns the foot paths into joint targets.</p>
            <GaitEditor session={session} gait={gait} />
          </>
        ) : (
          <>
            <h3 className="subhead">Pose the robot</h3>
            <p className="note muted">Each slider sets one joint's target. The robot follows under its joint controller in the live simulation, so it can lose its balance, as a real one would.</p>
            <PoseSliders session={session} />
            <h3 className="subhead">Keyframes ({d.keyframes.length})</h3>
            <Keyframes session={session} />
          </>
        )}
        <p className="note" data-testid="teach-playback">
          {gait
            ? `The gait steps for ${gait.durationS} s ${gait.feedbackKp > 0 || gait.feedbackKd > 0 ? 'and corrects body roll and pitch from the IMU (closed loop)' : 'open loop'}${lp ? `; its learned policy reads the IMU and the gait phase every 10 ms and adds up to ${RESIDUAL_SCALE_RAD} rad to each joint target` : ''}.`
            : `The keyframes play ${lp ? `with a learned feedback policy that reads the IMU, the foot contacts and the joint encoders every 10 ms and moves each joint target by up to ${POSE_RESIDUAL_RAD} rad (closed loop)` : 'open loop'} for ${motion.keyframes.length ? motionDuration(motion).toFixed(1) : '0'} s.`}
        </p>
      </Section>
      <Section title={`2. Success criteria (${d.goals.length})`}>
        <Goals session={session} />
      </Section>
      <Section title="3. Try">
        <p className="note muted">One live run in the simulation, judged by the goals. It is not an evaluation and adds nothing to the library; the run is listed under Attempts.</p>
        <div className="row">
          <button className="btn" onClick={() => void session.previewDraft()} disabled={busy || !playable} data-testid="teach-preview">Try once</button>
        </div>
        <LastPlay session={session} />
      </Section>
      <Section title="4. Evaluate and save">
        <p className="note muted">Runs 20 randomized trials (friction, body mass, joint offsets) in a background worker and saves the result to the library, whatever it is. Text commands use the skill only once some trials meet every goal.</p>
        <div className="row">
          <button className="btn primary" onClick={() => void session.evaluateDraft()} disabled={busy || !playable} data-testid="teach-evaluate">Evaluate 20 trials and save</button>
        </div>
        {tr && (
          <p className="note" data-testid="teach-training">
            Improved by trial and error: the mean goal reward on {tr.heldOutTrials} held-out trials rose from {tr.baselineReward.toFixed(3)} to {tr.improvedReward.toFixed(3)} (trials meeting every goal: {tr.baselineSuccesses} to {tr.improvedSuccesses}; {tr.rollouts} simulated runs).
          </p>
        )}
        {lp && lp.kind === 'gait' && (
          <p className="note" data-testid="teach-policy">
            Learned feedback policy ({lp.parameters} parameters, augmented random search, {lp.rollouts} simulated runs): on {lp.heldOutTrials} held-out trials the training return rose from {lp.baseline.meanReturn.toFixed(3)} to {lp.result.meanReturn.toFixed(3)}, goal reward {lp.baseline.meanGoalReward.toFixed(3)} to {lp.result.meanGoalReward.toFixed(3)}, velocity error {cms(lp.baseline.trackingErrorMps ?? 0)} to {cms(lp.result.trackingErrorMps ?? 0)}, trials meeting every goal {lp.baseline.successes} to {lp.result.successes}, falls {lp.baseline.falls} to {lp.result.falls}. Any change to the gait or the goals removes it.
          </p>
        )}
        {lp && lp.kind === 'pose' && (
          <p className="note" data-testid="teach-policy">
            Learned feedback policy ({lp.policy.layers.length > 2 ? 'nonlinear' : 'linear'}, {lp.parameters} parameters, reinforcement learning by augmented random search, {lp.rollouts} simulated runs): on {lp.heldOutTrials} held-out runs, each with a push of {lp.pushImpulseNs?.[0]} to {lp.pushImpulseNs?.[1]} N*s, every goal was met in {lp.result.successes} with the policy and {lp.baseline.successes} with the keyframes alone; falls {lp.result.falls} and {lp.baseline.falls}; goal reward {lp.result.meanGoalReward.toFixed(3)} and {lp.baseline.meanGoalReward.toFixed(3)}. Any change to the keyframes or the goals removes it.
          </p>
        )}
      </Section>
      <details className="disclosure" data-testid="teach-advanced">
        <summary>Improve or train (optional)</summary>
        <div className="disclosure-body">
          {!gait && s.robot.base.kind === 'floating' && (
            <Choice label="Feedback model" value={architecture} options={[['linear', 'Linear'], ['mlp', 'Neural (experimental)']]} onChange={(value) => setArchitecture(value === 'mlp' ? 'mlp' : 'linear')} disabled={busy} testId="teach-policy-architecture" />
          )}
          <div className="row wrap">
            <button className="btn" onClick={() => void session.improveDraft()} disabled={busy || !playable || d.goals.length === 0} data-testid="teach-improve">Improve by trial and error</button>
            {(gait || s.robot.base.kind === 'floating') && (
              <button
                className="btn"
                onClick={() => void session.trainDraftPolicy(gait ? 'linear' : architecture)}
                disabled={busy || !playable || d.goals.length === 0}
                title={gait ? 'Random search trains a network that corrects every joint target of the gait from the IMU, every 10 ms' : 'Reinforcement learning in randomized runs with pushes trains a network that corrects every joint target from the IMU, the foot contacts and the joint encoders, every 10 ms'}
                data-testid="teach-train-policy"
              >
                Train a learned feedback policy
              </button>
            )}
          </div>
          <p className="note muted">
            Improving runs evolution strategies in the background: it varies the {gait ? "gait's parameters" : 'poses and timing'}, keeps changes that raise the goal reward in randomized trials, and replaces the draft only when the result scores better on held-out trials.{gait ? ' Training a learned feedback policy runs random search on a linear network rewarded for keeping the gait\'s own speed where it steps, no drift elsewhere, staying upright and meeting the goals; the draft keeps it only when it beats the gait alone on held-out trials.' : ''} The goals are judged from simulator ground truth; the evaluation runs 20 trials with random friction, body mass and joint offsets, and the result goes to the library with its success rate.
          </p>
        </div>
      </details>
    </>
  );
}
