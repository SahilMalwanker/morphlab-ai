import { NumberField, Section } from '../components';
import { Explain } from '../learn';
import { useStore } from '../store';
import { walkOutcomeText, type SkillsSession } from './session';

const cm = (m: number) => `${(m * 100).toFixed(1)} cm`;
const deg = (r: number) => `${((r * 180) / Math.PI).toFixed(0)} deg`;
const range = ([lo, hi]: [number, number]) => `${lo} to ${hi}`;

/** Walking by speed with this body's learned walking controller (D28). */
export function WalkPanel({ session }: { session: SkillsSession }) {
  const s = useStore(session.store);
  const c = s.walking;
  const busy = s.job !== null;
  const legged = s.robot.base.kind === 'floating' && s.robot.limbs.filter((l) => l.role === 'leg').length >= 2;
  const w = s.walkCommand;
  const r = s.lastWalk;
  const e = c?.evaluation;
  return (
    <>
      <Section title="Walk by command">
        {!legged ? (
          <p className="note warn">{s.robot.name} has no legs to walk with.</p>
        ) : c && e ? (
          <p className="note" data-testid="walk-controller">
            Learned walking controller for this body, trained {new Date(c.createdAt).toLocaleString()}: forward {range(c.ranges.forwardMps)} m/s, sideways {range(c.ranges.leftMps)} m/s, turning {range(c.ranges.turnRadps)} rad/s.
            Over {e.trials} randomized commands, {e.final.successes} ended within tolerance (95 % interval {Math.round(e.final.wilson95[0] * 100)} to {Math.round(e.final.wilson95[1] * 100)} %), mean position error {cm(e.final.meanPositionErrorM)}, heading error {deg(e.final.meanHeadingErrorRad)}.
            The gait generator given the same commands without the learned calibration: {e.uncalibrated.successes} of {e.trials}, {cm(e.uncalibrated.meanPositionErrorM)}, {deg(e.uncalibrated.meanHeadingErrorRad)}.
            {c.policy ? ` A feedback policy learned by random search reads the IMU and a sensor-only estimate of where the body is, and changes the gait command every 10 ms${c.policy.layers[c.policy.layers.length - 1]! > 3 ? ' and corrects every joint' : ''}.` : ' The feedback policy did not beat the calibrated generator on held-out commands, so the generator walks alone.'} Training: {c.training.rollouts} simulated runs on {c.training.workers} worker{c.training.workers === 1 ? '' : 's'}, {c.training.wallTimeS} s.
          </p>
        ) : (
          <p className="note muted">No walking controller for this body yet. Training measures how the gait generator moves this body, learns a calibration from the speed asked for to the generator's command, trains a feedback policy by random search over random commands, and evaluates the result. It runs in background workers and takes four to eight minutes.</p>
        )}
        <div className="row">
          <button className="btn" onClick={() => void session.trainLocomotion()} disabled={busy || !legged} data-testid="walk-train">
            {c ? 'Train again' : 'Train walking controller'}
          </button>
          {c && (
            <button className="btn" onClick={() => session.deleteLocomotion()} disabled={busy} data-testid="walk-delete">
              Delete controller
            </button>
          )}
        </div>
      </Section>
      {c && (
        <Section title="Command">
          <div className="row" style={{ flexWrap: 'wrap' }}>
            <div style={{ width: 'calc(50% - 4px)' }}>
              <NumberField label="Forward (m/s)" value={w.forwardMps} digits={3} step={0.01} min={c.ranges.forwardMps[0]} max={c.ranges.forwardMps[1]} onCommit={(v) => session.setWalkCommand({ forwardMps: v })} testId="walk-forward" />
            </div>
            <div style={{ width: 'calc(50% - 4px)' }}>
              <NumberField label="Left (m/s)" value={w.leftMps} digits={3} step={0.01} min={c.ranges.leftMps[0]} max={c.ranges.leftMps[1]} onCommit={(v) => session.setWalkCommand({ leftMps: v })} testId="walk-left" />
            </div>
            <div style={{ width: 'calc(50% - 4px)' }}>
              <NumberField label="Turn left (rad/s)" value={w.turnRadps} digits={2} step={0.05} min={c.ranges.turnRadps[0]} max={c.ranges.turnRadps[1]} onCommit={(v) => session.setWalkCommand({ turnRadps: v })} testId="walk-turn" />
            </div>
            <div style={{ width: 'calc(50% - 4px)' }}>
              <NumberField label="Duration (s)" value={w.durationS} digits={1} step={0.5} min={1} max={10} onCommit={(v) => session.setWalkCommand({ durationS: v })} testId="walk-duration" />
            </div>
          </div>
          <button className="btn primary" onClick={() => void session.walkFromPanel()} disabled={busy || s.playing !== null} data-testid="walk-run">
            Walk
          </button>
          {r && (
            <p className="note" data-testid="walk-result">
              Walked {r.command.forwardMps} m/s forward, {r.command.leftMps} m/s left, {r.command.turnRadps} rad/s for {r.durationS} s: {walkOutcomeText(r)}. Where the command leads: {r.judgement.expected[0].toFixed(3)} m forward, {r.judgement.expected[1].toFixed(3)} m left, turned {deg(r.judgement.expected[2])}; measured (simulator ground truth): {r.judgement.measured[0].toFixed(3)} m, {r.judgement.measured[1].toFixed(3)} m, {deg(r.judgement.measured[2])}. Position error {cm(r.judgement.positionErrorM)} (tolerance {cm(r.judgement.toleranceM)}), heading error {deg(r.judgement.headingErrorRad)} (tolerance {deg(r.judgement.toleranceRad)}).
              {r.clamped ? ' The generator command hit its bounds, so the body could not reach the commanded speed.' : ''}
            </p>
          )}
        </Section>
      )}
      <Explain>
        <p>The command sets the gait generator's stride, side step and turn per cycle through the learned calibration, which also cancels the drift one direction causes in another. The feedback policy reads the IMU, the gait phase and how far a sensor-only estimate of the body's position and heading (leg odometry: the feet on the ground, measured by the joint encoders) is from where the command leads; every 10 ms it changes the stride, side step and turn per cycle (a policy trained with joint corrections also adds at most 0.15 rad to each joint target). The controller sees sensors only; training and the referee use simulator ground truth.</p>
        <p>A walk counts as within tolerance when the body stays up and ends within a quarter of the commanded travel and turn (at least 5 cm and 9 deg) of where the command leads; the stepping fades in and out over 0.5 s, so the distance is about the speed times the duration less 0.5 s.</p>
      </Explain>
    </>
  );
}
