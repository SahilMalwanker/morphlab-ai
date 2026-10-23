import { useId } from 'react';
import { modeInfo, type Mode } from './modes';

const STEPS: { mode: Mode; text: string }[] = [
  { mode: 'skills', text: 'teach a skill of your own (pose keyframes, set its goals, try it, evaluate it), then type its name; or evaluate the foot lift example and type "lift the front left leg and hold for three seconds".' },
  { mode: 'build', text: 'start from the quadruped or generate a robot; every edit is a new morphology version. Teach skills hands it to the playground.' },
  { mode: 'kinematics', text: 'set a target and read the IK status, residual and Jacobian.' },
  { mode: 'physics', text: 'watch the robot stand under PD control, push it and tune the gains.' },
  { mode: 'control', text: 'compare PD with gravity compensation on a step response.' },
  { mode: 'navigate', text: 'let the rover, the quadruped or the hexapod map and plan from its sensors, or drive it with the keyboard.' },
];

/** First-visit guide along the flagship path; hidden on request and remembered locally. */
export function Guide({ onOpen, onHide }: { onOpen: (m: Mode) => void; onHide: () => void }) {
  const id = useId();
  return (
    <section className="guide" aria-labelledby={id} data-testid="guide">
      <div className="row">
        <h2 id={id} className="grow">Getting started</h2>
        <button className="btn" type="button" onClick={onHide}>Hide guide</button>
      </div>
      <ol className="guide-steps">
        {STEPS.map((s) => {
          const label = modeInfo(s.mode).title;
          return (
            <li key={s.mode}>
              <button className="link-button" type="button" aria-label={`Go to ${label}`} onClick={() => onOpen(s.mode)}>{label}</button>: {s.text}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
