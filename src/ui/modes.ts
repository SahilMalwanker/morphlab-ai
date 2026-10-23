export type Mode = 'build' | 'kinematics' | 'physics' | 'control' | 'skills' | 'navigate';

export interface ModeInfo {
  id: Mode;
  /** Name used in messages ("the Skills tab"). */
  label: string;
  /** Name shown in the navigation and the work area. */
  title: string;
  /** One-line description shown in the command bar while the mode is open. */
  hint: string;
}

export const MODES: ModeInfo[] = [
  { id: 'build', label: 'Build', title: 'Build', hint: 'Every accepted edit is validated, saved locally and creates a new morphology version. Refused edits leave the robot unchanged.' },
  { id: 'kinematics', label: 'Kinematics', title: 'Kinematics', hint: 'Drag the target sphere or edit its coordinates. The solver uses damped least squares with joint limits.' },
  { id: 'physics', label: 'Physics', title: 'Physics', hint: 'Joint PD control runs in TypeScript at 500 Hz on encoder readings; torques go to MuJoCo motors with effort limits.' },
  { id: 'control', label: 'Control lab', title: 'Control lab', hint: 'Each run is a fresh MuJoCo simulation. Compare two controllers on the same step, then try the challenges. With the AI proxy running, ask the coach about a result or to try gains on set B.' },
  { id: 'skills', label: 'Skills', title: 'AI Playground', hint: 'Teach the robot skills of your own: pose keyframes, set goals that mean success, evaluate over 20 randomized trials, then run them by name. Every skill is labeled by method and bound to the morphology it was evaluated on.' },
  { id: 'navigate', label: 'Navigate', title: 'Navigate', hint: 'The rover maps with range rays and plans with A*. Sensor-only mode never gives the controller simulator state.' },
];

/** The primary destination; the app opens here (D34). */
export const PLAYGROUND: Mode = 'skills';

/** Secondary engineering tools, in navigation order. */
export const LAB_TOOLS: Mode[] = ['build', 'kinematics', 'physics', 'control', 'navigate'];

export function modeInfo(id: Mode): ModeInfo {
  return MODES.find((m) => m.id === id)!;
}
