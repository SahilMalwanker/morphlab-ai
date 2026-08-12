import type { JointType } from './types';

/** SI units of a joint's quantities: revolute and continuous joints rotate, prismatic joints slide. */
export interface JointUnits {
  position: 'rad' | 'm';
  velocity: 'rad/s' | 'm/s';
  effort: 'N*m' | 'N';
  /** What the effort is called: torque for rotation, force for sliding. */
  effortName: 'Torque' | 'Force';
  stiffness: 'N*m/rad' | 'N/m';
  damping: 'N*m*s/rad' | 'N*s/m';
  inertia: 'kg*m^2' | 'kg';
  /** Display unit (angles in degrees, lengths in metres), the factor from SI, and display digits. */
  display: { unit: 'deg' | 'm'; scale: number; digits: number };
}

const ANGULAR: JointUnits = {
  position: 'rad', velocity: 'rad/s', effort: 'N*m', effortName: 'Torque', stiffness: 'N*m/rad', damping: 'N*m*s/rad', inertia: 'kg*m^2',
  display: { unit: 'deg', scale: 180 / Math.PI, digits: 1 },
};
const LINEAR: JointUnits = {
  position: 'm', velocity: 'm/s', effort: 'N', effortName: 'Force', stiffness: 'N/m', damping: 'N*s/m', inertia: 'kg',
  display: { unit: 'm', scale: 1, digits: 3 },
};

export function jointUnits(type: JointType): JointUnits {
  return type === 'prismatic' ? LINEAR : ANGULAR;
}

/** Unit labels for gains shared by every joint of a robot; robots with both joint kinds name both. */
export function sharedGainUnits(types: readonly JointType[]): { stiffness: string; damping: string } {
  const angular = types.some((t) => t !== 'prismatic');
  const linear = types.some((t) => t === 'prismatic');
  if (angular && linear) return { stiffness: 'N*m/rad; N/m for prismatic joints', damping: 'N*m*s/rad; N*s/m for prismatic joints' };
  const u = linear ? LINEAR : ANGULAR;
  return { stiffness: u.stiffness, damping: u.damping };
}
