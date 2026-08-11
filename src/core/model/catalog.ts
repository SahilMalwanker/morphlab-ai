import { buildKinematicModel, clampToLimits } from '../kinematics/fk';
import { arm3, hexapod, quadruped, rover, serialChain, type ChainAxis } from './presets';
import type { RobotDescription } from './types';

export type PresetId = 'arm3' | 'arm6' | 'planar2' | 'quadruped' | 'hexapod' | 'rover';

export interface PresetInfo {
  id: PresetId;
  label: string;
  make: () => RobotDescription;
}

export const ARM6_SPEC: { axes: ChainAxis[]; lengths: number[] } = { axes: ['z', 'y', 'y', 'x', 'y', 'x'], lengths: [0.2, 0.3, 0.25, 0.1, 0.08, 0.06] };

export const PRESETS: PresetInfo[] = [
  { id: 'arm3', label: 'Three-joint arm', make: arm3 },
  { id: 'arm6', label: 'Six-joint arm', make: () => ({ ...serialChain(ARM6_SPEC), id: 'arm6', name: 'Six-joint arm' }) },
  { id: 'planar2', label: 'Planar two-link arm', make: () => ({ ...serialChain({ axes: ['y', 'y'], lengths: [0.35, 0.3] }), id: 'planar2', name: 'Planar two-link arm' }) },
  { id: 'quadruped', label: 'Quadruped', make: quadruped },
  { id: 'hexapod', label: 'Hexapod', make: hexapod },
  { id: 'rover', label: 'Wheeled rover', make: rover },
];

/**
 * A comfortable, non-singular starting configuration: the robot's stance
 * where defined, otherwise a heuristic by limb role. Clamped to limits.
 */
export function homeConfiguration(desc: RobotDescription): number[] {
  const km = buildKinematicModel(desc);
  const q: number[] = [];
  for (const limb of desc.limbs) {
    limb.joints.forEach((j, i) => {
      const s = desc.stance?.[j.id];
      if (s !== undefined) q.push(s);
      else if (limb.role === 'wheel') q.push(0);
      else if (limb.role === 'arm') q.push(i === 0 ? 0 : Math.abs(j.axis[1]) > 0.5 ? (i === 1 ? 0.5 : 0.8) : 0);
      else q.push(i === 0 ? 0 : i === 1 ? 0.3 : 1.2);
    });
  }
  return clampToLimits(km, q);
}
