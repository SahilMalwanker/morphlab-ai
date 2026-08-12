import type { Vec3 } from '../math/vec3';
import type { JointDescription, JointType, LimbDescription, RobotDescription } from './types';

interface JointSpec {
  id: string;
  name: string;
  type?: JointType;
  xyz: Vec3;
  rpy?: Vec3;
  axis: Vec3;
  lower: number;
  upper: number;
  effort: number;
  velocity?: number;
  radius: number;
  mass: number;
}

export function joint(s: JointSpec): JointDescription {
  return {
    id: s.id,
    name: s.name,
    type: s.type ?? 'revolute',
    origin: { xyz: s.xyz, rpy: s.rpy ?? [0, 0, 0] },
    axis: s.axis,
    limit: { lower: s.lower, upper: s.upper, effort: s.effort, velocity: s.velocity ?? 6 },
    link: { radius: s.radius, mass: s.mass },
    damping: 0.05,
    armature: 0.01,
  };
}

/** Fixed-base arm: base yaw, shoulder pitch, elbow pitch. Links point up (+Z) at q = 0. */
export function arm3(): RobotDescription {
  return {
    schemaVersion: 1,
    id: 'arm3',
    name: 'Three-joint arm',
    base: { kind: 'fixed', size: [0.16, 0.16, 0.06], mass: 2, pose: { xyz: [0, 0, 0.03], rpy: [0, 0, 0] } },
    limbs: [
      {
        id: 'arm',
        name: 'Arm',
        role: 'arm',
        joints: [
          joint({ id: 'base_yaw', name: 'Base yaw', xyz: [0, 0, 0.03], axis: [0, 0, 1], lower: -2.9, upper: 2.9, effort: 20, radius: 0.04, mass: 0.8 }),
          joint({ id: 'shoulder', name: 'Shoulder pitch', xyz: [0, 0, 0.12], axis: [0, 1, 0], lower: -1.75, upper: 1.75, effort: 20, radius: 0.03, mass: 0.6 }),
          joint({ id: 'elbow', name: 'Elbow pitch', xyz: [0, 0, 0.35], axis: [0, 1, 0], lower: -2.5, upper: 2.5, effort: 12, radius: 0.025, mass: 0.4 }),
        ],
        tip: { xyz: [0, 0, 0.3], radius: 0.02 },
      },
    ],
  };
}

export type ChainAxis = 'x' | 'y' | 'z';
const AXIS: Record<ChainAxis, Vec3> = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };

export interface SerialChainSpec {
  axes: ChainAxis[];
  /** Length of the link following each joint (m). */
  lengths: number[];
  lower?: number;
  upper?: number;
}

/** Configurable fixed-base serial chain; every link extends along +Z of its frame at q = 0. */
export function serialChain(spec: SerialChainSpec): RobotDescription {
  const n = spec.axes.length;
  if (n < 1 || spec.lengths.length !== n) throw new RangeError('serialChain needs one length per joint');
  const joints: JointDescription[] = spec.axes.map((ax, i) =>
    joint({
      id: `j${i + 1}`,
      name: `Joint ${i + 1}`,
      xyz: i === 0 ? [0, 0, 0.03] : [0, 0, spec.lengths[i - 1]!],
      axis: AXIS[ax],
      lower: spec.lower ?? -2.6,
      upper: spec.upper ?? 2.6,
      effort: 20,
      radius: Math.max(0.015, 0.035 - 0.004 * i),
      mass: Math.max(0.15, 0.7 - 0.1 * i),
    }),
  );
  return {
    schemaVersion: 1,
    id: 'chain',
    name: `${n}-joint chain`,
    base: { kind: 'fixed', size: [0.16, 0.16, 0.06], mass: 2, pose: { xyz: [0, 0, 0.03], rpy: [0, 0, 0] } },
    limbs: [{ id: 'arm', name: 'Arm', role: 'arm', joints, tip: { xyz: [0, 0, spec.lengths[n - 1]!], radius: 0.02 } }],
  };
}

export const QUADRUPED_LEGS = ['front_left', 'front_right', 'rear_left', 'rear_right'] as const;
export const QUADRUPED_STANCE: Record<'hip_roll' | 'hip_pitch' | 'knee', number> = { hip_roll: 0, hip_pitch: 0.7, knee: -1.4 };

function quadLeg(id: string, name: string, x: number, side: 1 | -1): LimbDescription {
  return {
    id,
    name,
    role: 'leg',
    joints: [
      joint({ id: `${id}_hip_roll`, name: 'Hip roll', xyz: [x, 0.1 * side, 0], axis: [1, 0, 0], lower: -0.7, upper: 0.7, effort: 12, velocity: 20, radius: 0.025, mass: 0.12 }),
      joint({ id: `${id}_hip_pitch`, name: 'Hip pitch', xyz: [0, 0.03 * side, 0], axis: [0, 1, 0], lower: -1.8, upper: 1.8, effort: 12, velocity: 20, radius: 0.022, mass: 0.25 }),
      joint({ id: `${id}_knee`, name: 'Knee', xyz: [0, 0, -0.16], axis: [0, 1, 0], lower: -2.7, upper: -0.15, effort: 12, velocity: 20, radius: 0.018, mass: 0.12 }),
    ],
    tip: { xyz: [0, 0, -0.16], radius: 0.022 },
  };
}

/** Floating-base quadruped with 3 joints per leg (12 actuated joints). Knees bend backward. */
export function quadruped(): RobotDescription {
  const stance: Record<string, number> = {};
  for (const leg of QUADRUPED_LEGS) {
    stance[`${leg}_hip_roll`] = QUADRUPED_STANCE.hip_roll;
    stance[`${leg}_hip_pitch`] = QUADRUPED_STANCE.hip_pitch;
    stance[`${leg}_knee`] = QUADRUPED_STANCE.knee;
  }
  return {
    schemaVersion: 1,
    id: 'quadruped',
    name: 'Quadruped',
    base: { kind: 'floating', size: [0.4, 0.2, 0.08], mass: 4, pose: { xyz: [0, 0, 0.28], rpy: [0, 0, 0] } },
    limbs: [
      quadLeg('front_left', 'Front left leg', 0.17, 1),
      quadLeg('front_right', 'Front right leg', 0.17, -1),
      quadLeg('rear_left', 'Rear left leg', -0.17, 1),
      quadLeg('rear_right', 'Rear right leg', -0.17, -1),
    ],
    stance,
  };
}

/** Joint vector for the quadruped stance, ordered limb by limb. */
export function quadrupedStance(): number[] {
  return QUADRUPED_LEGS.flatMap(() => [QUADRUPED_STANCE.hip_roll, QUADRUPED_STANCE.hip_pitch, QUADRUPED_STANCE.knee]);
}

interface RadialLegSpec {
  id: string;
  name: string;
  mount: Vec3;
  yaw: number;
  dof: number;
}

/** Leg whose frame has +X pointing outward; first joint yaws (if dof >= 3), the rest pitch. */
function radialLeg(s: RadialLegSpec): LimbDescription {
  const joints: JointDescription[] = [];
  const lengths = s.dof === 1 ? [0.16] : s.dof === 2 ? [0.08, 0.12] : [0.05, 0.1, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04];
  for (let i = 0; i < s.dof; i++) {
    const yawJoint = s.dof >= 3 && i === 0;
    joints.push(
      joint({
        id: `${s.id}_j${i + 1}`,
        name: yawJoint ? 'Coxa yaw' : `Pitch ${i + (s.dof >= 3 ? 0 : 1)}`,
        xyz: i === 0 ? s.mount : [lengths[i - 1]!, 0, 0],
        rpy: i === 0 ? [0, 0, s.yaw] : [0, 0, 0],
        axis: yawJoint ? [0, 0, 1] : [0, 1, 0],
        lower: yawJoint ? -0.8 : -1.2,
        upper: yawJoint ? 0.8 : 2.4,
        effort: 8,
        velocity: 20,
        radius: 0.018,
        mass: 0.12,
      }),
    );
  }
  return { id: s.id, name: s.name, role: 'leg', joints, tip: { xyz: [lengths[s.dof - 1]!, 0, 0], radius: 0.02 } };
}

/** Floating-base robot with legs spread radially around an elliptical body outline. */
export function radialLegged(legs: number, dofPerLeg: number, id = `legged_${legs}x${dofPerLeg}`): RobotDescription {
  if (legs < 1 || dofPerLeg < 1) throw new RangeError('radialLegged needs at least one leg and one joint');
  const limbs: LimbDescription[] = [];
  for (let k = 0; k < legs; k++) {
    const a = (2 * Math.PI * (k + 0.5)) / legs;
    limbs.push(
      radialLeg({ id: `leg${k + 1}`, name: `Leg ${k + 1}`, mount: [0.18 * Math.cos(a), 0.11 * Math.sin(a), 0], yaw: a, dof: dofPerLeg }),
    );
  }
  return {
    schemaVersion: 1,
    id,
    name: `${legs}-leg robot (${dofPerLeg} DOF per leg)`,
    base: { kind: 'floating', size: [0.36, 0.22, 0.06], mass: 3, pose: { xyz: [0, 0, 0.3], rpy: [0, 0, 0] } },
    limbs,
  };
}

/** Six-legged walker: three legs per side, 3 joints each (coxa yaw, femur, tibia). */
export function hexapod(): RobotDescription {
  const limbs: LimbDescription[] = [];
  const xs = [0.15, 0, -0.15];
  const names = ['front', 'middle', 'rear'];
  for (const [side, sign] of [['left', 1], ['right', -1]] as const) {
    xs.forEach((x, i) => {
      limbs.push(radialLeg({ id: `${names[i]}_${side}`, name: `${names[i]![0]!.toUpperCase()}${names[i]!.slice(1)} ${side} leg`, mount: [x, 0.12 * sign, 0], yaw: (sign * Math.PI) / 2, dof: 3 }));
    });
  }
  return {
    schemaVersion: 1,
    id: 'hexapod',
    name: 'Hexapod',
    base: { kind: 'floating', size: [0.36, 0.24, 0.06], mass: 3, pose: { xyz: [0, 0, 0.22], rpy: [0, 0, 0] } },
    limbs,
  };
}

/** Four-wheel skid-steer rover. Wheels are continuous joints about +Y. */
export function rover(): RobotDescription {
  const wheel = (id: string, name: string, x: number, y: number): LimbDescription => ({
    id,
    name,
    role: 'wheel',
    joints: [
      joint({ id: `${id}_spin`, name: 'Wheel', type: 'continuous', xyz: [x, y, -0.02], axis: [0, 1, 0], lower: -1, upper: 1, effort: 2.5, velocity: 25, radius: 0.06, mass: 0.3 }),
    ],
    tip: { xyz: [0, 0, 0], radius: 0 },
    wheel: { radius: 0.06, width: 0.04 },
  });
  return {
    schemaVersion: 1,
    id: 'rover',
    name: 'Wheeled rover',
    base: { kind: 'floating', size: [0.4, 0.26, 0.08], mass: 4, pose: { xyz: [0, 0, 0.1], rpy: [0, 0, 0] } },
    limbs: [
      wheel('wheel_fl', 'Front left wheel', 0.14, 0.17),
      wheel('wheel_fr', 'Front right wheel', 0.14, -0.17),
      wheel('wheel_rl', 'Rear left wheel', -0.14, 0.17),
      wheel('wheel_rr', 'Rear right wheel', -0.14, -0.17),
    ],
  };
}
