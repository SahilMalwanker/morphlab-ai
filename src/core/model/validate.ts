import { norm } from '../math/vec3';
import type { RobotDescription, ValidationIssue } from './types';

const ID_PATTERN = /^[A-Za-z][A-Za-z0-9_]{0,40}$/;
export const MAX_LIMBS = 12;
export const MAX_JOINTS_PER_LIMB = 8;
/** Sanity bounds that keep generated models inside what the simulator handles well. */
export const BOUNDS = { offset: 2, baseSize: 3, mass: 200, radius: 0.25, effort: 1000, velocity: 100 } as const;

function finite(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

function vec(v: unknown): v is [number, number, number] {
  return Array.isArray(v) && v.length === 3 && v.every(finite);
}

function withinOffset(v: [number, number, number]): boolean {
  return v.every((c) => Math.abs(c) <= BOUNDS.offset);
}

/** Returns every problem found; an empty array means the description is valid. */
export function validateRobot(desc: RobotDescription): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const push = (path: string, message: string) => issues.push({ path, message });

  if (desc.schemaVersion !== 1) push('schemaVersion', 'Unsupported schema version');
  if (!ID_PATTERN.test(desc.id ?? '')) push('id', 'Robot id must start with a letter and use only letters, digits and underscores');
  if (!desc.name?.trim()) push('name', 'Robot name is required');

  const b = desc.base;
  if (!b || (b.kind !== 'fixed' && b.kind !== 'floating')) push('base.kind', 'Base kind must be fixed or floating');
  if (!vec(b?.size) || b.size.some((s) => s <= 0)) push('base.size', 'Base size must be three positive lengths in metres');
  else if (b.size.some((s) => s > BOUNDS.baseSize)) push('base.size', `Base size must be at most ${BOUNDS.baseSize} m per side`);
  if (!finite(b?.mass) || b.mass <= 0) push('base.mass', 'Base mass must be a positive number of kilograms');
  else if (b.mass > BOUNDS.mass) push('base.mass', `Base mass must be at most ${BOUNDS.mass} kg`);
  if (!vec(b?.pose?.xyz) || !vec(b?.pose?.rpy)) push('base.pose', 'Base pose must contain finite xyz and rpy');

  if (!Array.isArray(desc.limbs)) {
    push('limbs', 'Limbs must be a list');
    return issues;
  }
  if (desc.limbs.length > MAX_LIMBS) push('limbs', `At most ${MAX_LIMBS} limbs are supported`);

  const limbIds = new Set<string>();
  const jointIds = new Set<string>();
  desc.limbs.forEach((limb, li) => {
    const lp = `limbs[${li}]`;
    if (!ID_PATTERN.test(limb.id ?? '')) push(`${lp}.id`, 'Limb id must start with a letter and use only letters, digits and underscores');
    else if (limbIds.has(limb.id)) push(`${lp}.id`, `Duplicate limb id "${limb.id}"`);
    limbIds.add(limb.id);
    if (!['arm', 'leg', 'wheel'].includes(limb.role)) push(`${lp}.role`, 'Limb role must be arm, leg or wheel');
    if (!Array.isArray(limb.joints) || limb.joints.length === 0) {
      push(`${lp}.joints`, 'A limb needs at least one joint');
      return;
    }
    if (limb.joints.length > MAX_JOINTS_PER_LIMB) push(`${lp}.joints`, `At most ${MAX_JOINTS_PER_LIMB} joints per limb are supported`);
    if (!vec(limb.tip?.xyz) || !finite(limb.tip?.radius) || limb.tip.radius < 0) push(`${lp}.tip`, 'Tip needs a finite position and a non-negative radius');
    else if (!withinOffset(limb.tip.xyz) || limb.tip.radius > BOUNDS.radius) push(`${lp}.tip`, `Tip offsets must be within ${BOUNDS.offset} m and the tip radius at most ${BOUNDS.radius} m`);
    if (limb.role === 'wheel') {
      if (limb.joints.length !== 1 || limb.joints[0]!.type !== 'continuous') push(`${lp}.joints`, 'A wheel limb has exactly one continuous joint');
      if (!limb.wheel || !(limb.wheel.radius > 0) || !(limb.wheel.width > 0)) push(`${lp}.wheel`, 'A wheel limb needs a positive wheel radius and width');
    } else if (limb.wheel) {
      push(`${lp}.wheel`, 'Only wheel limbs may define wheel geometry');
    }

    limb.joints.forEach((j, ji) => {
      const jp = `${lp}.joints[${ji}]`;
      if (!ID_PATTERN.test(j.id ?? '')) push(`${jp}.id`, 'Joint id must start with a letter and use only letters, digits and underscores');
      else if (jointIds.has(j.id)) push(`${jp}.id`, `Duplicate joint id "${j.id}"`);
      jointIds.add(j.id);
      if (!['revolute', 'prismatic', 'continuous'].includes(j.type)) push(`${jp}.type`, 'Joint type must be revolute, prismatic or continuous');
      if (!vec(j.origin?.xyz) || !vec(j.origin?.rpy)) push(`${jp}.origin`, 'Joint origin must contain finite xyz and rpy');
      else if (!withinOffset(j.origin.xyz)) push(`${jp}.origin`, `Joint offsets must be within ${BOUNDS.offset} m`);
      if (!vec(j.axis)) push(`${jp}.axis`, 'Joint axis must be three finite numbers');
      else if (norm(j.axis) < 1e-9) push(`${jp}.axis`, 'Joint axis must not be zero');
      const lim = j.limit;
      if (!lim || ![lim.lower, lim.upper, lim.effort, lim.velocity].every(finite)) {
        push(`${jp}.limit`, 'Joint limits must be finite numbers');
      } else {
        if (j.type !== 'continuous' && !(lim.lower < lim.upper)) push(`${jp}.limit`, 'Lower limit must be below upper limit');
        if (!(lim.effort > 0)) push(`${jp}.limit.effort`, 'Effort limit must be positive');
        else if (lim.effort > BOUNDS.effort) push(`${jp}.limit.effort`, `Effort limit must be at most ${BOUNDS.effort}`);
        if (!(lim.velocity > 0)) push(`${jp}.limit.velocity`, 'Velocity limit must be positive');
        else if (lim.velocity > BOUNDS.velocity) push(`${jp}.limit.velocity`, `Velocity limit must be at most ${BOUNDS.velocity}`);
      }
      if (!finite(j.link?.radius) || j.link.radius <= 0) push(`${jp}.link.radius`, 'Link radius must be positive');
      else if (j.link.radius > BOUNDS.radius) push(`${jp}.link.radius`, `Link radius must be at most ${BOUNDS.radius} m`);
      if (!finite(j.link?.mass) || j.link.mass <= 0) push(`${jp}.link.mass`, 'Link mass must be positive');
      else if (j.link.mass > BOUNDS.mass) push(`${jp}.link.mass`, `Link mass must be at most ${BOUNDS.mass} kg`);
      if (!finite(j.damping) || j.damping < 0) push(`${jp}.damping`, 'Damping must be zero or positive');
      if (!finite(j.armature) || j.armature < 0) push(`${jp}.armature`, 'Armature must be zero or positive');
    });
  });

  if (desc.stance !== undefined) {
    if (!desc.stance || typeof desc.stance !== 'object' || Array.isArray(desc.stance)) {
      push('stance', 'Stance must map joint ids to positions');
    } else {
      const joints = new Map(desc.limbs.flatMap((l) => (Array.isArray(l?.joints) ? l.joints : [])).map((j) => [j?.id, j]));
      for (const [id, v] of Object.entries(desc.stance)) {
        const j = joints.get(id);
        if (!j) push(`stance.${id}`, `Stance refers to unknown joint "${id}"`);
        else if (!finite(v)) push(`stance.${id}`, 'Stance positions must be finite numbers');
        else if (j.type !== 'continuous' && j.limit && (v < j.limit.lower || v > j.limit.upper)) push(`stance.${id}`, 'Stance position must lie within the joint limits');
      }
    }
  }
  return issues;
}

export function assertValidRobot(desc: RobotDescription): void {
  const issues = validateRobot(desc);
  if (issues.length > 0) {
    throw new Error(`Invalid robot description: ${issues.map((i) => `${i.path}: ${i.message}`).join('; ')}`);
  }
}
