import type { Vec3 } from '../math/vec3';
import { describeIssues } from './builder';
import type { JointDescription, LimbDescription, RobotDescription } from './types';

export type ParseResult<T> = { ok: true; value: T } | { ok: false; errors: string[] };

const ID_KEY = /^[A-Za-z][A-Za-z0-9_]{0,40}$/;

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);

/**
 * Reads untrusted JSON into a RobotDescription. Only known fields are
 * copied; wrong shapes produce path-specific errors, and the result must
 * also pass `validateRobot`.
 */
export function parseRobot(raw: unknown): ParseResult<RobotDescription> {
  const errors: string[] = [];
  const obj = (v: unknown, path: string): Obj => {
    if (isObj(v)) return v;
    errors.push(`${path} must be an object.`);
    return {};
  };
  const num = (o: Obj, k: string, path: string): number => {
    const v = o[k];
    if (typeof v === 'number') return v;
    errors.push(`${path}.${k} must be a number.`);
    return Number.NaN;
  };
  const str = (o: Obj, k: string, path: string): string => {
    const v = o[k];
    if (typeof v === 'string') return v;
    errors.push(`${path}.${k} must be a string.`);
    return '';
  };
  const vec3 = (o: Obj, k: string, path: string): Vec3 => {
    const v = o[k];
    if (Array.isArray(v) && v.length === 3 && v.every((c) => typeof c === 'number')) return [v[0]!, v[1]!, v[2]!];
    errors.push(`${path}.${k} must be a list of three numbers.`);
    return [Number.NaN, Number.NaN, Number.NaN];
  };
  const list = (o: Obj, k: string, path: string): unknown[] => {
    const v = o[k];
    if (Array.isArray(v)) return v;
    errors.push(`${path}.${k} must be a list.`);
    return [];
  };

  const r = obj(raw, 'robot');
  if (r.schemaVersion !== 1) errors.push('robot.schemaVersion must be 1.');
  const baseRaw = obj(r.base, 'robot.base');
  const poseRaw = obj(baseRaw.pose, 'robot.base.pose');
  const kind = str(baseRaw, 'kind', 'robot.base');
  const desc: RobotDescription = {
    schemaVersion: 1,
    id: str(r, 'id', 'robot'),
    name: str(r, 'name', 'robot'),
    base: {
      kind: kind as RobotDescription['base']['kind'],
      size: vec3(baseRaw, 'size', 'robot.base'),
      mass: num(baseRaw, 'mass', 'robot.base'),
      pose: { xyz: vec3(poseRaw, 'xyz', 'robot.base.pose'), rpy: vec3(poseRaw, 'rpy', 'robot.base.pose') },
    },
    limbs: list(r, 'limbs', 'robot').map((lv, li): LimbDescription => {
      const lp = `robot.limbs[${li}]`;
      const l = obj(lv, lp);
      const tip = obj(l.tip, `${lp}.tip`);
      const limb: LimbDescription = {
        id: str(l, 'id', lp),
        name: str(l, 'name', lp),
        role: str(l, 'role', lp) as LimbDescription['role'],
        joints: list(l, 'joints', lp).map((jv, ji): JointDescription => {
          const jp = `${lp}.joints[${ji}]`;
          const j = obj(jv, jp);
          const origin = obj(j.origin, `${jp}.origin`);
          const limit = obj(j.limit, `${jp}.limit`);
          const link = obj(j.link, `${jp}.link`);
          return {
            id: str(j, 'id', jp),
            name: str(j, 'name', jp),
            type: str(j, 'type', jp) as JointDescription['type'],
            origin: { xyz: vec3(origin, 'xyz', `${jp}.origin`), rpy: vec3(origin, 'rpy', `${jp}.origin`) },
            axis: vec3(j, 'axis', jp),
            limit: {
              lower: num(limit, 'lower', `${jp}.limit`),
              upper: num(limit, 'upper', `${jp}.limit`),
              effort: num(limit, 'effort', `${jp}.limit`),
              velocity: num(limit, 'velocity', `${jp}.limit`),
            },
            link: { radius: num(link, 'radius', `${jp}.link`), mass: num(link, 'mass', `${jp}.link`) },
            damping: num(j, 'damping', jp),
            armature: num(j, 'armature', jp),
          };
        }),
        tip: { xyz: vec3(tip, 'xyz', `${lp}.tip`), radius: num(tip, 'radius', `${lp}.tip`) },
      };
      if (l.wheel !== undefined) {
        const w = obj(l.wheel, `${lp}.wheel`);
        limb.wheel = { radius: num(w, 'radius', `${lp}.wheel`), width: num(w, 'width', `${lp}.wheel`) };
      }
      return limb;
    }),
  };
  if (r.stance !== undefined) {
    const s = obj(r.stance, 'robot.stance');
    const stance: Record<string, number> = {};
    for (const [k, v] of Object.entries(s)) {
      if (!ID_KEY.test(k)) errors.push(`robot.stance has an invalid joint id "${k.slice(0, 40)}".`);
      else if (typeof v !== 'number') errors.push(`robot.stance.${k} must be a number.`);
      else stance[k] = v;
    }
    desc.stance = stance;
  }
  if (errors.length) return { ok: false, errors };
  const issues = describeIssues(desc);
  return issues.length ? { ok: false, errors: issues } : { ok: true, value: desc };
}
