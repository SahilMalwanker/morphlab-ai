import loadMujoco, { type MainModule, type MjData, type MjModel } from '@mujoco/mujoco';
import { beforeAll, describe, expect, it } from 'vitest';
import { joint, quadruped, quadrupedStance } from '../../src/core/model/presets';
import type { RobotDescription } from '../../src/core/model/types';
import { generateMjcf, mjcfNames } from '../../src/physics/mjcf';
import { MujocoRuntime } from '../../src/physics/runtime';

/**
 * Characterizes the collision policy of the generated models (see
 * docs/decisions.md): the base never collides with limb links, MuJoCo
 * filters parent-child pairs, and every other pair collides (links of
 * different limbs, non-adjacent links of one limb, links and the floor).
 */

let mj: MainModule;
beforeAll(async () => {
  mj = await loadMujoco();
});

/** Sorted body-name pairs of the current contacts. Frees the copied contact objects. */
function contactBodies(model: MjModel, data: MjData): string[] {
  const BODY = mj.mjtObj.mjOBJ_BODY.value;
  const out: string[] = [];
  const vec = data.contact;
  try {
    for (let i = 0; i < data.ncon; i++) {
      const c = vec.get(i)!;
      try {
        const names = [c.geom1, c.geom2].map((g) => mj.mj_id2name(model, BODY, model.geom_bodyid[g] as number) || 'world').sort();
        out.push(names.join(' + '));
      } finally {
        c.delete();
      }
    }
  } finally {
    vec.delete();
  }
  return [...new Set(out)].sort();
}

function contactsOf(desc: RobotDescription, q: number[]): string[] {
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q });
  try {
    return contactBodies(rt.model, rt.data);
  } finally {
    rt.dispose();
  }
}

/** The same model with the generated base exclusions removed, to show what they suppress. */
function contactsWithoutExclusions(desc: RobotDescription, q: number[]): string[] {
  // The reference runtime places the base and joints; its qpos layout is identical without the exclusions.
  const rt = MujocoRuntime.create(mj, desc, { initialQ: q });
  const xml = generateMjcf(desc).replace(/\s*<contact>[\s\S]*?<\/contact>/, '');
  const model = mj.MjModel.from_xml_string(xml);
  const data = new mj.MjData(model);
  try {
    expect(model.nq).toBe(rt.model.nq);
    (data.qpos as Float64Array).set(rt.data.qpos as Float64Array);
    mj.mj_forward(model, data);
    return contactBodies(model, data);
  } finally {
    data.delete();
    model.delete();
    rt.dispose();
  }
}

/** Fixed base held 0.5 m above the floor so nothing touches the ground. */
function rig(limbs: RobotDescription['limbs'], size: [number, number, number] = [0.4, 0.4, 0.1]): RobotDescription {
  return { schemaVersion: 1, id: 'rig', name: 'Rig', base: { kind: 'fixed', size, mass: 2, pose: { xyz: [0, 0, 0.5], rpy: [0, 0, 0] } }, limbs };
}

const link = (id: string, xyz: [number, number, number], axis: [number, number, number]) =>
  joint({ id, name: id, xyz, axis, lower: -3, upper: 3, effort: 10, radius: 0.02, mass: 0.2 });

describe('collision policy', () => {
  it('a limb link inside the base does not collide with it: base-limb pairs are excluded', () => {
    // A 0.12 m link hinged at the centre of a 0.4 m box lies entirely inside it.
    const desc = rig([{ id: 'inner', name: 'Inner', role: 'arm', joints: [link('yaw', [0, 0, 0], [0, 0, 1])], tip: { xyz: [0.12, 0, 0], radius: 0.02 } }]);
    expect(contactsOf(desc, [0])).toEqual([]);
    // Without the generated exclusions MuJoCo would report the overlap.
    expect(contactsWithoutExclusions(desc, [0])).toEqual([`${mjcfNames.base} + ${mjcfNames.linkBody('yaw')}`]);
  });

  it('a quadruped leg swung into the body passes through it', () => {
    const q = quadruped();
    const stance = quadrupedStance();
    const rt = MujocoRuntime.create(mj, q, { initialQ: stance });
    const model = mj.MjModel.from_xml_string(generateMjcf(q).replace(/\s*<contact>[\s\S]*?<\/contact>/, ''));
    const data = new mj.MjData(model);
    const isBasePair = (p: string) => p.split(' + ').includes(mjcfNames.base) && !p.includes('world');
    let overlapping = 0;
    try {
      // Scan the front-left leg over its joint ranges (hip roll, hip pitch, knee).
      for (const roll of [-0.7, -0.35, 0, 0.35, 0.7]) {
        for (const pitch of [-1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8]) {
          for (const knee of [-2.7, -1.5, -0.15]) {
            const cfg = stance.slice();
            cfg.splice(0, 3, roll, pitch, knee);
            rt.setState(cfg);
            (data.qpos as Float64Array).set(rt.data.qpos as Float64Array);
            mj.mj_forward(model, data);
            if (!contactBodies(model, data).some(isBasePair)) continue;
            overlapping++;
            // Where the geometry overlaps the body, the generated model reports nothing.
            expect(contactBodies(rt.model, rt.data).filter(isBasePair)).toEqual([]);
          }
        }
      }
    } finally {
      data.delete();
      model.delete();
      rt.dispose();
    }
    expect(overlapping).toBeGreaterThan(0);
  });

  it('links of different limbs collide with each other', () => {
    const a = link('a', [0, -0.15, 0.3], [1, 0, 0]);
    const b = link('b', [0, 0.15, 0.3], [1, 0, 0]);
    const desc = rig(
      [
        { id: 'left', name: 'Left', role: 'arm', joints: [a], tip: { xyz: [0, 0.17, 0], radius: 0.02 } },
        { id: 'right', name: 'Right', role: 'arm', joints: [b], tip: { xyz: [0, -0.17, 0], radius: 0.02 } },
      ],
      [0.1, 0.1, 0.1],
    );
    // Both links reach 2 cm past the midline: they overlap.
    expect(contactsOf(desc, [0, 0])).toEqual([`${mjcfNames.linkBody('a')} + ${mjcfNames.linkBody('b')}`]);
    // Swing one up out of the way: no contact.
    expect(contactsOf(desc, [1.2, 0])).toEqual([]);
  });

  it('non-adjacent links of one limb collide; parent and child do not', () => {
    // Three 0.15 m links; folding the second and third back brings link 3 onto link 1.
    const j1 = link('j1', [0, 0, 0.3], [0, 1, 0]);
    const j2 = link('j2', [0, 0, 0.15], [0, 1, 0]);
    const j3 = link('j3', [0, 0, 0.15], [0, 1, 0]);
    const desc = rig([{ id: 'arm', name: 'Arm', role: 'arm', joints: [j1, j2, j3], tip: { xyz: [0, 0, 0.15], radius: 0.02 } }], [0.1, 0.1, 0.1]);
    expect(contactsOf(desc, [0, 0, 0])).toEqual([]);
    const folded = contactsOf(desc, [0, 2.9, 0]);
    // Link 3 folds back over link 1 (non-adjacent): reported. Link 2 against links 1 and 3 (parent, child): filtered.
    expect(folded).toContain(`${mjcfNames.linkBody('j1')} + ${mjcfNames.linkBody('j3')}`);
    expect(folded).not.toContain(`${mjcfNames.linkBody('j1')} + ${mjcfNames.linkBody('j2')}`);
    expect(folded).not.toContain(`${mjcfNames.linkBody('j2')} + ${mjcfNames.linkBody('j3')}`);
  });
});
