import { describe, expect, it } from 'vitest';
import { baseTransform, buildKinematicModel, forwardKinematics, tipPosition } from '../../src/core/kinematics/fk';
import { mulberry32 } from '../../src/core/math/rng';
import { arm3, hexapod, quadruped } from '../../src/core/model/presets';
import { RobotView } from '../../src/render/robotView';
import { buildChain, isEditableChain } from '../../src/ui/kinematics/ChainEditor';
import { KinematicsSession } from '../../src/ui/kinematics/session';

describe('render adapter', () => {
  it('places every link group exactly at the core FK transform', () => {
    const rng = mulberry32(2);
    for (const desc of [arm3(), quadruped(), hexapod()]) {
      const km = buildKinematicModel(desc);
      const q = km.joints.map((j) => (Number.isFinite(j.lower) ? j.lower + (j.upper - j.lower) * rng() : 0));
      const fk = forwardKinematics(km, q, baseTransform(km));
      const view = new RobotView(desc);
      view.applyTransforms(fk.base, fk.links);
      view.root.updateMatrixWorld(true);
      view.links.forEach((g, i) => {
        const e = g.matrixWorld.elements;
        for (let k = 0; k < 16; k++) expect(Math.abs(e[k]! - fk.links[i]![k]!)).toBeLessThan(1e-12);
      });
      const b = view.base.matrixWorld.elements;
      for (let k = 0; k < 16; k++) expect(Math.abs(b[k]! - fk.base[k]!)).toBeLessThan(1e-12);
      view.dispose();
    }
  });

  it('rejects a transform list that does not match the robot', () => {
    const view = new RobotView(arm3());
    expect(() => view.applyTransforms(new Float64Array(16), [])).toThrow(/does not match/);
    view.dispose();
  });
});

describe('kinematics session', () => {
  it('starts at a non-singular home pose with the target on the tip', () => {
    const s = new KinematicsSession(arm3());
    const tip = tipPosition(s.state.model, 0, s.state.q);
    expect(s.state.target).toEqual(tip);
    expect(s.state.manip!.inverseCondition).toBeGreaterThan(0.05);
  });

  it('solves reachable targets and keeps joints within limits', () => {
    const s = new KinematicsSession(arm3());
    s.setTarget([0.3, 0.2, 0.4]);
    expect(s.state.ik!.status).toBe('converged');
    expect(s.state.ik!.residual).toBeLessThan(1e-4);
    const tip = tipPosition(s.state.model, 0, s.state.q);
    expect(Math.hypot(tip[0] - 0.3, tip[1] - 0.2, tip[2] - 0.4)).toBeLessThan(1e-4);
  });

  it('rejects non-finite targets and leaves the pose unchanged', () => {
    const s = new KinematicsSession(arm3());
    const q = s.state.q.slice();
    s.setTarget([Number.NaN, 0, 0.3]);
    expect(s.state.inputError).toMatch(/finite/);
    expect(s.state.q).toEqual(q);
    s.setJoint(1, Number.POSITIVE_INFINITY);
    expect(s.state.q).toEqual(q);
  });

  it('clamps slider input to joint limits', () => {
    const s = new KinematicsSession(arm3());
    s.setJoint(1, 10);
    expect(s.state.q[1]).toBe(1.75);
    expect(s.state.limits).toEqual([{ jointIndex: 1, side: 'upper' }]);
  });

  it('refuses an invalid robot without replacing the current one', () => {
    const s = new KinematicsSession(arm3());
    const bad = arm3();
    bad.limbs[0]!.joints[0]!.limit.lower = 5;
    const errors = s.setRobot(bad);
    expect(errors.length).toBeGreaterThan(0);
    expect(s.state.description.limbs[0]!.joints[0]!.limit.lower).toBe(-2.9);
  });

  it('targets a quadruped foot without moving other legs', () => {
    const s = new KinematicsSession(quadruped());
    s.selectLimb(2);
    const before = s.state.q.slice();
    const t = s.state.target;
    s.setTarget([t[0] + 0.03, t[1], t[2] + 0.04]);
    expect(s.state.ik!.status).toBe('converged');
    s.state.q.forEach((v, i) => {
      if (i < 6 || i >= 9) expect(v).toBe(before[i]);
    });
  });
});

describe('chain editor', () => {
  it('builds a valid chain from editor rows', () => {
    const r = buildChain(arm3(), [
      { axis: 'z', length: '0.12', lower: '-160', upper: '160' },
      { axis: 'y', length: '0.3', lower: '-90', upper: '90' },
      { axis: 'x', length: '0.2', lower: '-45', upper: '45' },
      { axis: 'y', length: '0.1', lower: '-120', upper: '120' },
    ]);
    expect(r.errors).toEqual([]);
    expect(r.desc!.limbs[0]!.joints.length).toBe(4);
    expect(isEditableChain(r.desc!)).toBe(true);
    expect(r.desc!.limbs[0]!.joints[2]!.limit.upper).toBeCloseTo(Math.PI / 4, 12);
  });

  it('reports specific errors and builds nothing for invalid rows', () => {
    const r = buildChain(arm3(), [
      { axis: 'z', length: 'abc', lower: '10', upper: '-10' },
    ]);
    expect(r.desc).toBeUndefined();
    expect(r.errors).toEqual(['Joint 1: link length must be between 0.02 and 1 m.', 'Joint 1: lower limit must be below upper limit.']);
  });
});
