import { baseTransform, forwardKinematics, type KinematicModel } from '../kinematics/fk';
import { getTranslation, transformDirection, transformPoint, type Mat4 } from '../math/mat4';
import { cross, dot, scale, sub, type Vec3 } from '../math/vec3';
import { FOOT_MASS } from '../model/types';

/** Standard gravity used by the simulator (MuJoCo default), m/s^2. */
export const GRAVITY = 9.81;

interface PointMass {
  /** Index of the joint whose link carries this mass (global joint order). */
  joint: number;
  p: Vec3;
  m: number;
}

/** World positions of the link and foot masses at q, following the mass model in `types.ts`. */
function pointMasses(model: KinematicModel, links: Mat4[]): PointMass[] {
  const out: PointMass[] = [];
  model.description.limbs.forEach((limb, li) => {
    const start = model.limbs[li]!.start;
    limb.joints.forEach((j, ji) => {
      const T = links[start + ji]!;
      const next: Vec3 = ji + 1 < limb.joints.length ? limb.joints[ji + 1]!.origin.xyz : limb.tip.xyz;
      const centred = (limb.role === 'wheel' && limb.wheel !== undefined) || Math.hypot(...next) < 1e-6;
      out.push({ joint: start + ji, p: transformPoint(T, centred ? [0, 0, 0] : scale(next, 0.5)), m: j.link.mass });
      if (ji === limb.joints.length - 1 && limb.role !== 'wheel' && limb.tip.radius > 0) {
        out.push({ joint: start + ji, p: transformPoint(T, limb.tip.xyz), m: FOOT_MASS });
      }
    });
  });
  return out;
}

/**
 * Joint torques (or forces for prismatic joints) that statically balance
 * gravity at configuration q with the base held at T_W_B: tau = -J^T m g
 * summed over every mass distal to each joint. For a fixed base this equals
 * the gravity part of the bias force in the equations of motion.
 */
export function gravityTorques(model: KinematicModel, q: ArrayLike<number>, T_W_B: Mat4 = baseTransform(model), g = GRAVITY): number[] {
  const fk = forwardKinematics(model, q, T_W_B);
  const masses = pointMasses(model, fk.links);
  const tau = new Array<number>(model.dof).fill(0);
  for (const limb of model.limbs) {
    for (let i = limb.start; i < limb.start + limb.count; i++) {
      const T = fk.links[i]!;
      const kj = model.joints[i]!;
      const o = getTranslation(T);
      const a = transformDirection(T, kj.axis);
      let generalized = 0;
      for (const pm of masses) {
        if (pm.joint < i || pm.joint >= limb.start + limb.count) continue;
        const f: Vec3 = [0, 0, -g * pm.m];
        generalized += kj.type === 'prismatic' ? dot(a, f) : dot(cross(a, sub(pm.p, o)), f);
      }
      tau[i] = -generalized;
    }
  }
  return tau;
}
