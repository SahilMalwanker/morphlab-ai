import * as THREE from 'three';
import type { Mat4 } from '../core/math/mat4';
import type { Vec3 } from '../core/math/vec3';
import type { RobotDescription } from '../core/model/types';
import { disposeObject } from './viewport';

export interface RobotViewOverlays {
  jointAxes: boolean;
  frames: boolean;
}

const Y_UP = new THREE.Vector3(0, 1, 0);

function placeBetween(mesh: THREE.Object3D, a: Vec3, b: Vec3): void {
  const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
  const d = vb.clone().sub(va);
  const len = d.length();
  mesh.position.copy(va).addScaledVector(d, 0.5);
  if (len > 1e-9) mesh.quaternion.setFromUnitVectors(Y_UP, d.normalize());
}

/**
 * Renders a robot description. Link groups carry no kinematics of their own:
 * their matrices are copied verbatim from transforms computed by the core
 * (kinematics mode) or read from the simulator (physics mode).
 */
export class RobotView {
  readonly root = new THREE.Group();
  readonly base = new THREE.Group();
  readonly links: THREE.Group[] = [];
  private readonly overlayObjects: { axes: THREE.Object3D[]; frames: THREE.Object3D[] } = { axes: [], frames: [] };
  private readonly materials: THREE.Material[] = [];

  constructor(readonly description: RobotDescription) {
    this.root.name = `robot:${description.id}`;
    const mat = (color: number, metalness = 0.15, roughness = 0.55) => {
      const m = new THREE.MeshStandardMaterial({ color, metalness, roughness });
      this.materials.push(m);
      return m;
    };
    const baseMat = mat(0x6f9589);
    const linkMat = mat(0xcdd7d3);
    const jointMat = mat(0x303938);
    const tipMat = mat(0xe5c58b, 0.1, 0.4);
    const wheelMat = mat(0x252a29, 0.05, 0.85);

    this.base.matrixAutoUpdate = false;
    const box = new THREE.Mesh(new THREE.BoxGeometry(...description.base.size), baseMat);
    box.castShadow = true;
    this.base.add(box);
    this.base.add(this.makeFrame(Math.max(...description.base.size) * 0.6));
    this.root.add(this.base);

    for (const limb of description.limbs) {
      limb.joints.forEach((j, i) => {
        const g = new THREE.Group();
        g.name = `link:${j.id}`;
        g.matrixAutoUpdate = false;
        const next: Vec3 = i + 1 < limb.joints.length ? limb.joints[i + 1]!.origin.xyz : limb.tip.xyz;
        const len = Math.hypot(...next);
        if (limb.role === 'wheel' && limb.wheel) {
          const cyl = new THREE.Mesh(new THREE.CylinderGeometry(limb.wheel.radius, limb.wheel.radius, limb.wheel.width, 32), wheelMat);
          placeBetween(cyl, [0, 0, 0], j.axis);
          cyl.position.set(0, 0, 0);
          cyl.castShadow = true;
          g.add(cyl);
        } else if (len > 1e-6) {
          const cap = new THREE.Mesh(new THREE.CapsuleGeometry(j.link.radius, len, 6, 16), linkMat);
          placeBetween(cap, [0, 0, 0], next);
          cap.castShadow = true;
          g.add(cap);
        }
        const hub = new THREE.Mesh(new THREE.SphereGeometry(j.link.radius * 1.25, 20, 12), jointMat);
        hub.castShadow = true;
        g.add(hub);
        if (i === limb.joints.length - 1 && limb.role !== 'wheel' && limb.tip.radius > 0) {
          const tip = new THREE.Mesh(new THREE.SphereGeometry(limb.tip.radius, 20, 12), tipMat);
          tip.position.set(...limb.tip.xyz);
          tip.castShadow = true;
          g.add(tip);
        }
        const axis = new THREE.ArrowHelper(new THREE.Vector3(...j.axis).normalize(), new THREE.Vector3(), j.link.radius * 2 + 0.06, 0xf2b84b, 0.025, 0.015);
        axis.visible = false;
        this.overlayObjects.axes.push(axis);
        g.add(axis);
        const frame = this.makeFrame(0.07);
        g.add(frame);
        this.links.push(g);
        this.root.add(g);
      });
    }
    this.setOverlays({ jointAxes: false, frames: false });
  }

  private makeFrame(size: number): THREE.Object3D {
    const f = new THREE.AxesHelper(size);
    f.visible = false;
    this.overlayObjects.frames.push(f);
    return f;
  }

  setOverlays(o: RobotViewOverlays): void {
    this.overlayObjects.axes.forEach((a) => (a.visible = o.jointAxes));
    this.overlayObjects.frames.forEach((f) => (f.visible = o.frames));
  }

  /** Copies world transforms (column-major, identical layout to Matrix4.elements). */
  applyTransforms(base: Mat4, links: readonly Mat4[]): void {
    if (links.length !== this.links.length) throw new RangeError('Link transform count does not match the robot');
    this.base.matrix.fromArray(base);
    this.base.matrixWorldNeedsUpdate = true;
    for (let i = 0; i < links.length; i++) {
      const g = this.links[i]!;
      g.matrix.fromArray(links[i]!);
      g.matrixWorldNeedsUpdate = true;
    }
  }

  dispose(): void {
    this.root.removeFromParent();
    disposeObject(this.root);
    this.materials.forEach((m) => m.dispose());
  }
}
