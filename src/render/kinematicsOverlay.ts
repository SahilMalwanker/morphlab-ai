import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import type { Manipulability } from '../core/kinematics/jacobian';
import type { Vec3 } from '../core/math/vec3';
import type { Viewport } from './viewport';

export type TargetStatus = 'idle' | 'converged' | 'unreachable' | 'stalled' | 'max-iterations' | 'invalid-input';

const STATUS_COLOR: Record<TargetStatus, number> = {
  idle: 0x4ea1ff,
  converged: 0x3fb68b,
  unreachable: 0xe3b341,
  stalled: 0xe3b341,
  'max-iterations': 0xe3b341,
  'invalid-input': 0xf0616d,
};

/** Draggable IK target plus optional analysis overlays (residual, ellipsoid, workspace). */
export class KinematicsOverlay {
  readonly group = new THREE.Group();
  private readonly target: THREE.Mesh;
  private readonly targetMat = new THREE.MeshStandardMaterial({ color: STATUS_COLOR.idle, emissive: 0x0, roughness: 0.4 });
  private readonly controls: TransformControls;
  private readonly helper: THREE.Object3D;
  private readonly residual: THREE.Line;
  private readonly ellipsoid: THREE.Mesh;
  private workspace: THREE.Points | null = null;
  private readonly workspaceMat = new THREE.PointsMaterial({ color: 0x4ea1ff, size: 0.006, transparent: true, opacity: 0.35, depthWrite: false });
  private suppress = false;

  constructor(private readonly viewport: Viewport, onTargetMoved: (p: Vec3) => void) {
    this.group.name = 'kinematics-overlay';
    this.target = new THREE.Mesh(new THREE.SphereGeometry(0.022, 24, 16), this.targetMat);
    this.target.name = 'ik-target';
    this.group.add(this.target);

    const lineGeom = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
    this.residual = new THREE.Line(lineGeom, new THREE.LineDashedMaterial({ color: 0xe3b341, dashSize: 0.01, gapSize: 0.008 }));
    this.residual.visible = false;
    this.group.add(this.residual);

    this.ellipsoid = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 16),
      new THREE.MeshStandardMaterial({ color: 0xa78bfa, transparent: true, opacity: 0.25, depthWrite: false }),
    );
    this.ellipsoid.visible = false;
    this.ellipsoid.matrixAutoUpdate = false;
    this.group.add(this.ellipsoid);

    viewport.scene.add(this.group);
    this.controls = new TransformControls(viewport.camera, viewport.renderer.domElement);
    this.controls.setMode('translate');
    this.controls.setSize(0.7);
    this.controls.attach(this.target);
    this.helper = this.controls.getHelper();
    viewport.scene.add(this.helper);
    this.controls.addEventListener('dragging-changed', (e) => {
      viewport.controls.enabled = !e.value;
    });
    this.controls.addEventListener('objectChange', () => {
      if (this.suppress) return;
      const p = this.target.position;
      onTargetMoved([p.x, p.y, p.z]);
    });
  }

  setTarget(p: Vec3, status: TargetStatus): void {
    this.suppress = true;
    this.target.position.set(p[0], p[1], p[2]);
    this.suppress = false;
    this.targetMat.color.setHex(STATUS_COLOR[status]);
  }

  setTargetVisible(visible: boolean): void {
    this.target.visible = visible;
    this.helper.visible = visible;
    this.controls.enabled = visible;
  }

  /** Line from the limb tip to the target, shown when the residual is noticeable. */
  setResidual(tip: Vec3, target: Vec3, visible: boolean): void {
    const pos = this.residual.geometry.getAttribute('position') as THREE.BufferAttribute;
    pos.setXYZ(0, tip[0], tip[1], tip[2]);
    pos.setXYZ(1, target[0], target[1], target[2]);
    pos.needsUpdate = true;
    this.residual.computeLineDistances();
    this.residual.geometry.computeBoundingSphere();
    this.residual.visible = visible;
  }

  /** Velocity manipulability ellipsoid at the tip: semi-axes sigma_i * u_i, scaled for display. */
  setEllipsoid(tip: Vec3, m: Manipulability | null, visible: boolean, scale = 0.2): void {
    if (!m || !visible) {
      this.ellipsoid.visible = false;
      return;
    }
    const [u0, u1, u2] = m.directions;
    const s = m.singularValues.map((v) => Math.max(v * scale, 0.002));
    const M = new THREE.Matrix4().makeBasis(new THREE.Vector3(...u0), new THREE.Vector3(...u1), new THREE.Vector3(...u2));
    M.multiply(new THREE.Matrix4().makeScale(s[0]!, s[1]!, s[2]!));
    M.setPosition(tip[0], tip[1], tip[2]);
    this.ellipsoid.matrix.copy(M);
    this.ellipsoid.matrixWorldNeedsUpdate = true;
    this.ellipsoid.visible = true;
  }

  setWorkspace(points: Float32Array | null): void {
    if (this.workspace) {
      this.workspace.removeFromParent();
      this.workspace.geometry.dispose();
      this.workspace = null;
    }
    if (!points) return;
    const g = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(points, 3));
    this.workspace = new THREE.Points(g, this.workspaceMat);
    this.workspace.name = 'workspace-samples';
    this.group.add(this.workspace);
  }

  dispose(): void {
    this.controls.detach();
    this.controls.dispose();
    this.helper.removeFromParent();
    this.setWorkspace(null);
    this.group.removeFromParent();
    this.group.traverse((o) => {
      const mesh = o as THREE.Mesh;
      mesh.geometry?.dispose();
      const mat = mesh.material as THREE.Material | undefined;
      mat?.dispose();
    });
    this.workspaceMat.dispose();
    this.viewport.controls.enabled = true;
  }
}
