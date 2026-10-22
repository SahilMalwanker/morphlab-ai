import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export interface ViewportOptions {
  background?: number;
  cameraPosition?: [number, number, number];
  target?: [number, number, number];
}

/**
 * Three.js viewport in the MorphLab world frame (right-handed, Z up). No axis
 * conversion is applied: the camera up vector is +Z and the grid lies in XY.
 */
export class Viewport {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  private readonly resizeObserver: ResizeObserver;
  private frame = 0;
  private lastTime = 0;
  private disposed = false;
  /** Optional extra pass after the main render (for example an inset camera). */
  afterRender: ((renderer: THREE.WebGLRenderer, width: number, height: number) => void) | null = null;

  constructor(private readonly container: HTMLElement, opts: ViewportOptions = {}) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(opts.background ?? 0x111415);
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(...(opts.cameraPosition ?? [1.1, -1.3, 0.8]));
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(...(opts.target ?? [0, 0, 0.25]));
    this.controls.enableDamping = true;
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0xf2f5f3, 0x242b2d, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(1.5, -1, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const s = sun.shadow.camera as THREE.OrthographicCamera;
    s.left = -2; s.right = 2; s.top = 2; s.bottom = -2; s.near = 0.1; s.far = 10;
    this.scene.add(sun);

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), new THREE.MeshStandardMaterial({ color: 0x202727, roughness: 0.95 }));
    ground.receiveShadow = true;
    ground.name = 'ground';
    this.scene.add(ground);
    const grid = new THREE.GridHelper(10, 50, 0x4b5b56, 0x303d38);
    grid.rotation.x = Math.PI / 2; // GridHelper lies in XZ; rotate into the XY plane (Z up).
    grid.position.z = 0.0005;
    this.scene.add(grid);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  resize(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Starts the render loop; `onFrame` receives the elapsed wall time in seconds (capped at 0.1 s). */
  start(onFrame: (dt: number) => void): void {
    const loop = (t: number) => {
      if (this.disposed) return;
      const dt = this.lastTime ? Math.min(0.1, (t - this.lastTime) / 1000) : 0;
      this.lastTime = t;
      onFrame(dt);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      if (this.afterRender) this.afterRender(this.renderer, this.container.clientWidth, this.container.clientHeight);
      this.frame = requestAnimationFrame(loop);
    };
    this.frame = requestAnimationFrame(loop);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    disposeObject(this.scene);
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }
}

/** Disposes geometries and materials of an object tree. */
export function disposeObject(root: THREE.Object3D): void {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
    else mat?.dispose();
  });
}
