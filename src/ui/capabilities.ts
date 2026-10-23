export interface Capability {
  name: string;
  ok: boolean;
  /** What does not work without it. */
  consequence: string;
}

function hasWebGl(): boolean {
  try {
    const c = document.createElement('canvas');
    return Boolean(c.getContext('webgl2') ?? c.getContext('webgl'));
  } catch {
    return false;
  }
}

/** Browser features MorphLab depends on, checked once at startup. */
export function checkCapabilities(): Capability[] {
  return [
    { name: 'WebAssembly', ok: typeof WebAssembly === 'object', consequence: 'the physics engine (MuJoCo) cannot run, so Physics, Control lab and Skills are unavailable' },
    { name: 'Web Workers', ok: typeof Worker === 'function', consequence: 'skill evaluation and training cannot run' },
    { name: 'IndexedDB', ok: typeof indexedDB !== 'undefined', consequence: 'robots and skills are kept only until the page is closed' },
    { name: 'WebGL', ok: hasWebGl(), consequence: '3D views cannot be drawn' },
  ];
}
