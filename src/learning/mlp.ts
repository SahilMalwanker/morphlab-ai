import { gaussian } from '../core/math/rng';

/**
 * Small dense network: tanh hidden layers, linear output layer.
 * Parameters are stored flat, layer by layer: W (out x in, row-major) then b (out).
 */
export class Mlp {
  readonly sizes: number[];
  readonly params: Float64Array;

  constructor(sizes: number[], params?: Float64Array) {
    if (sizes.length < 2 || sizes.some((s) => !Number.isInteger(s) || s < 1)) throw new RangeError('Invalid layer sizes');
    this.sizes = sizes.slice();
    const n = Mlp.paramCount(sizes);
    if (params && params.length !== n) throw new RangeError(`Expected ${n} parameters, got ${params.length}`);
    this.params = params ? new Float64Array(params) : new Float64Array(n);
  }

  static paramCount(sizes: number[]): number {
    let n = 0;
    for (let l = 0; l + 1 < sizes.length; l++) n += sizes[l + 1]! * sizes[l]! + sizes[l + 1]!;
    return n;
  }

  /** Glorot-style initialization with a seeded generator; output layer scaled down. */
  static initialized(sizes: number[], rng: () => number, outputScale = 0.1): Mlp {
    const m = new Mlp(sizes);
    let o = 0;
    for (let l = 0; l + 1 < sizes.length; l++) {
      const fanIn = sizes[l]!, fanOut = sizes[l + 1]!;
      const std = Math.sqrt(2 / (fanIn + fanOut)) * (l + 2 === sizes.length ? outputScale : 1);
      for (let k = 0; k < fanIn * fanOut; k++) m.params[o++] = std * gaussian(rng);
      o += fanOut; // biases start at zero
    }
    return m;
  }

  get inputSize(): number {
    return this.sizes[0]!;
  }

  get outputSize(): number {
    return this.sizes[this.sizes.length - 1]!;
  }

  /** Forward pass with explicit parameters (used by evolution strategies without copying). */
  static forwardWith(sizes: number[], params: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
    let a = Float64Array.from(x as ArrayLike<number>);
    let o = 0;
    for (let l = 0; l + 1 < sizes.length; l++) {
      const nin = sizes[l]!, nout = sizes[l + 1]!;
      const z = new Float64Array(nout);
      for (let r = 0; r < nout; r++) {
        let s = 0;
        const row = o + r * nin;
        for (let c = 0; c < nin; c++) s += params[row + c]! * a[c]!;
        z[r] = s + params[o + nout * nin + r]!;
      }
      o += nout * nin + nout;
      if (l + 2 < sizes.length) for (let r = 0; r < nout; r++) z[r] = Math.tanh(z[r]!);
      a = z;
    }
    return a;
  }

  forward(x: ArrayLike<number>): Float64Array {
    if (x.length !== this.inputSize) throw new RangeError('Input size mismatch');
    return Mlp.forwardWith(this.sizes, this.params, x);
  }

  /**
   * Gradient of a scalar loss w.r.t. all parameters, given dLoss/dOutput.
   * Accumulates into `grad` (same length as params).
   */
  backward(x: ArrayLike<number>, dOut: ArrayLike<number>, grad: Float64Array): void {
    const L = this.sizes.length - 1;
    const acts: Float64Array[] = [Float64Array.from(x as ArrayLike<number>)];
    const offsets: number[] = [];
    let o = 0;
    for (let l = 0; l < L; l++) {
      offsets.push(o);
      const nin = this.sizes[l]!, nout = this.sizes[l + 1]!;
      const a = acts[l]!;
      const z = new Float64Array(nout);
      for (let r = 0; r < nout; r++) {
        let s = 0;
        for (let c = 0; c < nin; c++) s += this.params[o + r * nin + c]! * a[c]!;
        z[r] = s + this.params[o + nout * nin + r]!;
      }
      if (l + 1 < L) for (let r = 0; r < nout; r++) z[r] = Math.tanh(z[r]!);
      acts.push(z);
      o += nout * nin + nout;
    }
    let delta = Float64Array.from(dOut as ArrayLike<number>);
    for (let l = L - 1; l >= 0; l--) {
      const nin = this.sizes[l]!, nout = this.sizes[l + 1]!;
      const off = offsets[l]!;
      const aIn = acts[l]!;
      for (let r = 0; r < nout; r++) {
        const d = delta[r]!;
        for (let c = 0; c < nin; c++) grad[off + r * nin + c]! += d * aIn[c]!;
        grad[off + nout * nin + r]! += d;
      }
      if (l === 0) break;
      const prev = new Float64Array(nin);
      for (let c = 0; c < nin; c++) {
        let s = 0;
        for (let r = 0; r < nout; r++) s += this.params[off + r * nin + c]! * delta[r]!;
        const h = aIn[c]!; // tanh activation of the previous layer
        prev[c] = s * (1 - h * h);
      }
      delta = prev;
    }
  }

  toJSON(): { sizes: number[]; activation: 'tanh'; params: number[] } {
    return { sizes: this.sizes.slice(), activation: 'tanh', params: Array.from(this.params) };
  }

  static fromJSON(j: { sizes: number[]; activation: string; params: number[] }): Mlp {
    if (j.activation !== 'tanh') throw new Error(`Unsupported activation ${j.activation}`);
    if (!j.params.every(Number.isFinite)) throw new Error('Policy weights contain non-finite values');
    return new Mlp(j.sizes, Float64Array.from(j.params));
  }
}

/** Adam optimizer minimizing a loss: params -= lr * mHat / (sqrt(vHat) + eps). */
export class Adam {
  private readonly m: Float64Array;
  private readonly v: Float64Array;
  private t = 0;

  constructor(n: number, private readonly lr: number, private readonly beta1 = 0.9, private readonly beta2 = 0.999, private readonly eps = 1e-8) {
    this.m = new Float64Array(n);
    this.v = new Float64Array(n);
  }

  step(params: Float64Array, grad: ArrayLike<number>): void {
    this.t++;
    const b1t = 1 - Math.pow(this.beta1, this.t), b2t = 1 - Math.pow(this.beta2, this.t);
    for (let i = 0; i < params.length; i++) {
      const g = grad[i]!;
      this.m[i] = this.beta1 * this.m[i]! + (1 - this.beta1) * g;
      this.v[i] = this.beta2 * this.v[i]! + (1 - this.beta2) * g * g;
      params[i]! -= (this.lr * (this.m[i]! / b1t)) / (Math.sqrt(this.v[i]! / b2t) + this.eps);
    }
  }
}
