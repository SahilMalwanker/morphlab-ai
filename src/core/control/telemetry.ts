/** Fixed-capacity ring buffer of time-stamped multi-channel samples (oldest overwritten). */
export class Telemetry {
  private readonly data: Float64Array;
  private readonly stride: number;
  private head = 0;
  private count = 0;

  constructor(readonly channels: readonly string[], readonly capacity: number) {
    if (!(capacity > 1)) throw new RangeError('Capacity must exceed 1');
    this.stride = channels.length + 1;
    this.data = new Float64Array(capacity * this.stride);
  }

  get length(): number {
    return this.count;
  }

  push(t: number, values: ArrayLike<number>): void {
    if (values.length !== this.channels.length) throw new RangeError('Channel count mismatch');
    const o = this.head * this.stride;
    this.data[o] = t;
    for (let i = 0; i < values.length; i++) this.data[o + 1 + i] = values[i]!;
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
  }

  private slot(i: number): number {
    return ((this.head - this.count + i + this.capacity) % this.capacity) * this.stride;
  }

  /** Time of the i-th oldest sample. */
  time(i: number): number {
    return this.data[this.slot(i)]!;
  }

  value(i: number, channel: number): number {
    return this.data[this.slot(i) + 1 + channel]!;
  }

  channelIndex(name: string): number {
    const i = this.channels.indexOf(name);
    if (i < 0) throw new RangeError(`Unknown channel ${name}`);
    return i;
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }
}
