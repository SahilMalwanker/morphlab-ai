import { useSyncExternalStore } from 'react';

/** Minimal observable state container; snapshots are replaced, never mutated. */
export class Store<T> {
  private listeners = new Set<() => void>();

  constructor(private state: T) {}

  readonly get = (): T => this.state;

  set(next: T): void {
    this.state = next;
    this.listeners.forEach((l) => l());
  }

  update(patch: Partial<T>): void {
    this.set({ ...this.state, ...patch });
  }

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };
}

export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

interface Versioned {
  subscribe(listener: () => void): () => void;
  version(): number;
}

const noSubscription = (): (() => void) => () => {};
const noVersion = (): number => 0;

/** Re-renders when a versioned source (for example long-term memory) changes; null subscribes to nothing. */
export function useVersion(source: Versioned | null): number {
  return useSyncExternalStore(source ? source.subscribe : noSubscription, source ? source.version : noVersion, source ? source.version : noVersion);
}
