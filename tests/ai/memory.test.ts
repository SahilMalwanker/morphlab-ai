import { describe, expect, it, vi } from 'vitest';
import { LongTermMemory, MAX_PLACES_PER_MAP, parseMemoryRecord, type MemoryPersistence, type MemoryRecord } from '../../src/ai/memory.ts';

const NOW = '2026-10-04T12:00:00.000Z';
const HASH = 'a'.repeat(64);

describe('world memory: places', () => {
  it('remembers, finds by name regardless of case and spacing, replaces and forgets places per map', () => {
    const m = new LongTermMemory([], () => NOW);
    const r = m.rememberPlace({ mapId: 'open', name: '  Loading   dock ', x: 1.5, y: -0.25, source: 'sensor-estimate' });
    expect(r).toEqual({ ok: true, replaced: false, place: { id: 'place:open:loading dock', kind: 'place', mapId: 'open', name: 'Loading dock', x: 1.5, y: -0.25, source: 'sensor-estimate', savedAt: NOW } });
    expect(m.findPlace('open', 'LOADING DOCK')?.name).toBe('Loading dock');
    expect(m.findPlace('block', 'loading dock')).toBeNull();
    const again = m.rememberPlace({ mapId: 'open', name: 'loading dock', x: 2, y: 0, source: 'given' });
    expect(again.ok && again.replaced).toBe(true);
    expect(m.places('open')).toEqual([expect.objectContaining({ name: 'loading dock', x: 2, y: 0, source: 'given' })]);
    expect(m.forgetPlace('open', 'Loading Dock')?.x).toBe(2);
    expect(m.places('open')).toEqual([]);
    expect(m.forgetPlace('open', 'loading dock')).toBeNull();
  });

  it('refuses unusable names, non-finite positions and more than the limit of places on one map', () => {
    const m = new LongTermMemory([], () => NOW);
    for (const name of ['', '   ', '-dock', 'dock;rm', 'x'.repeat(41), '<b>dock</b>']) {
      expect(m.rememberPlace({ mapId: 'open', name, x: 0, y: 0, source: 'given' }).ok, name).toBe(false);
    }
    expect(m.rememberPlace({ mapId: 'open', name: 'a', x: Number.NaN, y: 0, source: 'given' })).toEqual({ ok: false, reason: 'The position is not a finite number.' });
    expect(m.rememberPlace({ mapId: 'open', name: 'a', x: 0, y: Number.POSITIVE_INFINITY, source: 'given' }).ok).toBe(false);
    for (let i = 0; i < MAX_PLACES_PER_MAP; i++) expect(m.rememberPlace({ mapId: 'open', name: `p${i}`, x: i * 0.1, y: 0, source: 'given' }).ok).toBe(true);
    expect(m.rememberPlace({ mapId: 'open', name: 'one more', x: 0, y: 0, source: 'given' })).toEqual({ ok: false, reason: '20 places are already remembered on this map; forget one first.' });
    // Replacing a place and using another map stay possible.
    expect(m.rememberPlace({ mapId: 'open', name: 'P0', x: 1, y: 1, source: 'given' }).ok).toBe(true);
    expect(m.rememberPlace({ mapId: 'slalom', name: 'one more', x: 0, y: 0, source: 'given' }).ok).toBe(true);
    expect(m.places('open')).toHaveLength(MAX_PLACES_PER_MAP);
  });
});

describe('skill memory: live runs', () => {
  it('counts completed, successful and early-stopped runs and keeps the last one', () => {
    const m = new LongTermMemory([], () => NOW);
    m.recordSkillRun('skill_a', HASH, { outcome: 'succeeded', holdS: 2.1, requiredHoldS: 2, failures: [] });
    m.recordSkillRun('skill_a', HASH, { outcome: 'failed', holdS: 1.4, requiredHoldS: 3, failures: ['hold-too-short'] });
    m.recordSkillRun('skill_a', HASH, { outcome: 'aborted', holdS: 0, requiredHoldS: 2, failures: ['aborted'] });
    const r = m.recordSkillRun('skill_a', HASH, { outcome: 'interrupted', holdS: Number.NaN, requiredHoldS: 2, failures: [] });
    expect(r).toEqual({
      id: 'skill-runs:skill_a',
      kind: 'skill-runs',
      skillId: 'skill_a',
      morphologyHash: HASH,
      completed: 3,
      succeeded: 1,
      interrupted: 1,
      last: { outcome: 'interrupted', holdS: null, requiredHoldS: 2, failures: [], at: NOW },
    });
    m.recordSkillRun('skill_a', HASH, { outcome: 'faulted', holdS: 0.2, requiredHoldS: 2, failures: ['numerical-instability'] });
    expect(m.skillRuns('skill_a')).toMatchObject({ completed: 3, succeeded: 1, interrupted: 2, last: { outcome: 'faulted', failures: ['numerical-instability'] } });
    expect(m.skillRuns('skill_b')).toBeNull();
    expect(m.forgetSkillRuns('skill_a')).toBe(true);
    expect(m.skillRuns('skill_a')).toBeNull();
    expect(m.forgetSkillRuns('skill_a')).toBe(false);
  });
});

describe('memory storage and change notices', () => {
  it('saves and deletes through storage, notifies listeners and restores from the stored records', () => {
    const saved = new Map<string, MemoryRecord>();
    const store: MemoryPersistence = {
      put: async (r) => void saved.set(r.id, r),
      remove: async (id) => void saved.delete(id),
    };
    const m = new LongTermMemory([], () => NOW);
    m.attach(store);
    let calls = 0;
    const unsubscribe = m.subscribe(() => calls++);
    const v0 = m.version();
    m.rememberPlace({ mapId: 'open', name: 'dock', x: 1, y: 1.5, source: 'ground-truth' });
    m.recordSkillRun('skill_a', HASH, { outcome: 'succeeded', holdS: 2, requiredHoldS: 2, failures: [] });
    m.rememberPlace({ mapId: 'open', name: 'gone', x: 0, y: 0, source: 'given' });
    m.forgetPlace('open', 'gone');
    expect(calls).toBe(4);
    expect(m.version()).toBe(v0 + 4);
    unsubscribe();
    m.forgetSkillRuns('nothing');
    expect(calls).toBe(4);
    expect([...saved.keys()].sort()).toEqual(['place:open:dock', 'skill-runs:skill_a']);
    for (const r of saved.values()) expect(parseMemoryRecord(JSON.parse(JSON.stringify(r)))).toEqual({ ok: true, value: r });
    const restarted = new LongTermMemory([...saved.values()]);
    expect(restarted.findPlace('open', 'Dock')).toEqual(saved.get('place:open:dock'));
    expect(restarted.skillRuns('skill_a')?.succeeded).toBe(1);
  });

  it('reports a storage failure, keeps the record for this page, and clears the report after the next save', async () => {
    let fail = true;
    const m = new LongTermMemory([], () => NOW);
    m.attach({
      put: async () => {
        if (fail) throw new Error('quota exceeded');
      },
      remove: async () => {},
    });
    m.rememberPlace({ mapId: 'open', name: 'dock', x: 1, y: 2, source: 'given' });
    await vi.waitFor(() => expect(m.error).toBe('Could not save memory: quota exceeded'));
    expect(m.findPlace('open', 'dock')).not.toBeNull();
    fail = false;
    m.rememberPlace({ mapId: 'open', name: 'dock', x: 1, y: 2, source: 'given' });
    await vi.waitFor(() => expect(m.error).toBeNull());
  });
});

describe('stored memory records', () => {
  const place = { id: 'place:open:dock', kind: 'place', mapId: 'open', name: 'dock', x: 1, y: 2, source: 'given', savedAt: NOW };
  const runs = {
    id: 'skill-runs:skill_a',
    kind: 'skill-runs',
    skillId: 'skill_a',
    morphologyHash: HASH,
    completed: 2,
    succeeded: 1,
    interrupted: 0,
    last: { outcome: 'failed', holdS: 1.2, requiredHoldS: 2, failures: ['hold-too-short'], at: NOW },
  };

  it('accepts valid records', () => {
    expect(parseMemoryRecord(place)).toEqual({ ok: true, value: place });
    expect(parseMemoryRecord(runs)).toEqual({ ok: true, value: runs });
  });

  it('refuses tampered or malformed records with reasons and never repairs them', () => {
    const bad: unknown[] = [
      { ...place, id: 'place:open:other' },
      { ...place, x: 'far' },
      { ...place, y: null },
      { ...place, source: 'model' },
      { ...place, note: 'extra' },
      { ...place, savedAt: 'yesterday' },
      { ...place, name: '<script>' },
      { ...runs, succeeded: 3 },
      { ...runs, completed: -1 },
      { ...runs, completed: 1.5 },
      { ...runs, morphologyHash: 'abc' },
      { ...runs, id: 'skill-runs:skill_b' },
      { ...runs, last: { ...runs.last, outcome: 'great' } },
      { kind: 'note', id: 'x' },
      null,
    ];
    for (const b of bad) {
      const r = parseMemoryRecord(b);
      expect(r.ok, JSON.stringify(b)).toBe(false);
      if (!r.ok) expect(r.errors.length).toBeGreaterThan(0);
    }
    expect(parseMemoryRecord({ ...place, id: 'place:open:other' })).toEqual({ ok: false, errors: ['record: The record id does not match its map and name.'] });
  });
});
