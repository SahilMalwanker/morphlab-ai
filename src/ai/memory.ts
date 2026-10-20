/**
 * Long-term memory for the AI path and the UI, kept in this browser:
 * world memory (named places on a Navigate map) and skill memory (measured
 * outcomes of live skill runs). Conversation memory lives in the
 * controller and is never stored. Records come from measurements and user
 * requests, never from model text, and the user can see and delete them.
 */

/** Place names: short plain text, starting with a letter or digit. */
export const PLACE_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 _'-]{0,39}$/;
export const MAX_PLACES_PER_MAP = 20;

export type PlaceSource = 'sensor-estimate' | 'ground-truth' | 'given';

export const PLACE_SOURCE_LABELS: Record<PlaceSource, string> = {
  'sensor-estimate': 'the sensor-only position estimate',
  'ground-truth': 'simulator ground truth',
  given: 'coordinates given in a request',
};

export type RunOutcome = 'succeeded' | 'failed' | 'aborted' | 'interrupted' | 'faulted';

export interface PlaceRecord {
  id: string;
  kind: 'place';
  mapId: string;
  name: string;
  /** World frame, m. */
  x: number;
  y: number;
  source: PlaceSource;
  savedAt: string;
}

export interface LastRun {
  outcome: RunOutcome;
  /** Longest measured hold, s; null when the run produced no finite value. */
  holdS: number | null;
  requiredHoldS: number;
  failures: string[];
  at: string;
}

export interface SkillRunsRecord {
  id: string;
  kind: 'skill-runs';
  skillId: string;
  morphologyHash: string;
  /** Runs that reached their end: succeeded, failed the criteria, or aborted by the skill. */
  completed: number;
  succeeded: number;
  /** Runs stopped early: interrupted, or ended by a simulation fault. */
  interrupted: number;
  last: LastRun;
}

export type MemoryRecord = PlaceRecord | SkillRunsRecord;

export interface MemoryPersistence {
  put(record: MemoryRecord): Promise<void>;
  remove(id: string): Promise<void>;
}

export function normalizePlaceName(name: string): string {
  return name.trim().replace(/\s+/g, ' ');
}

export function placeId(mapId: string, name: string): string {
  return `place:${mapId}:${normalizePlaceName(name).toLowerCase()}`;
}

export function skillRunsId(skillId: string): string {
  return `skill-runs:${skillId}`;
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isFiniteNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isNonNegInt = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v) && v >= 0;
const isIsoTime = (v: unknown): v is string => typeof v === 'string' && v.length <= 40 && !Number.isNaN(Date.parse(v));
const onlyKeys = (o: Record<string, unknown>, keys: readonly string[]): boolean => Object.keys(o).every((k) => keys.includes(k));
const PLACE_KEYS = ['id', 'kind', 'mapId', 'name', 'x', 'y', 'source', 'savedAt'] as const;
const RUNS_KEYS = ['id', 'kind', 'skillId', 'morphologyHash', 'completed', 'succeeded', 'interrupted', 'last'] as const;
const LAST_KEYS = ['outcome', 'holdS', 'requiredHoldS', 'failures', 'at'] as const;
const PLACE_SOURCES = new Set<string>(['sensor-estimate', 'ground-truth', 'given']);
const RUN_OUTCOMES = new Set<string>(['succeeded', 'failed', 'aborted', 'interrupted', 'faulted']);

/** Validates a stored record; invalid records are reported with reasons, never repaired. */
export function parseMemoryRecord(raw: unknown): { ok: true; value: MemoryRecord } | { ok: false; errors: string[] } {
  if (!isObj(raw)) return { ok: false, errors: ['record: Expected an object.'] };
  const errors: string[] = [];
  if (raw['kind'] === 'place') {
    if (!onlyKeys(raw, PLACE_KEYS)) errors.push('record: Unrecognized key in place record.');
    if (typeof raw['id'] !== 'string' || raw['id'].length > 200) errors.push('id: Invalid place id.');
    if (typeof raw['mapId'] !== 'string' || raw['mapId'].length < 1 || raw['mapId'].length > 100 || !/^[A-Za-z0-9_.:-]+$/.test(raw['mapId'])) errors.push('mapId: Invalid map id.');
    if (typeof raw['name'] !== 'string' || !PLACE_NAME_PATTERN.test(raw['name'])) errors.push('name: Invalid place name.');
    if (!isFiniteNum(raw['x'])) errors.push('x: Expected a finite number.');
    if (!isFiniteNum(raw['y'])) errors.push('y: Expected a finite number.');
    if (typeof raw['source'] !== 'string' || !PLACE_SOURCES.has(raw['source'])) errors.push('source: Invalid place source.');
    if (!isIsoTime(raw['savedAt'])) errors.push('savedAt: must be a date');
    if (errors.length === 0 && raw['id'] !== placeId(raw['mapId'] as string, raw['name'] as string)) {
      errors.push('record: The record id does not match its map and name.');
    }
    if (errors.length > 0) return { ok: false, errors: errors.slice(0, 3) };
    return { ok: true, value: structuredClone(raw) as unknown as PlaceRecord };
  }
  if (raw['kind'] === 'skill-runs') {
    if (!onlyKeys(raw, RUNS_KEYS)) errors.push('record: Unrecognized key in skill-runs record.');
    if (typeof raw['id'] !== 'string' || raw['id'].length > 300) errors.push('id: Invalid record id.');
    if (typeof raw['skillId'] !== 'string' || raw['skillId'].length < 1 || raw['skillId'].length > 200) errors.push('skillId: Invalid skill id.');
    if (typeof raw['morphologyHash'] !== 'string' || !/^[0-9a-f]{64}$/.test(raw['morphologyHash'])) errors.push('morphologyHash: Invalid morphology hash.');
    if (!isNonNegInt(raw['completed'])) errors.push('completed: Expected a non-negative integer.');
    if (!isNonNegInt(raw['succeeded'])) errors.push('succeeded: Expected a non-negative integer.');
    if (!isNonNegInt(raw['interrupted'])) errors.push('interrupted: Expected a non-negative integer.');
    const last = raw['last'];
    if (!isObj(last) || !onlyKeys(last, LAST_KEYS)) {
      errors.push('last: Invalid last run object.');
    } else {
      if (typeof last['outcome'] !== 'string' || !RUN_OUTCOMES.has(last['outcome'])) errors.push('last.outcome: Invalid run outcome.');
      if (last['holdS'] !== null && (!isFiniteNum(last['holdS']) || last['holdS'] < 0)) errors.push('last.holdS: Expected a non-negative number or null.');
      if (!isFiniteNum(last['requiredHoldS']) || last['requiredHoldS'] < 0) errors.push('last.requiredHoldS: Expected a non-negative number.');
      if (!Array.isArray(last['failures']) || last['failures'].length > 10 || !last['failures'].every((f) => typeof f === 'string' && f.length <= 60)) {
        errors.push('last.failures: Expected up to 10 failure strings.');
      }
      if (!isIsoTime(last['at'])) errors.push('last.at: must be a date');
    }
    if (errors.length === 0) {
      if (raw['id'] !== skillRunsId(raw['skillId'] as string)) errors.push('record: The record id does not match its skill.');
      if ((raw['succeeded'] as number) > (raw['completed'] as number)) errors.push('record: More successes than completed runs.');
    }
    if (errors.length > 0) return { ok: false, errors: errors.slice(0, 3) };
    return { ok: true, value: structuredClone(raw) as unknown as SkillRunsRecord };
  }
  return { ok: false, errors: ['kind: Invalid memory record kind.'] };
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export type RememberResult = { ok: true; place: PlaceRecord; replaced: boolean } | { ok: false; reason: string };

export class LongTermMemory {
  readonly #places = new Map<string, PlaceRecord>();
  readonly #runs = new Map<string, SkillRunsRecord>();
  readonly #listeners = new Set<() => void>();
  readonly #now: () => string;
  #persistence: MemoryPersistence | null = null;
  #version = 0;
  #error: string | null = null;

  constructor(records: readonly MemoryRecord[] = [], now: () => string = () => new Date().toISOString()) {
    for (const r of records) {
      if (r.kind === 'place') this.#places.set(r.id, r);
      else this.#runs.set(r.id, r);
    }
    this.#now = now;
  }

  /** Saves later changes through storage; without it, memory lasts until the page closes. */
  attach(p: MemoryPersistence | null): void {
    this.#persistence = p;
  }

  /** Increases with every change (for UI subscriptions). */
  readonly version = (): number => this.#version;

  /** The last storage failure, if a change could not be saved or deleted. */
  get error(): string | null {
    return this.#error;
  }

  readonly subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  };

  places(mapId: string): PlaceRecord[] {
    return [...this.#places.values()].filter((p) => p.mapId === mapId).sort((a, b) => a.name.localeCompare(b.name));
  }

  findPlace(mapId: string, name: string): PlaceRecord | null {
    return this.#places.get(placeId(mapId, name)) ?? null;
  }

  /** Stores or replaces a named place; the caller has checked that the position is valid on the map. */
  rememberPlace(input: { mapId: string; name: string; x: number; y: number; source: PlaceSource }): RememberResult {
    const name = normalizePlaceName(input.name);
    if (!PLACE_NAME_PATTERN.test(name)) {
      return { ok: false, reason: 'A place name has 1 to 40 letters, digits, spaces, hyphens, underscores or apostrophes and starts with a letter or digit.' };
    }
    if (!Number.isFinite(input.x) || !Number.isFinite(input.y)) return { ok: false, reason: 'The position is not a finite number.' };
    const id = placeId(input.mapId, name);
    const replaced = this.#places.has(id);
    if (!replaced && this.places(input.mapId).length >= MAX_PLACES_PER_MAP) {
      return { ok: false, reason: `${MAX_PLACES_PER_MAP} places are already remembered on this map; forget one first.` };
    }
    const place: PlaceRecord = { id, kind: 'place', mapId: input.mapId, name, x: input.x, y: input.y, source: input.source, savedAt: this.#now() };
    this.#places.set(id, place);
    this.#changed(place);
    return { ok: true, place, replaced };
  }

  forgetPlace(mapId: string, name: string): PlaceRecord | null {
    const id = placeId(mapId, name);
    const p = this.#places.get(id);
    if (!p) return null;
    this.#places.delete(id);
    this.#removed(id);
    return p;
  }

  skillRuns(skillId: string): SkillRunsRecord | null {
    return this.#runs.get(skillRunsId(skillId)) ?? null;
  }

  /** Adds one live run of a skill, as measured by the referee. */
  recordSkillRun(skillId: string, morphologyHash: string, run: { outcome: RunOutcome; holdS: number; requiredHoldS: number; failures: readonly string[] }): SkillRunsRecord {
    const id = skillRunsId(skillId);
    const prev = this.#runs.get(id);
    const early = run.outcome === 'interrupted' || run.outcome === 'faulted';
    const rec: SkillRunsRecord = {
      id,
      kind: 'skill-runs',
      skillId,
      morphologyHash,
      completed: (prev?.completed ?? 0) + (early ? 0 : 1),
      succeeded: (prev?.succeeded ?? 0) + (run.outcome === 'succeeded' ? 1 : 0),
      interrupted: (prev?.interrupted ?? 0) + (early ? 1 : 0),
      last: {
        outcome: run.outcome,
        holdS: Number.isFinite(run.holdS) && run.holdS >= 0 ? run.holdS : null,
        requiredHoldS: run.requiredHoldS,
        failures: run.failures.slice(0, 10).map((f) => f.slice(0, 60)),
        at: this.#now(),
      },
    };
    this.#runs.set(id, rec);
    this.#changed(rec);
    return rec;
  }

  forgetSkillRuns(skillId: string): boolean {
    const id = skillRunsId(skillId);
    if (!this.#runs.delete(id)) return false;
    this.#removed(id);
    return true;
  }

  #changed(r: MemoryRecord): void {
    this.#notify();
    this.#persistence?.put(r).then(
      () => this.#clearError(),
      (e: unknown) => this.#fail(`Could not save memory: ${errorText(e)}`),
    );
  }

  #removed(id: string): void {
    this.#notify();
    this.#persistence?.remove(id).then(
      () => this.#clearError(),
      (e: unknown) => this.#fail(`Could not delete from memory: ${errorText(e)}`),
    );
  }

  #clearError(): void {
    if (this.#error === null) return;
    this.#error = null;
    this.#notify();
  }

  #fail(message: string): void {
    this.#error = message;
    this.#notify();
  }

  #notify(): void {
    this.#version++;
    this.#listeners.forEach((l) => l());
  }
}
