import { openDB, type IDBPDatabase } from 'idb';
import { parseMemoryRecord, type MemoryRecord } from '../ai/memory';
import { morphologyHash } from '../core/model/hash';
import { parseRobot } from '../core/model/parse';
import type { RobotDescription } from '../core/model/types';
import { parseSkillArtifact, type SkillArtifact } from '../skills/artifact';
import { parseLocomotionController, type LocomotionController } from '../skills/locomotion';

export const DB_NAME = 'morphlab';
/** Version 2 added the memories store; upgrades keep every existing record. */
export const DB_VERSION = 2;

type StoreName = 'robots' | 'skills' | 'settings' | 'memories';

export interface StoredRobot {
  id: string;
  morphologyHash: string;
  name: string;
  savedAt: string;
  robot: RobotDescription;
}

/** A stored record that could not be read back; it is kept and reported, never dropped silently. */
export interface CorruptRecord {
  store: StoreName;
  /** Display form of the record key. */
  key: string;
  /** The raw IndexedDB key, for removal. */
  rawKey: IDBValidKey;
  reasons: string[];
}

export interface LoadResult<T> {
  records: T[];
  corrupted: CorruptRecord[];
}

/** Settings must never hold credentials. */
const SECRET_KEY = /api.?key|token|secret|password|credential/i;
const LOCOMOTION_PREFIX = 'locomotion.';

function keyText(k: IDBValidKey): string {
  return Array.isArray(k) ? k.map((x) => String(x)).join(' / ') : String(k);
}

/**
 * IndexedDB storage for robots (one record per morphology version, keyed by
 * [id, morphologyHash]), skills (keyed by id, indexed by morphology hash),
 * settings and memories (places and skill run records, keyed by id). Every
 * record is validated when read; invalid records are reported with reasons
 * and left in place until the user removes them.
 */
export class MorphStore {
  private constructor(private readonly db: IDBPDatabase) {}

  static async open(name = DB_NAME): Promise<MorphStore> {
    if (typeof indexedDB === 'undefined') throw new Error('IndexedDB is not available in this browser.');
    const db = await openDB(name, DB_VERSION, {
      upgrade(d) {
        if (!d.objectStoreNames.contains('robots')) d.createObjectStore('robots', { keyPath: ['id', 'morphologyHash'] });
        if (!d.objectStoreNames.contains('skills')) d.createObjectStore('skills', { keyPath: 'id' }).createIndex('morphologyHash', 'robot.morphologyHash');
        if (!d.objectStoreNames.contains('settings')) d.createObjectStore('settings', { keyPath: 'key' });
        if (!d.objectStoreNames.contains('memories')) d.createObjectStore('memories', { keyPath: 'id' });
      },
    });
    return new MorphStore(db);
  }

  close(): void {
    this.db.close();
  }

  async saveRobot(robot: RobotDescription): Promise<StoredRobot> {
    const parsed = parseRobot(JSON.parse(JSON.stringify(robot)));
    if (!parsed.ok) throw new Error(`Refusing to store an invalid robot: ${parsed.errors[0]}`);
    const rec: StoredRobot = { id: robot.id, morphologyHash: await morphologyHash(robot), name: robot.name, savedAt: new Date().toISOString(), robot: parsed.value };
    await this.db.put('robots', rec);
    return rec;
  }

  private async readAll(store: StoreName): Promise<{ key: IDBValidKey; value: unknown }[]> {
    const out: { key: IDBValidKey; value: unknown }[] = [];
    let cursor = await this.db.transaction(store).store.openCursor();
    while (cursor) {
      out.push({ key: cursor.primaryKey, value: cursor.value });
      cursor = await cursor.continue();
    }
    return out;
  }

  async loadRobots(): Promise<LoadResult<StoredRobot>> {
    const records: StoredRobot[] = [];
    const corrupted: CorruptRecord[] = [];
    for (const { key, value } of await this.readAll('robots')) {
      const v = value as Partial<StoredRobot> | null;
      const parsed = parseRobot(v?.robot);
      if (!parsed.ok) {
        corrupted.push({ store: 'robots', key: keyText(key), rawKey: key, reasons: parsed.errors });
        continue;
      }
      const hash = await morphologyHash(parsed.value);
      if (hash !== v!.morphologyHash || parsed.value.id !== v!.id) {
        corrupted.push({ store: 'robots', key: keyText(key), rawKey: key, reasons: ['The stored morphology hash or id does not match the robot it contains.'] });
        continue;
      }
      records.push({ id: parsed.value.id, morphologyHash: hash, name: parsed.value.name, savedAt: typeof v!.savedAt === 'string' ? v!.savedAt : '', robot: parsed.value });
    }
    records.sort((a, b) => b.savedAt.localeCompare(a.savedAt));
    return { records, corrupted };
  }

  async getRobot(id: string, hash: string): Promise<RobotDescription | null> {
    const v = (await this.db.get('robots', [id, hash])) as Partial<StoredRobot> | undefined;
    if (!v) return null;
    const parsed = parseRobot(v.robot);
    return parsed.ok && (await morphologyHash(parsed.value)) === hash ? parsed.value : null;
  }

  async deleteRobot(id: string, hash: string): Promise<void> {
    await this.db.delete('robots', [id, hash]);
  }

  async saveSkill(a: SkillArtifact): Promise<void> {
    const parsed = parseSkillArtifact(JSON.parse(JSON.stringify(a)));
    if (!parsed.ok) throw new Error(`Refusing to store an invalid skill: ${parsed.errors[0]}`);
    await this.db.put('skills', parsed.value);
  }

  async loadSkills(): Promise<LoadResult<SkillArtifact>> {
    const records: SkillArtifact[] = [];
    const corrupted: CorruptRecord[] = [];
    for (const { key, value } of await this.readAll('skills')) {
      const parsed = parseSkillArtifact(value);
      if (parsed.ok) records.push(parsed.value);
      else corrupted.push({ store: 'skills', key: keyText(key), rawKey: key, reasons: parsed.errors });
    }
    records.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    return { records, corrupted };
  }

  async deleteSkill(id: string): Promise<void> {
    await this.db.delete('skills', id);
  }

  async loadMemories(): Promise<LoadResult<MemoryRecord>> {
    const records: MemoryRecord[] = [];
    const corrupted: CorruptRecord[] = [];
    for (const { key, value } of await this.readAll('memories')) {
      const parsed = parseMemoryRecord(value);
      if (parsed.ok) records.push(parsed.value);
      else corrupted.push({ store: 'memories', key: keyText(key), rawKey: key, reasons: parsed.errors });
    }
    return { records, corrupted };
  }

  async putMemory(r: MemoryRecord): Promise<void> {
    const parsed = parseMemoryRecord(JSON.parse(JSON.stringify(r)));
    if (!parsed.ok) throw new Error(`Refusing to store an invalid memory record: ${parsed.errors[0]}`);
    await this.db.put('memories', parsed.value);
  }

  async deleteMemory(id: string): Promise<void> {
    await this.db.delete('memories', id);
  }

  /** Walking controllers live in the settings store, one per morphology hash. */
  async saveLocomotion(c: LocomotionController): Promise<void> {
    const parsed = parseLocomotionController(JSON.parse(JSON.stringify(c)));
    if (!parsed.ok) throw new Error(`Refusing to store an invalid walking controller: ${parsed.errors[0]}`);
    await this.db.put('settings', { key: `${LOCOMOTION_PREFIX}${c.robot.morphologyHash}`, value: parsed.value });
  }

  async loadLocomotion(): Promise<LoadResult<LocomotionController>> {
    const records: LocomotionController[] = [];
    const corrupted: CorruptRecord[] = [];
    for (const { key, value } of await this.readAll('settings')) {
      if (typeof key !== 'string' || !key.startsWith(LOCOMOTION_PREFIX)) continue;
      const parsed = parseLocomotionController((value as { value?: unknown } | null)?.value);
      if (parsed.ok && `${LOCOMOTION_PREFIX}${parsed.value.robot.morphologyHash}` === key) records.push(parsed.value);
      else corrupted.push({ store: 'settings', key, rawKey: key, reasons: parsed.ok ? ['The key does not match the morphology the controller was trained on.'] : parsed.errors });
    }
    return { records, corrupted };
  }

  async deleteLocomotion(morphologyHash: string): Promise<void> {
    await this.db.delete('settings', `${LOCOMOTION_PREFIX}${morphologyHash}`);
  }

  /** Removes a record that was reported as corrupted. */
  async removeRecord(rec: CorruptRecord): Promise<void> {
    await this.db.delete(rec.store, rec.rawKey);
  }

  async getSetting<T>(key: string): Promise<T | undefined> {
    const v = (await this.db.get('settings', key)) as { key: string; value: T } | undefined;
    return v?.value;
  }

  async setSetting(key: string, value: unknown): Promise<void> {
    if (SECRET_KEY.test(key)) throw new Error('Credentials are never stored.');
    await this.db.put('settings', { key, value });
  }

  /** Every stored value as text, for audits (for example, checking that no key was persisted). */
  async dumpText(): Promise<string> {
    const parts: string[] = [];
    for (const s of ['robots', 'skills', 'settings', 'memories'] as const) for (const r of await this.readAll(s)) parts.push(JSON.stringify(r.value));
    return parts.join('\n');
  }

  /** Writes a raw value (tests use this to simulate corruption). */
  async putRaw(store: StoreName, value: unknown): Promise<void> {
    await this.db.put(store, value);
  }
}
