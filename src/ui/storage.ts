import { MorphStore, type CorruptRecord, type StoredRobot } from '../persistence/db';
import type { MemoryPersistence, MemoryRecord } from '../ai/memory';
import type { RobotDescription } from '../core/model/types';
import type { SkillArtifact } from '../skills/artifact';
import type { LocomotionController } from '../skills/locomotion';
import { Store } from './store';

export interface StorageState {
  status: 'opening' | 'ready' | 'unavailable';
  error: string | null;
  robots: StoredRobot[];
  corrupted: CorruptRecord[];
  /** Problems found while restoring the last workspace. Nothing was deleted because of them. */
  notices: string[];
}

export interface RestoredWorkspace {
  builderRobot: RobotDescription | null;
  skillsRobot: RobotDescription | null;
  skills: SkillArtifact[];
  /** Walking controllers, one per morphology. */
  locomotion: LocomotionController[];
  /** Remembered places and skill run records. */
  memories: MemoryRecord[];
  prefs: UiPrefs;
}

/** Interface preferences kept in local storage. */
export interface UiPrefs {
  guideHidden: boolean;
  learnMode: boolean;
  /** Read command results aloud with a local voice. */
  speakReplies: boolean;
  /** The user agreed that voice input goes to the browser's speech service. */
  voiceInputConsent: boolean;
  /** The AI Playground side panel is hidden, giving the robot view the full width. */
  playgroundPanelCollapsed: boolean;
}

export const DEFAULT_PREFS: UiPrefs = { guideHidden: false, learnMode: false, speakReplies: false, voiceInputConsent: false, playgroundPanelCollapsed: false };

/** Reads stored preferences; anything malformed falls back to the defaults. */
export function parsePrefs(raw: unknown): UiPrefs {
  const o = typeof raw === 'object' && raw !== null ? (raw as Record<string, unknown>) : {};
  const flag = (k: keyof UiPrefs): boolean => (typeof o[k] === 'boolean' ? (o[k] as boolean) : DEFAULT_PREFS[k]);
  return { guideHidden: flag('guideHidden'), learnMode: flag('learnMode'), speakReplies: flag('speakReplies'), voiceInputConsent: flag('voiceInputConsent'), playgroundPanelCollapsed: flag('playgroundPanelCollapsed') };
}

/** Settings keys for the robots last open in each workspace (values are [id, hash]). */
const BUILDER_KEY = 'workspace.builderRobot';
const SKILLS_KEY = 'workspace.skillsRobot';
const PREFS_KEY = 'ui.preferences';

/** A stored [robot id, morphology hash] pointer; anything else is ignored and reported. */
export function isRobotPointer(v: unknown): v is [string, string] {
  return Array.isArray(v) && v.length === 2 && typeof v[0] === 'string' && v[0].length > 0 && v[0].length <= 200 && typeof v[1] === 'string' && /^[0-9a-f]{64}$/.test(v[1]);
}

function withTimeout<T>(p: Promise<T>, ms: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return Promise.race([p, new Promise<never>((_, reject) => (timer = setTimeout(() => reject(new Error(message)), ms)))]).finally(() => clearTimeout(timer));
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * App-level storage: opens IndexedDB, restores the last workspace, and keeps
 * a list of saved robot versions and of records that could not be read.
 * When storage is unavailable the app keeps working in memory and says so;
 * when the saved workspace cannot be read it starts fresh and says so.
 */
export class AppStorage {
  readonly store = new Store<StorageState>({ status: 'opening', error: null, robots: [], corrupted: [], notices: [] });
  private db: MorphStore | null = null;
  private opening: Promise<RestoredWorkspace> | null = null;

  constructor(private readonly dbName?: string, private readonly timeoutMs = 4000) {}

  get state(): StorageState {
    return this.store.get();
  }

  /** Opens storage once and returns what to restore; never throws. */
  open(): Promise<RestoredWorkspace> {
    this.opening ??= this.doOpen();
    return this.opening;
  }

  private async doOpen(): Promise<RestoredWorkspace> {
    const empty: RestoredWorkspace = { builderRobot: null, skillsRobot: null, skills: [], locomotion: [], memories: [], prefs: { ...DEFAULT_PREFS } };
    const opening = MorphStore.open(this.dbName);
    try {
      this.db = await withTimeout(opening, this.timeoutMs, 'Opening local storage timed out.');
    } catch (e) {
      // A database that opens after the timeout is closed again, so it cannot block later upgrades.
      opening.then((late) => late.close(), () => {});
      this.store.update({ status: 'unavailable', error: `${errorText(e)} Robots and skills will not survive a reload; export them to keep them.` });
      return empty;
    }
    try {
      return await withTimeout(this.restore(this.db), this.timeoutMs, 'Reading the saved workspace timed out.');
    } catch (e) {
      this.store.update({
        status: 'ready',
        notices: [...this.state.notices, `The saved workspace could not be restored (${errorText(e)}). The app started with the default robots; stored records were not changed.`],
      });
      return empty;
    }
  }

  private async restore(db: MorphStore): Promise<RestoredWorkspace> {
    const notices: string[] = [];
    const skills = await db.loadSkills();
    const memories = await db.loadMemories();
    const locomotion = await db.loadLocomotion();
    await this.refreshRobots([...skills.corrupted, ...memories.corrupted, ...locomotion.corrupted]);
    const pick = async (key: string, where: string): Promise<RobotDescription | null> => {
      const k = await db.getSetting<unknown>(key);
      if (k === undefined || k === null) return null;
      if (!isRobotPointer(k)) {
        notices.push(`The record of the robot last open in ${where} is malformed and was ignored; the default robot is shown.`);
        return null;
      }
      const robot = await db.getRobot(k[0], k[1]);
      if (!robot) notices.push(`The robot last open in ${where} (${k[0]}, morphology ${k[1].slice(0, 12)}) is missing or unreadable; the default robot is shown.`);
      return robot;
    };
    const builderRobot = await pick(BUILDER_KEY, 'Build');
    const skillsRobot = await pick(SKILLS_KEY, 'Skills');
    const prefs = parsePrefs(await db.getSetting(PREFS_KEY));
    this.store.update({ status: 'ready', notices: [...this.state.notices, ...notices] });
    return { builderRobot, skillsRobot, skills: skills.records, locomotion: locomotion.records, memories: memories.records, prefs };
  }

  dismissNotices(): void {
    this.store.update({ notices: [] });
  }

  /** Saves interface preferences; without storage they last until the page closes. */
  async savePrefs(p: UiPrefs): Promise<void> {
    await this.db?.setSetting(PREFS_KEY, p);
  }

  private async refreshRobots(extraCorrupted: CorruptRecord[] = this.state.corrupted.filter((c) => c.store !== 'robots')): Promise<void> {
    if (!this.db) return;
    const robots = await this.db.loadRobots();
    this.store.update({ robots: robots.records, corrupted: [...robots.corrupted, ...extraCorrupted] });
  }

  private async saveRobotAs(key: string, robot: RobotDescription): Promise<void> {
    if (!this.db) return;
    const rec = await this.db.saveRobot(robot);
    await this.db.setSetting(key, [rec.id, rec.morphologyHash]);
    await this.refreshRobots();
  }

  /** Hooks for the builder workspace. */
  builderHooks() {
    return { saveRobot: (r: RobotDescription) => this.saveRobotAs(BUILDER_KEY, r) };
  }

  /** Hooks for the skills workspace. */
  skillsHooks() {
    return {
      saveRobot: (r: RobotDescription) => this.saveRobotAs(SKILLS_KEY, r),
      storeRobot: async (r: RobotDescription) => {
        if (!this.db) return;
        await this.db.saveRobot(r);
        await this.refreshRobots();
      },
      findRobot: (id: string, hash: string) => this.state.robots.find((x) => x.id === id && x.morphologyHash === hash)?.robot ?? this.state.robots.find((x) => x.morphologyHash === hash)?.robot ?? null,
      saveSkill: async (a: SkillArtifact) => {
        await this.db?.saveSkill(a);
      },
      deleteSkill: async (id: string) => {
        await this.db?.deleteSkill(id);
      },
      saveLocomotion: async (c: LocomotionController) => {
        await this.db?.saveLocomotion(c);
      },
      deleteLocomotion: async (morphologyHash: string) => {
        await this.db?.deleteLocomotion(morphologyHash);
      },
    };
  }

  async deleteRobot(id: string, hash: string): Promise<void> {
    await this.db?.deleteRobot(id, hash);
    await this.refreshRobots();
  }

  /** Hooks for long-term memory (places and skill run records). */
  memoryHooks(): MemoryPersistence {
    return {
      put: async (r) => {
        await this.db?.putMemory(r);
      },
      remove: async (id) => {
        await this.db?.deleteMemory(id);
      },
    };
  }

  async removeCorrupted(rec: CorruptRecord): Promise<void> {
    await this.db?.removeRecord(rec);
    this.store.update({ corrupted: this.state.corrupted.filter((c) => !(c.store === rec.store && c.key === rec.key)) });
  }
}
