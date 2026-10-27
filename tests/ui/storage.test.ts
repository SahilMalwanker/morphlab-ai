import 'fake-indexeddb/auto';
import { describe, expect, it, vi } from 'vitest';
import type { MemoryRecord } from '../../src/ai/memory';
import { quadruped } from '../../src/core/model/presets';
import { MorphStore } from '../../src/persistence/db';
import { AppStorage, DEFAULT_PREFS, isRobotPointer, parsePrefs } from '../../src/ui/storage';

let n = 0;
const dbName = () => `morphlab-storage-test-${n++}`;

describe('app storage preferences', () => {
  it('malformed stored preferences fall back to the defaults field by field', () => {
    expect(parsePrefs(undefined)).toEqual(DEFAULT_PREFS);
    expect(parsePrefs('learn')).toEqual(DEFAULT_PREFS);
    expect(parsePrefs({ guideHidden: 'yes', learnMode: true })).toEqual({ ...DEFAULT_PREFS, guideHidden: false, learnMode: true });
    expect(parsePrefs({ guideHidden: true, learnMode: 1 })).toEqual({ ...DEFAULT_PREFS, guideHidden: true, learnMode: false });
    expect(parsePrefs({ speakReplies: true, voiceInputConsent: 'yes' })).toEqual({ ...DEFAULT_PREFS, speakReplies: true, voiceInputConsent: false });
    expect(DEFAULT_PREFS).toMatchObject({ speakReplies: false, voiceInputConsent: false });
  });

  it('the playground panel starts open and only a stored boolean collapses it', () => {
    expect(DEFAULT_PREFS.playgroundPanelCollapsed).toBe(false);
    expect(parsePrefs({ playgroundPanelCollapsed: true }).playgroundPanelCollapsed).toBe(true);
    expect(parsePrefs({ playgroundPanelCollapsed: 'true' }).playgroundPanelCollapsed).toBe(false);
    expect(parsePrefs({ playgroundPanelCollapsed: 1 }).playgroundPanelCollapsed).toBe(false);
    // Preferences stored before the field existed restore with the panel open.
    expect(parsePrefs({ guideHidden: true, learnMode: false })).toEqual({ ...DEFAULT_PREFS, guideHidden: true });
  });

  it('preferences survive reopening the database', async () => {
    const name = dbName();
    const first = new AppStorage(name);
    expect((await first.open()).prefs).toEqual(DEFAULT_PREFS);
    await first.savePrefs({ guideHidden: true, learnMode: true, speakReplies: true, voiceInputConsent: true, playgroundPanelCollapsed: true });
    const second = new AppStorage(name);
    expect((await second.open()).prefs).toEqual({ guideHidden: true, learnMode: true, speakReplies: true, voiceInputConsent: true, playgroundPanelCollapsed: true });
  });
});

describe('workspace restore', () => {
  it('validates robot pointers', () => {
    expect(isRobotPointer(['quadruped', 'a'.repeat(64)])).toBe(true);
    expect(isRobotPointer(['quadruped', 'A'.repeat(64)])).toBe(false);
    expect(isRobotPointer(['', 'a'.repeat(64)])).toBe(false);
    expect(isRobotPointer([1, 'a'.repeat(64)])).toBe(false);
    expect(isRobotPointer(['quadruped', { hash: 'x' }])).toBe(false);
    expect(isRobotPointer('quadruped')).toBe(false);
  });

  it('malformed or dangling pointers fall back to the defaults with a notice and touch nothing', async () => {
    const name = dbName();
    const db = await MorphStore.open(name);
    const q = quadruped();
    await db.saveRobot(q);
    await db.setSetting('workspace.builderRobot', ['quadruped', { not: 'a hash' }]);
    await db.setSetting('workspace.skillsRobot', ['gone', 'b'.repeat(64)]);
    db.close();
    const s = new AppStorage(name);
    const r = await s.open();
    expect(r.builderRobot).toBeNull();
    expect(r.skillsRobot).toBeNull();
    expect(s.state.status).toBe('ready');
    expect(s.state.notices).toEqual([
      'The record of the robot last open in Build is malformed and was ignored; the default robot is shown.',
      `The robot last open in Skills (gone, morphology ${'b'.repeat(12)}) is missing or unreadable; the default robot is shown.`,
    ]);
    // The saved robot is still there.
    expect(s.state.robots.map((x) => x.id)).toEqual(['quadruped']);
    s.dismissNotices();
    expect(s.state.notices).toEqual([]);
  });

  it('a restore that fails or hangs starts fresh with a notice instead of blocking the app', async () => {
    const name = dbName();
    const failing = new AppStorage(name);
    const spy = vi.spyOn(MorphStore.prototype, 'loadSkills').mockRejectedValueOnce(new Error('read error'));
    const r = await failing.open();
    spy.mockRestore();
    expect(r).toEqual({ builderRobot: null, skillsRobot: null, skills: [], locomotion: [], memories: [], prefs: DEFAULT_PREFS });
    expect(failing.state.status).toBe('ready');
    expect(failing.state.notices[0]).toBe('The saved workspace could not be restored (read error). The app started with the default robots; stored records were not changed.');

    const hanging = new AppStorage(dbName(), 50);
    const hang = vi.spyOn(MorphStore.prototype, 'loadSkills').mockReturnValueOnce(new Promise(() => {}));
    const h = await hanging.open();
    hang.mockRestore();
    expect(h.skills).toEqual([]);
    expect(hanging.state.notices[0]).toContain('Reading the saved workspace timed out.');
  });

  it('restores memory records, lists unreadable ones with the other corrupted records, and writes through its hooks', async () => {
    const name = dbName();
    const db = await MorphStore.open(name);
    const place: MemoryRecord = { id: 'place:open:dock', kind: 'place', mapId: 'open', name: 'dock', x: 1, y: 1.5, source: 'sensor-estimate', savedAt: '2026-10-04T12:00:00.000Z' };
    await db.putMemory(place);
    await db.putRaw('memories', { id: 'place:open:broken', kind: 'place' });
    db.close();
    const s = new AppStorage(name);
    const r = await s.open();
    expect(r.memories).toEqual([place]);
    expect(s.state.corrupted.map((c) => [c.store, c.key])).toEqual([['memories', 'place:open:broken']]);
    const hooks = s.memoryHooks();
    await hooks.put({ ...place, id: 'place:open:home', name: 'home' });
    await hooks.remove('place:open:dock');
    const again = await MorphStore.open(name);
    expect((await again.loadMemories()).records.map((x) => x.id)).toEqual(['place:open:home']);
    again.close();
  });
});
