import type { LongTermMemory, RememberResult } from '../../ai/memory';
import type { NavSession } from './session';

/**
 * Remembers a place on the current map: the given position (checked
 * against the arena and obstacles), or where the controller believes the
 * robot is now, labeled with that source.
 */
export function rememberPlace(n: NavSession, memory: LongTermMemory, name: string, at?: { x: number; y: number }): RememberResult {
  const mapId = n.state.mapId;
  if (at) {
    if (!n.isFreeFloor(at.x, at.y)) return { ok: false, reason: 'That position is outside the arena or too close to an obstacle.' };
    return memory.rememberPlace({ mapId, name, x: at.x, y: at.y, source: 'given' });
  }
  const p = n.believedPosition();
  if (!p) return { ok: false, reason: 'The robot position is not known yet.' };
  return memory.rememberPlace({ mapId, name, x: p.x, y: p.y, source: p.source });
}
