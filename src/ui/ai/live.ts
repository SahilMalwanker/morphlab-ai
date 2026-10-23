import type { TaskManager } from '../../ai/tasks';
import { parseLiveCommand } from '../../language/live';
import type { NavSession } from '../nav/session';

export interface LiveTargets {
  tasks: TaskManager;
  /** The local Stop control; returns what it did. */
  stop: () => string;
  /** The rover, when the Navigate tab is open. */
  nav: NavSession | null;
}

/**
 * Applies a short live command (slower, faster, stop) to what is running now:
 * the active AI task, or a goal run started in the Navigate tab. Returns the
 * reply, or null when the text is not a live command for anything running.
 */
export function applyLiveCommand(text: string, t: LiveTargets): string | null {
  const c = parseLiveCommand(text);
  if (c.kind === 'unknown') return null;
  if (c.kind === 'stop') return t.stop();
  const active = t.tasks.active;
  if (active) {
    const r = t.tasks.modify(active.id, t.tasks.stateVersion, { change: c.change, amount: c.amount });
    return r.status === 'running' ? `${active.tool} (${active.id}): ${r.reason}` : `Not changed: ${r.reason}`;
  }
  const n = t.nav;
  if (n && n.state.driving === 'auto') {
    if (c.change !== 'slower' && c.change !== 'faster') return `Not changed: a navigation run can go slower or faster, not ${c.change}.`;
    const r = n.changeSpeed(c.change, c.amount);
    return r.ok ? r.description : `Not changed: ${r.reason}`;
  }
  return null;
}
