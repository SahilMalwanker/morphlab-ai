/** Keys that drive the rover (KeyboardEvent.code values, so W A S D follow key positions on any layout). */
export const DRIVE_KEYS = { forward: ['KeyW', 'ArrowUp'], back: ['KeyS', 'ArrowDown'], left: ['KeyA', 'ArrowLeft'], right: ['KeyD', 'ArrowRight'] };

/** Focused elements that take typed text: no key drives while one has focus. */
export const TEXT_OWNERS =
  'input:not([type="range"]):not([type="checkbox"]):not([type="radio"]):not([type="button"]):not([type="submit"]):not([type="reset"]), select, textarea, [contenteditable=""], [contenteditable="true"], [role="spinbutton"], [role="listbox"], [role="menu"], [role="combobox"]';

/** Focused elements that use the arrow keys themselves: W A S D still drive there. */
export const ARROW_OWNERS = 'input[type="range"], input[type="radio"], [role="slider"], [role="radiogroup"], [role="tab"], [role="tablist"]';

const ARROWS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);

interface KeyLike {
  code: string;
  target: unknown;
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
}

/**
 * Whether a key event drives the rover rather than the focused control or a
 * browser shortcut. W A S D work everywhere except in text entry, so a
 * keyboard-only user can drive right after opening the tab from the tab list.
 */
export function isDrivingKey(e: KeyLike): boolean {
  if (e.ctrlKey || e.metaKey || e.altKey) return false;
  if (!Object.values(DRIVE_KEYS).flat().includes(e.code)) return false;
  const t = e.target as { closest?: (selector: string) => unknown } | null;
  const inside = (selector: string) => !!t && typeof t.closest === 'function' && !!t.closest(selector);
  if (inside(TEXT_OWNERS)) return false;
  return !(ARROWS.has(e.code) && inside(ARROW_OWNERS));
}
