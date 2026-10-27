import { describe, expect, it } from 'vitest';
import { ARROW_OWNERS, isDrivingKey, TEXT_OWNERS } from '../../src/ui/nav/keys';

/** A stand-in for a focused element: `closest` matches only the given selector list. */
function inside(list: string) {
  return { closest: (selector: string) => (selector === list ? {} : null) };
}
const button = { closest: () => null };

const key = (code: string, target: unknown = null, mods: Partial<{ ctrlKey: boolean; metaKey: boolean; altKey: boolean }> = {}) => ({ code, target, ctrlKey: false, metaKey: false, altKey: false, ...mods });

describe('keyboard driving keys', () => {
  it('drives with arrows and W A S D from the page or a button', () => {
    for (const c of ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']) {
      expect(isDrivingKey(key(c))).toBe(true);
      expect(isDrivingKey(key(c, button))).toBe(true);
    }
    expect(isDrivingKey(key('KeyQ'))).toBe(false);
    expect(isDrivingKey(key('Space'))).toBe(false);
  });

  it('nothing drives while text is being typed', () => {
    for (const c of ['KeyW', 'KeyA', 'ArrowUp', 'ArrowLeft']) expect(isDrivingKey(key(c, inside(TEXT_OWNERS)))).toBe(false);
  });

  it('on a focused tab or slider the arrows keep their meaning and W A S D still drive', () => {
    const tab = inside(ARROW_OWNERS);
    for (const c of ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']) expect(isDrivingKey(key(c, tab))).toBe(false);
    for (const c of ['KeyW', 'KeyA', 'KeyS', 'KeyD']) expect(isDrivingKey(key(c, tab))).toBe(true);
  });

  it('shortcuts with modifier keys are left to the browser', () => {
    expect(isDrivingKey(key('KeyS', null, { ctrlKey: true }))).toBe(false);
    expect(isDrivingKey(key('ArrowLeft', null, { altKey: true }))).toBe(false);
    expect(isDrivingKey(key('KeyW', null, { metaKey: true }))).toBe(false);
  });
});
