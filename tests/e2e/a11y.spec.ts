import { expect, test, type Page } from '@playwright/test';

const MODES = ['AI Playground', 'Build', 'Kinematics', 'Physics', 'Control lab', 'Navigate'] as const;
type ModeName = (typeof MODES)[number];

function navButton(page: Page, name: ModeName) {
  return page.getByRole('navigation', { name: 'Workspace' }).getByRole('button', { name, exact: true });
}

/** Opens a view from the navigation with the keyboard and waits until its controls are in their ready state. */
async function openMode(page: Page, name: ModeName) {
  const button = navButton(page, name);
  await button.focus();
  await page.keyboard.press('Enter');
  await expect(button).toHaveAttribute('aria-current', 'page');
  const ready: Record<ModeName, () => Promise<void>> = {
    'AI Playground': () => expect(page.getByTestId('teach-new')).toBeEnabled({ timeout: 30_000 }),
    Build: () => expect(page.getByTestId('morph-hash')).toHaveText(/^[0-9a-f]{12}$/),
    Kinematics: () => expect(page.getByTestId('viewport').locator('canvas')).toBeVisible(),
    Physics: () => expect(page.getByTestId('sim-status')).toHaveText('Running', { timeout: 30_000 }),
    'Control lab': () => expect(page.getByTestId('run-step')).toBeEnabled({ timeout: 30_000 }),
    Navigate: () => expect(page.getByTestId('nav-go')).toBeEnabled({ timeout: 30_000 }),
  };
  await ready[name]();
}

/** Marks every keyboard-focusable, visible element and returns their count. */
function markFocusable(page: Page): Promise<number> {
  return page.evaluate(() => {
    document.querySelectorAll<HTMLElement>('[data-focus-id]').forEach((el) => delete el.dataset.focusId);
    const els = Array.from(document.querySelectorAll<HTMLElement>('a[href], button, input, select, textarea, [tabindex]')).filter((el) => {
      if ((el as HTMLButtonElement).disabled || el.getAttribute('tabindex') === '-1' || el.closest('[hidden], [inert]')) return false;
      if (el instanceof HTMLInputElement && el.type === 'hidden') return false;
      return el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
    });
    els.forEach((el, i) => (el.dataset.focusId = String(i)));
    return els.length;
  });
}

/** WCAG contrast of every visible text node against its composited background; returns failures below 4.5:1. */
function lowContrastText(page: Page) {
  return page.evaluate(() => {
    type C = { r: number; g: number; b: number; a: number };
    const parse = (c: string): C | null => {
      const m = /rgba?\(([^)]+)\)/.exec(c);
      if (!m) return null;
      const p = m[1]!.split(/[\s,/]+/).filter(Boolean).map(Number);
      return { r: p[0]!, g: p[1]!, b: p[2]!, a: p.length > 3 ? p[3]! : 1 };
    };
    const blend = (top: C, bottom: C): C => ({ r: top.r * top.a + bottom.r * (1 - top.a), g: top.g * top.a + bottom.g * (1 - top.a), b: top.b * top.a + bottom.b * (1 - top.a), a: 1 });
    const lum = (c: C) => {
      const f = (v: number) => ((v /= 255) <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
      return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
    };
    const background = (el: Element): C => {
      const layers: C[] = [];
      for (let e: Element | null = el; e; e = e.parentElement) {
        const c = parse(getComputedStyle(e).backgroundColor);
        if (c && c.a > 0) {
          layers.push(c);
          if (c.a >= 1) break;
        }
      }
      let color: C = parse(getComputedStyle(document.body).backgroundColor) ?? { r: 0, g: 0, b: 0, a: 1 };
      for (let i = layers.length - 1; i >= 0; i--) color = blend(layers[i]!, color);
      return color;
    };
    const failures: { text: string; ratio: number }[] = [];
    const seen = new Set<Element>();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const el = n.parentElement;
      if (!el || !n.textContent?.trim() || seen.has(el)) continue;
      seen.add(el);
      if (!el.getClientRects().length || getComputedStyle(el).visibility === 'hidden') continue;
      // Text of inactive controls is exempt (WCAG 1.4.3).
      if (el.closest(':disabled')) continue;
      let opacity = 1;
      for (let e: Element | null = el; e; e = e.parentElement) opacity *= Number(getComputedStyle(e).opacity);
      const bg = background(el);
      const fg0 = parse(getComputedStyle(el).color)!;
      const fg = blend({ ...fg0, a: fg0.a * opacity }, bg);
      const [hi, lo] = [lum(fg), lum(bg)].sort((x, y) => y - x) as [number, number];
      const ratio = (hi + 0.05) / (lo + 0.05);
      if (ratio < 4.5) failures.push({ text: n.textContent.trim().slice(0, 60), ratio: +ratio.toFixed(2) });
    }
    return failures;
  });
}

test.describe('M8 accessibility', () => {
  test('the navigation puts the AI Playground first, groups the lab tools and marks the open view @smoke', async ({ page }) => {
    await page.goto('./');
    const nav = page.getByRole('navigation', { name: 'Workspace' });
    await expect(page.getByRole('tab')).toHaveCount(0);
    await expect(navButton(page, 'AI Playground')).toHaveAttribute('aria-current', 'page');
    await expect(page.getByRole('main', { name: 'AI Playground' })).toBeVisible();
    // Above 1000 px the list is always shown, so the Menu disclosure is not.
    await expect(page.getByTestId('nav-menu')).toBeHidden();
    await expect(nav.getByRole('list', { name: 'Lab tools' }).getByRole('button')).toHaveText(['Build', 'Kinematics', 'Physics', 'Control lab', 'Navigate']);
    const physics = navButton(page, 'Physics');
    await physics.focus();
    await page.keyboard.press('Enter');
    await expect(physics).toHaveAttribute('aria-current', 'page');
    await expect(physics).toBeFocused();
    expect(await navButton(page, 'AI Playground').getAttribute('aria-current')).toBeNull();
    await expect(page.getByRole('main', { name: 'Physics' })).toBeVisible();
    const back = page.getByTestId('back-to-playground');
    await back.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('main', { name: 'AI Playground' })).toBeVisible();
    await expect(navButton(page, 'AI Playground')).toHaveAttribute('aria-current', 'page');
    // The Back button leaves with the lab view; focus moves to the Playground instead of the page body.
    await expect(page.getByRole('main', { name: 'AI Playground' })).toBeFocused();
  });

  test('every control in every mode is reached with Tab and shows a visible focus indicator @smoke', async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto('./');
    for (const mode of MODES) {
      await openMode(page, mode);
      const count = await markFocusable(page);
      // Start from a temporary sentinel at the top of the page: after blur() Firefox continues from
      // the blurred element, Chromium from the start, so a sentinel makes both start at the top.
      await page.evaluate(() => {
        const s = document.createElement('div');
        s.tabIndex = 0;
        s.id = 'tab-order-sentinel';
        document.body.prepend(s);
        s.focus();
        s.addEventListener('blur', () => s.remove(), { once: true });
      });
      const reached = new Set<string>();
      const noIndicator: string[] = [];
      for (let i = 0; i < count + 15 && reached.size < count; i++) {
        await page.keyboard.press('Tab');
        const f = await page.evaluate(() => {
          const el = document.activeElement as HTMLElement | null;
          if (!el || el === document.body) return null;
          const cs = getComputedStyle(el);
          return { id: el.dataset.focusId ?? null, name: `${el.tagName.toLowerCase()} ${(el.getAttribute('aria-label') ?? el.textContent ?? '').trim().slice(0, 40)}`, visible: cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) >= 2 };
        });
        if (!f) continue;
        if (f.id !== null) reached.add(f.id);
        if (!f.visible) noIndicator.push(f.name);
      }
      expect.soft(reached.size, `${mode}: focusable elements reached with Tab`).toBe(count);
      if (reached.size !== count) {
        const missed = await page.evaluate((ids) => Array.from(document.querySelectorAll<HTMLElement>('[data-focus-id]')).filter((el) => !ids.includes(el.dataset.focusId!)).map((el) => el.outerHTML.slice(0, 160)), [...reached]);
        expect.soft(missed, `${mode}: elements never reached with Tab`).toEqual([]);
      }
      expect.soft(noIndicator, `${mode}: focused elements without a visible indicator`).toEqual([]);
      // Moving through fields without typing must not edit the robot.
      if (mode === 'Build') await expect(page.getByTestId('morph-version')).toHaveText('1');
      // Leave the view from the keyboard as well, through its navigation entry.
      await navButton(page, mode).focus();
    }
  });

  test('text contrast is at least 4.5:1 in every mode, with the guide and learn mode on @smoke', async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto('./');
    await page.getByTestId('learn-toggle').click();
    await expect(page.getByTestId('guide')).toBeVisible();
    for (const mode of MODES) {
      await openMode(page, mode);
      await expect(page.getByTestId('explain').first()).toBeVisible();
      expect.soft(await lowContrastText(page), `${mode}: text below 4.5:1`).toEqual([]);
    }
  });

  test('narrow window: the menu reaches every view, closes with Escape and nothing scrolls sideways', async ({ page }) => {
    await page.setViewportSize({ width: 740, height: 900 });
    await page.goto('./');
    const menu = page.getByTestId('nav-menu');
    await expect(menu).toBeVisible();
    await expect(menu).toHaveAttribute('aria-expanded', 'false');
    await expect(navButton(page, 'Build')).toBeHidden();
    await menu.click();
    await expect(navButton(page, 'Build')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(menu).toHaveAttribute('aria-expanded', 'false');
    await expect(menu).toBeFocused();
    for (const mode of MODES) {
      await menu.click();
      await expect(menu).toHaveAttribute('aria-expanded', 'true');
      const button = navButton(page, mode);
      await button.click();
      await expect(button).toHaveAttribute('aria-current', 'page');
      await expect(menu).toHaveAttribute('aria-expanded', 'false');
      await expect(menu).toBeFocused();
      expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    }
  });

  test('short desktop: the robot view keeps its height and the command bar never overlaps it', async ({ page }) => {
    await page.setViewportSize({ width: 1100, height: 480 });
    await page.goto('./');
    const viewport = page.getByTestId('viewport');
    await expect(viewport.locator('canvas')).toBeVisible();
    const view = (await viewport.boundingBox())!;
    const command = (await page.getByRole('contentinfo', { name: 'Command bar' }).boundingBox())!;
    expect(view.height).toBeGreaterThanOrEqual(200);
    expect(command.y).toBeGreaterThanOrEqual(view.y + view.height - 1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  });

  test('phone-sized window: the command field and Stop task come before the Playground side panel', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('./');
    const side = page.getByRole('complementary', { name: 'Teach' });
    await expect(side).toBeVisible();
    // Visual order: the command row sits above the side panel, so a long teaching form is never scrolled past to reach it.
    const sideTop = (await side.boundingBox())!.y;
    expect((await page.getByTestId('command-input').boundingBox())!.y).toBeLessThan(sideTop);
    expect((await page.getByTestId('command-stop').boundingBox())!.y).toBeLessThan(sideTop);
    // Reading and Tab order match the visual order: the panel follows the command bar in the document.
    const panelFollows = await page.evaluate(() => {
      const stop = document.querySelector('[data-testid="command-stop"]');
      const aside = document.querySelector('aside.pg-side');
      return !!stop && !!aside && (stop.compareDocumentPosition(aside) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
    });
    expect(panelFollows).toBe(true);
  });
});
