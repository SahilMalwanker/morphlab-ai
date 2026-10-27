import { expect, test } from '@playwright/test';

/**
 * Load-time measurements of the production build. Budgets are loose (they
 * catch regressions such as the physics engine landing in the first bundle);
 * the measured numbers are printed for docs/status.md.
 */
test.describe('M8 performance', () => {
  test('first view, script size, playground and physics start-up stay within budget @smoke', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByTestId('viewport').locator('canvas')).toBeVisible();
    const load = await page.evaluate(() => {
      const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      const scripts = performance.getEntriesByType('resource').filter((r) => /\.js(\?|$)/.test(r.name)) as PerformanceResourceTiming[];
      return {
        firstCanvasMs: Math.round(performance.now()),
        domContentLoadedMs: Math.round(nav.domContentLoadedEventEnd),
        scriptBytes: scripts.reduce((n, r) => n + r.decodedBodySize, 0),
      };
    });
    // The AI Playground opens first and loads the engine on demand for its live robot (D34).
    await expect(page.getByTestId('pg-engine')).toHaveText('Live simulation', { timeout: 30_000 });
    const playgroundReadyMs = await page.evaluate(() => Math.round(performance.now()));
    const wasmRequested = await page.evaluate(() => performance.getEntriesByType('resource').some((r) => r.name.endsWith('.wasm')));
    const t0 = Date.now();
    await page.getByTestId('nav-physics').click();
    await expect(page.getByTestId('sim-status')).toHaveText('Running', { timeout: 30_000 });
    const physicsReadyMs = Date.now() - t0;
    console.log(`perf ${JSON.stringify({ ...load, playgroundReadyMs, physicsReadyMs })}`);
    expect(wasmRequested).toBe(true);
    expect(load.scriptBytes).toBeLessThan(1_200_000);
    expect(load.firstCanvasMs).toBeLessThan(8_000);
    expect(playgroundReadyMs).toBeLessThan(20_000);
    expect(physicsReadyMs).toBeLessThan(20_000);
  });
});
