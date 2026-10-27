import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

test.describe('M8 onboarding, learn mode and attempt comparison', () => {
  test('view switches clear old output and discard a late AI answer', async ({ page }) => {
    await page.route('**/api/ai/status', (route) => route.fulfill({ json: { ok: true, provider: 'test', capabilities: null, reason: null } }));
    let release: () => void = () => {};
    const delayed = new Promise<void>((resolve) => { release = resolve; });
    let requested = false;
    await page.route('**/api/ai/chat', async (route) => {
      requested = true;
      await delayed;
      await route.fulfill({ json: { content: 'Old view answer', toolCalls: [], finishReason: 'stop', model: 'test', usage: null } });
    });
    await page.goto('./');
    await page.getByTestId('command-stop').click();
    await expect(page.getByTestId('stop-message')).toBeVisible();
    await page.getByTestId('nav-kinematics').click();
    await expect(page.getByTestId('stop-message')).toHaveCount(0);
    await page.getByTestId('command-input').fill('what is the current robot state?');
    await page.getByTestId('command-run').click();
    await expect.poll(() => requested).toBe(true);
    await page.getByTestId('nav-build').click();
    release();
    await expect(page.getByTestId('command-run')).toHaveText('Run');
    await expect(page.getByTestId('command-output')).not.toContainText('Old view answer');
    await expect(page.getByTestId('command-kind')).toHaveCount(0);
  });

  test('repeated viewport replacement does not exhaust WebGL contexts', async ({ page }) => {
    const warnings: string[] = [];
    page.on('console', (message) => {
      if (/Too many active WebGL contexts/.test(message.text())) warnings.push(message.text());
    });
    await page.goto('./');
    for (let cycle = 0; cycle < 12; cycle++) {
      for (const view of ['build', 'kinematics']) {
        await page.getByTestId(`nav-${view}`).click();
        await expect(page.getByTestId('viewport').locator('canvas')).toBeVisible();
      }
    }
    expect(warnings).toEqual([]);
    await expect(page.locator('canvas')).toHaveCount(1);
  });

  test('the guide opens modes, hides on request and stays hidden after a reload', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('./');
    const guide = page.getByTestId('guide');
    await expect(guide).toContainText('Getting started');
    await guide.getByRole('button', { name: 'Go to Control lab' }).click();
    await expect(page.getByTestId('nav-control')).toHaveAttribute('aria-current', 'page');
    await guide.getByRole('button', { name: 'Hide guide' }).click();
    await expect(guide).toHaveCount(0);
    await expect(page.getByTestId('guide-toggle')).toHaveAttribute('aria-pressed', 'false');
    await page.reload();
    await expect(page.getByTestId('nav-skills')).toHaveAttribute('aria-current', 'page');
    await expect(page.getByTestId('guide')).toHaveCount(0);
    await page.getByTestId('guide-toggle').click();
    await expect(page.getByTestId('guide')).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('learn mode adds explanations and is remembered after a reload', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByTestId('explain')).toHaveCount(0);
    await page.getByTestId('learn-toggle').click();
    await expect(page.getByTestId('learn-toggle')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('explain').first()).toContainText('teaching by demonstration');
    await page.reload();
    await expect(page.getByTestId('learn-toggle')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByTestId('explain').first()).toBeVisible();
    await page.getByTestId('nav-navigate').click();
    await expect(page.getByRole('note', { name: 'Explanation' }).first()).toContainText('Odometry integrates');
  });

  test('two recorded attempts replay side by side on one timeline @smoke', async ({ page }) => {
    test.setTimeout(240_000);
    const errors = collectErrors(page);
    await page.goto('./');
    await page.getByTestId('skills-panel-foot-lift').click();
    await page.getByTestId('evaluate-state-machine').click();
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('skill-card')).toHaveCount(1, { timeout: 90_000 });
    for (let k = 1; k <= 2; k++) {
      await page.getByTestId('pg-view-library').click();
      await page.getByTestId('skill-play').click();
      await page.getByTestId('pg-view-attempts').click();
      await expect(page.getByTestId('attempts').locator('li')).toHaveCount(k, { timeout: 60_000 });
    }
    await expect(page.getByTestId('attempts')).toContainText('Attempt 2: Lift front left leg (state-machine): hold');
    await page.getByTestId('compare-open').click();
    const view = page.getByTestId('compare-view');
    await expect(view).toContainText('Replay of recorded simulator states, display only: nothing is simulated.');
    await expect(page.getByTestId('compare-left')).toContainText('A: Attempt 1');
    await expect(page.getByTestId('compare-right')).toContainText('B: Attempt 2');
    const clock = page.getByTestId('compare-clock');
    const seconds = async () => Number.parseFloat((await clock.textContent())!);
    await expect.poll(seconds).toBeGreaterThan(0.5);
    await page.getByTestId('compare-play').click();
    await expect(page.getByTestId('compare-play')).toHaveText('Play');
    await page.getByTestId('compare-time').fill('1.5');
    await expect(clock).toHaveText(/^1\.50 \/ \d+\.\d\d s$/);
    const png = await page.getByTestId('compare-plot').screenshot();
    expect(png.length).toBeGreaterThan(5_000);
    await page.getByTestId('compare-close').click();
    await expect(page.getByText('Skills: live MuJoCo simulation.', { exact: false })).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('the navigation goal can be typed; goals inside obstacles are refused and the field reverts', async ({ page }) => {
    await page.goto('./');
    await page.getByTestId('nav-navigate').click();
    await expect(page.getByTestId('nav-go')).toBeEnabled({ timeout: 30_000 });
    const x = page.getByTestId('nav-goal-x');
    await x.fill('2.5');
    await x.press('Enter');
    await expect(page.getByText('(2.50, 0.00) m')).toBeVisible();
    // The single-block map has a box around (1.5, 0).
    await x.fill('1.5');
    await x.press('Enter');
    await expect(page.getByTestId('nav-message')).toHaveText('That goal is outside the arena or too close to an obstacle.');
    await expect(x).toHaveValue('2.50');
  });
});
