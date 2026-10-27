import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function openNavigate(page: Page) {
  await page.goto('./');
  await page.getByTestId('nav-navigate').click();
  await expect(page.getByTestId('nav-go')).toBeEnabled({ timeout: 30_000 });
}

const truthX = async (page: Page) => Number.parseFloat((await page.getByTestId('nav-truth-x').textContent())!.replace('(', ''));

test.describe('M7 sensors and navigation', () => {
  test('leaving Navigate stops a panel-started run and idle Stop reports its scope', async ({ page }) => {
    await openNavigate(page);
    await page.getByTestId('command-stop').click();
    await expect(page.getByTestId('stop-message')).toContainText('No navigation run was active');
    await expect(page.getByTestId('nav-message')).toHaveCount(0);
    await page.getByTestId('nav-go').click();
    await expect(page.getByTestId('nav-driving')).not.toHaveText('stopped');
    await page.getByTestId('nav-kinematics').click();
    await page.getByTestId('nav-navigate').click();
    await expect(page.getByTestId('nav-driving')).toHaveText('stopped');
  });

  test('place feedback clears on Forget and map changes; legs walk to places', async ({ page }) => {
    await openNavigate(page);
    await page.getByTestId('place-name').fill('audit-dock');
    await page.getByTestId('place-remember').click();
    await expect(page.getByTestId('place-note')).toContainText('Remembered');
    await page.getByTestId('nav-robot').selectOption('quadruped');
    await expect(page.getByRole('button', { name: 'Walk here', exact: true })).toBeVisible();
    await page.getByTestId('place-forget').click();
    await expect(page.getByTestId('place-note')).toHaveCount(0);
    await page.getByTestId('place-name').fill('audit-dock');
    await page.getByTestId('place-remember').click();
    await expect(page.getByTestId('place-note')).toBeVisible();
    await page.getByTestId('nav-map').selectOption('open');
    await expect(page.getByTestId('place-note')).toHaveCount(0);
  });

  test('the sensor-only rover reaches the goal on at least 4 of the 5 test maps', async ({ page }) => {
    const errors = collectErrors(page);
    await openNavigate(page);
    await expect(page.getByText('It never receives simulator state.')).toBeVisible();
    await page.getByTestId('nav-evaluate').click();
    const summary = page.getByTestId('nav-summary');
    await expect(summary).toBeVisible({ timeout: 120_000 });
    await expect(summary).toContainText('(sensor-only)');
    const reached = Number.parseInt((await summary.textContent())!, 10);
    expect(reached).toBeGreaterThanOrEqual(4);
    await expect(page.getByTestId('nav-results').locator('tbody tr')).toHaveCount(5);
    expect(errors).toEqual([]);
  });

  test('drive to goal in the live view, then switch to the labeled ground-truth mode @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await openNavigate(page);
    await page.getByTestId('nav-map').selectOption('block');
    await page.getByTestId('nav-go').click();
    await expect(page.getByTestId('nav-message')).toContainText('Reached the goal', { timeout: 60_000 });
    expect(await truthX(page)).toBeGreaterThan(2.6);
    await page.getByTestId('nav-mode').selectOption('ground-truth');
    await expect(page.getByText('Use it as a reference, not as perception.')).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('keyboard remote control drives the rover and is ignored while typing a command', async ({ page }) => {
    const errors = collectErrors(page);
    await openNavigate(page);
    const x0 = await truthX(page);
    await page.getByTestId('nav-remote').click();
    await expect(page.getByTestId('nav-remote')).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.down('ArrowUp');
    await expect(page.getByTestId('nav-driving')).toHaveText('remote (keyboard)');
    await page.waitForTimeout(1500);
    await page.keyboard.up('ArrowUp');
    await expect.poll(() => truthX(page)).toBeGreaterThan(x0 + 0.2);

    // Keys typed into the command field must not move the rover.
    await page.waitForTimeout(800);
    const x1 = await truthX(page);
    await page.getByRole('textbox', { name: 'Command' }).click();
    await page.keyboard.type('wwwwwwwwww');
    await page.waitForTimeout(800);
    expect(Math.abs((await truthX(page)) - x1)).toBeLessThan(0.02);

    await page.getByTestId('nav-remote').click();
    await expect(page.getByTestId('nav-driving')).toHaveText('stopped');
    expect(errors).toEqual([]);
  });

  test('keyboard only: open Navigate from the navigation and drive with W A S D @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('./');
    const entry = page.getByRole('navigation', { name: 'Workspace' }).getByRole('button', { name: 'Navigate', exact: true });
    await entry.focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('nav-go')).toBeEnabled({ timeout: 30_000 });
    await expect(entry).toBeFocused();
    await expect(entry).toHaveAttribute('aria-current', 'page');
    const x0 = await truthX(page);
    await page.keyboard.down('KeyW');
    await expect(page.getByTestId('nav-driving')).toHaveText('remote (keyboard)');
    await page.waitForTimeout(1500);
    await page.keyboard.up('KeyW');
    await expect.poll(() => truthX(page)).toBeGreaterThan(x0 + 0.2);
    // The way back to the primary view is always one control away.
    await page.getByTestId('back-to-playground').click();
    await expect(page.getByRole('main', { name: 'AI Playground' })).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('the robot picker lists every preset, explains which cannot move, and takes a legged design from Build @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await openNavigate(page);
    const picker = page.getByTestId('nav-robot');
    await expect(picker).toHaveValue('rover');
    await expect(picker.locator('option')).toHaveText(['Three-joint arm (cannot drive)', 'Six-joint arm (cannot drive)', 'Planar two-link arm (cannot drive)', 'Quadruped', 'Hexapod', 'Wheeled rover']);
    await expect(picker.locator('option[disabled]')).toHaveCount(3);
    const reasons = page.getByTestId('nav-robot-reasons');
    await expect(reasons).toContainText('Three-joint arm, Six-joint arm, Planar two-link arm: its base is fixed in place');
    await expect(page.getByTestId('nav-robot-geometry')).toContainText('4 wheels, radius 6.0 cm, track 0.34 m');
    await picker.selectOption('hexapod');
    await expect(page.getByTestId('nav-robot-geometry')).toContainText('6 legs, trot gait');
    await expect(page.getByTestId('nav-walker')).toContainText('Walks with');
    await expect(page.getByTestId('nav-go')).toHaveText('Walk to goal');

    // Build starts with the quadruped; Navigate walks it.
    await page.getByTestId('nav-build').click();
    await expect(page.getByTestId('builder-navigate')).toBeEnabled();
    await page.getByTestId('builder-navigate').click();
    await expect(page.getByRole('main', { name: 'Navigate' })).toBeVisible();
    await expect(picker).toHaveValue('custom');
    await expect(picker.locator('option:checked')).toHaveText('Build robot: Quadruped');
    await expect(page.getByTestId('nav-robot-geometry')).toContainText('4 legs');
    await expect(page.getByTestId('nav-go')).toBeEnabled({ timeout: 30_000 });
    expect(errors).toEqual([]);
  });
});
