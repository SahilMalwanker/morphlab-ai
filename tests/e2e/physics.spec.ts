import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function openPhysics(page: Page) {
  await page.goto('./');
  await page.getByTestId('nav-physics').click();
  // Includes fetching and compiling the MuJoCo WASM module (about 4 s in headless Chromium).
  await expect(page.getByTestId('sim-status')).toHaveText('Running', { timeout: 20_000 });
}

function seconds(text: string | null): number {
  return Number.parseFloat((text ?? '').replace(' s', ''));
}

test.describe('M2 physical body', () => {
  test('the quadruped stands under PD control in MuJoCo and is rendered @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await openPhysics(page);
    await expect(page.getByText('Physics mode: MuJoCo simulation with gravity and contact. Rendered from simulator state (ground truth).')).toBeVisible();
    await expect.poll(async () => seconds(await page.getByTestId('sim-time').textContent()), { timeout: 15_000 }).toBeGreaterThan(2);
    const height = Number.parseFloat((await page.getByTestId('base-height').textContent())!);
    expect(height).toBeGreaterThan(0.24);
    expect(height).toBeLessThan(0.29);
    const png = await page.getByTestId('viewport').locator('canvas').screenshot();
    expect(png.length).toBeGreaterThan(20_000);
    expect(errors).toEqual([]);
  });

  test('push is logged, unstable gains are refused, reset restarts time', async ({ page }) => {
    const errors = collectErrors(page);
    await openPhysics(page);
    await page.getByRole('button', { name: 'Push +Y' }).click();
    await expect(page.getByTestId('events')).toContainText('Push 2.0 N*s along (0, 1, 0) at the base');

    const kp = page.getByRole('slider', { name: /Proportional gain/ });
    await kp.fill('150');
    await expect(page.getByText(/^Refused: with the lightest joint/)).toBeVisible();
    await expect(kp).toHaveValue('40');
    await kp.fill('80');
    await expect(page.getByText(/^Refused/)).toHaveCount(0);
    await expect(kp).toHaveValue('80');

    await expect.poll(async () => seconds(await page.getByTestId('sim-time').textContent()), { timeout: 15_000 }).toBeGreaterThan(1.5);
    await page.getByRole('button', { name: 'Pause' }).click();
    await expect(page.getByRole('button', { name: 'Run simulation', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Run', exact: true })).toHaveCount(1);
    await page.getByRole('button', { name: 'Reset' }).click();
    await expect(page.getByTestId('sim-time')).toHaveText('0.00 s');
    await expect(page.getByTestId('events')).toContainText('Reset to the initial state.');
    expect(errors).toEqual([]);
  });

  test('switching presets rebuilds the model; fixed arms hide base readouts', async ({ page }) => {
    const errors = collectErrors(page);
    await openPhysics(page);
    await page.getByRole('combobox', { name: 'Preset' }).selectOption({ label: 'Six-joint arm' });
    await expect(page.getByRole('combobox', { name: 'Selected joint' }).locator('option')).toHaveCount(6);
    await expect(page.getByTestId('base-height')).toHaveCount(0);
    await page.getByTestId('nav-kinematics').click();
    await page.getByTestId('nav-physics').click();
    await expect(page.getByTestId('sim-status')).toHaveText('Running');
    // Only the open view renders: the Physics canvas and its two joint plots, no hidden Playground or Kinematics canvas.
    await expect(page.locator('canvas')).toHaveCount(3);
    expect(errors).toEqual([]);
  });
});
