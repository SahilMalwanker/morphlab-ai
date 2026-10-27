import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function setField(page: Page, id: string, value: string) {
  const f = page.getByTestId(id);
  await f.fill(value);
  await f.press('Enter');
}

/** The app opens on the AI Playground; Kinematics is a lab tool in the navigation. */
async function openKinematics(page: Page) {
  await page.goto('./');
  await page.getByTestId('nav-kinematics').click();
}

test.describe('M1 kinematics playground', () => {
  test('wheel-only robots disable IK and identify every wheel joint', async ({ page }) => {
    await openKinematics(page);
    await page.getByLabel('Preset', { exact: true }).selectOption('rover');
    for (const field of ['target-x', 'target-y', 'target-z']) await expect(page.getByTestId(field)).toBeDisabled();
    for (const label of ['Solve while the target moves', 'Manipulability ellipsoid', 'Reachable workspace samples']) await expect(page.getByLabel(label)).toBeDisabled();
    await expect(page.getByTestId('tip-x')).toHaveCount(0);
    for (const limb of ['Front left', 'Front right', 'Rear left', 'Rear right']) await expect(page.getByRole('slider', { name: `${limb} wheel: Wheel`, exact: true })).toBeVisible();
    await page.getByLabel('Preset', { exact: true }).selectOption('arm3');
    await expect(page.getByTestId('target-x')).toBeEnabled();
    await expect(page.getByTestId('tip-x')).toBeVisible();
  });

  test('loads without console errors and renders the viewport @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('./');
    await expect(page.getByRole('heading', { name: 'MorphLab' })).toBeVisible();
    await page.getByTestId('nav-kinematics').click();
    await expect(page.getByText('Kinematics mode: poses computed by forward kinematics. No physics.')).toBeVisible();
    const box = await page.locator('canvas').boundingBox();
    expect(box!.width).toBeGreaterThan(500);
    expect(box!.height).toBeGreaterThan(500);
    // A rendered scene compresses far worse than a uniformly cleared canvas (a few KB).
    await page.waitForTimeout(500);
    const png = await page.locator('canvas').screenshot();
    expect(png.length).toBeGreaterThan(20_000);
    expect(errors).toEqual([]);
  });

  test('IK reports converged, unreachable and invalid input honestly @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await openKinematics(page);
    await setField(page, 'target-x', '0.3');
    await setField(page, 'target-y', '0.2');
    await setField(page, 'target-z', '0.4');
    await expect(page.getByTestId('ik-status')).toHaveText('Converged');
    const residualMm = parseFloat((await page.getByTestId('ik-residual').textContent())!);
    expect(residualMm).toBeLessThan(0.1);
    await expect(page.getByTestId('tip-x')).toHaveText('0.3000 m');
    await expect(page.getByTestId('tip-z')).toHaveText('0.4000 m');

    await setField(page, 'target-z', '2');
    await expect(page.getByTestId('ik-status')).toHaveText('Unreachable');
    await expect(page.getByTestId('ik-message')).toContainText('outside the maximum reach');

    const tipBefore = await page.getByTestId('tip-z').textContent();
    // 1e400 overflows to Infinity and must be rejected without touching the pose.
    await setField(page, 'target-x', '1e400');
    await expect(page.getByTestId('target-x')).toHaveAttribute('aria-invalid', 'true');
    await expect(page.getByTestId('tip-z')).toHaveText(tipBefore!);
    expect(errors).toEqual([]);
  });

  test('joint sliders drive forward kinematics', async ({ page }) => {
    await openKinematics(page);
    await page.getByRole('button', { name: 'Reset pose' }).click();
    await page.getByLabel('Shoulder pitch').fill('0');
    await page.getByLabel('Elbow pitch').fill('0');
    // Fully stretched arm points straight up: tip at z = 0.06 + 0.12 + 0.35 + 0.30.
    await expect(page.getByTestId('tip-z')).toHaveText('0.8300 m');
    await expect(page.getByTestId('singular-note')).toBeVisible();
  });

  test('quadruped foot IK and chain editor validation', async ({ page }) => {
    const errors = collectErrors(page);
    await openKinematics(page);
    await page.getByRole('button', { name: 'Add joint' }).click();
    await page.getByLabel('Joint 4 lower limit').fill('50');
    await page.getByLabel('Joint 4 upper limit').fill('10');
    await page.getByRole('button', { name: 'Apply' }).click();
    await expect(page.getByRole('alert').first()).toHaveText('Joint 4: lower limit must be below upper limit.');
    await page.getByLabel('Joint 4 lower limit').fill('-90');
    await page.getByLabel('Joint 4 upper limit').fill('90');
    await page.getByRole('button', { name: 'Apply' }).click();
    await expect(page.getByText('Fixed base. 4 joints.')).toBeVisible();

    await page.getByLabel('Preset').selectOption('quadruped');
    await page.getByLabel('Target limb').selectOption({ label: 'Rear right leg' });
    const z = parseFloat(await page.getByTestId('target-z').inputValue());
    await setField(page, 'target-z', String(z + 0.05));
    await expect(page.getByTestId('ik-status')).toHaveText('Converged');
    expect(errors).toEqual([]);
  });
});
