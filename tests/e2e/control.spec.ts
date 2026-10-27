import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function openLab(page: Page) {
  await page.goto('./');
  await page.getByTestId('nav-control').click();
  await expect(page.getByTestId('run-step')).toBeEnabled({ timeout: 20_000 });
}

test.describe('M4 control lab', () => {
  test('comparison changes and edited challenge gains invalidate displayed results', async ({ page }) => {
    await openLab(page);
    await page.getByTestId('run-step').click();
    await expect(page.getByTestId('control-running')).toHaveCount(0);
    await expect(page.getByTestId('step-metrics').locator('th')).toHaveCount(5);
    await page.getByLabel('Compare with gain set B').uncheck();
    await expect(page.getByTestId('step-stale')).toBeVisible();
    await expect(page.getByTestId('step-metrics').locator('th')).toHaveText(['Metric', 'A measured', 'A linear']);
    await page.getByLabel('Compare with gain set B').check();
    await expect(page.getByTestId('step-stale')).toHaveCount(0);
    await page.getByTestId('challenge-hold').getByRole('button', { name: 'Run', exact: true }).click();
    await expect(page.getByTestId('challenge-hold').getByRole('status')).toHaveCount(2);
    await page.getByTestId('kp-a').fill('80');
    await page.getByTestId('kp-a').press('Enter');
    await expect(page.getByTestId('challenge-stale-hold')).toContainText('settings have changed');
    await page.getByTestId('kp-a').fill('60');
    await page.getByTestId('kp-a').press('Enter');
    await expect(page.getByTestId('challenge-stale-hold')).toHaveCount(0);
  });

  test('step test compares PD with and without gravity compensation @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await openLab(page);
    await page.getByTestId('run-step').click();
    const table = page.getByTestId('step-metrics');
    await expect(table.locator('th')).toHaveCount(5);
    await expect(table.locator('th')).toHaveText(['Metric', 'A measured', 'A linear', 'B measured', 'B linear']);
    const sse = table.locator('tr', { hasText: 'Steady-state error' }).locator('td');
    // A (PD only) sags under gravity; B (with gravity compensation) does not.
    expect(Math.abs(Number.parseFloat((await sse.nth(1).textContent())!))).toBeGreaterThan(1);
    expect(Math.abs(Number.parseFloat((await sse.nth(3).textContent())!))).toBeLessThan(0.01);
    await expect(page.getByTestId('saturation-fraction')).toContainText('Time at the effort limit after the step: A');
    // A rests where kp times the error balances gravity; the note says that kd cannot fix it.
    await expect(page.getByTestId('rest-note')).toContainText('A came to rest');
    await expect(page.getByTestId('rest-note')).toContainText('kd does not change it');
    await expect(page.getByTestId('rest-note')).not.toContainText('B came to rest');
    const png = await page.getByTestId('lab-position').screenshot();
    expect(png.length).toBeGreaterThan(10_000);
    expect(errors).toEqual([]);
  });

  test('challenges report objective verdicts per gain set', async ({ page }) => {
    const errors = collectErrors(page);
    await openLab(page);
    const hold = page.getByTestId('challenge-hold');
    await hold.getByRole('button', { name: 'Run' }).click();
    await expect(hold.getByRole('status')).toHaveText(['Failed', 'Passed'], { timeout: 20_000 });
    await expect(hold).toContainText('Largest joint error in the last second');

    await page.getByTestId('control-robot').selectOption({ label: 'Quadruped' });
    await expect(page.getByText('Floating base: experiments other than the push run with the base fixed on a test stand, legs clear of the ground.')).toBeVisible();
    const reach = page.getByTestId('challenge-reach');
    await reach.getByRole('button', { name: 'Run' }).click();
    await expect(reach).toContainText('Reaching needs a fixed-base arm; this robot has a floating base.');
    const push = page.getByTestId('challenge-push');
    await push.getByRole('button', { name: 'Run' }).click();
    await expect(push.getByRole('status')).toHaveText(['Passed', 'Passed'], { timeout: 20_000 });
    await expect(push).toContainText('Lowest base height / pre-push height');
    expect(errors).toEqual([]);
  });
});
