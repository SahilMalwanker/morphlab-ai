import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function openBuild(page: Page) {
  await page.goto('./');
  await page.getByTestId('nav-build').click();
  await expect(page.getByTestId('morph-hash')).toHaveText(/^[0-9a-f]{12}$/);
}

async function commit(page: Page, testId: string, value: string) {
  const f = page.getByTestId(testId);
  await f.fill(value);
  await f.press('Enter');
}

test.describe('M3 robot builder', () => {
  test('generator, refused edit, mirrored edit and versioning', async ({ page }) => {
    const errors = collectErrors(page);
    await openBuild(page);
    await expect(page.getByText('Builder preview: home configuration computed by forward kinematics. No physics.')).toBeVisible();
    await page.getByTestId('gen-legs').selectOption({ label: '6' });
    await page.getByTestId('gen-dof').selectOption({ label: '2' });
    await page.getByRole('button', { name: 'Generate legged robot' }).click();
    await expect(page.getByTestId('morph-version')).toHaveText('2');
    await expect(page.getByTestId('limb-select').locator('option')).toHaveCount(6);
    await expect(page.getByTestId('morph-hash')).toHaveText(/^[0-9a-f]{12}$/);
    const hash = await page.getByTestId('morph-hash').textContent();

    await page.getByTestId('joint-select').selectOption({ index: 1 });
    await commit(page, 'joint-lower', '10');
    await expect(page.getByTestId('builder-issues')).toContainText('Front left leg, Knee: Lower limit must be below upper limit.');
    await expect(page.getByTestId('builder-issues')).toContainText('The change was not applied.');
    await expect(page.getByTestId('morph-version')).toHaveText('2');
    await expect(page.getByTestId('morph-hash')).toHaveText(hash!);
    await expect(page.getByTestId('joint-lower')).toHaveValue('-154.7');

    await commit(page, 'joint-z', '-0.2');
    await expect(page.getByTestId('morph-version')).toHaveText('3');
    await expect(page.getByTestId('builder-issues')).toHaveCount(0);
    await expect(page.getByTestId('morph-hash')).not.toHaveText(hash!);
    await expect(page.getByTestId('morph-history')).toContainText('Front left leg, Knee: xyz (mirrored)');
    await page.getByTestId('limb-select').selectOption({ label: 'Front right leg (2)' });
    await page.getByTestId('joint-select').selectOption({ index: 1 });
    await expect(page.getByTestId('joint-z')).toHaveValue('-0.200');
    expect(errors).toEqual([]);
  });

  test('export then import restores the same morphology; broken files are refused @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await openBuild(page);
    await commit(page, 'base-mass', '5.5');
    await expect(page.getByTestId('morph-version')).toHaveText('2');
    await expect(page.getByTestId('morph-hash')).toHaveText(/^[0-9a-f]{12}$/);
    const hash = await page.getByTestId('morph-hash').textContent();
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Export JSON' }).click()]);
    expect(download.suggestedFilename()).toBe(`quadruped-${hash!.slice(0, 8)}.morphlab.json`);
    const text = readFileSync((await download.path())!, 'utf8');
    expect(JSON.parse(text).robots[0].base.mass).toBe(5.5);

    await page.getByRole('button', { name: 'Load preset' }).click();
    await expect(page.getByTestId('morph-hash')).not.toHaveText(hash!);
    await page.getByTestId('import-file').setInputFiles({ name: 'robot.json', mimeType: 'application/json', buffer: Buffer.from(text) });
    await expect(page.getByTestId('builder-notice')).toHaveText('Imported Quadruped.');
    await expect(page.getByTestId('morph-hash')).toHaveText(hash!);
    const [again] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Export JSON' }).click()]);
    const text2 = readFileSync((await again.path())!, 'utf8');
    expect(JSON.parse(text2).robots).toEqual(JSON.parse(text).robots);

    await page.getByTestId('import-file').setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"kind": "something else"}') });
    await expect(page.getByTestId('builder-issues')).toContainText('The file is not a MorphLab bundle (expected kind "morphlab.bundle").');
    await expect(page.getByTestId('morph-hash')).toHaveText(hash!);
    expect(errors).toEqual([]);
  });

  test('a built robot simulates in physics and opens in kinematics @smoke', async ({ page }) => {
    const errors = collectErrors(page);
    await openBuild(page);
    await page.getByTestId('gen-legs').selectOption({ label: '6' });
    await page.getByRole('button', { name: 'Generate legged robot' }).click();
    await expect(page.getByTestId('morph-version')).toHaveText('2');
    await page.getByRole('button', { name: 'Simulate in Physics' }).click();
    await expect(page.getByTestId('sim-status')).toHaveText('Running', { timeout: 20_000 });
    await expect(page.getByRole('combobox', { name: 'Preset' }).locator('option:checked')).toHaveText('Builder robot: 6-legged robot (3 DOF per leg)');
    await expect(page.getByRole('combobox', { name: 'Selected joint' }).locator('option')).toHaveCount(18);
    await expect.poll(async () => Number.parseFloat((await page.getByTestId('sim-time').textContent()) ?? '0'), { timeout: 15_000 }).toBeGreaterThan(1.5);
    const height = Number.parseFloat((await page.getByTestId('base-height').textContent())!);
    expect(height).toBeGreaterThan(0.2);
    await page.getByTestId('nav-build').click();
    await page.getByRole('button', { name: 'Open in Kinematics' }).click();
    await expect(page.getByRole('combobox', { name: 'Preset' }).locator('option:checked')).toHaveText('Custom: 6-legged robot (3 DOF per leg)');
    await expect(page.getByRole('combobox', { name: 'Target limb' }).locator('option')).toHaveCount(6);
    expect(errors).toEqual([]);
  });
});
