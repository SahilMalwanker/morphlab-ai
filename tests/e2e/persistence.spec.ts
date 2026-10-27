import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function evaluateStateMachine(page: Page) {
  await page.getByTestId('nav-skills').click();
  await page.getByTestId('pg-view-teach').click();
  await page.getByTestId('skills-panel-foot-lift').click();
  await page.getByTestId('evaluate-state-machine').click();
  await page.getByTestId('pg-view-library').click();
  await expect(page.getByTestId('skill-card')).toHaveCount(1, { timeout: 90_000 });
}

/** Every value in the app's IndexedDB database, as text. */
async function dumpIndexedDb(page: Page): Promise<string> {
  return page.evaluate(
    () =>
      new Promise<string>((resolve, reject) => {
        const req = indexedDB.open('morphlab');
        req.onerror = () => reject(req.error);
        req.onsuccess = () => {
          const db = req.result;
          const names = Array.from(db.objectStoreNames);
          const tx = db.transaction(names, 'readonly');
          const parts: string[] = [];
          for (const n of names) {
            const r = tx.objectStore(n).getAll();
            r.onsuccess = () => parts.push(JSON.stringify(r.result));
          }
          tx.oncomplete = () => {
            db.close();
            resolve(parts.join('\n'));
          };
        };
      }),
  );
}

test.describe('M6 persistence and language', () => {
  test('robots and skills survive a reload; the flagship command runs a stored skill and reports the hold', async ({ page }) => {
    test.setTimeout(240_000);
    const errors = collectErrors(page);
    await page.goto('./');
    await page.getByTestId('nav-build').click();
    const mass = page.getByTestId('base-mass');
    await mass.fill('4.4');
    await mass.press('Enter');
    await expect(page.getByTestId('morph-version')).toHaveText('2');
    await expect(page.getByTestId('morph-hash')).toHaveText(/^[0-9a-f]{12}$/);
    const hash = await page.getByTestId('morph-hash').textContent();
    await evaluateStateMachine(page);

    await page.reload();
    await page.getByTestId('nav-build').click();
    await expect(page.getByTestId('morph-hash')).toHaveText(hash!);
    await expect(page.getByTestId('base-mass')).toHaveValue('4.40');
    await expect(page.getByTestId('saved-robots')).toContainText(hash!.slice(0, 8));
    await page.getByTestId('nav-skills').click();
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('skill-card')).toHaveCount(1);

    await page.getByTestId('command-input').fill('lift the front left leg and hold for three seconds');
    await page.getByTestId('command-run').click();
    const out = page.getByTestId('command-output');
    await expect(out).toContainText('Measured hold at least 3 cm above the ground', { timeout: 60_000 });
    const measured = Number(/ground: (\d+\.\d+) s/.exec((await out.textContent())!)![1]);
    expect(measured).toBeGreaterThan(3);
    expect(measured).toBeLessThan(3.6);
    await expect(out).toContainText('requested 3 s; simulator ground truth');

    await page.getByTestId('command-input').fill('walk forward');
    await page.getByTestId('command-run').click();
    await expect(out).toContainText('Refused: Walking is not a skill this robot has learned yet.');
    await expect(out).toContainText('Suggestion:');
    expect(errors).toEqual([]);
  });

  test('the skill library exports to JSON and imports back, labeled as imported, and survives a reload @smoke', async ({ page }) => {
    test.setTimeout(180_000);
    const errors = collectErrors(page);
    await page.goto('./');
    await evaluateStateMachine(page);
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('skills-export').click()]);
    expect(download.suggestedFilename()).toMatch(/^morphlab-skills-\d{4}-\d{2}-\d{2}\.morphlab\.json$/);
    const file = await download.path();
    const bundle = JSON.parse(readFileSync(file, 'utf8')) as { kind: string; skills: { id: string; method: string }[] };
    expect(bundle.kind).toBe('morphlab.bundle');
    expect(bundle.skills.map((s) => s.method)).toEqual(['state-machine']);

    await page.getByTestId('skill-card').getByRole('button', { name: 'Delete' }).click();
    await expect(page.getByTestId('skill-card')).toHaveCount(0);
    await page.getByTestId('skills-import').setInputFiles(file);
    await expect(page.getByTestId('skills-message')).toHaveText('Imported 1 skill. Their evaluation results come from the file and were not re-run here; text commands will not use them until you re-evaluate them here. Saved 1 robot the skills were evaluated on.');
    await expect(page.getByTestId('skill-origin')).toContainText('the evaluation was not re-run here');
    await expect(page.getByTestId('skill-card')).toContainText('(this robot)');
    await expect(page.getByTestId('skill-card')).toContainText('Unverified here');

    // A command will not run on a claim it cannot verify; re-evaluating here makes it usable.
    await page.getByTestId('command-input').fill('lift the front left leg');
    await page.getByTestId('command-run').click();
    await expect(page.getByTestId('command-output')).toContainText('has not been reproduced here');
    await page.getByTestId('skill-reevaluate').click();
    await expect(page.getByTestId('skill-origin')).toContainText('re-evaluated here (the file claimed', { timeout: 60_000 });
    await expect(page.getByTestId('skill-card')).not.toContainText('Unverified here');

    await page.reload();
    await expect(page.getByRole('main', { name: 'AI Playground' })).toBeVisible();
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('skill-card')).toHaveCount(1);
    await expect(page.getByTestId('skill-origin')).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('missing browser features are reported and the rest of the app still works', async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(window, 'Worker', { value: undefined, configurable: true });
      Object.defineProperty(window, 'indexedDB', { value: undefined, configurable: true });
    });
    await page.goto('./');
    const notices = page.getByRole('region', { name: 'Notices' });
    await expect(notices).toContainText('Web Workers is not available in this browser: skill evaluation and training cannot run.');
    await expect(notices).toContainText('IndexedDB is not available in this browser: robots and skills are kept only until the page is closed.');
    await expect(page.getByTestId('storage-unavailable')).toContainText('Local storage is unavailable.');
    await page.getByTestId('nav-kinematics').click();
    await expect(page.locator('canvas').first()).toBeVisible();
  });

  test('corrupted records are reported and kept until removed', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByTestId('nav-kinematics')).toBeVisible();
    await page.evaluate(
      () =>
        new Promise<void>((resolve, reject) => {
          const req = indexedDB.open('morphlab');
          req.onsuccess = () => {
            const tx = req.result.transaction('skills', 'readwrite');
            tx.objectStore('skills').put({ id: 'skill_bad', kind: 'morphlab.skill', method: 'teleport' });
            tx.oncomplete = () => {
              req.result.close();
              resolve();
            };
            tx.onerror = () => reject(tx.error);
          };
        }),
    );
    await page.reload();
    const notice = page.getByTestId('corrupted-records');
    await expect(notice).toContainText('1 stored record could not be read. They are kept until you remove them.');
    await expect(notice).toContainText('Skill skill_bad');
    await page.reload();
    await expect(notice).toContainText('Skill skill_bad');
    await notice.getByRole('button', { name: 'Remove' }).click();
    await expect(notice).toHaveCount(0);
    await page.reload();
    await expect(page.getByTestId('nav-kinematics')).toBeVisible();
    await expect(page.getByTestId('corrupted-records')).toHaveCount(0);
  });

  test('the page never asks for a provider key; without the proxy AI is off and Stop still works @smoke', async ({ page }) => {
    test.setTimeout(120_000);
    let aiRequests = 0;
    await page.route('**/api/ai/**', (route) => {
      aiRequests++;
      return route.abort();
    });
    await page.goto('./');
    await expect(page.getByTestId('ai-status')).toHaveText('AI: unavailable');
    expect(aiRequests).toBeGreaterThan(0);
    await expect(page.locator('input[type="password"]')).toHaveCount(0);
    await page.getByTestId('command-input').fill('make me a sandwich');
    await page.getByTestId('command-run').click();
    await expect(page.getByTestId('command-output')).toContainText('This command is not recognized.', { timeout: 30_000 });
    await expect(page.getByTestId('command-output')).toContainText('(local rules)');
    await page.getByTestId('command-stop').click();
    await expect(page.getByTestId('stop-message')).toContainText('No playback or demonstration was running.');
    expect(await dumpIndexedDb(page)).not.toMatch(/api[-_ ]?key/i);
    // Nothing that looks like a provider key or endpoint ships in the production bundle.
    const dir = join(process.cwd(), 'dist', 'assets');
    for (const f of readdirSync(dir).filter((x) => x.endsWith('.js'))) {
      const code = readFileSync(join(dir, f), 'utf8');
      expect(code).not.toMatch(/sk-[A-Za-z0-9_-]{20,}/);
      expect(code).not.toContain('openai.azure.com');
    }
  });
});
