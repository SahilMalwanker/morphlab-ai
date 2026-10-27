import { expect, test, type Page } from '@playwright/test';

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

async function openSkills(page: Page) {
  await page.goto('./');
  // The app opens on the AI Playground (D34); the foot lift example is a workflow of its Teach view.
  await expect(page.getByRole('main', { name: 'AI Playground' })).toBeVisible();
  await page.getByTestId('skills-panel-foot-lift').click();
  await expect(page.getByTestId('evaluate-state-machine')).toBeEnabled();
}

test.describe('M5 skills', () => {
  test('state machine and replay are evaluated over 20 trials in a worker and bound to the morphology', async ({ page }) => {
    test.setTimeout(180_000);
    const errors = collectErrors(page);
    const workers: string[] = [];
    page.on('worker', (w) => workers.push(w.url()));
    await openSkills(page);
    await page.getByTestId('evaluate-state-machine').click();
    await page.getByTestId('pg-view-library').click();
    const card = page.getByTestId('skill-card').first();
    await expect(card).toBeVisible({ timeout: 90_000 });
    expect(workers.some((u) => u.includes('skill.worker'))).toBe(true);
    await expect(card.getByTestId('skill-method')).toHaveText('State machine (hand-designed)');
    await expect(card.getByTestId('skill-success')).toHaveText(/^\d+ \/ 20 \(/);
    await expect(card).toContainText('(this robot)');
    await expect(card).toContainText('friction 0.6-1.2, mass 3.5-4.5 kg');

    await page.getByTestId('pg-view-teach').click();
    await page.getByTestId('record-expert').click();
    await expect(page.getByTestId('demonstration-status')).toContainText('Demonstration: state-machine expert', { timeout: 60_000 });
    await page.getByTestId('evaluate-replay').click();
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('skill-card')).toHaveCount(2, { timeout: 90_000 });
    await expect(page.getByTestId('skill-card').first().getByTestId('skill-method')).toHaveText('Replay of a recorded demonstration (open loop)');

    await page.getByTestId('skill-play').first().click();
    await expect(page.getByTestId('skill-card').first()).toContainText('Playing');
    // Leaving the playground ends the playback; coming back never shows it as still running.
    await page.getByTestId('nav-kinematics').click();
    await page.getByTestId('back-to-playground').click();
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('skill-card').first()).not.toContainText('Playing');
    await page.getByTestId('pg-view-attempts').click();
    await expect(page.getByTestId('last-play')).toContainText('Last playback');

    await page.getByTestId('pg-view-library').click();
    await page.getByTestId('skills-robot').selectOption({ label: 'Hexapod' });
    await page.getByTestId('skill-play').first().click();
    await expect(page.getByTestId('skills-message')).toContainText('A skill is never reused on a different body');
    await expect(page.getByTestId('skill-card').first()).toContainText('(other robot)');
    expect(errors).toEqual([]);
  });

  test('imitation learning and reward-based improvement run in the worker and are labeled', async ({ page }) => {
    // Training took 15 s in a GPU-backed browser but over 120 s in headless Chromium, where software WebGL
    // for the live view competes with the worker for the CPU. The assertions are unchanged; only the wait is longer.
    test.setTimeout(600_000);
    const errors = collectErrors(page);
    await openSkills(page);
    await page.getByTestId('train-imitation').click();
    await expect(page.getByTestId('skills-job')).toContainText('Imitation learning');
    await expect(page.getByTestId('skills-message')).toContainText('Trained a neural policy', { timeout: 300_000 });
    await page.getByTestId('evaluate-imitation-mlp').click();
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('skill-card')).toHaveCount(1, { timeout: 90_000 });
    await expect(page.getByTestId('skill-card').first().getByTestId('skill-method')).toHaveText('Imitation-learned neural policy');
    await expect(page.getByTestId('skill-card').first()).toContainText('behavior cloning');

    await page.getByTestId('pg-view-teach').click();
    await page.getByTestId('improve').click();
    await expect(page.getByTestId('improvement')).toContainText('Held-out reward', { timeout: 240_000 });
    await page.getByTestId('evaluate-es-improved').click();
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('skill-card')).toHaveCount(2, { timeout: 90_000 });
    await expect(page.getByTestId('skill-card').first().getByTestId('skill-method')).toHaveText('State machine improved by reward (evolution strategies)');
    expect(errors).toEqual([]);
  });

  test('a skill the user teaches: pose keyframes, set goals, evaluate, run it by name', async ({ page }) => {
    test.setTimeout(180_000);
    const errors = collectErrors(page);
    await page.goto('./');
    await page.getByTestId('teach-new').click();
    await expect(page.getByTestId('teach-draft-state')).toContainText('Draft, not saved');
    await page.getByTestId('teach-name').fill('crouch');
    // Every leg: hip pitch 1.1 rad, knee -2.2 rad (joints are ordered hip roll, hip pitch, knee per leg).
    for (const leg of [0, 1, 2, 3]) {
      await page.getByTestId(`pose-joint-${3 * leg + 1}`).fill('1.1');
      await page.getByTestId(`pose-joint-${3 * leg + 2}`).fill('-2.2');
    }
    await page.getByTestId('teach-add-keyframe').click();
    await expect(page.getByTestId('teach-keyframes').locator('li')).toHaveCount(1);
    await page.getByTestId('teach-goal-kind').selectOption('baseHeight');
    await page.getByTestId('teach-goal-add').click();
    await expect(page.getByTestId('teach-goal')).toHaveCount(3);
    await page.getByTestId('teach-evaluate').click();
    await page.getByTestId('pg-view-library').click();
    const card = page.getByTestId('skill-card').first();
    await expect(card).toBeVisible({ timeout: 90_000 });
    await expect(card.getByTestId('skill-method')).toHaveText('Keyframe motion you posed (open loop)');
    await expect(card.getByTestId('skill-goals')).toContainText('the body is below 0.2 m for 1 s');
    await expect(card.getByTestId('skill-success')).toHaveText(/^\d+ \/ 20 \(/);

    await page.getByTestId('command-input').fill('crouch');
    await page.getByTestId('command-run').click();
    await expect(page.getByTestId('command-output')).toContainText('Measured (simulator ground truth)', { timeout: 60_000 });
    expect(errors).toEqual([]);
  });

  test('Stop task leaves a training job running; the Cancel next to its progress ends it', async ({ page }) => {
    test.setTimeout(120_000);
    await openSkills(page);
    await page.getByTestId('train-imitation').click();
    await expect(page.getByTestId('skills-job')).toContainText('Imitation learning');
    await page.getByTestId('command-stop').click();
    await expect(page.getByTestId('stop-message')).toContainText('Training and evaluation jobs are not affected');
    await expect(page.getByTestId('skills-job')).toBeVisible();
    await page.getByTestId('skills-job-cancel').click();
    await expect(page.getByTestId('skills-job')).toHaveCount(0);
  });

  test('the side panel collapses for a wider robot view, stays collapsed after a reload, and a view button restores it', async ({ page }) => {
    await page.goto('./');
    const toggle = page.getByTestId('pg-panel-toggle');
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByRole('complementary', { name: 'Teach' })).toBeVisible();
    await expect(page.getByTestId('viewport').locator('canvas')).toBeVisible();
    const before = (await page.getByTestId('viewport').boundingBox())!.width;
    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await expect(page.getByRole('complementary', { name: 'Teach' })).toBeHidden();
    await expect.poll(async () => (await page.getByTestId('viewport').boundingBox())!.width).toBeGreaterThan(before + 200);
    // Resizing reuses the one renderer: no second canvas appears.
    await expect(page.locator('canvas')).toHaveCount(1);
    await page.reload();
    await expect(page.getByTestId('pg-panel-toggle')).toHaveAttribute('aria-expanded', 'false');
    await page.getByTestId('pg-view-library').click();
    await expect(page.getByTestId('pg-panel-toggle')).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByRole('complementary', { name: 'Skill library' })).toBeVisible();
  });

  test('if the physics engine cannot load, the playground says so and the lab tools still work', async ({ page }) => {
    await page.route('**/*.wasm', (route) => route.abort());
    await page.goto('./');
    await expect(page.getByTestId('pg-engine')).toHaveText('Simulation unavailable', { timeout: 30_000 });
    await expect(page.getByTestId('pg-engine-notice')).toContainText('Build and Kinematics work without it.');
    await page.getByTestId('nav-kinematics').click();
    await expect(page.getByText('Kinematics mode: poses computed by forward kinematics. No physics.')).toBeVisible();
    await page.getByTestId('nav-build').click();
    await expect(page.getByTestId('morph-hash')).toHaveText(/^[0-9a-f]{12}$/);
  });
});
