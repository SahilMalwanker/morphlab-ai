import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  // One browser at a time: parallel headless browsers with software WebGL starve
  // the 10 MB MuJoCo WASM compile (measured 3.8 s alone, over 10 s in parallel),
  // which is not representative of a single user.
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:4173',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    // The smoke suite against the same build on a plain static server, under a
    // sub-path and a strict Content Security Policy (deployment check).
    { name: 'static', grep: /@smoke/, use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 }, baseURL: 'http://localhost:4174/morphlab/' } },
    // The smoke suite in Firefox (needs the Firefox build that matches @playwright/test).
    { name: 'firefox', grep: /@smoke/, use: { ...devices['Desktop Firefox'], viewport: { width: 1440, height: 900 } } },
  ],
  webServer: [
    {
      command: 'npm run build && npx vite preview --port 4173 --strictPort',
      url: 'http://localhost:4173',
      reuseExistingServer: false,
      timeout: 180_000,
    },
    {
      command: 'node scripts/serve-static.mjs dist 4174 /morphlab/',
      url: 'http://localhost:4174/morphlab/',
      reuseExistingServer: false,
      timeout: 180_000,
    },
  ],
});
