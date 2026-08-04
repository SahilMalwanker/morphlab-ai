import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['spikes/**/*.spike.test.ts'],
    testTimeout: 600_000,
  },
});
