import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { inspect } from 'node:util';
import { afterAll, describe, expect, it } from 'vitest';
import { ConfigError, configFromValues, detectProvider, loadConfig } from '../../server/config.ts';

// Fake values only; real credentials never appear in tests.
const KEY = 'fake-key-0123456789abcdef';
const ENDPOINT = 'https://fake-resource.openai.azure.com/openai/deployments/fake-model/chat/completions?api-version=2024-10-21';

const dir = mkdtempSync(join(tmpdir(), 'morphlab-config-'));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

function expectConfigError(fn: () => unknown, message: RegExp): void {
  let caught: unknown = null;
  try {
    fn();
  } catch (e) {
    caught = e;
  }
  expect(caught).toBeInstanceOf(ConfigError);
  const text = (caught as Error).message;
  expect(text).toMatch(message);
  expect(text).not.toContain(KEY);
  expect(text).not.toContain('fake-resource');
}

describe('detectProvider', () => {
  it('recognizes an Azure OpenAI chat completions deployment URL', () => {
    expect(detectProvider(new URL(ENDPOINT))).toBe('azure-openai');
  });

  it('rejects other shapes', () => {
    expect(detectProvider(new URL(ENDPOINT.replace('https:', 'http:')))).toBeNull();
    expect(detectProvider(new URL(ENDPOINT.replace('openai.azure.com', 'example.com')))).toBeNull();
    expect(detectProvider(new URL(ENDPOINT.replace('?api-version=2024-10-21', '')))).toBeNull();
    expect(detectProvider(new URL(ENDPOINT.replace('/chat/completions', '/embeddings')))).toBeNull();
    expect(detectProvider(new URL('https://fake-resource.openai.azure.com.attacker.example/openai/deployments/x/chat/completions?api-version=1'))).toBeNull();
  });
});

describe('configFromValues', () => {
  it('fails fast and names only the variable', () => {
    expectConfigError(() => configFromValues({ ENDPOINT }), /^API_KEY is missing or empty/);
    expectConfigError(() => configFromValues({ API_KEY: KEY, ENDPOINT: '  ' }), /^ENDPOINT is missing or empty/);
    expectConfigError(() => configFromValues({ API_KEY: KEY, ENDPOINT: 'not a url fake-resource' }), /^ENDPOINT in \.env is not a valid URL\.$/);
    expectConfigError(() => configFromValues({ API_KEY: KEY, ENDPOINT: 'https://fake-resource.example.com/v1/chat' }), /^ENDPOINT in \.env is not a supported provider URL/);
  });

  it('keeps the credentials out of serialization, inspection and string conversion', () => {
    const config = configFromValues({ API_KEY: KEY, ENDPOINT });
    expect(config.provider).toBe('azure-openai');
    expect(config.apiKey).toBe(KEY);
    expect(config.endpoint.searchParams.get('api-version')).toBe('2024-10-21');
    for (const text of [JSON.stringify(config), inspect(config, { showHidden: true, depth: 5 }), String(config), JSON.stringify(Object.entries(config))]) {
      expect(text).not.toContain(KEY);
      expect(text).not.toContain('fake-resource');
    }
  });
});

describe('loadConfig', () => {
  it('reads API_KEY and ENDPOINT from a .env file', () => {
    const file = join(dir, '.env');
    writeFileSync(file, `# comment\nAPI_KEY="${KEY}"\nENDPOINT="${ENDPOINT}"\n`);
    expect(loadConfig(file).provider).toBe('azure-openai');
  });

  it('reports a missing file without a path or value', () => {
    expectConfigError(() => loadConfig(join(dir, 'missing.env')), /^The \.env file was not found\./);
  });
});
