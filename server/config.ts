import { readFileSync } from 'node:fs';
import { inspect, parseEnv } from 'node:util';

export type ProviderKind = 'azure-openai';

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigError';
  }
}

/** Provider credentials. Stays inside the proxy process; prints and serializes without the secrets. */
export class ProviderConfig {
  readonly provider: ProviderKind;
  readonly #apiKey: string;
  readonly #endpoint: string;

  constructor(provider: ProviderKind, apiKey: string, endpoint: URL) {
    this.provider = provider;
    this.#apiKey = apiKey;
    this.#endpoint = endpoint.href;
  }

  get apiKey(): string {
    return this.#apiKey;
  }

  get endpoint(): URL {
    return new URL(this.#endpoint);
  }

  toJSON(): { provider: ProviderKind } {
    return { provider: this.provider };
  }

  toString(): string {
    return `ProviderConfig(${this.provider})`;
  }

  [inspect.custom](): string {
    return this.toString();
  }
}

/** Detects the API style from the URL shape; only Azure OpenAI chat completion deployments are supported. */
export function detectProvider(url: URL): ProviderKind | null {
  const azureHost = /\.openai\.azure\.com$/i.test(url.hostname);
  const deploymentPath = /^\/openai\/deployments\/[^/]+\/chat\/completions\/?$/.test(url.pathname);
  if (url.protocol === 'https:' && azureHost && deploymentPath && url.searchParams.has('api-version')) return 'azure-openai';
  return null;
}

/** Builds the configuration from the two variables; errors name the variable, never its value. */
export function configFromValues(values: Record<string, string | undefined>): ProviderConfig {
  const apiKey = values['API_KEY']?.trim() ?? '';
  const endpoint = values['ENDPOINT']?.trim() ?? '';
  if (!apiKey) throw new ConfigError('API_KEY is missing or empty in .env.');
  if (!endpoint) throw new ConfigError('ENDPOINT is missing or empty in .env.');
  let url: URL;
  try {
    url = new URL(endpoint);
  } catch {
    throw new ConfigError('ENDPOINT in .env is not a valid URL.');
  }
  const provider = detectProvider(url);
  if (!provider) {
    throw new ConfigError('ENDPOINT in .env is not a supported provider URL. Supported: Azure OpenAI chat completions (https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=...).');
  }
  return new ProviderConfig(provider, apiKey, url);
}

export function loadConfig(envFile: URL | string): ProviderConfig {
  let content: string;
  try {
    content = readFileSync(envFile, 'utf8');
  } catch {
    throw new ConfigError('The .env file was not found. Copy .env.example to .env and set API_KEY and ENDPOINT.');
  }
  return configFromValues(parseEnv(content));
}
