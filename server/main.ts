import { buildProxy } from './app.ts';
import { createAzureOpenAIAdapter } from './azure.ts';
import { ConfigError, loadConfig, type ProviderConfig } from './config.ts';

const HOST = '127.0.0.1';
const PORT = 8787;
const ORIGINS = ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:4173', 'http://127.0.0.1:4173'];
const HOSTS = [`${HOST}:${PORT}`, `localhost:${PORT}`];

let config: ProviderConfig;
try {
  config = loadConfig(new URL('../.env', import.meta.url));
} catch (e) {
  if (!(e instanceof ConfigError)) throw e;
  console.error(`MorphLab AI proxy not started: ${e.message}`);
  process.exit(1);
}

const app = buildProxy({
  adapter: createAzureOpenAIAdapter(config, fetch),
  allowedOrigins: ORIGINS,
  allowedHosts: HOSTS,
});

try {
  await app.listen({ host: HOST, port: PORT });
} catch (e) {
  const code = typeof e === 'object' && e !== null && 'code' in e ? String(e.code) : 'unknown';
  console.error(`MorphLab AI proxy not started: cannot listen on ${HOST}:${PORT} (${code}).`);
  process.exit(1);
}
console.log(`MorphLab AI proxy on http://${HOST}:${PORT} (provider: ${config.provider}).`);

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.once(signal, () => {
    void app.close().then(() => process.exit(0));
  });
}
