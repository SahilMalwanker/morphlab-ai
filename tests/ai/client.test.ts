import { describe, expect, it } from 'vitest';
import { AiError, newSessionId, PROXY_OFFLINE, ProxyClient, type FetchFn } from '../../src/ai/client.ts';
import type { ChatRequest } from '../../src/ai/protocol.ts';

const request: ChatRequest = { messages: [{ role: 'user', content: 'Drive to the far wall.' }] };
const answer = { content: 'Done.', toolCalls: [], finishReason: 'stop', model: 'fake-model', usage: null };

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });

function scripted(responses: (() => Response | Promise<Response>)[]) {
  const calls: { url: string; init: RequestInit }[] = [];
  const waits: number[] = [];
  const fetchImpl: FetchFn = async (url, init) => {
    calls.push({ url, init });
    const next = responses.shift();
    if (!next) throw new Error('unexpected request');
    return next();
  };
  const client = new ProxyClient({ fetchImpl, sessionId: 'session-test-1', sleep: async (ms) => void waits.push(ms) });
  return { client, calls, waits };
}

async function failure(p: Promise<unknown>): Promise<AiError> {
  try {
    await p;
  } catch (e) {
    if (e instanceof AiError) return e;
    throw e;
  }
  throw new Error('expected an AiError');
}

describe('ProxyClient.status', () => {
  it('reports the provider when the proxy is configured', async () => {
    const { client, calls } = scripted([() => json(200, { ok: true, provider: 'azure-openai', reason: null, capabilities: null })]);
    expect(await client.status()).toEqual({ state: 'available', provider: 'azure-openai', capabilities: null });
    expect(calls[0]?.url).toBe('/api/ai/status');
  });

  it('passes on why an unconfigured proxy cannot serve AI', async () => {
    const { client } = scripted([() => json(200, { ok: false, provider: null, reason: 'API_KEY is missing or empty in .env.', capabilities: null })]);
    expect(await client.status()).toEqual({ state: 'unavailable', reason: 'API_KEY is missing or empty in .env.' });
  });

  it('treats a missing proxy (network error, 404 page, non-JSON) as offline', async () => {
    const { client } = scripted([
      () => Promise.reject(new TypeError('Failed to fetch')),
      () => new Response('<html>Not found</html>', { status: 404 }),
      () => new Response('<html>app</html>', { status: 200 }),
      () => json(200, { ok: true, provider: 'x', reason: null, capabilities: 'yes' }),
    ]);
    for (let i = 0; i < 4; i++) expect(await client.status()).toEqual({ state: 'unavailable', reason: PROXY_OFFLINE });
  });
});

describe('ProxyClient.chat', () => {
  it('posts JSON with the session header and validates the answer', async () => {
    const { client, calls } = scripted([() => json(200, answer)]);
    expect(await client.chat(request, new AbortController().signal)).toEqual(answer);
    const init = calls[0]!.init;
    expect(calls[0]!.url).toBe('/api/ai/chat');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'content-type': 'application/json', 'x-morphlab-session': 'session-test-1' });
    expect(JSON.parse(String(init.body))).toEqual(request);
  });

  it('retries 429, 502, 503 and 504 with backoff or Retry-After, then succeeds', async () => {
    const { client, waits } = scripted([() => json(503, { error: 'busy' }), () => json(429, { error: 'rate' }, { 'retry-after': '2' }), () => json(200, answer)]);
    expect(await client.chat(request, new AbortController().signal)).toEqual(answer);
    expect(waits).toEqual([1000, 2000]);
  });

  it('gives up after the retries and reports the proxy message', async () => {
    const { client, waits } = scripted([() => json(502, { error: 'The AI provider failed (HTTP 500).' }), () => json(502, { error: 'x' }), () => json(504, { error: 'The AI provider did not answer within 30 s.' })]);
    const e = await failure(client.chat(request, new AbortController().signal));
    expect(e.message).toBe('The AI provider did not answer within 30 s.');
    expect(e.status).toBe(504);
    expect(waits).toEqual([1000, 2000]);
  });

  it('does not retry refusals or waits longer than 10 s', async () => {
    const refused = scripted([() => json(400, { error: 'Invalid chat request: messages: Too small' })]);
    const e = await failure(refused.client.chat(request, new AbortController().signal));
    expect(e).toMatchObject({ message: 'Invalid chat request: messages: Too small', status: 400, cancelled: false });
    const slow = scripted([() => json(429, { error: 'Too many AI requests. Wait a moment and try again.' }, { 'retry-after': '60' })]);
    expect((await failure(slow.client.chat(request, new AbortController().signal))).status).toBe(429);
    expect(slow.waits).toEqual([]);
  });

  it('reports an unreachable proxy, an unexpected answer and a dev-server error page plainly', async () => {
    const down = scripted([() => Promise.reject(new TypeError('Failed to fetch'))]);
    expect((await failure(down.client.chat(request, new AbortController().signal))).message).toBe(PROXY_OFFLINE);
    const odd = scripted([() => json(200, { content: 5 })]);
    expect((await failure(odd.client.chat(request, new AbortController().signal))).message).toBe('The AI proxy returned an unexpected answer.');
    const page = scripted([() => new Response('proxy error', { status: 500 })]);
    expect((await failure(page.client.chat(request, new AbortController().signal))).message).toBe(PROXY_OFFLINE);
  });

  it('stops when cancelled', async () => {
    const c = new AbortController();
    c.abort();
    const { client, calls } = scripted([() => json(200, answer)]);
    expect(await failure(client.chat(request, c.signal))).toMatchObject({ cancelled: true });
    expect(calls).toHaveLength(0);
    const live = new AbortController();
    const hanging = new ProxyClient({
      fetchImpl: (_url, init) => new Promise((_resolve, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))),
    });
    const pending = failure(hanging.chat(request, live.signal));
    live.abort();
    expect(await pending).toMatchObject({ cancelled: true });
  });
});

it('creates random session IDs that fit the proxy header pattern', () => {
  const a = newSessionId();
  expect(a).toMatch(/^[0-9a-f]{32}$/);
  expect(newSessionId()).not.toBe(a);
});
