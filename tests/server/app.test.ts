import { afterEach, describe, expect, it } from 'vitest';
import type { FastifyInstance } from 'fastify';
import type { ChatRequest, ChatResult } from '../../src/ai/protocol.ts';
import { UpstreamError, type ChatAdapter, type FetchLike } from '../../server/adapter.ts';
import { buildProxy, type ProxyOptions } from '../../server/app.ts';
import { createAzureOpenAIAdapter } from '../../server/azure.ts';
import { configFromValues } from '../../server/config.ts';

// Fake values only; real credentials never appear in tests.
const KEY = 'fake-key-0123456789abcdef';
const ENDPOINT = 'https://fake-resource.openai.azure.com/openai/deployments/fake-model/chat/completions?api-version=2024-10-21';
const ORIGIN = 'http://localhost:5173';
const SESSION = 'session-0001';

const answer: ChatResult = { content: 'Done.', toolCalls: [], finishReason: 'stop', model: 'fake-model', usage: { promptTokens: 10, completionTokens: 2 } };
const userTurn = { messages: [{ role: 'user', content: 'Drive to the red cube.' }] };

function fakeAdapter(impl?: (r: ChatRequest, s: AbortSignal) => Promise<ChatResult>): ChatAdapter & { calls: ChatRequest[] } {
  const calls: ChatRequest[] = [];
  return {
    provider: 'fake',
    calls,
    capabilities: () => ({
      toolCalling: { state: 'yes', note: '' },
      imageInput: { state: 'no', note: '' },
      streaming: { state: 'no', note: '' },
      realtimeVoice: { state: 'no', note: '' },
    }),
    chat: async (r, s) => {
      calls.push(r);
      return impl ? impl(r, s) : answer;
    },
  };
}

let app: FastifyInstance | null = null;
afterEach(async () => {
  await app?.close();
  app = null;
});

function proxy(over: Partial<ProxyOptions> = {}): FastifyInstance {
  app = buildProxy({ adapter: fakeAdapter(), allowedOrigins: [ORIGIN], ...over });
  return app;
}

function post(a: FastifyInstance, payload: unknown, headers: Record<string, string> = {}) {
  return a.inject({ method: 'POST', url: '/api/ai/chat', headers: { origin: ORIGIN, 'x-morphlab-session': SESSION, ...headers }, payload: payload as Record<string, unknown> });
}

describe('AI proxy routes', () => {
  it('reports status without secrets, and why AI is unavailable when unconfigured', async () => {
    const ok = await proxy().inject({ method: 'GET', url: '/api/ai/status' });
    expect(ok.statusCode).toBe(200);
    expect(ok.json()).toMatchObject({ ok: true, provider: 'fake', reason: null });
    expect(ok.headers['cache-control']).toBe('no-store');
    await app?.close();
    app = null;
    const offline = proxy({ adapter: null, unavailableReason: 'API_KEY is missing or empty in .env.' });
    const off = await offline.inject({ method: 'GET', url: '/api/ai/status' });
    expect(off.json()).toEqual({ ok: false, provider: null, reason: 'API_KEY is missing or empty in .env.', capabilities: null });
    expect((await post(offline, userTurn)).statusCode).toBe(503);
  });

  it('forwards a valid chat request and returns the sanitized result', async () => {
    const adapter = fakeAdapter();
    const a = proxy({ adapter });
    const res = await post(a, { ...userTurn, tools: [{ name: 'navigateTo', description: 'Drive.', parameters: { type: 'object' } }], toolChoice: 'auto' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual(answer);
    expect(adapter.calls[0]?.tools?.[0]?.name).toBe('navigateTo');
  });

  it('refuses images: the app never sends one (D31)', async () => {
    const adapter = fakeAdapter();
    const a = proxy({ adapter });
    const img = 'data:image/jpeg;base64,/9j/4AAQSkZJRg==';
    const bad: unknown[][] = [[{ role: 'user', content: 'Camera image.', image: img }], [{ role: 'system', content: 'x', image: img }]];
    for (const messages of bad) expect((await post(a, { messages })).statusCode, JSON.stringify(messages).slice(0, 80)).toBe(400);
    expect(adapter.calls).toHaveLength(0);
  });

  it('refuses other origins, cross-site requests, foreign hosts and POSTs without an Origin', async () => {
    const a = proxy({ allowedHosts: ['127.0.0.1:8787'] });
    const host = { host: '127.0.0.1:8787' };
    expect((await post(a, userTurn, { ...host, origin: 'https://evil.example' })).statusCode).toBe(403);
    expect((await post(a, userTurn, { ...host, 'sec-fetch-site': 'cross-site' })).statusCode).toBe(403);
    expect((await post(a, userTurn, { host: 'evil.example:8787' })).statusCode).toBe(403);
    const noOrigin = await a.inject({ method: 'POST', url: '/api/ai/chat', headers: { ...host, 'x-morphlab-session': SESSION }, payload: userTurn });
    expect(noOrigin.statusCode).toBe(403);
    expect((await post(a, userTurn, host)).statusCode).toBe(200);
  });

  it('requires a well-formed session header', async () => {
    const a = proxy();
    expect((await post(a, userTurn, { 'x-morphlab-session': 'short' })).statusCode).toBe(400);
    expect((await post(a, userTurn, { 'x-morphlab-session': 'bad session id!' })).statusCode).toBe(400);
  });

  it('validates the body: schema, unknown tools, size, media type and JSON syntax', async () => {
    const a = proxy({ bodyLimitBytes: 2048 });
    const badRole = await post(a, { messages: [{ role: 'root', content: 'x' }] });
    expect(badRole.statusCode).toBe(400);
    expect(badRole.json().error).toMatch(/^Invalid chat request: /);
    const extraKey = await post(a, { ...userTurn, temperature: 2 });
    expect(extraKey.statusCode).toBe(400);
    const foreignTool = await post(a, { ...userTurn, tools: [{ name: 'shell_exec', description: 'Run a command.', parameters: {} }] });
    expect(foreignTool.statusCode).toBe(400);
    expect(foreignTool.json().error).toBe('shell_exec is not a MorphLab tool.');
    const big = await post(a, { messages: [{ role: 'user', content: 'x'.repeat(4000) }] });
    expect(big.statusCode).toBe(413);
    expect(big.json()).toEqual({ error: 'The request body is too large.' });
    const text = await a.inject({ method: 'POST', url: '/api/ai/chat', headers: { origin: ORIGIN, 'x-morphlab-session': SESSION, 'content-type': 'text/plain' }, payload: 'hello' });
    expect(text.statusCode).toBe(415);
    const broken = await a.inject({ method: 'POST', url: '/api/ai/chat', headers: { origin: ORIGIN, 'x-morphlab-session': SESSION, 'content-type': 'application/json' }, payload: '{"messages":' });
    expect(broken.statusCode).toBe(400);
    expect(broken.json()).toEqual({ error: 'The request body could not be parsed.' });
  });

  it('limits requests per session and overall, then recovers after the window', async () => {
    let t = 0;
    const a = proxy({ requestsPerWindow: 2, globalRequestsPerWindow: 3, windowMs: 60_000, now: () => t });
    expect((await post(a, userTurn)).statusCode).toBe(200);
    expect((await post(a, userTurn)).statusCode).toBe(200);
    const limited = await post(a, userTurn);
    expect(limited.statusCode).toBe(429);
    expect(limited.headers['retry-after']).toBe('60');
    expect((await post(a, userTurn, { 'x-morphlab-session': 'session-0002' })).statusCode).toBe(200);
    expect((await post(a, userTurn, { 'x-morphlab-session': 'session-0003' })).statusCode).toBe(429);
    t = 60_000;
    expect((await post(a, userTurn)).statusCode).toBe(200);
  });

  it('hides unexpected errors and answers unknown routes with 404', async () => {
    const a = proxy({ adapter: fakeAdapter(async () => Promise.reject(new Error(`internal detail ${KEY}`))) });
    const res = await post(a, userTurn);
    expect(res.statusCode).toBe(500);
    expect(res.json()).toEqual({ error: 'Internal proxy error.' });
    const missing = await a.inject({ method: 'GET', url: '/api/other' });
    expect(missing.statusCode).toBe(404);
    expect(missing.json()).toEqual({ error: 'Not found.' });
  });

  it('passes upstream errors through as safe messages', async () => {
    const a = proxy({ adapter: fakeAdapter(async () => Promise.reject(new UpstreamError(429, 'The AI provider rate limit was reached. Try again shortly.', 7))) });
    const res = await post(a, userTurn);
    expect(res.statusCode).toBe(429);
    expect(res.headers['retry-after']).toBe('7');
  });
});

interface Captured {
  url: string;
  headers: Record<string, string>;
  body: Record<string, unknown>;
}

function mockFetch(respond: (signal: AbortSignal | null | undefined) => Promise<Response>): FetchLike & { captured: Captured[] } {
  const captured: Captured[] = [];
  const fn = async (input: URL, init: RequestInit): Promise<Response> => {
    captured.push({ url: input.href, headers: init.headers as Record<string, string>, body: JSON.parse(String(init.body)) as Record<string, unknown> });
    return respond(init.signal);
  };
  return Object.assign(fn, { captured });
}

function azureProxy(respond: (signal: AbortSignal | null | undefined) => Promise<Response>, endpoint = ENDPOINT, timeoutMs = 30_000) {
  const fetchImpl = mockFetch(respond);
  const adapter = createAzureOpenAIAdapter(configFromValues({ API_KEY: KEY, ENDPOINT: endpoint }), fetchImpl, timeoutMs);
  return { a: proxy({ adapter }), fetchImpl };
}

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } }));

function expectNoSecrets(text: string): void {
  expect(text).not.toContain(KEY);
  expect(text).not.toContain('fake-resource');
  expect(text).not.toContain('openai.azure.com');
}

describe('Azure OpenAI adapter behind the proxy (mocked upstream)', () => {
  it('sends the api-key header and the mapped body, and maps tool calls back', async () => {
    const { a, fetchImpl } = azureProxy(() =>
      json(200, {
        model: 'fake-model-2024',
        choices: [{ finish_reason: 'tool_calls', message: { content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'navigateTo', arguments: '{"x":1,"y":0}' } }] } }],
        usage: { prompt_tokens: 120, completion_tokens: 18, total_tokens: 138 },
      }),
    );
    const res = await post(a, {
      messages: [
        { role: 'system', content: 'You control a simulated robot through tools.' },
        { role: 'user', content: 'Drive forward one meter.' },
        { role: 'assistant', content: null, toolCalls: [{ id: 'call_0', name: 'getRobotState', arguments: '{}' }] },
        { role: 'tool', toolCallId: 'call_0', content: '{"status":"succeeded"}' },
      ],
      tools: [{ name: 'navigateTo', description: 'Drive.', parameters: { type: 'object', properties: {} } }],
      maxOutputTokens: 256,
    });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({
      content: null,
      toolCalls: [{ id: 'call_1', name: 'navigateTo', arguments: '{"x":1,"y":0}' }],
      finishReason: 'tool_calls',
      model: 'fake-model-2024',
      usage: { promptTokens: 120, completionTokens: 18 },
    });
    expectNoSecrets(res.body);
    const sent = fetchImpl.captured[0];
    expect(sent?.url).toBe(ENDPOINT);
    expect(sent?.headers['api-key']).toBe(KEY);
    expect(sent?.headers).not.toHaveProperty('authorization');
    expect(sent?.body).toEqual({
      messages: [
        { role: 'system', content: 'You control a simulated robot through tools.' },
        { role: 'user', content: 'Drive forward one meter.' },
        { role: 'assistant', content: null, tool_calls: [{ id: 'call_0', type: 'function', function: { name: 'getRobotState', arguments: '{}' } }] },
        { role: 'tool', tool_call_id: 'call_0', content: '{"status":"succeeded"}' },
      ],
      tools: [{ type: 'function', function: { name: 'navigateTo', description: 'Drive.', parameters: { type: 'object', properties: {} } } }],
      tool_choice: 'auto',
      max_completion_tokens: 256,
    });
  });

  it('uses max_tokens for api-versions before 2024-09-01', async () => {
    const { a, fetchImpl } = azureProxy(() => json(200, { choices: [{ finish_reason: 'stop', message: { content: 'Hi.' } }] }), ENDPOINT.replace('2024-10-21', '2024-06-01'));
    expect((await post(a, userTurn)).statusCode).toBe(200);
    expect(fetchImpl.captured[0]?.body).toMatchObject({ max_tokens: 1024 });
    expect(fetchImpl.captured[0]?.body).not.toHaveProperty('max_completion_tokens');
  });

  it('maps upstream failures to safe messages without echoing the upstream body', async () => {
    const leakyBody = { error: { code: 'BadRequest', message: `deployment fake-model at fake-resource rejected key ${KEY}`, param: 'max_completion_tokens' } };
    const cases: [() => Promise<Response>, number, string][] = [
      [() => json(401, { error: { code: '401', message: `bad key ${KEY}` } }), 502, 'The AI provider rejected the proxy credentials. Check API_KEY and ENDPOINT in .env.'],
      [() => json(404, { error: { code: 'DeploymentNotFound', message: 'fake-model' } }), 502, 'The AI provider did not find the deployment. Check ENDPOINT in .env.'],
      [() => json(400, leakyBody), 502, 'The AI provider rejected the request (HTTP 400, code BadRequest, parameter max_completion_tokens).'],
      [() => json(400, { error: { code: 'content_filter', message: 'filtered' } }), 422, 'The provider content filter blocked this request.'],
      [() => json(503, { error: { message: 'fake-resource overloaded' } }), 502, 'The AI provider failed (HTTP 503).'],
      [() => json(200, { unexpected: true }), 502, 'The AI provider returned an unexpected response.'],
      [() => Promise.resolve(new Response('not json', { status: 200 })), 502, 'The AI provider could not be reached or returned an unreadable answer.'],
      [() => Promise.reject(new TypeError(`fetch failed for ${ENDPOINT}`)), 502, 'The AI provider could not be reached or returned an unreadable answer.'],
    ];
    for (const [respond, status, message] of cases) {
      const { a } = azureProxy(respond);
      const res = await post(a, userTurn);
      expect(res.statusCode, message).toBe(status);
      expect(res.json()).toEqual({ error: message });
      expectNoSecrets(res.body);
      await a.close();
      app = null;
    }
  });

  it('passes the upstream Retry-After on 429', async () => {
    const { a } = azureProxy(() => json(429, { error: { code: '429' } }, { 'retry-after': '12' }));
    const res = await post(a, userTurn);
    expect(res.statusCode).toBe(429);
    expect(res.headers['retry-after']).toBe('12');
  });

  it('times out a slow provider with 504', async () => {
    const { a } = azureProxy(
      (signal) =>
        new Promise<Response>((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError')));
        }),
      ENDPOINT,
      50,
    );
    const res = await post(a, userTurn);
    expect(res.statusCode).toBe(504);
    expect(res.json()).toEqual({ error: 'The AI provider did not answer within 0.05 s.' });
  });
});
