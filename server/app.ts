import Fastify, { type FastifyInstance } from 'fastify';
import { ChatRequestSchema, type AiStatus } from '../src/ai/protocol.ts';
import { TOOL_NAMES } from '../src/ai/tools.ts';
import { UpstreamError, type ChatAdapter } from './adapter.ts';

export interface ProxyOptions {
  adapter: ChatAdapter | null;
  /** Shown by the status route when no adapter is configured. */
  unavailableReason?: string;
  /** Exact browser origins allowed to call the proxy. */
  allowedOrigins: readonly string[];
  /** When set, requests whose Host header is not listed are refused (DNS rebinding guard). */
  allowedHosts?: readonly string[];
  requestsPerWindow?: number;
  globalRequestsPerWindow?: number;
  windowMs?: number;
  bodyLimitBytes?: number;
  now?: () => number;
}

export const SESSION_HEADER = 'x-morphlab-session';
const SESSION_PATTERN = /^[A-Za-z0-9-]{8,64}$/;
const MAX_SESSIONS = 1000;

function statusCodeOf(error: unknown): number | null {
  if (typeof error === 'object' && error !== null && 'statusCode' in error && typeof error.statusCode === 'number') return error.statusCode;
  return null;
}

function summarizeIssues(issues: ReadonlyArray<{ path: ReadonlyArray<PropertyKey>; message: string }>): string {
  return issues
    .slice(0, 3)
    .map((i) => `${i.path.map(String).join('.') || 'body'}: ${i.message}`)
    .join('; ');
}

/** The local AI proxy: narrow routes, same-origin only, bounded requests, sanitized errors. */
export function buildProxy(opts: ProxyOptions): FastifyInstance {
  const app = Fastify({ logger: false, bodyLimit: opts.bodyLimitBytes ?? 256 * 1024 });
  const origins = new Set(opts.allowedOrigins);
  const hosts = opts.allowedHosts ? new Set(opts.allowedHosts) : null;
  const perSession = opts.requestsPerWindow ?? 20;
  const perWindow = opts.globalRequestsPerWindow ?? perSession * 5;
  const windowMs = opts.windowMs ?? 60_000;
  const now = opts.now ?? Date.now;
  const sessions = new Map<string, { start: number; count: number }>();
  const overall = { start: now(), count: 0 };

  /** Fixed-window limits per session and overall; returns seconds to wait, or null when allowed. */
  function limited(session: string): number | null {
    const t = now();
    if (t - overall.start >= windowMs) {
      overall.start = t;
      overall.count = 0;
    }
    if (sessions.size >= MAX_SESSIONS) {
      for (const [id, w] of sessions) if (t - w.start >= windowMs) sessions.delete(id);
    }
    let w = sessions.get(session);
    if (!w || t - w.start >= windowMs) {
      if (!w && sessions.size >= MAX_SESSIONS) return Math.max(1, Math.ceil((overall.start + windowMs - t) / 1000));
      w = { start: t, count: 0 };
      sessions.set(session, w);
    }
    if (w.count >= perSession) return Math.max(1, Math.ceil((w.start + windowMs - t) / 1000));
    if (overall.count >= perWindow) return Math.max(1, Math.ceil((overall.start + windowMs - t) / 1000));
    w.count++;
    overall.count++;
    return null;
  }

  app.removeContentTypeParser('text/plain');

  app.addHook('onRequest', async (request, reply) => {
    if (hosts && !hosts.has(request.headers.host ?? '')) return reply.code(403).send({ error: 'This host is not allowed.' });
    if (request.headers['sec-fetch-site'] === 'cross-site') return reply.code(403).send({ error: 'Cross-site requests are not allowed.' });
    const origin = request.headers.origin;
    if (origin !== undefined && !origins.has(origin)) return reply.code(403).send({ error: 'This origin is not allowed.' });
    if (request.method !== 'GET' && request.method !== 'HEAD' && origin === undefined) {
      return reply.code(403).send({ error: 'An Origin header is required.' });
    }
  });

  app.addHook('onSend', async (_request, reply, payload) => {
    reply.header('cache-control', 'no-store');
    reply.header('x-content-type-options', 'nosniff');
    return payload;
  });

  app.get('/api/ai/status', async (_request, reply) => {
    const status: AiStatus = opts.adapter
      ? { ok: true, provider: opts.adapter.provider, reason: null, capabilities: opts.adapter.capabilities() }
      : { ok: false, provider: null, reason: opts.unavailableReason ?? 'The AI proxy is not configured.', capabilities: null };
    return reply.send(status);
  });

  app.post('/api/ai/chat', async (request, reply) => {
    const adapter = opts.adapter;
    if (!adapter) return reply.code(503).send({ error: opts.unavailableReason ?? 'The AI proxy is not configured.' });
    const session = request.headers[SESSION_HEADER];
    if (typeof session !== 'string' || !SESSION_PATTERN.test(session)) {
      return reply.code(400).send({ error: `A valid ${SESSION_HEADER} header is required.` });
    }
    const wait = limited(session);
    if (wait !== null) return reply.code(429).header('retry-after', String(wait)).send({ error: 'Too many AI requests. Wait a moment and try again.' });
    const parsed = ChatRequestSchema.safeParse(request.body);
    if (!parsed.success) return reply.code(400).send({ error: `Invalid chat request: ${summarizeIssues(parsed.error.issues)}` });
    // Only MorphLab's own tools may be offered to the model, so the proxy is not a general relay.
    const unknown = (parsed.data.tools ?? []).map((t) => t.name).find((n) => !TOOL_NAMES.has(n));
    if (unknown !== undefined) return reply.code(400).send({ error: `${unknown} is not a MorphLab tool.` });

    const controller = new AbortController();
    const onClose = (): void => {
      if (!reply.raw.writableEnded) controller.abort();
    };
    reply.raw.on('close', onClose);
    try {
      const result = await adapter.chat(parsed.data, controller.signal);
      return reply.send(result);
    } catch (e) {
      if (!(e instanceof UpstreamError)) throw e;
      if (e.retryAfterSeconds !== null) reply.header('retry-after', String(e.retryAfterSeconds));
      return reply.code(e.status).send({ error: e.message });
    } finally {
      reply.raw.off('close', onClose);
    }
  });

  app.setNotFoundHandler((_request, reply) => {
    void reply.code(404).send({ error: 'Not found.' });
  });

  app.setErrorHandler((error: unknown, _request, reply) => {
    const status = statusCodeOf(error);
    if (status === 413) return reply.code(413).send({ error: 'The request body is too large.' });
    if (status === 415) return reply.code(415).send({ error: 'Send JSON with content-type application/json.' });
    if (status === 400) return reply.code(400).send({ error: 'The request body could not be parsed.' });
    if (status !== null && status > 400 && status < 500) return reply.code(status).send({ error: 'The request was refused.' });
    return reply.code(500).send({ error: 'Internal proxy error.' });
  });

  return app;
}
