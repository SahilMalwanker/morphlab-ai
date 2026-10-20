import type { AiStatus, CapabilityState, ChatRequest, ChatResult, ProviderCapabilities } from './protocol.ts';

const CAP_STATES = new Set<string>(['yes', 'no', 'unknown']);

function parseCap(v: unknown): { state: CapabilityState; note: string } | null {
  if (typeof v !== 'object' || v === null) return null;
  const o = v as Record<string, unknown>;
  if (typeof o['state'] !== 'string' || !CAP_STATES.has(o['state']) || typeof o['note'] !== 'string' || o['note'].length > 500) return null;
  return { state: o['state'] as CapabilityState, note: o['note'] };
}

function parseAiStatus(raw: unknown): AiStatus | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const o = raw as Record<string, unknown>;
  if (typeof o['ok'] !== 'boolean') return null;
  if (o['provider'] !== null && (typeof o['provider'] !== 'string' || o['provider'].length > 64)) return null;
  if (o['reason'] !== null && (typeof o['reason'] !== 'string' || o['reason'].length > 500)) return null;
  let capabilities: ProviderCapabilities | null = null;
  if (o['capabilities'] !== null) {
    if (typeof o['capabilities'] !== 'object') return null;
    const c = o['capabilities'] as Record<string, unknown>;
    const toolCalling = parseCap(c['toolCalling']);
    const imageInput = parseCap(c['imageInput']);
    const streaming = parseCap(c['streaming']);
    const realtimeVoice = parseCap(c['realtimeVoice']);
    if (!toolCalling || !imageInput || !streaming || !realtimeVoice) return null;
    capabilities = { toolCalling, imageInput, streaming, realtimeVoice };
  }
  return { ok: o['ok'], provider: o['provider'], reason: o['reason'], capabilities };
}

export type AiAvailability =
  | { state: 'unknown' }
  | { state: 'available'; provider: string; capabilities: ProviderCapabilities | null }
  | { state: 'unavailable'; reason: string };

/** A failed AI request; `message` is safe to show (the proxy never forwards upstream bodies). */
export class AiError extends Error {
  readonly status: number | null;
  readonly cancelled: boolean;

  constructor(message: string, status: number | null, cancelled = false) {
    super(message);
    this.name = 'AiError';
    this.status = status;
    this.cancelled = cancelled;
  }
}

export type FetchFn = (input: string, init: RequestInit) => Promise<Response>;

export interface ProxyClientOptions {
  baseUrl?: string;
  fetchImpl?: FetchFn;
  sessionId?: string;
  /** Retries after 429, 502, 503 and 504 answers. */
  maxRetries?: number;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  statusTimeoutMs?: number;
  chatTimeoutMs?: number;
}

export const PROXY_OFFLINE = 'The AI proxy is not running here. Start the app with npm run dev to use AI commands; everything else works without it.';
const RETRYABLE = new Set([429, 502, 503, 504]);
const MAX_RETRY_WAIT_MS = 10_000;

/** A random session ID for the proxy's per-session rate limit (not a credential). */
export function newSessionId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function cancelled(): AiError {
  return new AiError('Cancelled.', null, true);
}

function defaultSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(cancelled());
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(cancelled());
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (typeof body === 'object' && body !== null && 'error' in body && typeof body.error === 'string') return body.error.slice(0, 300);
  } catch {
    // Not a proxy answer (for example the dev server's own error page).
  }
  return res.status >= 500 ? PROXY_OFFLINE : `The AI proxy refused the request (HTTP ${res.status}).`;
}

/** Talks to the local AI proxy (same origin). Holds no credentials. */
export class ProxyClient {
  readonly #base: string;
  readonly #fetch: FetchFn;
  readonly #session: string;
  readonly #maxRetries: number;
  readonly #sleep: (ms: number, signal: AbortSignal) => Promise<void>;
  readonly #statusTimeoutMs: number;
  readonly #chatTimeoutMs: number;

  constructor(opts: ProxyClientOptions = {}) {
    this.#base = opts.baseUrl ?? '/api/ai/';
    this.#fetch = opts.fetchImpl ?? ((input, init) => fetch(input, init));
    this.#session = opts.sessionId ?? newSessionId();
    this.#maxRetries = opts.maxRetries ?? 2;
    this.#sleep = opts.sleep ?? defaultSleep;
    this.#statusTimeoutMs = opts.statusTimeoutMs ?? 3000;
    this.#chatTimeoutMs = opts.chatTimeoutMs ?? 60_000;
  }

  async status(): Promise<AiAvailability> {
    try {
      const res = await this.#fetch(`${this.#base}status`, { method: 'GET', headers: { accept: 'application/json' }, signal: AbortSignal.timeout(this.#statusTimeoutMs) });
      if (!res.ok) return { state: 'unavailable', reason: res.status === 404 || res.status >= 500 ? PROXY_OFFLINE : `The AI proxy answered with HTTP ${res.status}.` };
      const s = parseAiStatus(await res.json());
      if (!s) return { state: 'unavailable', reason: PROXY_OFFLINE };
      if (!s.ok || !s.provider) return { state: 'unavailable', reason: s.reason ?? 'The AI proxy is not configured.' };
      return { state: 'available', provider: s.provider, capabilities: s.capabilities };
    } catch {
      return { state: 'unavailable', reason: PROXY_OFFLINE };
    }
  }

  async chat(request: ChatRequest, signal: AbortSignal): Promise<ChatResult> {
    const { ChatResultSchema } = await import('./protocol.ts');
    for (let attempt = 0; ; attempt++) {
      if (signal.aborted) throw cancelled();
      let res: Response;
      try {
        res = await this.#fetch(`${this.#base}chat`, {
          method: 'POST',
          headers: { 'content-type': 'application/json', 'x-morphlab-session': this.#session },
          body: JSON.stringify(request),
          signal: AbortSignal.any([signal, AbortSignal.timeout(this.#chatTimeoutMs)]),
        });
      } catch {
        if (signal.aborted) throw cancelled();
        throw new AiError(PROXY_OFFLINE, null);
      }
      if (res.ok) {
        let body: unknown = null;
        try {
          body = await res.json();
        } catch {
          if (signal.aborted) throw cancelled();
        }
        const parsed = ChatResultSchema.safeParse(body);
        if (!parsed.success) throw new AiError('The AI proxy returned an unexpected answer.', res.status);
        return parsed.data;
      }
      const message = await errorMessage(res);
      if (RETRYABLE.has(res.status) && attempt < this.#maxRetries) {
        const header = Number(res.headers.get('retry-after'));
        const wait = Number.isFinite(header) && header > 0 ? header * 1000 : 1000 * 2 ** attempt;
        if (wait <= MAX_RETRY_WAIT_MS) {
          await this.#sleep(wait, signal);
          continue;
        }
      }
      throw new AiError(message, res.status);
    }
  }
}
