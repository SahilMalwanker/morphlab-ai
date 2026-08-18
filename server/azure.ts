import * as z from 'zod';
import type { ChatMessage, ChatRequest, ChatResult, ProviderCapabilities } from '../src/ai/protocol.ts';
import { UpstreamError, type ChatAdapter, type FetchLike } from './adapter.ts';
import type { ProviderConfig } from './config.ts';

const DEFAULT_MAX_OUTPUT_TOKENS = 1024;
const SAFE_TOKEN = /^[A-Za-z0-9_.-]{1,64}$/;

const AzureResponseSchema = z.object({
  model: z.string().optional(),
  choices: z
    .array(
      z.object({
        finish_reason: z.string().nullable().optional(),
        message: z
          .object({
            content: z.string().nullable().optional(),
            tool_calls: z
              .array(z.object({ id: z.string(), function: z.object({ name: z.string(), arguments: z.string() }) }))
              .optional(),
          })
          .optional(),
      }),
    )
    .min(1),
  usage: z.object({ prompt_tokens: z.number(), completion_tokens: z.number() }).optional(),
});

const AzureErrorSchema = z.object({ error: z.object({ code: z.string().nullable().optional(), param: z.string().nullable().optional() }) });

function toAzureMessage(m: ChatMessage): Record<string, unknown> {
  switch (m.role) {
    case 'system':
    case 'user':
      return { role: m.role, content: m.content };
    case 'assistant':
      if (m.toolCalls && m.toolCalls.length > 0) {
        return {
          role: 'assistant',
          content: m.content,
          tool_calls: m.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } })),
        };
      }
      return { role: 'assistant', content: m.content ?? '' };
    case 'tool':
      return { role: 'tool', tool_call_id: m.toolCallId, content: m.content };
  }
}

/** Builds the Azure OpenAI chat completions body. `max_completion_tokens` replaces `max_tokens` from api-version 2024-09-01. */
export function buildAzureBody(request: ChatRequest, apiVersion: string): Record<string, unknown> {
  const body: Record<string, unknown> = { messages: request.messages.map(toAzureMessage) };
  if (request.tools && request.tools.length > 0) {
    body['tools'] = request.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
    body['tool_choice'] = request.toolChoice ?? 'auto';
  }
  const limit = request.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS;
  body[apiVersion >= '2024-09-01' ? 'max_completion_tokens' : 'max_tokens'] = limit;
  return body;
}

function retryAfter(res: Response): number | null {
  const raw = Number(res.headers.get('retry-after'));
  return Number.isFinite(raw) && raw > 0 ? Math.min(Math.ceil(raw), 300) : null;
}

async function errorDetail(res: Response): Promise<{ code: string | null; param: string | null }> {
  try {
    const parsed = AzureErrorSchema.safeParse(await res.json());
    if (!parsed.success) return { code: null, param: null };
    const { code, param } = parsed.data.error;
    return { code: code && SAFE_TOKEN.test(code) ? code : null, param: param && SAFE_TOKEN.test(param) ? param : null };
  } catch {
    return { code: null, param: null };
  }
}

async function mapHttpError(res: Response): Promise<UpstreamError> {
  if (res.status === 401 || res.status === 403) {
    return new UpstreamError(502, 'The AI provider rejected the proxy credentials. Check API_KEY and ENDPOINT in .env.');
  }
  if (res.status === 404) return new UpstreamError(502, 'The AI provider did not find the deployment. Check ENDPOINT in .env.');
  if (res.status === 429) return new UpstreamError(429, 'The AI provider rate limit was reached. Try again shortly.', retryAfter(res));
  if (res.status >= 500) return new UpstreamError(502, `The AI provider failed (HTTP ${res.status}).`);
  const { code, param } = await errorDetail(res);
  if (code === 'content_filter') return new UpstreamError(422, 'The provider content filter blocked this request.');
  const details = [code && `code ${code}`, param && `parameter ${param}`].filter(Boolean).join(', ');
  return new UpstreamError(502, `The AI provider rejected the request (HTTP ${res.status}${details ? `, ${details}` : ''}).`);
}

export function createAzureOpenAIAdapter(config: ProviderConfig, fetchImpl: FetchLike, timeoutMs = 30_000): ChatAdapter {
  const apiVersion = config.endpoint.searchParams.get('api-version') ?? '';
  return {
    provider: config.provider,
    capabilities(): ProviderCapabilities {
      return {
        toolCalling: { state: 'unknown', note: 'Not probed yet; depends on the deployed model.' },
        imageInput: { state: 'unknown', note: 'Not probed yet; depends on the deployed model.' },
        streaming: { state: 'no', note: 'This proxy version returns complete answers.' },
        realtimeVoice: { state: 'no', note: 'A chat completions deployment has no realtime audio; voice uses the browser speech API.' },
      };
    },
    async chat(request: ChatRequest, signal: AbortSignal): Promise<ChatResult> {
      const timeout = AbortSignal.timeout(timeoutMs);
      let res: Response;
      let json: unknown;
      try {
        res = await fetchImpl(config.endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json', 'api-key': config.apiKey },
          body: JSON.stringify(buildAzureBody(request, apiVersion)),
          signal: AbortSignal.any([signal, timeout]),
        });
        if (!res.ok) throw await mapHttpError(res);
        json = await res.json();
      } catch (e) {
        if (e instanceof UpstreamError) throw e;
        if (timeout.aborted) throw new UpstreamError(504, `The AI provider did not answer within ${timeoutMs / 1000} s.`);
        if (signal.aborted) throw new UpstreamError(499, 'The request was cancelled.');
        throw new UpstreamError(502, 'The AI provider could not be reached or returned an unreadable answer.');
      }
      const parsed = AzureResponseSchema.safeParse(json);
      if (!parsed.success) throw new UpstreamError(502, 'The AI provider returned an unexpected response.');
      const choice = parsed.data.choices[0];
      const message = choice?.message;
      return {
        content: message?.content ?? null,
        toolCalls: (message?.tool_calls ?? []).map((c) => ({ id: c.id, name: c.function.name, arguments: c.function.arguments })),
        finishReason: choice?.finish_reason ?? 'unknown',
        model: parsed.data.model ?? null,
        usage: parsed.data.usage ? { promptTokens: parsed.data.usage.prompt_tokens, completionTokens: parsed.data.usage.completion_tokens } : null,
      };
    },
  };
}
