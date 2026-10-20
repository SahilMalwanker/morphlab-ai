import * as z from 'zod';

/** Browser-to-proxy chat contract. The proxy accepts nothing outside this schema. */
export const MAX_MESSAGE_CHARS = 16_000;
export const MAX_TOTAL_CHARS = 48_000;
export const MAX_ARGUMENT_CHARS = 4000;
export const MAX_MESSAGES = 40;
export const MAX_TOOLS = 24;
export const MAX_TOOL_CALLS = 8;
export const MAX_DESCRIPTION_CHARS = 1024;
export const OUTPUT_TOKEN_RANGE = [16, 2048] as const;

const ident = z.string().min(1).max(64).regex(/^[A-Za-z0-9_.:-]+$/);
const toolName = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const text = z.string().max(MAX_MESSAGE_CHARS);

const ToolCallSchema = z.strictObject({ id: ident, name: toolName, arguments: z.string().max(MAX_ARGUMENT_CHARS) });

export const ChatMessageSchema = z.discriminatedUnion('role', [
  z.strictObject({ role: z.literal('system'), content: text.min(1) }),
  z.strictObject({ role: z.literal('user'), content: text.min(1) }),
  z.strictObject({ role: z.literal('assistant'), content: text.nullable(), toolCalls: z.array(ToolCallSchema).max(MAX_TOOL_CALLS).optional() }),
  z.strictObject({ role: z.literal('tool'), toolCallId: ident, content: text }),
]);

export const ModelToolSpecSchema = z.strictObject({
  name: toolName,
  description: z.string().max(MAX_DESCRIPTION_CHARS),
  parameters: z.record(z.string(), z.unknown()),
});

export const ChatRequestSchema = z
  .strictObject({
    messages: z.array(ChatMessageSchema).min(1).max(MAX_MESSAGES),
    tools: z.array(ModelToolSpecSchema).max(MAX_TOOLS).optional(),
    toolChoice: z.enum(['auto', 'none', 'required']).optional(),
    maxOutputTokens: z.int().min(OUTPUT_TOKEN_RANGE[0]).max(OUTPUT_TOKEN_RANGE[1]).optional(),
  })
  .refine((r) => requestChars(r) <= MAX_TOTAL_CHARS, { message: `The request is longer than ${MAX_TOTAL_CHARS} characters.` });

export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatRequest = z.infer<typeof ChatRequestSchema>;

/** Characters a request counts against MAX_TOTAL_CHARS: message texts, tool-call arguments and the tool list. */
export function requestChars(r: { messages: ChatMessage[]; tools?: unknown[] | undefined }): number {
  let n = r.tools ? proxyJsonLength(JSON.stringify(r.tools)) : 0;
  for (const m of r.messages) {
    n += m.content?.length ?? 0;
    if (m.role === 'assistant') for (const c of m.toolCalls ?? []) n += c.arguments.length;
  }
  return n;
}

/**
 * Length of JSON text as the proxy measures it: System.Text.Json writes an
 * escaped quote and every character in <>&'+` or outside ASCII as \uXXXX.
 * Seen in the browser: a request within the limit here was refused there.
 */
export function proxyJsonLength(json: string): number {
  let n = json.length;
  for (let i = 0; i < json.length; i++) {
    const c = json.charCodeAt(i);
    if (c > 0x7e || c === 0x3c || c === 0x3e || c === 0x26 || c === 0x27 || c === 0x2b || c === 0x60) n += 5;
    else if (c === 0x5c && json.charCodeAt(i + 1) === 0x22) {
      n += 4;
      i++;
    }
  }
  return n;
}

export interface ChatToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** Sanitized model answer returned by the proxy. */
export interface ChatResult {
  content: string | null;
  toolCalls: ChatToolCall[];
  finishReason: string;
  model: string | null;
  usage: { promptTokens: number; completionTokens: number } | null;
}

export type CapabilityState = 'yes' | 'no' | 'unknown';

export interface ProviderCapabilities {
  toolCalling: { state: CapabilityState; note: string };
  imageInput: { state: CapabilityState; note: string };
  streaming: { state: CapabilityState; note: string };
  realtimeVoice: { state: CapabilityState; note: string };
}

/** Body of GET /api/ai/status. Never contains secrets. */
export interface AiStatus {
  ok: boolean;
  provider: string | null;
  reason: string | null;
  capabilities: ProviderCapabilities | null;
}

// The browser checks the proxy's answers as untrusted input too.
const capability = z.object({ state: z.enum(['yes', 'no', 'unknown']), note: z.string().max(500) });

export const AiStatusSchema = z.object({
  ok: z.boolean(),
  provider: z.string().max(64).nullable(),
  reason: z.string().max(500).nullable(),
  capabilities: z.object({ toolCalling: capability, imageInput: capability, streaming: capability, realtimeVoice: capability }).nullable(),
});

export const ChatResultSchema = z.object({
  content: z.string().max(20_000).nullable(),
  toolCalls: z.array(z.object({ id: z.string().max(200), name: z.string().max(200), arguments: z.string().max(20_000) })).max(32),
  finishReason: z.string().max(64),
  model: z.string().max(200).nullable(),
  usage: z.object({ promptTokens: z.number(), completionTokens: z.number() }).nullable(),
});
