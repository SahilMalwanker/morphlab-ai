import type { ChatRequest, ChatResult, ProviderCapabilities } from '../src/ai/protocol.ts';

/** An error whose message is safe to show to the browser: it never contains secrets or upstream bodies. */
export class UpstreamError extends Error {
  readonly status: number;
  readonly retryAfterSeconds: number | null;

  constructor(status: number, message: string, retryAfterSeconds: number | null = null) {
    super(message);
    this.name = 'UpstreamError';
    this.status = status;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export interface ChatAdapter {
  readonly provider: string;
  capabilities(): ProviderCapabilities;
  chat(request: ChatRequest, signal: AbortSignal): Promise<ChatResult>;
}

export type FetchLike = (input: URL, init: RequestInit) => Promise<Response>;
