import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  MAX_ARGUMENT_CHARS,
  MAX_DESCRIPTION_CHARS,
  MAX_MESSAGE_CHARS,
  MAX_MESSAGES,
  MAX_TOOL_CALLS,
  MAX_TOOLS,
  MAX_TOTAL_CHARS,
  OUTPUT_TOKEN_RANGE,
} from '../../src/ai/protocol.ts';
import { TOOL_NAMES } from '../../src/ai/tools.ts';

// The PowerShell server runs where Node.js may not; CI cannot run it, so its copy of the contract is checked here.
const ps1 = readFileSync(new URL('../../server/serve.ps1', import.meta.url), 'utf8');

function psNumber(name: string): number {
  const m = new RegExp(`^\\$${name} = (\\d+)\\s*$`, 'm').exec(ps1);
  if (!m) throw new Error(`$${name} is not a plain number in server/serve.ps1`);
  return Number(m[1]);
}

describe('server/serve.ps1 follows the TypeScript chat contract', () => {
  it('offers exactly the MorphLab tools', () => {
    const m = /^\$ToolNames = @\(([^)]*)\)/m.exec(ps1);
    expect(m).not.toBeNull();
    const names = [...(m?.[1] ?? '').matchAll(/'([^']+)'/g)].map((x) => x[1]);
    expect([...names].sort()).toEqual([...TOOL_NAMES].sort());
  });

  it('uses the same request limits', () => {
    expect(psNumber('MaxMessageChars')).toBe(MAX_MESSAGE_CHARS);
    expect(psNumber('MaxTotalChars')).toBe(MAX_TOTAL_CHARS);
    expect(psNumber('MaxMessages')).toBe(MAX_MESSAGES);
    expect(psNumber('MaxTools')).toBe(MAX_TOOLS);
    expect(psNumber('MaxToolCalls')).toBe(MAX_TOOL_CALLS);
    expect(psNumber('MaxArgumentChars')).toBe(MAX_ARGUMENT_CHARS);
    expect(psNumber('MaxDescriptionChars')).toBe(MAX_DESCRIPTION_CHARS);
    expect(psNumber('MinOutputTokens')).toBe(OUTPUT_TOKEN_RANGE[0]);
    expect(psNumber('MaxOutputTokens')).toBe(OUTPUT_TOKEN_RANGE[1]);
  });

  it('accepts no images, like the TypeScript contract (D31)', () => {
    expect(ps1).not.toMatch(/image_url|\$MaxImages|\$ImageDataUrlPattern/);
    expect(ps1).toContain("'user' { @('role', 'content') }");
  });

  it('keeps the proxy defaults: 20 requests per session and 100 overall per minute, 30 s upstream timeout', () => {
    expect(psNumber('PerSessionLimit')).toBe(20);
    expect(psNumber('OverallLimit')).toBe(100);
    expect(psNumber('WindowMs')).toBe(60_000);
    expect(psNumber('UpstreamTimeoutSeconds')).toBe(30);
  });

  it('allows only same-origin connections in its Content Security Policy', () => {
    expect(ps1).toContain("connect-src 'self';");
  });
});
