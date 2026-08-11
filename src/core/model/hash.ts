import type { RobotDescription } from './types';

/** Recursively sorts object keys so equal content always serializes identically. */
export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortKeys(value));
}

/** Canonical JSON with two-space indentation, for files people may read. */
export function canonicalJsonPretty(value: unknown): string {
  return JSON.stringify(sortKeys(value), null, 2);
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      const v = (value as Record<string, unknown>)[key];
      if (v !== undefined) out[key] = sortKeys(v);
    }
    return out;
  }
  return value;
}

/** The fields that define the physical body: names and the initial pose are excluded. */
export function morphologyPayload(desc: RobotDescription): unknown {
  return {
    schemaVersion: desc.schemaVersion,
    base: { kind: desc.base.kind, size: desc.base.size, mass: desc.base.mass },
    limbs: desc.limbs.map((l) => ({
      id: l.id,
      role: l.role,
      tip: l.tip,
      wheel: l.wheel,
      joints: l.joints.map((j) => ({
        id: j.id,
        type: j.type,
        origin: j.origin,
        axis: j.axis,
        limit: j.limit,
        link: j.link,
        damping: j.damping,
        armature: j.armature,
      })),
    })),
  };
}

/** SHA-256 (hex) of the canonical morphology payload. */
export async function morphologyHash(desc: RobotDescription): Promise<string> {
  const bytes = new TextEncoder().encode(canonicalJson(morphologyPayload(desc)));
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
