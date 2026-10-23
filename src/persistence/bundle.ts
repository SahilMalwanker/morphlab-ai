import { canonicalJsonPretty } from '../core/model/hash';
import { parseRobot } from '../core/model/parse';
import type { RobotDescription } from '../core/model/types';
import { parseSkillArtifact, type SkillArtifact } from '../skills/artifact';

export const BUNDLE_KIND = 'morphlab.bundle';
export const BUNDLE_SCHEMA_VERSION = 1;
/** Files larger than this are refused before parsing. */
export const MAX_BUNDLE_CHARS = 5_000_000;

export interface RejectedRecord {
  kind: 'robot' | 'skill';
  index: number;
  reasons: string[];
}

export interface ImportResult {
  /** File-level problems; when non-empty nothing was imported. */
  errors: string[];
  robots: RobotDescription[];
  skills: SkillArtifact[];
  rejected: RejectedRecord[];
  exportedAt: string | null;
}

/** Canonical (sorted keys, two-space indent) bundle text; equal content gives identical bytes. */
export function exportBundle(robots: RobotDescription[], exportedAt: string, skills: SkillArtifact[] = []): string {
  return `${canonicalJsonPretty({ kind: BUNDLE_KIND, schemaVersion: BUNDLE_SCHEMA_VERSION, exportedAt, robots, skills })}\n`;
}

/** Parses and validates a bundle. Every record is checked; rejected records are reported with reasons. */
export function importBundle(text: string): ImportResult {
  const fail = (msg: string): ImportResult => ({ errors: [msg], robots: [], skills: [], rejected: [], exportedAt: null });
  if (text.length > MAX_BUNDLE_CHARS) return fail(`The file is larger than ${MAX_BUNDLE_CHARS / 1_000_000} MB.`);
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return fail(`The file is not valid JSON (${e instanceof Error ? e.message : String(e)}).`);
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return fail('The file does not contain a MorphLab bundle object.');
  const b = raw as Record<string, unknown>;
  if (b.kind !== BUNDLE_KIND) return fail(`The file is not a MorphLab bundle (expected kind "${BUNDLE_KIND}").`);
  if (b.schemaVersion !== BUNDLE_SCHEMA_VERSION) return fail(`Unsupported bundle schema version ${String(b.schemaVersion)}; this version reads ${BUNDLE_SCHEMA_VERSION}.`);
  if (!Array.isArray(b.robots)) return fail('The bundle has no robots list.');
  const result: ImportResult = { errors: [], robots: [], skills: [], rejected: [], exportedAt: typeof b.exportedAt === 'string' ? b.exportedAt : null };
  b.robots.forEach((r, index) => {
    const parsed = parseRobot(r);
    if (parsed.ok) result.robots.push(parsed.value);
    else result.rejected.push({ kind: 'robot', index, reasons: parsed.errors });
  });
  if (b.skills !== undefined && !Array.isArray(b.skills)) return fail('The bundle skills entry must be a list.');
  const skills: unknown[] = Array.isArray(b.skills) ? b.skills : [];
  skills.forEach((s, index) => {
    const parsed = parseSkillArtifact(s);
    if (parsed.ok) result.skills.push(parsed.value);
    else result.rejected.push({ kind: 'skill', index, reasons: parsed.errors });
  });
  return result;
}
