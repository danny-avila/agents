// src/tools/skillCatalog.ts
import type { SkillCatalogEntry } from '@/types';

const HEADER = '## Available Skills';
const DEFAULT_CONTEXT_WINDOW_TOKENS = 200_000;
const DEFAULT_BUDGET_PERCENT = 0.01;
const DEFAULT_MAX_ENTRY_CHARS = 250;
const DEFAULT_MIN_DESC_LENGTH = 20;
const DEFAULT_CHARS_PER_TOKEN = 4;

export type SkillCatalogOptions = {
  /** Total context window in tokens. Default: 200_000 */
  contextWindowTokens?: number;
  /** Fraction of context budget for catalog. Default: 0.01 (1%) */
  budgetPercent?: number;
  /** Max chars per entry description. Default: 250 */
  maxEntryChars?: number;
  /** Descriptions below this length trigger names-only fallback. Default: 20 */
  minDescLength?: number;
  /** Approximate chars per token for budget calculation. Default: 4 */
  charsPerToken?: number;
};

/**
 * Formats a skill catalog for injection into agent context.
 * Uses a truncation ladder: full descriptions, proportional truncation, names-only.
 * Returns empty string for empty input.
 */
export function formatSkillCatalog(
  skills: SkillCatalogEntry[],
  opts?: SkillCatalogOptions
): string {
  if (skills.length === 0) return '';

  const contextWindowTokens =
    opts?.contextWindowTokens ?? DEFAULT_CONTEXT_WINDOW_TOKENS;
  const budgetPercent = opts?.budgetPercent ?? DEFAULT_BUDGET_PERCENT;
  const maxEntryChars = opts?.maxEntryChars ?? DEFAULT_MAX_ENTRY_CHARS;
  const minDescLength = opts?.minDescLength ?? DEFAULT_MIN_DESC_LENGTH;
  const charsPerToken = opts?.charsPerToken ?? DEFAULT_CHARS_PER_TOKEN;

  const budgetChars = Math.floor(
    contextWindowTokens * budgetPercent * charsPerToken
  );

  const capped = skills.map((s) => ({
    name: s.name,
    description:
      s.description.length > maxEntryChars
        ? s.description.slice(0, maxEntryChars - 1) + '\u2026'
        : s.description,
  }));

  const fullOutput = formatEntries(capped);
  if (fullOutput.length <= budgetChars) return fullOutput;

  const headerLen = HEADER.length + 2;
  const newlineChars = capped.length > 1 ? capped.length - 1 : 0;
  const availableChars = budgetChars - headerLen - newlineChars;
  const perEntryOverhead = 4;
  const nameCharsTotal = capped.reduce(
    (sum, s) => sum + s.name.length + perEntryOverhead,
    0
  );
  const availableForDescs = availableChars - nameCharsTotal;

  if (availableForDescs <= 0) {
    return formatEntries(
      capped.map((s) => ({ name: s.name, description: '' }))
    );
  }

  const maxDescPerEntry = Math.floor(availableForDescs / capped.length);

  if (maxDescPerEntry < minDescLength) {
    return formatEntries(
      capped.map((s) => ({ name: s.name, description: '' }))
    );
  }

  const truncated = capped.map((s) => ({
    name: s.name,
    description:
      s.description.length > maxDescPerEntry
        ? s.description.slice(0, maxDescPerEntry - 1) + '\u2026'
        : s.description,
  }));

  const result = formatEntries(truncated);
  if (result.length <= budgetChars) return result;
  return formatEntries(capped.map((s) => ({ name: s.name, description: '' })));
}

function formatEntries(
  entries: { name: string; description: string }[]
): string {
  const lines = entries.map((e) =>
    e.description ? `- ${e.name}: ${e.description}` : `- ${e.name}`
  );
  return `${HEADER}\n\n${lines.join('\n')}`;
}
