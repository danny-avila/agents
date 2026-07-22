/**
 * Default system prompt for fast-model activity labeling.
 *
 * Style synthesized from Claude Code's tool-use summary prompt (git-subject
 * register, past tense, distinctive nouns) and claude.ai's observed group
 * headers (5–9 words describing a mixed reasoning + tool block, e.g.
 * "Synthesized version data and curated comparative framework").
 */
export const ACTIVITY_LABEL_PROMPT = `Write a short label describing what this block of agent activity accomplished. It appears as the header of a collapsed activity group in a chat UI.

Rules:
- 5 to 9 words, past-tense verb first
- Name the most distinctive subject (file, API, topic); drop articles and filler
- Describe outcomes, not mechanics; if something failed, say so plainly
- Output only the label — no quotes, no punctuation at the end, no preamble

Examples:
- Searched Node.js release notes and changelogs
- Compared runtime versions across official sources
- Fixed failing auth middleware tests
- Read project config and dependency manifests
- Attempted database migration, hit permission errors`;

/** Truncates a serialized value for the label prompt. */
export function truncateForLabel(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  return value.slice(0, Math.max(0, maxLength - 1)) + '…';
}
