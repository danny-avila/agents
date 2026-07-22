import type { ResolvedLangfuseToolOutputTracingConfig } from '@/langfuseRuntimeContext';
import type { ActivityLabelToolEntry } from '@/types/activityLabel';
import { shouldRedactTool } from '@/langfuseToolOutputTracing';

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

const ABORT_SERIALIZATION = Symbol('abort-label-serialization');

/**
 * Serializes a tool value for the prompt WITHOUT materializing huge JSON:
 * the output is clipped to a few hundred characters anyway, so a multi-
 * megabyte tool result must not be stringified in full on the label path.
 * Strings clip immediately; structured values serialize under a character
 * budget and degrade to a shape summary once it is exhausted.
 */
function serializeForLabel(value: unknown, limit: number): string {
  if (value == null) {
    return '';
  }
  if (typeof value === 'string') {
    return value.length > limit ? value.slice(0, limit + 1) : value;
  }
  let budget = limit * 4;
  try {
    return (
      JSON.stringify(value, (_key, nested: unknown) => {
        if (budget <= 0) {
          throw ABORT_SERIALIZATION;
        }
        if (typeof nested === 'string') {
          const clipped =
            nested.length > limit ? nested.slice(0, limit) : nested;
          budget -= clipped.length;
          return clipped;
        }
        budget -= 8;
        return nested;
      }) ?? ''
    );
  } catch (error) {
    if (error === ABORT_SERIALIZATION) {
      return Array.isArray(value) ? `[Array(${value.length})]` : '[Object]';
    }
    return String(value);
  }
}

const INPUT_CONTEXT_LIMIT = 200;
const MAX_THINKING_EXCERPTS = 4;

export type BuildActivityLabelPromptParams = {
  entries: ActivityLabelToolEntry[];
  charLimit: number;
  thinkingExcerpts?: string[];
  lastAssistantText?: string;
  /**
   * Resolved tool-output tracing policy. The label prompt becomes Langfuse
   * generation input, so outputs/errors excluded from tracing (global
   * disable or `redactedToolNames`) must never appear in it — the same
   * redaction the span processor applies to structured tool observations.
   */
  redaction?: ResolvedLangfuseToolOutputTracingConfig;
};

/**
 * Builds the user prompt for a fast-model activity label. Pure — exported
 * for direct testing of redaction and truncation behavior.
 */
export function buildActivityLabelPrompt({
  entries,
  charLimit,
  thinkingExcerpts,
  lastAssistantText,
  redaction,
}: BuildActivityLabelPromptParams): string {
  const clip = truncateForLabel;
  /** Reasoning and intent text can quote tool output verbatim — including
   *  output from EARLIER calls to a redacted tool that this batch does not
   *  contain — so any active policy (global disable or a configured
   *  redacted-name list) drops both wholesale. There is no reliable way to
   *  scrub a quoted fragment out of free-form model prose. */
  const excerptsRedacted =
    redaction != null &&
    (redaction.enabled === false || redaction.redactedToolNames.size > 0);
  const sections: string[] = [];
  /** Intent text is free-form assistant prose that can quote a redacted
   *  tool result just as reasoning can, so it shares the excerpts' fate. */
  if (
    !excerptsRedacted &&
    lastAssistantText != null &&
    lastAssistantText.length > 0
  ) {
    sections.push(
      `Intent (assistant's last message): ${clip(lastAssistantText, INPUT_CONTEXT_LIMIT)}`
    );
  }
  if (
    !excerptsRedacted &&
    thinkingExcerpts != null &&
    thinkingExcerpts.length > 0
  ) {
    sections.push(
      'Reasoning excerpts:\n' +
        thinkingExcerpts
          .slice(0, MAX_THINKING_EXCERPTS)
          .map((excerpt) => `- ${clip(excerpt, charLimit)}`)
          .join('\n')
    );
  }
  if (entries.length > 0) {
    sections.push(
      'Tool calls:\n' +
        entries
          .map((entry) => {
            const input = clip(
              serializeForLabel(entry.toolInput, charLimit),
              charLimit
            );
            const redacted =
              redaction != null && shouldRedactTool(entry.toolName, redaction);
            let outcome: string;
            if (redacted) {
              outcome = redaction.redactionText;
            } else if (entry.status === 'error') {
              outcome = `ERROR: ${clip(entry.error ?? 'unknown error', charLimit)}`;
            } else {
              outcome = clip(
                serializeForLabel(entry.toolOutput, charLimit),
                charLimit
              );
            }
            return `- ${entry.toolName}(${input}) → ${outcome}`;
          })
          .join('\n')
    );
  }
  sections.push('Label:');
  return sections.join('\n\n');
}
