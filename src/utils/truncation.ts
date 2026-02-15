/**
 * Ingestion-time and pre-flight truncation utilities for tool results.
 *
 * Prevents oversized tool outputs from entering the message array and
 * consuming the entire context window. Inspired by openclaw's
 * session-tool-result-guard and tool-result-truncation patterns.
 */

/**
 * Absolute hard cap on tool result length (characters).
 * Even if the model has a 1M-token context, a single tool result
 * larger than this is almost certainly a bug (e.g., dumping a binary file).
 */
export const HARD_MAX_TOOL_RESULT_CHARS = 400_000;

/**
 * Computes the dynamic max tool result size based on the model's context window.
 * Uses 30% of the context window (in estimated characters, ~4 chars/token)
 * capped at HARD_MAX_TOOL_RESULT_CHARS.
 *
 * @param contextWindowTokens - The model's max context tokens (optional).
 * @returns Maximum allowed characters for a single tool result.
 */
export function calculateMaxToolResultChars(
  contextWindowTokens?: number
): number {
  if (contextWindowTokens == null || contextWindowTokens <= 0) {
    return HARD_MAX_TOOL_RESULT_CHARS;
  }
  return Math.min(
    Math.floor(contextWindowTokens * 0.3) * 4,
    HARD_MAX_TOOL_RESULT_CHARS
  );
}

/**
 * Truncates tool result content that exceeds `maxChars`.
 * Preserves newline boundaries where possible so the truncated output
 * remains structurally coherent (e.g., JSON or log output).
 *
 * Strategy: keep the head of the content (most tool results start with
 * structure/headers), append a truncation indicator with the original size.
 *
 * @param content - The tool result string content.
 * @param maxChars - Maximum allowed characters.
 * @returns The (possibly truncated) content string.
 */
export function truncateToolResultContent(
  content: string,
  maxChars: number
): string {
  if (content.length <= maxChars) {
    return content;
  }

  const indicator = `\n\n… [truncated: ${content.length} chars total, showing first ${maxChars} chars]`;
  const available = maxChars - indicator.length;
  if (available <= 0) {
    return content.slice(0, maxChars);
  }

  // Try to break at a newline boundary within the last 200 chars of the head
  const head = content.slice(0, available);
  const lastNewline = head.lastIndexOf('\n', available);
  const breakPoint =
    lastNewline > available - 200 && lastNewline > 0 ? lastNewline : available;

  return content.slice(0, breakPoint) + indicator;
}
