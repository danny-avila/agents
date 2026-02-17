/**
 * Ingestion-time and pre-flight truncation utilities for tool results.
 *
 * Prevents oversized tool outputs from entering the message array and
 * consuming the entire context window.
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
 * Truncates a tool-call input (the arguments/payload of a tool_use block)
 * using head+tail strategy. Returns an object with `_truncated` (the
 * truncated string) and `_originalChars` (for diagnostics).
 *
 * Accepts any type — objects are JSON-serialized before truncation.
 *
 * @param input - The tool input (string, object, etc.).
 * @param maxChars - Maximum allowed characters.
 */
export function truncateToolInput(
  input: unknown,
  maxChars: number
): { _truncated: string; _originalChars: number } {
  const serialized = typeof input === 'string' ? input : JSON.stringify(input);
  if (serialized.length <= maxChars) {
    return { _truncated: serialized, _originalChars: serialized.length };
  }
  const indicator = `\n… [truncated: ${serialized.length} → ${maxChars} chars] …\n`;
  const available = maxChars - indicator.length;

  if (available < 100) {
    return {
      _truncated: serialized.slice(0, maxChars) + indicator.trimEnd(),
      _originalChars: serialized.length,
    };
  }

  const headSize = Math.ceil(available * 0.7);
  const tailSize = available - headSize;

  return {
    _truncated:
      serialized.slice(0, headSize) +
      indicator +
      serialized.slice(serialized.length - tailSize),
    _originalChars: serialized.length,
  };
}

/**
 * Truncates tool result content that exceeds `maxChars` using a head+tail
 * strategy. Keeps the beginning (structure/headers) and end (return value /
 * conclusion) of the content so the model retains both the opening context
 * and the final outcome.
 *
 * Head gets ~70% of the budget, tail gets ~30%. Falls back to head-only
 * when the budget is too small for a meaningful tail.
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

  const indicator = `\n\n… [truncated: ${content.length} → ${maxChars} chars] …\n\n`;
  const available = maxChars - indicator.length;
  if (available <= 0) {
    return content.slice(0, maxChars);
  }

  // When budget is too small for a meaningful tail, fall back to head-only
  if (available < 200) {
    return content.slice(0, available) + indicator.trimEnd();
  }

  const headSize = Math.ceil(available * 0.7);
  const tailSize = available - headSize;

  // Try to break at newline boundaries for cleaner output
  let headEnd = headSize;
  const headNewline = content.lastIndexOf('\n', headSize);
  if (headNewline > headSize - 200 && headNewline > 0) {
    headEnd = headNewline;
  }

  let tailStart = content.length - tailSize;
  const tailNewline = content.indexOf('\n', tailStart);
  if (tailNewline > 0 && tailNewline < tailStart + 200) {
    tailStart = tailNewline + 1;
  }

  return content.slice(0, headEnd) + indicator + content.slice(tailStart);
}
