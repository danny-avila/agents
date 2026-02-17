/**
 * Context overflow error detection utilities.
 *
 * Identifies provider-specific error messages that indicate the request
 * exceeded the model's context window. Used by the overflow recovery loop
 * to decide whether to retry with truncation/compaction vs. propagating
 * the error.
 */

/**
 * Exact phrases that definitively indicate a context overflow error.
 * These are returned by various LLM providers when the prompt is too large.
 */
const CONTEXT_OVERFLOW_PHRASES = [
  'request_too_large',
  'context length exceeded',
  'maximum context length',
  'prompt is too long',
  'exceeds model context window',
  'exceeds the model',
  'too large for model',
  'context_length_exceeded',
  'max_tokens',
  'token limit',
  'input too long',
  'payload too large',
  'content_too_large',
] as const;

/**
 * HTTP status codes and broader hints that suggest context overflow.
 * Used by the less-strict `isLikelyContextOverflowError`.
 */
const CONTEXT_OVERFLOW_HINT_RE =
  /413|too large|too long|context.*exceed|exceed.*context|token.*limit|limit.*token|prompt.*size|size.*limit|maximum.*length|length.*maximum/i;

/**
 * Patterns that should NOT be treated as context overflow even if they
 * contain words like "limit" or "too large".
 */
const FALSE_POSITIVE_RE =
  /rate.?limit|too many requests|quota|billing|auth|permission|forbidden/i;

/**
 * Extracts a human-readable error message from an unknown error value.
 */
export function extractErrorMessage(error: unknown): string {
  if (error == null) {
    return '';
  }
  if (typeof error === 'string') {
    return error;
  }
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'object') {
    const record = error as Record<string, unknown>;
    if (typeof record.message === 'string') {
      return record.message;
    }
    if (typeof record.error === 'string') {
      return record.error;
    }
    if (
      typeof record.error === 'object' &&
      record.error != null &&
      typeof (record.error as Record<string, unknown>).message === 'string'
    ) {
      return (record.error as Record<string, unknown>).message as string;
    }
  }
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

/**
 * Returns true if the error message definitively indicates a context
 * overflow / prompt-too-large error from the provider.
 *
 * This is the strict check: only matches known, unambiguous phrases.
 * Use this when you want high confidence before taking recovery action.
 */
export function isContextOverflowError(errorMessage?: string): boolean {
  if (!errorMessage) {
    return false;
  }
  const lower = errorMessage.toLowerCase();
  if (FALSE_POSITIVE_RE.test(lower)) {
    return false;
  }
  return CONTEXT_OVERFLOW_PHRASES.some((phrase) => lower.includes(phrase));
}

/**
 * Returns true if the error message likely indicates a context overflow.
 * Uses broader heuristic matching (regex) in addition to exact phrases.
 *
 * May produce false positives for unusual error messages. Use this when
 * the cost of a false positive (one extra retry) is acceptable.
 */
export function isLikelyContextOverflowError(errorMessage?: string): boolean {
  if (!errorMessage) {
    return false;
  }
  if (isContextOverflowError(errorMessage)) {
    return true;
  }
  const lower = errorMessage.toLowerCase();
  if (FALSE_POSITIVE_RE.test(lower)) {
    return false;
  }
  return CONTEXT_OVERFLOW_HINT_RE.test(lower);
}
