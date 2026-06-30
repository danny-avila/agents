import type { AIMessageChunk, BaseMessage } from '@langchain/core/messages';

/**
 * Normalized truncation reasons across providers. Providers disagree on the
 * key and the value: Anthropic/Bedrock use `max_tokens`, OpenAI uses `length`,
 * Google uses `MAX_TOKENS`.
 */
export type TruncationStopReason = 'max_tokens' | 'length';

const MAX_TOKEN_VALUES = new Set(['max_tokens', 'max_token', 'maxtokens']);
const LENGTH_VALUES = new Set(['length']);

function normalizeStopValue(value: unknown): TruncationStopReason | null {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (MAX_TOKEN_VALUES.has(normalized)) {
    return 'max_tokens';
  }
  if (LENGTH_VALUES.has(normalized)) {
    return 'length';
  }
  return null;
}

/**
 * Reads the truncation stop reason off a message's `response_metadata`,
 * covering every provider shape we stream:
 * - `stopReason` (Bedrock Converse, non-streaming)
 * - `messageStop.stopReason` (Bedrock Converse, streaming `messageStop` event)
 * - `stop_reason` (Anthropic)
 * - `finish_reason` (OpenAI / OpenAI-compatible)
 * - `finishReason` (Google)
 *
 * Returns the normalized reason when the model stopped because it hit the
 * output token ceiling, otherwise `null`.
 */
export function getTruncationStopReason(
  message: Pick<BaseMessage, 'response_metadata'> | undefined | null
): TruncationStopReason | null {
  const meta = message?.response_metadata as
    | Record<string, unknown>
    | undefined;
  if (meta == null) {
    return null;
  }

  const messageStop = meta.messageStop as { stopReason?: unknown } | undefined;
  const candidates: unknown[] = [
    meta.stopReason,
    meta.stop_reason,
    meta.finish_reason,
    meta.finishReason,
    messageStop?.stopReason,
  ];

  for (const candidate of candidates) {
    const normalized = normalizeStopValue(candidate);
    if (normalized != null) {
      return normalized;
    }
  }
  return null;
}

function hasOpenToolCall(message: AIMessageChunk | BaseMessage): boolean {
  const m = message as AIMessageChunk & {
    tool_calls?: unknown[];
    tool_call_chunks?: unknown[];
    invalid_tool_calls?: unknown[];
  };
  return (
    (m.tool_calls?.length ?? 0) > 0 ||
    (m.tool_call_chunks?.length ?? 0) > 0 ||
    (m.invalid_tool_calls?.length ?? 0) > 0
  );
}

/**
 * Error raised when a model turn is cut off by the output token limit while it
 * was still emitting a tool call. The arguments are necessarily incomplete, so
 * executing or re-prompting the call would loop on a malformed request. Failing
 * fast with this surfaces an actionable message to the caller instead.
 */
export class OutputTruncationError extends Error {
  readonly stopReason: TruncationStopReason;
  readonly toolCallNames: string[];

  constructor(stopReason: TruncationStopReason, toolCallNames: string[]) {
    const named =
      toolCallNames.length > 0
        ? ` (tool call: ${toolCallNames.join(', ')})`
        : '';
    super(
      'The model response was truncated at the maximum output token limit before ' +
        `the tool call arguments were complete${named}. Increase the model's max ` +
        'output tokens, or have the model produce smaller arguments — for large ' +
        'files, write a lean main file and move bulky content into separate files.'
    );
    this.name = 'OutputTruncationError';
    this.stopReason = stopReason;
    this.toolCallNames = toolCallNames;
  }
}

function collectToolCallNames(message: AIMessageChunk | BaseMessage): string[] {
  const m = message as AIMessageChunk & {
    tool_calls?: Array<{ name?: string }>;
    tool_call_chunks?: Array<{ name?: string }>;
    invalid_tool_calls?: Array<{ name?: string }>;
  };
  const names = [
    ...(m.tool_calls ?? []),
    ...(m.tool_call_chunks ?? []),
    ...(m.invalid_tool_calls ?? []),
  ]
    .map((tc) => tc.name)
    .filter((name): name is string => typeof name === 'string' && name !== '');
  return [...new Set(names)];
}

/**
 * Throws {@link OutputTruncationError} when `message` was truncated by the
 * output token limit AND still carries a tool call. A tool call emitted under
 * truncation has incomplete arguments (the tool-use block is the last thing the
 * model streams), so letting it through makes the agent loop on a malformed
 * call. No-ops for normal completions and for truncated plain-text turns.
 */
export function assertNotTruncatedToolCall(
  message: AIMessageChunk | BaseMessage | undefined | null
): void {
  if (message == null) {
    return;
  }
  const stopReason = getTruncationStopReason(message);
  if (stopReason == null || !hasOpenToolCall(message)) {
    return;
  }
  throw new OutputTruncationError(stopReason, collectToolCallNames(message));
}
