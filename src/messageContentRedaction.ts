import type { BaseMessage, MessageContent } from '@langchain/core/messages';

export const DEFAULT_REDACTION_TEXT = '[REDACTED]';

export type SensitivePattern = {
  id: string;
  label: string;
  pattern: RegExp;
};

export type PatternMatch = {
  patternId: string;
  patternLabel: string;
  count: number;
};

export type MessageContentRedactionConfig = {
  patterns: SensitivePattern[];
  redactionText?: string;
};

type ResolvedConfig = {
  patterns: SensitivePattern[];
  redactionText: string;
};

function resolveConfig(config: MessageContentRedactionConfig): ResolvedConfig {
  return {
    patterns: config.patterns,
    redactionText: config.redactionText ?? DEFAULT_REDACTION_TEXT,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function recordMatch(
  aggregate: Map<string, PatternMatch>,
  pattern: SensitivePattern,
  count: number
): void {
  if (count === 0) {
    return;
  }
  const existing = aggregate.get(pattern.id);
  if (existing != null) {
    existing.count += count;
    return;
  }
  aggregate.set(pattern.id, {
    patternId: pattern.id,
    patternLabel: pattern.label,
    count,
  });
}

function redactStringInto(
  text: string,
  resolved: ResolvedConfig,
  aggregate: Map<string, PatternMatch>
): string {
  let next = text;
  for (const pattern of resolved.patterns) {
    pattern.pattern.lastIndex = 0;
    let count = 0;
    const replaced = next.replace(pattern.pattern, (...args: unknown[]) => {
      count += 1;
      const groups = args.slice(1, -2);
      const prefix = typeof groups[0] === 'string' ? groups[0] : '';
      return `${prefix}${resolved.redactionText}`;
    });
    if (count > 0) {
      next = replaced;
      recordMatch(aggregate, pattern, count);
    }
  }
  return next;
}

/**
 * Scrubs credential-shaped substrings from a single string using the
 * caller-supplied patterns. Each match is replaced with the configured
 * redaction text (default `[REDACTED]`) while the leading prefix capture
 * group is preserved so reviewers can still identify which family of
 * secret was matched.
 *
 * This is a primitive: the caller owns the pattern catalog. The agents
 * library ships no built-in patterns by design — consumers like LibreChat
 * define and maintain the patterns they care about.
 */
export function redactSensitiveText(
  text: string,
  config: MessageContentRedactionConfig
): { text: string; matches: PatternMatch[] } {
  const resolved = resolveConfig(config);
  const aggregate = new Map<string, PatternMatch>();
  const next = redactStringInto(text, resolved, aggregate);
  return { text: next, matches: Array.from(aggregate.values()) };
}

function redactValueInto(
  value: unknown,
  resolved: ResolvedConfig,
  aggregate: Map<string, PatternMatch>
): { value: unknown; changed: boolean } {
  if (typeof value === 'string') {
    const next = redactStringInto(value, resolved, aggregate);
    return { value: next, changed: next !== value };
  }

  if (Array.isArray(value)) {
    let changed = false;
    const next: unknown[] = [];
    for (const item of value) {
      const result = redactValueInto(item, resolved, aggregate);
      next.push(result.value);
      if (result.changed) {
        changed = true;
      }
    }
    return changed ? { value: next, changed } : { value, changed: false };
  }

  if (!isRecord(value)) {
    return { value, changed: false };
  }

  let changed = false;
  const next: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value)) {
    const result = redactValueInto(child, resolved, aggregate);
    next[key] = result.value;
    if (result.changed) {
      changed = true;
    }
  }
  return changed ? { value: next, changed } : { value, changed: false };
}

/**
 * Recursively walks an arbitrary value, scrubbing every string it
 * encounters. Returns the rewritten value (reference-equal to the original
 * when no matches fired) and the aggregated match summary.
 */
export function redactSensitiveValue(
  value: unknown,
  config: MessageContentRedactionConfig
): { value: unknown; matches: PatternMatch[] } {
  const resolved = resolveConfig(config);
  const aggregate = new Map<string, PatternMatch>();
  const { value: next } = redactValueInto(value, resolved, aggregate);
  return { value: next, matches: Array.from(aggregate.values()) };
}

function cloneMessageWithContent(
  message: BaseMessage,
  content: MessageContent
): BaseMessage {
  const clone = Object.assign(
    Object.create(Object.getPrototypeOf(message)),
    message
  );
  clone.content = content;
  return clone as BaseMessage;
}

function redactMessageContent(
  content: MessageContent,
  resolved: ResolvedConfig,
  aggregate: Map<string, PatternMatch>
): { content: MessageContent; changed: boolean } {
  if (typeof content === 'string') {
    const next = redactStringInto(content, resolved, aggregate);
    return { content: next, changed: next !== content };
  }

  const result = redactValueInto(content, resolved, aggregate);
  return {
    content: result.value as MessageContent,
    changed: result.changed,
  };
}

/**
 * Returns a new array of LangChain messages with credential-shaped
 * substrings scrubbed from each message's content. Messages with no
 * matches keep reference equality; only the affected messages are cloned
 * (prototype-preserving so `instanceof` checks still hold).
 */
export function filterMessageContent(
  messages: BaseMessage[],
  config: MessageContentRedactionConfig
): { messages: BaseMessage[]; matches: PatternMatch[] } {
  const resolved = resolveConfig(config);
  const aggregate = new Map<string, PatternMatch>();
  const next = messages.map((message) => {
    const { content, changed } = redactMessageContent(
      message.content,
      resolved,
      aggregate
    );
    return changed ? cloneMessageWithContent(message, content) : message;
  });
  return { messages: next, matches: Array.from(aggregate.values()) };
}
