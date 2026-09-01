import type { BaseMessage } from '@langchain/core/messages';
import type { SessionEntry } from './types';
import { deserializeMessage } from './messageSerialization';

export interface DerivedSessionMessages {
  initialSummary?: {
    text: string;
    tokenCount: number;
  };
  messages: BaseMessage[];
}

/** Derives the active model context from one append-only session-log path. */
export function deriveMessages(
  entries: readonly SessionEntry[]
): DerivedSessionMessages {
  const messages: BaseMessage[] = [];
  let initialSummary: DerivedSessionMessages['initialSummary'];
  for (const entry of entries) {
    if (entry.type === 'summary') {
      initialSummary = {
        text: entry.data.text,
        tokenCount:
          typeof entry.data.tokenCount === 'number' &&
          Number.isFinite(entry.data.tokenCount)
            ? entry.data.tokenCount
            : 0,
      };
      messages.length = 0;
      continue;
    }
    if (entry.type === 'message') {
      messages.push(deserializeMessage(entry.data.message));
    }
  }
  return { messages, ...(initialSummary != null && { initialSummary }) };
}
