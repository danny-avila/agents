import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type {
  SessionEntry,
  SessionMessageEntry,
  SessionSummaryEntry,
} from '@/session';
import { deriveMessages } from '@/session';
import { serializeMessage } from '@/session/messageSerialization';

function createMessageEntry(
  id: string,
  message: HumanMessage | AIMessage
): SessionMessageEntry {
  return {
    type: 'message',
    id,
    parentId: null,
    timestamp: '2026-08-24T00:00:00.000Z',
    data: {
      role: message.getType(),
      message: serializeMessage(message),
    },
  };
}

function createSummaryEntry(id: string): SessionSummaryEntry {
  return {
    type: 'summary',
    id,
    parentId: null,
    timestamp: '2026-08-24T00:00:00.000Z',
    data: {
      text: 'Earlier conversation summary',
      tokenCount: 42,
      retainedEntryIds: [],
      summarizedEntryIds: [],
    },
  };
}

describe('deriveMessages', () => {
  it('derives active messages after the latest compaction summary', () => {
    const entries: SessionEntry[] = [
      createMessageEntry('old-user', new HumanMessage('old question')),
      createMessageEntry('old-assistant', new AIMessage('old answer')),
      createSummaryEntry('summary'),
      {
        type: 'run_event',
        id: 'completed',
        parentId: 'summary',
        timestamp: '2026-08-24T00:01:00.000Z',
        data: { event: 'run.completed' },
      },
      createMessageEntry('recent-user', new HumanMessage('recent question')),
    ];

    const derived = deriveMessages(entries);

    expect(derived.initialSummary).toEqual({
      text: 'Earlier conversation summary',
      tokenCount: 42,
    });
    expect(derived.messages.map((message) => message.content)).toEqual([
      'recent question',
    ]);
  });

  it('uses a zero token count when a legacy summary omitted one', () => {
    const summary = createSummaryEntry('summary');
    delete summary.data.tokenCount;

    const derived = deriveMessages([summary]);

    expect(derived.initialSummary).toEqual({
      text: 'Earlier conversation summary',
      tokenCount: 0,
    });
    expect(derived.messages).toEqual([]);
  });

  it('keeps the full path when the log has no summary', () => {
    const derived = deriveMessages([
      createMessageEntry('user', new HumanMessage('hello')),
      createMessageEntry('assistant', new AIMessage('hi')),
    ]);

    expect(derived.initialSummary).toBeUndefined();
    expect(derived.messages.map((message) => message.content)).toEqual([
      'hello',
      'hi',
    ]);
  });
});
