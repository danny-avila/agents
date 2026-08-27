import type { MessageContentComplex, TPayload } from '@/types';
import { renderCompactionSemanticIndex } from '@/summarization/semanticIndex';
import { COMPACTION_SEMANTIC_INDEX_LIMITS, ContentTypes } from '@/common';
import { formatAgentMessages } from './format';

const semanticPayload = (): TPayload => [
  {
    role: 'assistant',
    messageId: 'assistant-semantic-source',
    content: [
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'search-1',
          name: 'search_docs',
          args: JSON.stringify({
            intent: 'Locate the cache implementation',
            query: 'cache',
          }),
          output: 'Found the implementation.',
          outcome: 'Located the cache implementation',
        },
      },
      {
        type: ContentTypes.THINK,
        think: 'Private reasoning remains raw-message data, not index text.',
        reasoning_label: 'Checking cache ownership',
        reasoning_label_step_id: 'reasoning-1',
        reasoning_label_revision: 2,
        reasoning_label_status: 'complete',
      },
      {
        type: ContentTypes.ACTIVITY_LABEL,
        activity_label: 'Mapped the cache request path',
        activity_label_type: 'phase',
        activity_start_index: 0,
        pending: false,
      },
      {
        type: ContentTypes.TEXT,
        text: 'The cache is initialized in the request adapter.',
      },
    ],
  },
];

type TestSemanticPart = MessageContentComplex & {
  activity_start_index?: number;
  pending?: boolean;
  reasoning_label_status?: string;
};

describe('formatAgentMessages compaction semantic index', () => {
  it('derives exact semantic guidance without changing provider messages', () => {
    const baseline = formatAgentMessages(
      semanticPayload(),
      undefined,
      undefined,
      undefined,
      { preserveReasoningContent: true }
    );
    const derived = formatAgentMessages(
      semanticPayload(),
      undefined,
      undefined,
      undefined,
      {
        preserveReasoningContent: true,
        compactionSemanticIndex: {
          intentToolNames: new Set(['search_docs']),
        },
      }
    );

    expect(derived.messages.map((message) => message.toDict())).toEqual(
      baseline.messages.map((message) => message.toDict())
    );
    expect(baseline.compactionSemanticIndex).toBeUndefined();
    expect(derived.compactionSemanticIndex).toEqual([
      {
        type: 'tool_intent',
        sourceMessageId: 'assistant-semantic-source',
        sourceContentIndex: 0,
        revision: 0,
        status: 'committed',
        text: 'Locate the cache implementation',
        toolCallId: 'search-1',
      },
      {
        type: 'tool_outcome',
        sourceMessageId: 'assistant-semantic-source',
        sourceContentIndex: 0,
        revision: 0,
        status: 'committed',
        text: 'Located the cache implementation',
        toolCallId: 'search-1',
      },
      {
        type: 'reasoning_label',
        sourceMessageId: 'assistant-semantic-source',
        sourceContentIndex: 1,
        revision: 2,
        status: 'committed',
        text: 'Checking cache ownership',
        reasoningStepId: 'reasoning-1',
      },
      {
        type: 'activity_phase',
        sourceMessageId: 'assistant-semantic-source',
        sourceContentIndex: 0,
        revision: 0,
        status: 'committed',
        text: 'Mapped the cache request path',
      },
    ]);

    const rendered = renderCompactionSemanticIndex(
      derived.compactionSemanticIndex,
      derived.messages
    );
    expect(rendered.entryCount).toBe(4);
    expect(rendered.appendix).toContain('Locate the cache implementation');
    expect(rendered.appendix).toContain('Mapped the cache request path');
    expect(rendered.appendix).toContain('Checking cache ownership');
    expect(rendered.appendix).not.toContain('Private reasoning remains');
  });

  it('requires host ownership before treating an intent argument as a label', () => {
    const result = formatAgentMessages(
      semanticPayload(),
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: {} }
    );

    expect(result.compactionSemanticIndex).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'tool_intent' })])
    );
    expect(result.compactionSemanticIndex).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'tool_outcome' }),
        expect.objectContaining({ type: 'reasoning_label' }),
        expect.objectContaining({ type: 'activity_phase' }),
      ])
    );
  });

  it('fails closed without stable source identity and preserves pending state', () => {
    const payload = semanticPayload();
    delete payload[0].messageId;
    const content = payload[0].content;
    if (Array.isArray(content)) {
      const reasoning = content[1] as TestSemanticPart;
      const activity = content[2] as TestSemanticPart;
      reasoning.reasoning_label_status = 'streaming';
      activity.pending = true;
    }

    const unidentified = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { intentToolNames: new Set(['search_docs']) } }
    );
    expect(unidentified.compactionSemanticIndex).toBeUndefined();

    payload[0].messageId = 'assistant-pending-source';
    const pending = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { intentToolNames: new Set(['search_docs']) } }
    );
    expect(pending.compactionSemanticIndex).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'reasoning_label', status: 'pending' }),
        expect.objectContaining({ type: 'activity_phase', status: 'pending' }),
      ])
    );
  });

  it('does not derive a phase whose evidence was removed by a summary slice', () => {
    const payload = semanticPayload();
    const content = payload[0].content;
    if (Array.isArray(content)) {
      content.splice(1, 0, {
        type: ContentTypes.SUMMARY,
        text: 'Covered prior tool work',
        tokenCount: 12,
      });
      const activity = content[3] as TestSemanticPart;
      activity.activity_start_index = 0;
    }

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { intentToolNames: new Set(['search_docs']) } }
    );

    expect(result.compactionSemanticIndex).toEqual([
      expect.objectContaining({
        type: 'reasoning_label',
        sourceContentIndex: 2,
      }),
    ]);
  });

  it('bounds entries while continuing ordinary message reconstruction', () => {
    const toolCallCount =
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries / 2 + 16;
    const payload: TPayload = [
      {
        role: 'assistant',
        messageId: 'assistant-bounded-source',
        content: Array.from({ length: toolCallCount }, (_, index) => ({
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: `tool-${index}`,
            name: 'search_docs',
            args: { intent: `Search ${index}` },
            output: `Result ${index}`,
            outcome: `Found ${index}`,
          },
        })),
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { intentToolNames: new Set(['search_docs']) } }
    );

    expect(result.compactionSemanticIndex).toHaveLength(
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries
    );
    expect(result.messages).toHaveLength(toolCallCount + 1);
  });
});
