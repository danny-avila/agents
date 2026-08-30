import type {
  CompactionSemanticIndex,
  MessageContentComplex,
  TPayload,
} from '@/types';
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

  it('rejects malformed base revisions before delta revision selection', () => {
    const baseIndex: CompactionSemanticIndex = Array.from(
      { length: COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries },
      (_, index) => ({
        type: 'activity_phase' as const,
        sourceMessageId:
          index === 200 ? 'malformed-revision-source' : `valid-base-${index}`,
        sourceContentIndex: 0,
        revision: index === 200 ? Number.MAX_SAFE_INTEGER + 1 : 1,
        status: 'committed' as const,
        text: index === 200 ? 'invalid newer label' : `Valid base ${index}`,
      })
    );
    const payload: TPayload = [
      {
        role: 'assistant',
        messageId: 'malformed-revision-source',
        content: [
          { type: ContentTypes.TEXT, text: 'Current answer' },
          {
            type: ContentTypes.ACTIVITY_LABEL,
            activity_label: 'valid delta label',
            activity_label_type: 'phase',
            activity_label_revision: 2,
            activity_start_index: 0,
            pending: false,
          } as MessageContentComplex,
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { baseIndex } }
    );

    expect(
      result.compactionSemanticIndex?.filter(
        ({ sourceMessageId }) =>
          sourceMessageId === 'malformed-revision-source'
      )
    ).toEqual([
      expect.objectContaining({ revision: 2, text: 'valid delta label' }),
    ]);
  });

  it('evolves a bounded prior snapshot without changing delta provider messages', () => {
    const baseIndex: CompactionSemanticIndex = [
      {
        type: 'activity_phase',
        sourceMessageId: 'prior-source',
        sourceContentIndex: 0,
        revision: 1,
        status: 'committed',
        text: 'Completed the prior phase',
      },
    ];
    const baseline = formatAgentMessages(
      semanticPayload(),
      undefined,
      undefined,
      undefined,
      { preserveReasoningContent: true }
    );
    const evolved = formatAgentMessages(
      semanticPayload(),
      undefined,
      undefined,
      undefined,
      {
        preserveReasoningContent: true,
        compactionSemanticIndex: {
          baseIndex,
          intentToolNames: new Set(['search_docs']),
        },
      }
    );

    baseIndex[0].text = 'caller mutation';

    expect(evolved.messages.map((message) => message.toDict())).toEqual(
      baseline.messages.map((message) => message.toDict())
    );
    expect(evolved.compactionSemanticIndex).toHaveLength(5);
    expect(evolved.compactionSemanticIndex?.[0]).toEqual(
      expect.objectContaining({
        sourceMessageId: 'prior-source',
        text: 'Completed the prior phase',
      })
    );
    expect(evolved.compactionSemanticIndex).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          sourceMessageId: 'assistant-semantic-source',
          type: 'tool_intent',
        }),
      ])
    );
  });

  it('fails an oversized prior snapshot closed while retaining the delta', () => {
    const oversizedBase: CompactionSemanticIndex = Array.from(
      { length: COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries + 1 },
      (_, index) => ({
        type: 'activity_phase' as const,
        sourceMessageId: `oversized-source-${index}`,
        sourceContentIndex: 0,
        revision: 0,
        status: 'committed' as const,
        text: `Oversized prior phase ${index}`,
      })
    );

    const result = formatAgentMessages(
      semanticPayload(),
      undefined,
      undefined,
      undefined,
      {
        compactionSemanticIndex: {
          baseIndex: oversizedBase,
          intentToolNames: new Set(['search_docs']),
        },
      }
    );

    expect(result.compactionSemanticIndex).toHaveLength(4);
    expect(result.compactionSemanticIndex).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          sourceMessageId: expect.stringContaining('oversized-source-'),
        }),
      ])
    );
    expect(
      renderCompactionSemanticIndex(
        result.compactionSemanticIndex,
        result.messages
      )
    ).toEqual(
      expect.objectContaining({
        providedEntryCount:
          oversizedBase.length +
          (result.compactionSemanticIndex?.length ?? 0),
        entryCount: 3,
        omittedEntryCount: oversizedBase.length + 1,
      })
    );
  });

  it('keeps serialized warm-turn evolution bounded and coverage balanced', () => {
    let evolved: CompactionSemanticIndex = Array.from(
      { length: COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries },
      (_, index) => ({
        type: 'activity_phase' as const,
        sourceMessageId: `base-source-${index}`,
        sourceContentIndex: 0,
        revision: 0,
        status: 'committed' as const,
        text: `Base phase ${index}`,
      })
    );

    for (let turn = 0; turn < 24; turn++) {
      const payload: TPayload = [
        {
          role: 'assistant',
          messageId: `warm-source-${turn}`,
          content: [
            { type: ContentTypes.TEXT, text: `Warm answer ${turn}` },
            {
              type: ContentTypes.ACTIVITY_LABEL,
              activity_label: `Warm phase ${turn}`,
              activity_label_type: 'phase',
              activity_start_index: 0,
              pending: false,
            } as MessageContentComplex,
          ],
        },
      ];
      const result = formatAgentMessages(
        payload,
        undefined,
        undefined,
        undefined,
        { compactionSemanticIndex: { baseIndex: evolved } }
      );
      evolved = JSON.parse(
        JSON.stringify(result.compactionSemanticIndex)
      ) as CompactionSemanticIndex;
      expect(evolved.length).toBeLessThanOrEqual(
        COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries
      );
    }

    expect(evolved).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ sourceMessageId: 'base-source-0' }),
        expect.objectContaining({ sourceMessageId: 'warm-source-23' }),
      ])
    );
  });

  it('applies newer delta tombstones over a full prior snapshot', () => {
    const baseIndex: CompactionSemanticIndex = Array.from(
      { length: COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries },
      (_, index) => ({
        type: 'activity_phase' as const,
        sourceMessageId:
          index === 200 ? 'continued-source' : `revision-base-${index}`,
        sourceContentIndex: 0,
        revision: 1,
        status: 'committed' as const,
        text: index === 200 ? 'stale label' : `Base label ${index}`,
      })
    );
    const payload: TPayload = [
      {
        role: 'assistant',
        messageId: 'continued-source',
        content: [
          { type: ContentTypes.TEXT, text: 'Continued answer' },
          {
            type: ContentTypes.ACTIVITY_LABEL,
            activity_label: 'new pending label',
            activity_label_type: 'phase',
            activity_label_revision: 2,
            activity_start_index: 0,
            pending: true,
          } as MessageContentComplex,
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { baseIndex } }
    );

    expect(
      result.compactionSemanticIndex?.filter(
        ({ sourceMessageId }) => sourceMessageId === 'continued-source'
      )
    ).toEqual([
      expect.objectContaining({
        revision: 2,
        status: 'pending',
        text: '',
      }),
    ]);
  });

  it('reuses the formatter parse for string-backed tool arguments', () => {
    const args = JSON.stringify({
      intent: 'Inspect the projection',
      query: 'projection',
    });
    const payload = semanticPayload();
    const content = payload[0].content;
    if (Array.isArray(content) && content[0].type === ContentTypes.TOOL_CALL) {
      content[0].tool_call = { ...content[0].tool_call, args };
    }
    const parse = jest.spyOn(JSON, 'parse');

    try {
      formatAgentMessages(payload, undefined, undefined, undefined, {
        compactionSemanticIndex: { intentToolNames: new Set(['search_docs']) },
      });

      expect(parse.mock.calls.filter(([value]) => value === args)).toHaveLength(
        1
      );
    } finally {
      parse.mockRestore();
    }
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

  it.each(['2', 1.5, Number.POSITIVE_INFINITY])(
    'rejects an explicitly malformed activity revision %p',
    (revision) => {
      const payload = semanticPayload();
      const content = payload[0].content;
      if (Array.isArray(content)) {
        Object.assign(content[2], { activity_label_revision: revision });
      }

      const result = formatAgentMessages(
        payload,
        undefined,
        undefined,
        undefined,
        { compactionSemanticIndex: {} }
      );

      expect(result.compactionSemanticIndex).not.toEqual(
        expect.arrayContaining([
          expect.objectContaining({ type: 'activity_phase' }),
        ])
      );
    }
  );

  it('turns a malformed activity pending flag into a suppressing tombstone', () => {
    const malformedPending = {
      type: ContentTypes.ACTIVITY_LABEL,
      activity_label: 'malformed current label',
      activity_label_type: 'phase',
      activity_label_revision: 2,
      activity_start_index: 0,
      pending: false,
    } as MessageContentComplex;
    Object.assign(malformedPending, { pending: 'true' });
    const payload: TPayload = [
      {
        role: 'assistant',
        messageId: 'malformed-pending-source',
        content: [
          { type: ContentTypes.TEXT, text: 'Old evidence' },
          {
            type: ContentTypes.ACTIVITY_LABEL,
            activity_label: 'stale committed label',
            activity_label_type: 'phase',
            activity_label_revision: 1,
            activity_start_index: 0,
            pending: false,
          } as MessageContentComplex,
        ],
      },
      {
        role: 'assistant',
        messageId: 'malformed-pending-source',
        content: [
          { type: ContentTypes.TEXT, text: 'Current evidence' },
          malformedPending,
        ],
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: {} }
    );

    expect(result.compactionSemanticIndex?.[1]).toMatchObject({
      revision: 2,
      status: 'pending',
      text: '',
      redacted: true,
    });
    expect(
      renderCompactionSemanticIndex(
        result.compactionSemanticIndex,
        result.messages
      ).appendix
    ).not.toContain('stale committed label');
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
    const content: MessageContentComplex[] = Array.from(
      { length: toolCallCount },
      (_, index) => ({
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: `tool-${index}`,
          name: 'search_docs',
          args: { intent: `Search ${index}` },
          output: `Result ${index}`,
          outcome: `Found ${index}`,
        },
      })
    );
    const middleToolIndex = toolCallCount / 2;
    content.splice(middleToolIndex + 1, 0, {
      type: ContentTypes.ACTIVITY_LABEL,
      activity_label: 'Rare middle activity',
      activity_label_type: 'phase',
      activity_start_index: middleToolIndex,
      pending: false,
    } as MessageContentComplex);
    const payload: TPayload = [
      {
        role: 'assistant',
        messageId: 'assistant-bounded-source',
        content,
      },
    ];

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { intentToolNames: new Set(['search_docs']) } }
    );

    expect(result.compactionSemanticIndex?.length).toBeLessThanOrEqual(
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries
    );
    expect(result.compactionSemanticIndex).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ text: 'Search 0' }),
        expect.objectContaining({ text: `Found ${toolCallCount - 1}` }),
        expect.objectContaining({ text: 'Rare middle activity' }),
      ])
    );
    expect(result.messages).toHaveLength(toolCallCount + 1);

    const rendered = renderCompactionSemanticIndex(
      result.compactionSemanticIndex,
      result.messages
    );
    expect(rendered.providedEntryCount).toBe(toolCallCount * 2 + 1);
    expect(rendered.appendix).toContain('Search 0');
    expect(rendered.appendix).toContain(`Found ${toolCallCount - 1}`);
    expect(rendered.appendix).toContain('Rare middle activity');
    expect(rendered.omittedEntryCount).toBe(
      rendered.providedEntryCount - rendered.entryCount
    );
  });

  it('retains newer suppressing revisions when balanced admission overflows', () => {
    const messageCount = COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries + 80;
    const payload: TPayload = Array.from(
      { length: messageCount },
      (_, index) => {
        const isOldRevision = index === 0;
        const isNewRevision = index === 100;
        let activityLabel = `Filler activity ${index}`;
        if (isOldRevision) {
          activityLabel = 'stale retained label';
        } else if (isNewRevision) {
          activityLabel = 'new pending label';
        }
        return {
          role: 'assistant',
          messageId:
            isOldRevision || isNewRevision
              ? 'revision-source'
              : `filler-source-${index}`,
          content: [
            { type: ContentTypes.TEXT, text: `Assistant text ${index}` },
            {
              type: ContentTypes.ACTIVITY_LABEL,
              activity_label: activityLabel,
              activity_label_type: 'phase',
              activity_label_revision: isNewRevision ? 2 : 1,
              activity_start_index: 0,
              pending: isNewRevision,
            } as MessageContentComplex,
          ],
        };
      }
    );

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: {} }
    );
    const revisions = result.compactionSemanticIndex?.filter(
      ({ sourceMessageId }) => sourceMessageId === 'revision-source'
    );

    expect(revisions).toEqual([
      expect.objectContaining({
        revision: 2,
        status: 'pending',
        text: '',
      }),
    ]);
    expect(
      renderCompactionSemanticIndex(
        result.compactionSemanticIndex,
        result.messages
      ).appendix
    ).not.toContain('stale retained label');
  });

  it('renews recent retention when an admitted identity advances', () => {
    const messageCount = COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries + 100;
    const payload: TPayload = Array.from(
      { length: messageCount },
      (_, index) => {
        const isOldRevision = index === 200;
        const isNewRevision = index === 257;
        let activityLabel = `Filler activity ${index}`;
        if (isOldRevision) {
          activityLabel = 'older guidance';
        } else if (isNewRevision) {
          activityLabel = 'latest retained guidance';
        }
        return {
          role: 'assistant',
          messageId:
            isOldRevision || isNewRevision
              ? 'recent-revision-source'
              : `recent-filler-source-${index}`,
          content: [
            { type: ContentTypes.TEXT, text: `Assistant text ${index}` },
            {
              type: ContentTypes.ACTIVITY_LABEL,
              activity_label: activityLabel,
              activity_label_type: 'phase',
              activity_label_revision: isNewRevision ? 2 : 1,
              activity_start_index: 0,
              pending: false,
            } as MessageContentComplex,
          ],
        };
      }
    );

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: {} }
    );

    expect(
      result.compactionSemanticIndex?.filter(
        ({ sourceMessageId }) => sourceMessageId === 'recent-revision-source'
      )
    ).toEqual([
      expect.objectContaining({
        revision: 2,
        text: 'latest retained guidance',
      }),
    ]);
  });

  it('balances revisions with renderer-normalized identities', () => {
    const messageCount = COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries + 80;
    const payload: TPayload = Array.from(
      { length: messageCount },
      (_, index) => {
        const isOldRevision = index === 0;
        const isNewRevision = index === 100;
        let reasoningLabel = `Filler reasoning ${index}`;
        let reasoningStepId = `filler-step-${index}`;
        if (isOldRevision) {
          reasoningLabel = 'stale normalized label';
          reasoningStepId = 'shared-step';
        } else if (isNewRevision) {
          reasoningLabel = 'new normalized pending label';
          reasoningStepId = ' shared-step ';
        }
        return {
          role: 'assistant',
          messageId:
            isOldRevision || isNewRevision
              ? 'normalized-revision-source'
              : `normalized-filler-source-${index}`,
          content: [
            {
              type: ContentTypes.THINK,
              think: `Reasoning ${index}`,
              reasoning_label: reasoningLabel,
              reasoning_label_step_id: reasoningStepId,
              reasoning_label_revision: isNewRevision ? 2 : 1,
              reasoning_label_status: isNewRevision ? 'streaming' : 'complete',
            } as MessageContentComplex,
          ],
        };
      }
    );

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { preserveReasoningContent: true, compactionSemanticIndex: {} }
    );
    const revisions = result.compactionSemanticIndex?.filter(
      ({ sourceMessageId }) => sourceMessageId === 'normalized-revision-source'
    );

    expect(revisions).toEqual([
      expect.objectContaining({
        revision: 2,
        status: 'pending',
        text: '',
      }),
    ]);
    expect(
      renderCompactionSemanticIndex(
        result.compactionSemanticIndex,
        result.messages
      ).appendix
    ).not.toContain('stale normalized label');
  });

  it('accepts renderer-equivalent text for an equal retained revision', () => {
    const messageCount = COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries + 80;
    const payload: TPayload = Array.from(
      { length: messageCount },
      (_, index) => {
        const isFirstRevision = index === 0;
        const isEquivalentRevision = index === 100;
        let activityLabel = `Filler activity ${index}`;
        if (isFirstRevision) {
          activityLabel = 'checking cache';
        } else if (isEquivalentRevision) {
          activityLabel = 'checking   cache';
        }
        return {
          role: 'assistant',
          messageId:
            isFirstRevision || isEquivalentRevision
              ? 'equivalent-text-source'
              : `equivalent-filler-source-${index}`,
          content: [
            { type: ContentTypes.TEXT, text: `Assistant text ${index}` },
            {
              type: ContentTypes.ACTIVITY_LABEL,
              activity_label: activityLabel,
              activity_label_type: 'phase',
              activity_label_revision: 1,
              activity_start_index: 0,
              pending: false,
            } as MessageContentComplex,
          ],
        };
      }
    );

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: {} }
    );

    expect(
      result.compactionSemanticIndex?.filter(
        ({ sourceMessageId }) => sourceMessageId === 'equivalent-text-source'
      )
    ).toEqual([
      expect.objectContaining({
        revision: 1,
        status: 'committed',
        text: 'checking cache',
      }),
    ]);
  });

  it('returns bounded tombstones instead of retaining oversized label text', () => {
    const payload = semanticPayload();
    const content = payload[0].content;
    if (Array.isArray(content) && content[0].type === ContentTypes.TOOL_CALL) {
      content[0].tool_call = {
        ...content[0].tool_call,
        outcome: 'x'.repeat(
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputTextChars + 1
        ),
      };
    }

    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      { compactionSemanticIndex: { intentToolNames: new Set(['search_docs']) } }
    );

    expect(result.compactionSemanticIndex).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: 'tool_outcome',
          text: '',
          redacted: true,
        }),
      ])
    );
  });
});
