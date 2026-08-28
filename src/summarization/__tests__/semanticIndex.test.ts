import { HumanMessage } from '@langchain/core/messages';
import type {
  CompactionActivitySemanticIndexEntry,
  CompactionSemanticIndex,
  CompactionSemanticIndexEntry,
} from '@/types';
import {
  COMPACTION_SEMANTIC_INDEX_LIMITS,
  renderCompactionSemanticIndex,
  snapshotCompactionSemanticIndex,
} from '@/summarization/semanticIndex';
import {
  setInvalidProviderMessageProvenance,
  setFreshProviderMessageProvenance,
} from '@/messages/provenance';

function activityEntry(
  overrides: Partial<CompactionActivitySemanticIndexEntry> = {}
): CompactionActivitySemanticIndexEntry {
  return {
    type: 'activity_phase',
    sourceMessageId: 'message-1',
    sourceContentIndex: 0,
    revision: 1,
    status: 'committed',
    text: 'Inspected the provider request path',
    ...overrides,
  };
}

function sourceMessage(
  id: string,
  content: string,
  sourceContentPartIndices: number[] = Array.from(
    { length: 128 },
    (_, index) => index
  )
): HumanMessage {
  const message = new HumanMessage({ id, content });
  setFreshProviderMessageProvenance(message, [
    {
      attribution: 'user',
      sourceMessageId: id,
      sourceContentPartIndices,
    },
  ]);
  return message;
}

describe('renderCompactionSemanticIndex', () => {
  const messages = [
    sourceMessage('message-1', 'first'),
    sourceMessage('message-2', 'second'),
  ];

  it('fails closed when a message carries no explicit source provenance', () => {
    const rendered = renderCompactionSemanticIndex(
      [activityEntry()],
      [new HumanMessage({ id: 'message-1', content: 'unstamped' })]
    );

    expect(rendered.entryCount).toBe(0);
    expect(rendered.appendix).toBe('');
  });

  it('fails closed when source provenance is explicitly malformed', () => {
    const malformed = new HumanMessage({
      id: 'message-1',
      content: 'malformed',
    });
    setInvalidProviderMessageProvenance(malformed);

    const rendered = renderCompactionSemanticIndex(
      [activityEntry()],
      [malformed]
    );

    expect(rendered.entryCount).toBe(0);
    expect(rendered.appendix).toBe('');
  });

  it('returns no appendix when the host supplies no index', () => {
    expect(renderCompactionSemanticIndex(undefined, messages)).toEqual({
      appendix: '',
      providedEntryCount: 0,
      entryCount: 0,
      charCount: 0,
      omittedEntryCount: 0,
    });
  });

  it('snapshots caller-owned entries and fails an oversized input closed', () => {
    const callerEntry = activityEntry();
    const callerIndex: CompactionActivitySemanticIndexEntry[] = [callerEntry];
    const snapshot = snapshotCompactionSemanticIndex(callerIndex);

    callerEntry.text = 'mutated after construction';
    callerIndex.push(activityEntry({ sourceContentIndex: 1 }));

    expect(snapshot).toHaveLength(1);
    expect(snapshot?.[0].text).toBe('Inspected the provider request path');
    expect(Object.isFrozen(snapshot)).toBe(true);
    expect(Object.isFrozen(snapshot?.[0])).toBe(true);
    expect(
      snapshotCompactionSemanticIndex([
        activityEntry({ text: 'secret', redacted: true }),
      ])?.[0].text
    ).toBe('');
    expect(
      snapshotCompactionSemanticIndex([
        activityEntry({ text: 'x'.repeat(10_000) }),
      ])
    ).toEqual([expect.objectContaining({ text: '', redacted: true })]);

    const oversized = Array.from(
      {
        length: COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries + 1,
      },
      (_, sourceContentIndex) => activityEntry({ sourceContentIndex })
    );
    const oversizedSnapshot = snapshotCompactionSemanticIndex(oversized);
    expect(oversizedSnapshot).toEqual([]);
    expect(
      renderCompactionSemanticIndex(oversizedSnapshot, messages)
    ).toMatchObject({
      appendix: '',
      entryCount: 0,
      omittedEntryCount: oversized.length,
    });
  });

  it('captures the admitted array length before reading accessor-backed entries', () => {
    const callerIndex = [activityEntry()];
    let expandedEntryRead = false;
    Object.defineProperty(callerIndex, 0, {
      get() {
        callerIndex.length = 2;
        Object.defineProperty(callerIndex, 1, {
          get() {
            expandedEntryRead = true;
            return activityEntry({ sourceContentIndex: 1 });
          },
        });
        return activityEntry();
      },
    });

    const snapshot = snapshotCompactionSemanticIndex(callerIndex);

    expect(snapshot).toHaveLength(1);
    expect(expandedEntryRead).toBe(false);
    expect(renderCompactionSemanticIndex(snapshot, messages)).toMatchObject({
      providedEntryCount: 1,
      entryCount: 1,
      omittedEntryCount: 0,
    });
  });

  it('rejects malformed and out-of-range source references', () => {
    const index: CompactionSemanticIndex = [
      activityEntry({ sourceMessageId: '' }),
      activityEntry({ sourceMessageId: 'missing' }),
      activityEntry({ sourceContentIndex: -1 }),
      activityEntry({
        sourceContentIndex:
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxSourceContentIndex + 1,
      }),
    ];

    const rendered = renderCompactionSemanticIndex(index, messages);

    expect(rendered.appendix).toBe('');
    expect(rendered.entryCount).toBe(0);
    expect(rendered.omittedEntryCount).toBe(index.length);
  });

  it('requires the exact source content part to remain in the compaction range', () => {
    const rendered = renderCompactionSemanticIndex(
      [
        activityEntry({ sourceContentIndex: 0, text: 'retained part' }),
        activityEntry({ sourceContentIndex: 2, text: 'omitted part' }),
      ],
      [sourceMessage('message-1', 'partial source', [0])]
    );

    expect(rendered.entryCount).toBe(1);
    expect(rendered.appendix).toContain('retained part');
    expect(rendered.appendix).not.toContain('omitted part');
  });

  it('uses only the highest revision and fails closed on a conflicting tie', () => {
    const index: CompactionSemanticIndex = [
      activityEntry({ revision: 1, text: 'old label' }),
      activityEntry({ revision: 2, text: 'settled label' }),
      activityEntry({ revision: 2, text: 'settled label' }),
      activityEntry({
        sourceContentIndex: 1,
        revision: 3,
        text: 'conflict a',
      }),
      activityEntry({
        sourceContentIndex: 1,
        revision: 3,
        text: 'conflict b',
      }),
    ];

    const rendered = renderCompactionSemanticIndex(index, messages);

    expect(rendered.entryCount).toBe(1);
    expect(rendered.appendix).toContain('settled label');
    expect(rendered.appendix).not.toContain('old label');
    expect(rendered.appendix).not.toContain('conflict a');
    expect(rendered.appendix).not.toContain('conflict b');
  });

  it('rejects oversized identities and tied text before bounded values can alias', () => {
    const oversizedIdentity = 'i'.repeat(
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars + 1
    );
    const sharedPrefix = 'x'.repeat(
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputTextChars
    );
    const index: CompactionSemanticIndex = [
      activityEntry({ sourceMessageId: oversizedIdentity }),
      activityEntry({
        sourceContentIndex: 2,
        revision: 3,
        text: `${sharedPrefix}a`,
      }),
      activityEntry({
        sourceContentIndex: 2,
        revision: 3,
        text: `${sharedPrefix}b`,
      }),
    ];

    expect(snapshotCompactionSemanticIndex(index)).toEqual([
      expect.objectContaining({ text: '', redacted: true }),
      expect.objectContaining({ text: '', redacted: true }),
    ]);
    expect(renderCompactionSemanticIndex(index, messages)).toMatchObject({
      appendix: '',
      entryCount: 0,
      omittedEntryCount: index.length,
    });
  });

  it('lets pending and redacted latest revisions suppress stale guidance', () => {
    const index: CompactionSemanticIndex = [
      activityEntry({ sourceContentIndex: 0, revision: 1, text: 'stale' }),
      activityEntry({
        sourceContentIndex: 0,
        revision: 2,
        status: 'pending',
        text: 'still changing'.repeat(1_000),
      }),
      activityEntry({
        sourceContentIndex: 1,
        revision: 1,
        text: 'sensitive old label',
      }),
      activityEntry({
        sourceContentIndex: 1,
        revision: 2,
        text: 'sensitive current label'.repeat(1_000),
        redacted: true,
      }),
    ];

    const rendered = renderCompactionSemanticIndex(index, messages);

    expect(rendered.appendix).toBe('');
    expect(rendered.entryCount).toBe(0);
    expect(snapshotCompactionSemanticIndex(index)).toHaveLength(index.length);
  });

  it('lets an oversized latest revision suppress stale guidance', () => {
    const snapshot = snapshotCompactionSemanticIndex([
      activityEntry({ revision: 1, text: 'stale label' }),
      activityEntry({
        revision: 2,
        text: 'x'.repeat(
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputTextChars + 1
        ),
      }),
    ]);

    expect(snapshot).toHaveLength(2);
    expect(snapshot?.[1]).toMatchObject({ text: '', redacted: true });
    expect(renderCompactionSemanticIndex(snapshot, messages)).toMatchObject({
      appendix: '',
      providedEntryCount: 2,
      entryCount: 0,
      omittedEntryCount: 2,
    });
  });

  it('preserves omission counts when snapshotting rejects entries', () => {
    const hostileEntry = activityEntry();
    Object.defineProperty(hostileEntry, 'text', {
      get() {
        throw new Error('hostile getter');
      },
    });
    const snapshot = snapshotCompactionSemanticIndex([
      activityEntry({
        sourceMessageId: 'x'.repeat(
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars + 1
        ),
      }),
      hostileEntry,
      activityEntry({ text: 'accepted' }),
    ]);

    expect(snapshot).toHaveLength(1);
    expect(renderCompactionSemanticIndex(snapshot, messages)).toMatchObject({
      providedEntryCount: 3,
      entryCount: 1,
      omittedEntryCount: 2,
    });
  });

  it('orders by source position, content position, semantic type, and local identity', () => {
    const index: CompactionSemanticIndex = [
      {
        type: 'reasoning_label',
        sourceMessageId: 'message-2',
        sourceContentIndex: 0,
        reasoningStepId: 'reasoning-2',
        revision: 1,
        status: 'committed',
        text: 'reasoning second',
      },
      {
        type: 'tool_outcome',
        sourceMessageId: 'message-1',
        sourceContentIndex: 2,
        toolCallId: 'tool-b',
        revision: 1,
        status: 'committed',
        text: 'outcome',
      },
      {
        type: 'tool_intent',
        sourceMessageId: 'message-1',
        sourceContentIndex: 2,
        toolCallId: 'tool-a',
        revision: 1,
        status: 'committed',
        text: 'intent',
      },
    ];

    const forward = renderCompactionSemanticIndex(index, messages).appendix;
    const reversed = renderCompactionSemanticIndex(
      [...index].reverse(),
      messages
    ).appendix;

    expect(reversed).toBe(forward);
    expect(forward.indexOf('intent')).toBeLessThan(forward.indexOf('outcome'));
    expect(forward.indexOf('outcome')).toBeLessThan(
      forward.indexOf('reasoning second')
    );
    expect(forward).not.toContain('message-1');
    expect(forward).not.toContain('tool-a');
    expect(forward).not.toContain('reasoning-2');
  });

  it('preserves contribution order when source messages are folded together', () => {
    const folded = new HumanMessage('folded context');
    setFreshProviderMessageProvenance(folded, [
      {
        attribution: 'user',
        sourceMessageId: 'message-1',
        sourceContentPartIndices: [2],
      },
      {
        attribution: 'user',
        sourceMessageId: 'message-2',
        sourceContentPartIndices: [0],
      },
    ]);
    const rendered = renderCompactionSemanticIndex(
      [
        activityEntry({
          sourceMessageId: 'message-2',
          sourceContentIndex: 0,
          text: 'later source',
        }),
        activityEntry({
          sourceMessageId: 'message-1',
          sourceContentIndex: 2,
          text: 'earlier source',
        }),
      ],
      [folded]
    );

    expect(rendered.appendix.indexOf('earlier source')).toBeLessThan(
      rendered.appendix.indexOf('later source')
    );
  });

  it('balances temporal and semantic-type coverage before spending the character budget', () => {
    const filler = 'x'.repeat(420);
    const index: CompactionSemanticIndex = [
      {
        type: 'tool_intent',
        sourceMessageId: 'message-1',
        sourceContentIndex: 0,
        toolCallId: 'tool-initial',
        revision: 1,
        status: 'committed',
        text: `initial goal ${filler}`,
      },
      ...Array.from(
        { length: 124 },
        (_, offset): CompactionSemanticIndexEntry => ({
          type: 'tool_intent',
          sourceMessageId: 'message-1',
          sourceContentIndex: offset + 1,
          toolCallId: `tool-${offset + 1}`,
          revision: 1,
          status: 'committed',
          text: `middle intent ${offset + 1} ${filler}`,
        })
      ),
      {
        type: 'reasoning_label',
        sourceMessageId: 'message-1',
        sourceContentIndex: 52,
        reasoningStepId: 'reasoning-52',
        revision: 1,
        status: 'committed',
        text: 'rare reasoning checkpoint',
      },
      activityEntry({
        sourceContentIndex: 78,
        text: 'rare activity checkpoint',
      }),
      {
        type: 'tool_outcome',
        sourceMessageId: 'message-1',
        sourceContentIndex: 127,
        toolCallId: 'tool-latest',
        revision: 1,
        status: 'committed',
        text: `latest outcome ${filler}`,
      },
    ];

    const rendered = renderCompactionSemanticIndex(index, [
      sourceMessage('message-1', 'long tool history'),
    ]);

    expect(rendered.entryCount).toBeLessThan(index.length);
    expect(rendered.appendix).toContain('initial goal');
    expect(rendered.appendix).toContain('rare reasoning checkpoint');
    expect(rendered.appendix).toContain('rare activity checkpoint');
    expect(rendered.appendix).toContain('latest outcome');
    expect(rendered.appendix.indexOf('initial goal')).toBeLessThan(
      rendered.appendix.indexOf('rare reasoning checkpoint')
    );
    expect(rendered.appendix.indexOf('rare activity checkpoint')).toBeLessThan(
      rendered.appendix.indexOf('latest outcome')
    );
  });

  it('escapes host text and enforces per-entry and total budgets', () => {
    const index: CompactionSemanticIndex = Array.from(
      { length: 100 },
      (_, sourceContentIndex) =>
        activityEntry({
          sourceContentIndex,
          text: `<instruction>${'&'.repeat(1_000)}</instruction>`,
        })
    );

    const rendered = renderCompactionSemanticIndex(index, messages);
    const lines = rendered.appendix
      .split('\n')
      .filter((line) => line.startsWith('- '));

    expect(rendered.appendix).not.toContain('<instruction>');
    expect(rendered.appendix).toContain('&lt;instruction&gt;');
    expect(rendered.appendix).not.toContain('&amp…');
    expect(rendered.appendix).not.toContain('&am…');
    expect(rendered.appendix).not.toContain('&a…');
    expect(rendered.charCount).toBe(rendered.appendix.length);
    expect(rendered.charCount).toBeLessThanOrEqual(
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxTotalChars
    );
    expect(
      lines.every(
        (line) => line.length <= COMPACTION_SEMANTIC_INDEX_LIMITS.maxEntryChars
      )
    ).toBe(true);
    expect(rendered.entryCount).toBeLessThan(index.length);
    expect(rendered.omittedEntryCount).toBe(index.length - rendered.entryCount);
  });
});
