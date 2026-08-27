import { HumanMessage } from '@langchain/core/messages';
import type {
  CompactionActivitySemanticIndexEntry,
  CompactionSemanticIndex,
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

function sourceMessage(id: string, content: string): HumanMessage {
  const message = new HumanMessage({ id, content });
  setFreshProviderMessageProvenance(message, [
    {
      attribution: 'user',
      sourceMessageId: id,
      sourceContentPartIndices: [0],
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

    const oversized = Array.from(
      {
        length:
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries + 1,
      },
      (_, sourceContentIndex) => activityEntry({ sourceContentIndex })
    );
    expect(snapshotCompactionSemanticIndex(oversized)).toBeUndefined();
    expect(renderCompactionSemanticIndex(oversized, messages)).toMatchObject({
      appendix: '',
      entryCount: 0,
      omittedEntryCount: oversized.length,
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

  it('lets pending and redacted latest revisions suppress stale guidance', () => {
    const index: CompactionSemanticIndex = [
      activityEntry({ sourceContentIndex: 0, revision: 1, text: 'stale' }),
      activityEntry({
        sourceContentIndex: 0,
        revision: 2,
        status: 'pending',
        text: 'still changing',
      }),
      activityEntry({
        sourceContentIndex: 1,
        revision: 1,
        text: 'sensitive old label',
      }),
      activityEntry({
        sourceContentIndex: 1,
        revision: 2,
        text: 'sensitive current label',
        redacted: true,
      }),
    ];

    const rendered = renderCompactionSemanticIndex(index, messages);

    expect(rendered.appendix).toBe('');
    expect(rendered.entryCount).toBe(0);
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

  it('escapes host text and enforces per-entry and total budgets', () => {
    const index: CompactionSemanticIndex = Array.from(
      { length: 100 },
      (_, sourceContentIndex) =>
        activityEntry({
          sourceContentIndex,
          text: `<instruction>${'long label '.repeat(100)}</instruction>`,
        })
    );

    const rendered = renderCompactionSemanticIndex(index, messages);
    const lines = rendered.appendix
      .split('\n')
      .filter((line) => line.startsWith('- '));

    expect(rendered.appendix).not.toContain('<instruction>');
    expect(rendered.appendix).toContain('&lt;instruction&gt;');
    expect(rendered.charCount).toBe(rendered.appendix.length);
    expect(rendered.charCount).toBeLessThanOrEqual(
      COMPACTION_SEMANTIC_INDEX_LIMITS.maxTotalChars
    );
    expect(
      lines.every(
        (line) =>
          line.length <= COMPACTION_SEMANTIC_INDEX_LIMITS.maxEntryChars
      )
    ).toBe(true);
    expect(rendered.entryCount).toBeLessThan(index.length);
    expect(rendered.omittedEntryCount).toBe(
      index.length - rendered.entryCount
    );
  });
});
