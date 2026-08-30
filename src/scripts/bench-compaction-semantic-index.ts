import { performance } from 'node:perf_hooks';
import { HumanMessage } from '@langchain/core/messages';
import type {
  CompactionSemanticIndex,
  CompactionSemanticIndexEntry,
  MessageContentComplex,
  TPayload,
} from '@/types';
import { setFreshProviderMessageProvenance } from '@/messages/provenance';
import { renderCompactionSemanticIndex } from '@/summarization/semanticIndex';
import { formatAgentMessages } from '@/messages/format';
import { COMPACTION_SEMANTIC_INDEX_LIMITS, ContentTypes } from '@/common';

const WARMUP_ITERATIONS = 5_000;
const MEASURED_ITERATIONS = 25_000;
const FORMAT_WARMUP_ITERATIONS = 500;
const FORMAT_SAMPLE_ITERATIONS = 300;
const FORMAT_SAMPLE_COUNT = 9;
const INCREMENTAL_WARMUP_ITERATIONS = 300;
const INCREMENTAL_SAMPLE_ITERATIONS = 300;
const INTENT_TOOL_NAMES: ReadonlySet<string> = new Set(['search_docs']);

const messages = Array.from({ length: 12 }, (_, index) => {
  const sourceMessageId = `message-${index}`;
  const message = new HumanMessage({
    id: sourceMessageId,
    content: `message ${index}`,
  });
  setFreshProviderMessageProvenance(message, [
    {
      attribution: 'user',
      sourceMessageId,
      sourceContentPartIndices: [0, 1, 2, 3],
    },
  ]);
  return message;
});
const semanticIndex: CompactionSemanticIndex = Array.from(
  { length: 48 },
  (_, index) => ({
    type: 'activity_phase' as const,
    sourceMessageId: `message-${Math.floor(index / 4)}`,
    sourceContentIndex: index % 4,
    revision: 1,
    status: 'committed' as const,
    text: `Completed bounded activity ${index} and recorded its outcome`,
  })
);

function run(
  iterations: number,
  index: CompactionSemanticIndex | undefined
): { elapsedMs: number; checksum: number } {
  let checksum = 0;
  const startedAt = performance.now();
  for (let iteration = 0; iteration < iterations; iteration++) {
    checksum += renderCompactionSemanticIndex(index, messages).charCount;
  }
  return { elapsedMs: performance.now() - startedAt, checksum };
}

function createPersistedMessage(messageIndex: number): TPayload[number] {
  return {
    role: 'assistant',
    messageId: `persisted-${messageIndex}`,
    content: [
      ...Array.from({ length: 2 }, (_, toolIndex) => ({
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: `tool-${messageIndex}-${toolIndex}`,
          name: 'search_docs',
          args: {
            intent: `Locate implementation ${messageIndex}-${toolIndex}`,
            query: `query ${messageIndex}-${toolIndex}`,
          },
          output: `result ${messageIndex}-${toolIndex}`,
          outcome: `Located implementation ${messageIndex}-${toolIndex}`,
        },
      })),
      {
        type: ContentTypes.THINK,
        think: `reasoning ${messageIndex}`,
        reasoning_label: `Checked ownership ${messageIndex}`,
        reasoning_label_step_id: `reasoning-${messageIndex}`,
        reasoning_label_revision: 1,
        reasoning_label_status: 'complete',
      },
      {
        type: ContentTypes.ACTIVITY_LABEL,
        activity_label: `Mapped request phase ${messageIndex}`,
        activity_label_type: 'phase',
        activity_start_index: 0,
        pending: false,
      },
      {
        type: ContentTypes.TEXT,
        text: `Completed request analysis ${messageIndex}`,
      },
    ],
  };
}

const persistedPayload: TPayload = Array.from(
  { length: 32 },
  (_, messageIndex) => createPersistedMessage(messageIndex)
);
const warmHistoryPayload: TPayload = Array.from(
  { length: 100 },
  (_, messageIndex) => createPersistedMessage(messageIndex)
);
const incrementalDeltaPayload: TPayload = [
  {
    role: 'assistant',
    messageId: 'persisted-warm-delta',
    content: [
      {
        type: ContentTypes.TOOL_CALL,
        tool_call: {
          id: 'tool-warm-delta',
          name: 'search_docs',
          args: {
            intent: 'Locate the warm continuation implementation',
            query: 'warm continuation',
          },
          output: 'result warm delta',
          outcome: 'Located the warm continuation implementation',
        },
      },
      {
        type: ContentTypes.ACTIVITY_LABEL,
        activity_label: 'Mapped the warm continuation phase',
        activity_label_type: 'phase',
        activity_start_index: 0,
        pending: false,
      },
      { type: ContentTypes.TEXT, text: 'Completed the warm continuation' },
    ],
  },
];

function stripActivityLabelParts(payload: TPayload): TPayload {
  return payload.map((message) => {
    if (!Array.isArray(message.content)) {
      return message;
    }
    const filtered = message.content.filter(
      (part) => part.type !== ContentTypes.ACTIVITY_LABEL
    );
    if (filtered.length === message.content.length) {
      return message;
    }
    return { ...message, content: filtered };
  });
}

type BenchmarkSemanticPart = MessageContentComplex & {
  activity_label?: string;
  activity_label_type?: string;
  activity_start_index?: number;
  pending?: boolean;
  reasoning_label?: string;
  reasoning_label_revision?: number;
  reasoning_label_status?: string;
  reasoning_label_step_id?: string;
};

function appendSeparateSemanticEntry(
  entries: CompactionSemanticIndexEntry[],
  entry: CompactionSemanticIndexEntry
): void {
  if (entries.length >= COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputEntries) {
    return;
  }
  if (entry.status === 'pending') {
    entries.push({ ...entry, text: '' });
    return;
  }
  if (entry.text.length > COMPACTION_SEMANTIC_INDEX_LIMITS.maxInputTextChars) {
    entries.push({ ...entry, text: '', redacted: true });
    return;
  }
  entries.push(entry);
}

function deriveSemanticIndexSeparately(
  payload: TPayload
): CompactionSemanticIndexEntry[] {
  const entries: CompactionSemanticIndexEntry[] = [];
  for (let messageIndex = 0; messageIndex < payload.length; messageIndex++) {
    const message = payload[messageIndex];
    const sourceMessageId =
      typeof message.messageId === 'string' ? message.messageId : undefined;
    if (
      sourceMessageId == null ||
      sourceMessageId.length >
        COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars ||
      !Array.isArray(message.content)
    ) {
      continue;
    }
    for (
      let contentIndex = 0;
      contentIndex < message.content.length;
      contentIndex++
    ) {
      const part = message.content[contentIndex] as BenchmarkSemanticPart;
      if (
        contentIndex > COMPACTION_SEMANTIC_INDEX_LIMITS.maxSourceContentIndex
      ) {
        continue;
      }
      if (
        part.type === ContentTypes.TOOL_CALL &&
        typeof part.tool_call?.id === 'string' &&
        part.tool_call.id !== '' &&
        part.tool_call.id.length <=
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars
      ) {
        const { id: toolCallId, args, outcome } = part.tool_call;
        const intent =
          part.tool_call.name != null &&
          INTENT_TOOL_NAMES.has(part.tool_call.name) &&
          args != null &&
          typeof args === 'object' &&
          !Array.isArray(args)
            ? args.intent
            : undefined;
        if (typeof intent === 'string' && intent !== '') {
          appendSeparateSemanticEntry(entries, {
            type: 'tool_intent',
            sourceMessageId,
            sourceContentIndex: contentIndex,
            revision: 0,
            status: 'committed',
            text: intent,
            toolCallId,
          });
        }
        if (typeof outcome === 'string' && outcome !== '') {
          appendSeparateSemanticEntry(entries, {
            type: 'tool_outcome',
            sourceMessageId,
            sourceContentIndex: contentIndex,
            revision: 0,
            status: 'committed',
            text: outcome,
            toolCallId,
          });
        }
        continue;
      }
      if (
        part.type === ContentTypes.THINK &&
        typeof part.reasoning_label === 'string' &&
        typeof part.reasoning_label_step_id === 'string' &&
        part.reasoning_label_step_id !== '' &&
        part.reasoning_label_step_id.length <=
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxIdentityChars &&
        typeof part.reasoning_label_revision === 'number' &&
        Number.isSafeInteger(part.reasoning_label_revision) &&
        part.reasoning_label_revision >= 0 &&
        (part.reasoning_label_status === 'complete' ||
          part.reasoning_label_status === 'streaming')
      ) {
        appendSeparateSemanticEntry(entries, {
          type: 'reasoning_label',
          sourceMessageId,
          sourceContentIndex: contentIndex,
          revision: part.reasoning_label_revision,
          status:
            part.reasoning_label_status === 'complete'
              ? 'committed'
              : 'pending',
          text: part.reasoning_label,
          reasoningStepId: part.reasoning_label_step_id,
        });
        continue;
      }
      if (
        part.type === ContentTypes.ACTIVITY_LABEL &&
        part.activity_label_type === 'phase' &&
        typeof part.activity_label === 'string' &&
        typeof part.activity_start_index === 'number' &&
        Number.isSafeInteger(part.activity_start_index) &&
        part.activity_start_index >= 0 &&
        part.activity_start_index < message.content.length &&
        part.activity_start_index <=
          COMPACTION_SEMANTIC_INDEX_LIMITS.maxSourceContentIndex
      ) {
        appendSeparateSemanticEntry(entries, {
          type: 'activity_phase',
          sourceMessageId,
          sourceContentIndex: part.activity_start_index,
          revision: 0,
          status: part.pending === true ? 'pending' : 'committed',
          text: part.activity_label,
        });
      }
    }
  }
  return entries;
}

type FormatBenchmarkMode =
  | 'disabled'
  | 'legacy-prestrip'
  | 'separate-projection'
  | 'one-pass';

function runFormatting(
  iterations: number,
  mode: FormatBenchmarkMode
): { elapsedMs: number; checksum: number } {
  let checksum = 0;
  const startedAt = performance.now();
  for (let iteration = 0; iteration < iterations; iteration++) {
    const payload =
      mode === 'legacy-prestrip' || mode === 'separate-projection'
        ? stripActivityLabelParts(persistedPayload)
        : persistedPayload;
    const separatelyDerived =
      mode === 'separate-projection'
        ? deriveSemanticIndexSeparately(persistedPayload)
        : undefined;
    const result = formatAgentMessages(
      payload,
      undefined,
      undefined,
      undefined,
      mode === 'one-pass'
        ? {
            preserveReasoningContent: true,
            compactionSemanticIndex: {
              intentToolNames: INTENT_TOOL_NAMES,
            },
          }
        : { preserveReasoningContent: true }
    );
    checksum +=
      result.messages.length +
      (result.compactionSemanticIndex?.length ??
        separatelyDerived?.length ??
        0);
  }
  return { elapsedMs: performance.now() - startedAt, checksum };
}

type IncrementalBenchmarkMode = 'full-history' | 'bounded-delta';

const incrementalBaseSnapshot = formatAgentMessages(
  warmHistoryPayload,
  undefined,
  undefined,
  undefined,
  {
    preserveReasoningContent: true,
    compactionSemanticIndex: { intentToolNames: INTENT_TOOL_NAMES },
  }
).compactionSemanticIndexSnapshot;
const fullWarmHistoryPayload = [
  ...warmHistoryPayload,
  ...incrementalDeltaPayload,
];

function runIncrementalEvolution(
  iterations: number,
  mode: IncrementalBenchmarkMode
): { elapsedMs: number; checksum: number } {
  let checksum = 0;
  const startedAt = performance.now();
  for (let iteration = 0; iteration < iterations; iteration++) {
    const result = formatAgentMessages(
      mode === 'full-history'
        ? fullWarmHistoryPayload
        : incrementalDeltaPayload,
      undefined,
      undefined,
      undefined,
      {
        preserveReasoningContent: true,
        compactionSemanticIndex: {
          baseSnapshot:
            mode === 'bounded-delta' ? incrementalBaseSnapshot : undefined,
          intentToolNames: INTENT_TOOL_NAMES,
        },
      }
    );
    checksum +=
      result.messages.length + (result.compactionSemanticIndex?.length ?? 0);
  }
  return { elapsedMs: performance.now() - startedAt, checksum };
}

function median(values: number[]): number {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor(ordered.length / 2)];
}

function formatTiming(
  results: Array<{
    elapsedMs: number;
    checksum: number;
  }>
): {
  medianTotalMs: number;
  microsecondsPerProjection: number;
  checksum: number;
} {
  const medianElapsedMs = median(results.map((result) => result.elapsedMs));
  return {
    medianTotalMs: Number(medianElapsedMs.toFixed(3)),
    microsecondsPerProjection: Number(
      ((medianElapsedMs * 1_000) / FORMAT_SAMPLE_ITERATIONS).toFixed(3)
    ),
    checksum: results.reduce((total, result) => total + result.checksum, 0),
  };
}

run(WARMUP_ITERATIONS, undefined);
run(WARMUP_ITERATIONS, semanticIndex);

const disabled = run(MEASURED_ITERATIONS, undefined);
const enabled = run(MEASURED_ITERATIONS, semanticIndex);
runFormatting(FORMAT_WARMUP_ITERATIONS, 'disabled');
runFormatting(FORMAT_WARMUP_ITERATIONS, 'legacy-prestrip');
runFormatting(FORMAT_WARMUP_ITERATIONS, 'separate-projection');
runFormatting(FORMAT_WARMUP_ITERATIONS, 'one-pass');
const formatModes: FormatBenchmarkMode[] = [
  'disabled',
  'legacy-prestrip',
  'separate-projection',
  'one-pass',
];
const formatSamples = new Map<
  FormatBenchmarkMode,
  Array<{ elapsedMs: number; checksum: number }>
>(formatModes.map((mode) => [mode, []]));
for (let sample = 0; sample < FORMAT_SAMPLE_COUNT; sample++) {
  for (let offset = 0; offset < formatModes.length; offset++) {
    const mode = formatModes[(sample + offset) % formatModes.length];
    formatSamples
      .get(mode)
      ?.push(runFormatting(FORMAT_SAMPLE_ITERATIONS, mode));
  }
}
const formatDisabled = formatTiming(formatSamples.get('disabled') ?? []);
const legacyPrestrip = formatTiming(formatSamples.get('legacy-prestrip') ?? []);
const separateProjection = formatTiming(
  formatSamples.get('separate-projection') ?? []
);
const onePass = formatTiming(formatSamples.get('one-pass') ?? []);
runIncrementalEvolution(INCREMENTAL_WARMUP_ITERATIONS, 'full-history');
runIncrementalEvolution(INCREMENTAL_WARMUP_ITERATIONS, 'bounded-delta');
const incrementalModes: IncrementalBenchmarkMode[] = [
  'full-history',
  'bounded-delta',
];
const incrementalSamples = new Map<
  IncrementalBenchmarkMode,
  Array<{ elapsedMs: number; checksum: number }>
>(incrementalModes.map((mode) => [mode, []]));
for (let sample = 0; sample < FORMAT_SAMPLE_COUNT; sample++) {
  for (let offset = 0; offset < incrementalModes.length; offset++) {
    const mode = incrementalModes[(sample + offset) % incrementalModes.length];
    incrementalSamples
      .get(mode)
      ?.push(runIncrementalEvolution(INCREMENTAL_SAMPLE_ITERATIONS, mode));
  }
}
const fullHistoryEvolution = formatTiming(
  incrementalSamples.get('full-history') ?? []
);
const boundedDeltaEvolution = formatTiming(
  incrementalSamples.get('bounded-delta') ?? []
);

// eslint-disable-next-line no-console
console.log(
  JSON.stringify(
    {
      iterations: MEASURED_ITERATIONS,
      entriesPerCompaction: semanticIndex.length,
      disabled: {
        totalMs: Number(disabled.elapsedMs.toFixed(3)),
        microsecondsPerCompaction: Number(
          ((disabled.elapsedMs * 1_000) / MEASURED_ITERATIONS).toFixed(3)
        ),
        checksum: disabled.checksum,
      },
      enabled: {
        totalMs: Number(enabled.elapsedMs.toFixed(3)),
        microsecondsPerCompaction: Number(
          ((enabled.elapsedMs * 1_000) / MEASURED_ITERATIONS).toFixed(3)
        ),
        checksum: enabled.checksum,
      },
      formatterProjection: {
        iterationsPerSample: FORMAT_SAMPLE_ITERATIONS,
        samples: FORMAT_SAMPLE_COUNT,
        messagesPerProjection: persistedPayload.length,
        derivedEntriesPerProjection: persistedPayload.length * 6,
        disabled: formatDisabled,
        legacyPrestripWithoutDerivation: legacyPrestrip,
        separateProjectionWithPrestrip: separateProjection,
        onePassWithDerivation: onePass,
        onePassDeltaVsSeparatePercent: Number(
          (
            ((onePass.microsecondsPerProjection -
              separateProjection.microsecondsPerProjection) /
              separateProjection.microsecondsPerProjection) *
            100
          ).toFixed(2)
        ),
      },
      warmTurnEvolution: {
        iterationsPerSample: INCREMENTAL_SAMPLE_ITERATIONS,
        samples: FORMAT_SAMPLE_COUNT,
        historicalMessages: warmHistoryPayload.length,
        retainedBaseEntries: incrementalBaseSnapshot?.entries.length ?? 0,
        deltaMessages: incrementalDeltaPayload.length,
        fullHistory: fullHistoryEvolution,
        boundedDelta: boundedDeltaEvolution,
        speedup: Number(
          (
            fullHistoryEvolution.microsecondsPerProjection /
            boundedDeltaEvolution.microsecondsPerProjection
          ).toFixed(2)
        ),
        latencyReductionPercent: Number(
          (
            ((fullHistoryEvolution.microsecondsPerProjection -
              boundedDeltaEvolution.microsecondsPerProjection) /
              fullHistoryEvolution.microsecondsPerProjection) *
            100
          ).toFixed(2)
        ),
      },
    },
    null,
    2
  )
);
