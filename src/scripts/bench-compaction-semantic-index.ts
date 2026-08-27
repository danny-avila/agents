import { performance } from 'node:perf_hooks';
import { HumanMessage } from '@langchain/core/messages';
import type { CompactionSemanticIndex } from '@/types';
import { setFreshProviderMessageProvenance } from '@/messages/provenance';
import { renderCompactionSemanticIndex } from '@/summarization/semanticIndex';

const WARMUP_ITERATIONS = 5_000;
const MEASURED_ITERATIONS = 25_000;

const messages = Array.from({ length: 12 }, (_, index) => {
  const sourceMessageId = `message-${index}`;
  const message = new HumanMessage({
    id: sourceMessageId,
    content: `message ${index}`,
  });
  setFreshProviderMessageProvenance(message, [
    { attribution: 'user', sourceMessageId },
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

run(WARMUP_ITERATIONS, undefined);
run(WARMUP_ITERATIONS, semanticIndex);

const disabled = run(MEASURED_ITERATIONS, undefined);
const enabled = run(MEASURED_ITERATIONS, semanticIndex);

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
    },
    null,
    2
  )
);
