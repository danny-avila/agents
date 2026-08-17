/**
 * An interrupted Responses turn never receives `response.completed`, so it has
 * no authoritative `response_metadata.output` to replay from. The only record
 * of its reasoning is the `additional_kwargs.reasoning` slot, which chunk
 * concatenation merges field-by-field. That slot holds one item, but a turn
 * can emit many: without a per-item boundary the merge welds every item's
 * `encrypted_content` into a single blob and pairs it with the last item's id,
 * which the provider rejects with "Encrypted content could not be decrypted or
 * parsed."
 */
import { concat } from '@langchain/core/utils/stream';
import { convertMessagesToResponsesInput } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';
import type { OpenAIClient } from '@langchain/openai';
import type { AIMessageChunk } from '@langchain/core/messages';
import { ChatOpenAI } from '@/llm/openai';

type StreamingResponsesDelegate = {
  completionWithRetry: (
    request: OpenAIClient.Responses.ResponseCreateParamsStreaming
  ) => Promise<
    AsyncIterable<OpenAIClient.Responses.ResponseStreamEvent> | undefined
  >;
};

type ReasoningItemFixture = {
  id: string;
  encryptedContent?: string;
};

const ENCRYPTED_CONTENT_BY_ID = new Map([
  ['rs_first', 'ENCRYPTED_FIRST'],
  ['rs_second', 'ENCRYPTED_SECOND'],
  ['rs_third', 'ENCRYPTED_THIRD'],
]);

function* interruptedReasoningEvents(
  items: readonly ReasoningItemFixture[]
): Generator<OpenAIClient.Responses.ResponseStreamEvent> {
  let sequence = 0;
  let outputIndex = 0;
  for (const item of items) {
    yield {
      type: 'response.output_item.added',
      sequence_number: sequence++,
      output_index: outputIndex,
      item: {
        id: item.id,
        type: 'reasoning',
        status: 'in_progress',
        summary: [],
      },
    } as OpenAIClient.Responses.ResponseStreamEvent;
    if (item.encryptedContent == null) {
      outputIndex++;
      continue;
    }
    yield {
      type: 'response.output_item.done',
      sequence_number: sequence++,
      output_index: outputIndex,
      item: {
        id: item.id,
        type: 'reasoning',
        status: 'completed',
        summary: [],
        encrypted_content: item.encryptedContent,
      },
    } as OpenAIClient.Responses.ResponseStreamEvent;
    outputIndex++;
  }
  yield {
    type: 'response.output_item.added',
    sequence_number: sequence++,
    output_index: outputIndex,
    item: {
      id: 'msg_interrupted',
      type: 'message',
      role: 'assistant',
      status: 'in_progress',
      content: [],
    },
  } as OpenAIClient.Responses.ResponseStreamEvent;
  yield {
    type: 'response.output_text.delta',
    sequence_number: sequence++,
    output_index: outputIndex,
    content_index: 0,
    item_id: 'msg_interrupted',
    delta: 'Partial answer.',
    logprobs: [],
  } as OpenAIClient.Responses.ResponseStreamEvent;
  /** The user interrupted here: no `response.completed`, no terminal output. */
}

async function streamInterruptedTurn(
  items: readonly ReasoningItemFixture[]
): Promise<AIMessageChunk> {
  const model = new ChatOpenAI({
    model: 'gpt-5.6',
    apiKey: 'test-key',
    useResponsesApi: true,
  });
  const responses = (
    model as unknown as { responses: StreamingResponsesDelegate }
  ).responses;
  responses.completionWithRetry = async () =>
    (async function* () {
      yield* interruptedReasoningEvents(items);
    })();

  let aggregate: AIMessageChunk | undefined;
  for await (const chunk of await model.stream([
    new HumanMessage('tell me a long story'),
  ])) {
    aggregate = aggregate == null ? chunk : concat(aggregate, chunk);
  }
  if (aggregate == null) {
    throw new Error('Expected the interrupted stream to yield chunks');
  }
  return aggregate;
}

function replayedReasoningItems(
  message: AIMessageChunk
): OpenAIClient.Responses.ResponseReasoningItem[] {
  return convertMessagesToResponsesInput({
    messages: [message],
    model: 'gpt-5.6',
    zdrEnabled: false,
  }).filter(
    (item): item is OpenAIClient.Responses.ResponseReasoningItem =>
      item.type === 'reasoning'
  );
}

describe('interrupted Responses reasoning replay', () => {
  it.each([
    [
      'every reasoning item completed before the interrupt',
      [
        { id: 'rs_first', encryptedContent: 'ENCRYPTED_FIRST' },
        { id: 'rs_second', encryptedContent: 'ENCRYPTED_SECOND' },
        { id: 'rs_third', encryptedContent: 'ENCRYPTED_THIRD' },
      ],
      'rs_third',
    ],
    [
      'the interrupt landed inside a later reasoning item',
      [
        { id: 'rs_first', encryptedContent: 'ENCRYPTED_FIRST' },
        { id: 'rs_second', encryptedContent: 'ENCRYPTED_SECOND' },
        { id: 'rs_third' },
      ],
      'rs_second',
    ],
    [
      'the interrupt landed inside the second reasoning item',
      [
        { id: 'rs_first', encryptedContent: 'ENCRYPTED_FIRST' },
        { id: 'rs_second' },
      ],
      'rs_first',
    ],
  ] as [string, ReasoningItemFixture[], string][])(
    'replays the last sealed item intact when %s',
    async (_label, items, expectedSealedId) => {
      const message = await streamInterruptedTurn(items);
      const replayed = replayedReasoningItems(message);

      const encrypted = replayed.filter(
        (item) => item.encrypted_content != null
      );
      expect(encrypted).toHaveLength(1);
      expect(encrypted[0].id).toBe(expectedSealedId);
      expect(encrypted[0].encrypted_content).toBe(
        ENCRYPTED_CONTENT_BY_ID.get(expectedSealedId)
      );

      /** Ids the provider cannot resolve must never ride along bare. */
      for (const item of replayed) {
        expect(item.encrypted_content).toBe(
          ENCRYPTED_CONTENT_BY_ID.get(item.id)
        );
      }
    }
  );

  it('keeps the replayed status a single provider-legal value', async () => {
    const message = await streamInterruptedTurn([
      { id: 'rs_first', encryptedContent: 'ENCRYPTED_FIRST' },
      { id: 'rs_second', encryptedContent: 'ENCRYPTED_SECOND' },
      { id: 'rs_third', encryptedContent: 'ENCRYPTED_THIRD' },
    ]);

    for (const item of replayedReasoningItems(message)) {
      if (item.status == null) {
        continue;
      }
      expect(['completed', 'in_progress', 'incomplete']).toContain(item.status);
    }
  });

  it('still replays reasoning when the turn emitted a single item', async () => {
    const message = await streamInterruptedTurn([
      { id: 'rs_first', encryptedContent: 'ENCRYPTED_FIRST' },
    ]);

    const replayed = replayedReasoningItems(message);
    expect(replayed).toHaveLength(1);
    expect(replayed[0].id).toBe('rs_first');
    expect(replayed[0].encrypted_content).toBe('ENCRYPTED_FIRST');
  });
});
