import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';

import type { AnthropicMessageStreamEvent } from './types';

import { CustomAnthropic } from './index';

type ToolArgChunk = NonNullable<AIMessageChunk['tool_call_chunks']>[number];
type MessageWithToolArgChunks = {
  content?: unknown;
  tool_call_chunks?: ToolArgChunk[];
};
type ToolArgStreamChunks = {
  contentInputs: string[];
  toolArgChunks: ToolArgChunk[];
};
type AnthropicEventStream = AsyncIterable<AnthropicMessageStreamEvent> & {
  controller: AbortController;
};

function anthropicEvent(event: unknown): AnthropicMessageStreamEvent {
  return event as AnthropicMessageStreamEvent;
}

function createAnthropicEventStream(
  events: AnthropicMessageStreamEvent[]
): AnthropicEventStream {
  return {
    controller: new AbortController(),
    async *[Symbol.asyncIterator](): AsyncGenerator<AnthropicMessageStreamEvent> {
      for (const event of events) {
        yield event;
      }
    },
  };
}

function createFakeStreamingAnthropic(
  events: AnthropicMessageStreamEvent[],
  streamDelay = 0
): CustomAnthropic {
  const model = new CustomAnthropic({
    apiKey: 'test-key',
    model: 'claude-haiku-4-5-20251001',
    streaming: true,
    _lc_stream_delay: streamDelay,
  });
  Object.defineProperty(model, 'createStreamWithRetry', {
    value: async () => createAnthropicEventStream(events),
  });
  return model;
}

function createToolArgEvents(
  partialJsonChunks: string[]
): AnthropicMessageStreamEvent[] {
  return [
    anthropicEvent({
      type: 'content_block_start',
      index: 0,
      content_block: {
        type: 'tool_use',
        id: 'toolu_lookup',
        name: 'lookup',
        input: {},
      },
    }),
    ...partialJsonChunks.map((partialJson) =>
      anthropicEvent({
        type: 'content_block_delta',
        index: 0,
        delta: {
          type: 'input_json_delta',
          partial_json: partialJson,
        },
      })
    ),
    anthropicEvent({ type: 'content_block_stop', index: 0 }),
    anthropicEvent({ type: 'message_stop' }),
  ];
}

async function collectToolArgChunks(
  partialJsonChunks: string[],
  streamDelay = 0
): Promise<ToolArgStreamChunks> {
  const model = createFakeStreamingAnthropic(
    createToolArgEvents(partialJsonChunks),
    streamDelay
  );
  const contentInputs: string[] = [];
  const toolArgChunks: ToolArgChunk[] = [];

  const stream = model._streamResponseChunks([new HumanMessage('lookup')], {
    tools: [
      {
        name: 'lookup',
        description: 'Lookup a query',
        input_schema: {
          type: 'object',
          properties: {
            q: { type: 'string' },
          },
        },
      },
    ],
  });
  for await (const chunk of stream) {
    const message = chunk.message as MessageWithToolArgChunks;
    toolArgChunks.push(...(message.tool_call_chunks ?? []));
    if (!Array.isArray(message.content)) {
      continue;
    }
    const [firstContentBlock] = message.content;
    if (
      typeof firstContentBlock === 'object' &&
      firstContentBlock !== null &&
      'input' in firstContentBlock &&
      typeof firstContentBlock.input === 'string'
    ) {
      contentInputs.push(firstContentBlock.input);
    }
  }

  return { contentInputs, toolArgChunks };
}

describe('CustomAnthropic tool argument streaming', () => {
  test('streams input_json_delta chunks before the final tool args are complete', async () => {
    const { contentInputs, toolArgChunks } = await collectToolArgChunks([
      '{"q":',
      '"sf"}',
    ]);

    expect(toolArgChunks.map((chunk) => chunk.args)).toEqual([
      '',
      '{"q":',
      '"sf"}',
    ]);
    expect(contentInputs).toEqual(['', '{"q":', '"sf"}']);
    expect(toolArgChunks.map((chunk) => chunk.args).join('')).toBe(
      '{"q":"sf"}'
    );
  });

  test('delays large input_json_delta chunks without splitting structured input', async () => {
    jest.useFakeTimers();
    try {
      const rawArgs = '{"query": "San Francisco", "unit": "fahrenheit"}';
      const collection = collectToolArgChunks([rawArgs], 20);
      let settled = false;
      collection.then(() => {
        settled = true;
      });

      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(19);
      expect(settled).toBe(false);

      await jest.advanceTimersByTimeAsync(1);
      const { contentInputs, toolArgChunks } = await collection;

      expect(toolArgChunks.map((chunk) => chunk.args)).toEqual(['', rawArgs]);
      expect(contentInputs).toEqual(['', rawArgs]);
      expect(toolArgChunks.every((chunk) => chunk.index === 0)).toBe(true);
    } finally {
      jest.useRealTimers();
    }
  });

  test('paces dense input_json_delta bursts without delaying every provider chunk', async () => {
    jest.useFakeTimers();
    try {
      const rawChunks = Array.from({ length: 10 }, () => 'abcdefghij');
      const collection = collectToolArgChunks(rawChunks, 20);
      let settled = false;
      collection.then(() => {
        settled = true;
      });

      await Promise.resolve();
      await jest.advanceTimersByTimeAsync(19);
      expect(settled).toBe(false);

      await jest.advanceTimersByTimeAsync(1);
      const { toolArgChunks } = await collection;

      expect(settled).toBe(true);
      expect(toolArgChunks.map((chunk) => chunk.args)).toEqual([
        '',
        ...rawChunks,
      ]);
    } finally {
      jest.useRealTimers();
    }
  });
});
