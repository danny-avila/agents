/**
 * Regression coverage for a generic `custom` OpenAI-compatible endpoint
 * (provider `openai`, default `reasoningKey`, non-standard model name) that
 * streams reasoning in the modern vLLM `reasoning` field — `reasoning_content`
 * is `null` throughout — and streams `tool_calls` with fragmented arguments
 * (vLLM `--reasoning-parser qwen3` + `--tool-call-parser qwen3_coder`).
 *
 * The wire shapes mirror the captures in LibreChat discussion #13849:
 *   - reasoning must surface as `think` content (the Thoughts block), and
 *   - the fragmented streamed tool call must accumulate into a structured call.
 */
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import type { OpenAIClient } from '@langchain/openai';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { createContentAggregator } from '@/stream';
import { ChatOpenAI } from '@/llm/openai';
import { Run } from '@/run';

type StreamingCompletionBackedModel = {
  completions: {
    completionWithRetry: () => Promise<
      AsyncIterable<OpenAIClient.Chat.Completions.ChatCompletionChunk>
    >;
  };
};

type DeltaConverterModel = {
  completions: {
    _convertCompletionsDeltaToBaseMessageChunk(
      delta: Record<string, unknown>,
      rawResponse: Record<string, unknown>,
      defaultRole?: string
    ): AIMessageChunk;
  };
};

const MODEL = 'custom_llm_thinking';

function completionChunk(
  delta: Record<string, unknown>,
  finishReason: string | null = null
): OpenAIClient.Chat.Completions.ChatCompletionChunk {
  return {
    id: 'chatcmpl-vllm',
    object: 'chat.completion.chunk',
    created: 0,
    model: MODEL,
    choices: [
      {
        index: 0,
        delta:
          delta as OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice['delta'],
        finish_reason:
          finishReason as OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice['finish_reason'],
      },
    ],
  };
}

function captureHandlers(
  aggregateContent: t.ContentAggregator,
  reasoningDeltas: t.ReasoningDeltaEvent[],
  messageDeltas: t.MessageDeltaEvent[]
): Record<string | GraphEvents, t.EventHandler> {
  return {
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.StreamEventData): void =>
        aggregateContent({ event, data: data as t.RunStep }),
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        messageDeltas.push(data as t.MessageDeltaEvent);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData
      ): void => {
        reasoningDeltas.push(data as t.ReasoningDeltaEvent);
        aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
      },
    },
  };
}

describe('custom OpenAI-compatible endpoint (vLLM reasoning + qwen3_coder tool calls)', () => {
  const config = {
    configurable: { thread_id: 'vllm-reasoning-toolcalls' },
    streamMode: 'values' as const,
    version: 'v2' as const,
  };

  it('renders reasoning from delta.reasoning when reasoning_content is null', async () => {
    const reasoningTokens = [
      'Here',
      '\'s a thinking',
      ' process: 17*23',
      ' = 391',
    ];
    const contentTokens = ['\n\nUm ', '17 ×', ' 23 = **391', '**.'];

    const reasoningDeltas: t.ReasoningDeltaEvent[] = [];
    const messageDeltas: t.MessageDeltaEvent[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();

    const run = await Run.create<t.IState>({
      runId: 'vllm-reasoning',
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI, streamUsage: false },
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: captureHandlers(
        aggregateContent,
        reasoningDeltas,
        messageDeltas
      ),
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const model = new ChatOpenAI({
      model: MODEL,
      apiKey: 'test-key',
      streaming: true,
    });
    const completions = (model as unknown as StreamingCompletionBackedModel)
      .completions;
    async function* streamChunks(): AsyncGenerator<OpenAIClient.Chat.Completions.ChatCompletionChunk> {
      yield completionChunk({ role: 'assistant', content: '' });
      for (const reasoning of reasoningTokens) {
        yield completionChunk({
          reasoning,
          reasoning_content: null,
          content: null,
        });
      }
      for (let i = 0; i < contentTokens.length; i++) {
        yield completionChunk(
          {
            reasoning: null,
            reasoning_content: null,
            content: contentTokens[i],
          },
          i === contentTokens.length - 1 ? 'stop' : null
        );
      }
    }
    completions.completionWithRetry = async () => streamChunks();
    run.Graph.overrideModel = model as t.ChatModel;

    await run.processStream(
      {
        messages: [
          new HumanMessage('Was ist 17*23? Denk Schritt für Schritt.'),
        ],
      },
      config
    );

    const thoughts = reasoningDeltas
      .flatMap((delta) => delta.delta.content ?? [])
      .map((part) => (part as { think?: string }).think ?? '')
      .join('');
    const answer = messageDeltas
      .flatMap((delta) => delta.delta.content ?? [])
      .map((part) => (part as { text?: string }).text ?? '')
      .join('');

    expect(reasoningDeltas).toHaveLength(reasoningTokens.length);
    expect(thoughts).toBe(reasoningTokens.join(''));
    expect(answer).toBe(contentTokens.join(''));
    expect(contentParts.map((part) => part?.type)).toEqual([
      ContentTypes.THINK,
      ContentTypes.TEXT,
    ]);
  });

  it('accumulates fragmented streamed tool_calls into a structured call', () => {
    const model = new ChatOpenAI({ model: MODEL, apiKey: 'test-key' });
    const converter = (model as unknown as DeltaConverterModel).completions;
    const rawResponse = {
      id: 'chatcmpl-vllm',
      object: 'chat.completion.chunk',
      created: 0,
      model: MODEL,
      choices: [],
    };

    const deltas: Record<string, unknown>[] = [
      { role: 'assistant', tool_calls: null, content: '\n\n' },
      {
        tool_calls: [
          {
            id: 'call_0065144d618f4f33be1491af',
            type: 'function',
            index: 0,
            function: { name: 'get_weather', arguments: '' },
          },
        ],
        content: null,
      },
      {
        tool_calls: [{ index: 0, function: { arguments: '{' } }],
        content: null,
      },
      {
        tool_calls: [
          { index: 0, function: { arguments: '"location": "Berlin"' } },
        ],
        content: null,
      },
      { tool_calls: [{ index: 0, function: { arguments: '}' } }] },
    ];

    const chunks = deltas.map((delta) =>
      converter._convertCompletionsDeltaToBaseMessageChunk(
        delta,
        rawResponse,
        'assistant'
      )
    );

    const argFragments = chunks
      .flatMap((chunk) => chunk.tool_call_chunks ?? [])
      .map((toolCallChunk) => toolCallChunk.args ?? '')
      .join('');
    expect(argFragments).toBe('{"location": "Berlin"}');

    let merged = chunks[0];
    for (let i = 1; i < chunks.length; i++) {
      merged = merged.concat(chunks[i]);
    }
    expect(merged.tool_calls).toEqual([
      {
        id: 'call_0065144d618f4f33be1491af',
        name: 'get_weather',
        args: { location: 'Berlin' },
        type: 'tool_call',
      },
    ]);
  });
});
