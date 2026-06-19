/**
 * Regression coverage for a generic `custom` OpenAI-compatible endpoint
 * (provider `openai`, non-OpenAI `baseURL`, default `reasoningKey`,
 * non-standard model name) that streams reasoning in the modern vLLM
 * `reasoning` field — `reasoning_content` is `null` throughout — and streams
 * `tool_calls` with fragmented arguments (vLLM `--reasoning-parser qwen3` +
 * `--tool-call-parser qwen3_coder`).
 *
 * A non-OpenAI `baseURL` is required so the model stays on the custom-endpoint
 * (final-signal) path rather than the official-OpenAI sequential-seal path.
 *
 * The wire shapes mirror the captures in LibreChat discussion #13849:
 *   - reasoning must surface as `think` content (the Thoughts block), and
 *   - the fragmented streamed tool call must be assembled by the graph into a
 *     structured call and executed.
 */
import { DynamicStructuredTool } from '@langchain/core/tools';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type { OpenAIClient } from '@langchain/openai';
import type * as t from '@/types';
import { ContentTypes, GraphEvents, Providers, StepTypes } from '@/common';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { createContentAggregator } from '@/stream';
import { ChatOpenAI } from '@/llm/openai';
import { Run } from '@/run';

type CompletionChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;
type CompletionChoice =
  OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice;
type CompletionDelta = CompletionChoice['delta'] & {
  reasoning?: string | null;
  reasoning_content?: string | null;
};
type StreamingCompletions = {
  completionWithRetry: () => Promise<AsyncIterable<CompletionChunk>>;
};

const MODEL = 'custom_llm_thinking';
const BASE_URL = 'http://vllm.internal:8000/v1';

function completionChunk(
  delta: CompletionDelta,
  finishReason: CompletionChoice['finish_reason'] = null
): CompletionChunk {
  return {
    id: 'chatcmpl-vllm',
    object: 'chat.completion.chunk',
    created: 0,
    model: MODEL,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

function customEndpointModel(): ChatOpenAI {
  return new ChatOpenAI({
    model: MODEL,
    apiKey: 'test-key',
    streaming: true,
    configuration: { baseURL: BASE_URL },
  });
}

function setCompletionStream(
  model: ChatOpenAI,
  stream: () => AsyncIterable<CompletionChunk>
): void {
  (
    model as unknown as { completions: StreamingCompletions }
  ).completions.completionWithRetry = async () => stream();
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
      customHandlers: {
        [GraphEvents.ON_RUN_STEP]: {
          handle: (
            event: GraphEvents.ON_RUN_STEP,
            data: t.StreamEventData
          ): void => aggregateContent({ event, data: data as t.RunStep }),
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
      },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const model = customEndpointModel();
    setCompletionStream(
      model,
      async function* streamChunks(): AsyncGenerator<CompletionChunk> {
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
    );
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

  it('assembles fragmented streamed tool_calls and executes the tool through the graph', async () => {
    let weatherArgs: unknown;
    const getWeather = new DynamicStructuredTool({
      name: 'get_weather',
      description: 'Get the current weather for a location',
      schema: {
        type: 'object',
        properties: { location: { type: 'string' } },
        required: ['location'],
      },
      func: async (input: unknown): Promise<string> => {
        weatherArgs = input;
        return JSON.stringify({
          location: 'Berlin',
          summary: 'sunny',
          temp_c: 18,
        });
      },
    });

    const { aggregateContent } = createContentAggregator();
    const collectedUsage: UsageMetadata[] = [];
    const runSteps: t.RunStep[] = [];

    const run = await Run.create<t.IState>({
      runId: 'vllm-tool-calls',
      graphConfig: {
        type: 'standard',
        llmConfig: { provider: Providers.OPENAI, streamUsage: false },
        tools: [getWeather],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.TOOL_END]: new ToolEndHandler(),
        [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
        [GraphEvents.ON_RUN_STEP]: {
          handle: (
            event: GraphEvents.ON_RUN_STEP,
            data: t.StreamEventData
          ): void => {
            runSteps.push(data as t.RunStep);
            aggregateContent({ event, data: data as t.RunStep });
          },
        },
        [GraphEvents.ON_RUN_STEP_COMPLETED]: {
          handle: (
            event: GraphEvents.ON_RUN_STEP_COMPLETED,
            data: t.StreamEventData
          ): void =>
            aggregateContent({
              event,
              data: data as unknown as { result: t.ToolEndEvent },
            }),
        },
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (
            event: GraphEvents.ON_MESSAGE_DELTA,
            data: t.StreamEventData
          ): void =>
            aggregateContent({ event, data: data as t.MessageDeltaEvent }),
        },
      },
    });
    if (!run.Graph) {
      throw new Error('Expected graph to be initialized');
    }

    const model = customEndpointModel();
    async function* toolCallStream(): AsyncGenerator<CompletionChunk> {
      yield completionChunk({ role: 'assistant', content: '\n\n' });
      yield completionChunk({
        tool_calls: [
          {
            id: 'call_0065144d618f4f33be1491af',
            type: 'function',
            index: 0,
            function: { name: 'get_weather', arguments: '' },
          },
        ],
      });
      yield completionChunk({
        tool_calls: [{ index: 0, function: { arguments: '{' } }],
      });
      yield completionChunk({
        tool_calls: [
          { index: 0, function: { arguments: '"location": "Berlin"' } },
        ],
      });
      yield completionChunk(
        { tool_calls: [{ index: 0, function: { arguments: '}' } }] },
        'tool_calls'
      );
    }
    async function* finalAnswerStream(): AsyncGenerator<CompletionChunk> {
      yield completionChunk({
        role: 'assistant',
        content: 'Das Wetter in Berlin ist sonnig.',
      });
      yield completionChunk({ content: '' }, 'stop');
    }
    let modelCall = 0;
    setCompletionStream(model, () => {
      modelCall += 1;
      return modelCall === 1 ? toolCallStream() : finalAnswerStream();
    });
    run.Graph.overrideModel = model as t.ChatModel;

    await run.processStream(
      { messages: [new HumanMessage('Wie ist das Wetter in Berlin?')] },
      config
    );

    const toolCallStepNames = runSteps
      .filter((step) => step.stepDetails.type === StepTypes.TOOL_CALLS)
      .flatMap(
        (step) => (step.stepDetails as t.ToolCallsDetails).tool_calls ?? []
      )
      .map((call) => ('function' in call ? call.function.name : call.name));

    const messages = run.getRunMessages() ?? [];
    const finalAnswer = messages
      .filter((message): message is AIMessage => message.getType() === 'ai')
      .map((message) =>
        typeof message.content === 'string' ? message.content : ''
      )
      .join('');

    expect(modelCall).toBe(2);
    expect(toolCallStepNames).toContain('get_weather');
    expect(weatherArgs).toEqual({ location: 'Berlin' });
    expect(messages.some((message) => message.getType() === 'tool')).toBe(true);
    expect(finalAnswer).toContain('Das Wetter in Berlin ist sonnig.');
  });
});
