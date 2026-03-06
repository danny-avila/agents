import { config } from 'dotenv';
config();
import { Calculator } from '@/tools/Calculator';
import {
  HumanMessage,
  BaseMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import type { ChatOpenRouterCallOptions } from '@/llm/openrouter';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { ContentTypes, GraphEvents, Providers, TitleMethod } from '@/common';
import { capitalizeFirstLetter } from './spec.utils';
import { createContentAggregator } from '@/stream';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

// Auto-skip if OpenRouter env is missing
const hasOpenRouter = (process.env.OPENROUTER_API_KEY ?? '').trim() !== '';
const describeIf = hasOpenRouter ? describe : describe.skip;

const provider = Providers.OPENROUTER;
describeIf(`${capitalizeFirstLetter(provider)} Streaming Tests`, () => {
  jest.setTimeout(60000);
  let run: Run<t.IState>;
  let collectedUsage: UsageMetadata[];
  let conversationHistory: BaseMessage[];
  let contentParts: t.MessageContentComplex[];

  const configV2 = {
    configurable: { thread_id: 'or-convo-1' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const baseLLMConfig = getLLMConfig(provider);

  beforeEach(async () => {
    conversationHistory = [];
    collectedUsage = [];
    const { contentParts: cp } = createContentAggregator();
    contentParts = cp as t.MessageContentComplex[];
  });

  const onMessageDeltaSpy = jest.fn();
  const onRunStepSpy = jest.fn();

  afterAll(() => {
    onMessageDeltaSpy.mockReset();
    onRunStepSpy.mockReset();
  });

  const setupCustomHandlers = (): Record<
    string | GraphEvents,
    t.EventHandler
  > => ({
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
  });

  /**
   * Helper: run a reasoning test against a specific model with the given reasoning config.
   * Asserts that reasoning tokens are reported and content is produced.
   */
  // eslint-disable-next-line @typescript-eslint/explicit-function-return-type
  async function runReasoningTest(opts: {
    model: string;
    reasoning?: ChatOpenRouterCallOptions['reasoning'];
    threadId: string;
    runId: string;
  }) {
    const { reasoning: _baseReasoning, ...baseWithoutReasoning } =
      baseLLMConfig as unknown as Record<string, unknown>;
    const llmConfig = {
      ...baseWithoutReasoning,
      model: opts.model,
      ...(opts.reasoning != null ? { reasoning: opts.reasoning } : {}),
    } as t.LLMConfig;
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: opts.runId,
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful AI assistant. Think step by step.',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    const userMessage = 'What is 15 * 37 + 128 / 4? Show your work.';
    conversationHistory.push(new HumanMessage(userMessage));

    const finalContentParts = await run.processStream(
      { messages: conversationHistory },
      { ...configV2, configurable: { thread_id: opts.threadId } }
    );

    expect(finalContentParts).toBeDefined();
    expect(finalContentParts?.length).toBeGreaterThan(0);

    // Verify usage metadata was collected
    expect(collectedUsage.length).toBeGreaterThan(0);
    const usage = collectedUsage[0];
    expect(usage.input_tokens).toBeGreaterThan(0);
    expect(usage.output_tokens).toBeGreaterThan(0);

    // Verify reasoning tokens are reported in output_token_details
    const reasoningTokens =
      (usage.output_token_details as Record<string, number> | undefined)
        ?.reasoning ?? 0;
    expect(reasoningTokens).toBeGreaterThan(0);

    // Verify the final message has content
    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    expect(finalMessages?.length).toBeGreaterThan(0);
    const assistantMsg = finalMessages?.[0];
    expect(typeof assistantMsg?.content).toBe('string');
    expect((assistantMsg?.content as string).length).toBeGreaterThan(0);

    return { usage, reasoningTokens, finalMessages };
  }

  test(`${capitalizeFirstLetter(provider)}: simple stream + title`, async () => {
    const { userName, location } = await getArgs();
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'or-run-1',
      graphConfig: {
        type: 'standard',
        llmConfig: baseLLMConfig,
        tools: [new Calculator()],
        instructions: 'You are a friendly AI assistant.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    const userMessage = 'hi';
    conversationHistory.push(new HumanMessage(userMessage));

    const finalContentParts = await run.processStream(
      { messages: conversationHistory },
      configV2
    );
    expect(finalContentParts).toBeDefined();
    const allTextParts = finalContentParts?.every(
      (part) => part.type === ContentTypes.TEXT
    );
    expect(allTextParts).toBe(true);
    expect(
      (collectedUsage[0]?.input_tokens ?? 0) +
        (collectedUsage[0]?.output_tokens ?? 0)
    ).toBeGreaterThan(0);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    conversationHistory.push(...(finalMessages ?? []));

    const titleRes = await run.generateTitle({
      provider,
      inputText: userMessage,
      titleMethod: TitleMethod.COMPLETION,
      contentParts,
    });
    expect(titleRes.title).toBeDefined();
  });

  test(`${capitalizeFirstLetter(provider)}: Anthropic does NOT reason by default (no config)`, async () => {
    const { reasoning: _baseReasoning, ...baseWithoutReasoning } =
      baseLLMConfig as unknown as Record<string, unknown>;
    const llmConfig = {
      ...baseWithoutReasoning,
      model: 'anthropic/claude-sonnet-4',
    } as t.LLMConfig;
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'or-anthropic-default-1',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful AI assistant.',
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers,
    });

    conversationHistory.push(
      new HumanMessage('What is 15 * 37 + 128 / 4? Show your work.')
    );

    await run.processStream(
      { messages: conversationHistory },
      { ...configV2, configurable: { thread_id: 'or-anthropic-default-1' } }
    );

    expect(collectedUsage.length).toBeGreaterThan(0);
    const usage = collectedUsage[0];
    // Anthropic requires explicit reasoning config — no reasoning tokens by default
    const reasoningTokens =
      (usage.output_token_details as Record<string, number> | undefined)
        ?.reasoning ?? 0;
    expect(reasoningTokens).toBe(0);
  });

  test(`${capitalizeFirstLetter(provider)}: Gemini 3 reasons by default (no config)`, async () => {
    await runReasoningTest({
      model: 'google/gemini-3-pro-preview',
      reasoning: undefined,
      threadId: 'or-gemini-default-1',
      runId: 'or-gemini-default-1',
    });
  });

  test(`${capitalizeFirstLetter(provider)}: Gemini reasoning with max_tokens`, async () => {
    await runReasoningTest({
      model: 'google/gemini-3-pro-preview',
      reasoning: { max_tokens: 4000 },
      threadId: 'or-gemini-reasoning-1',
      runId: 'or-gemini-reasoning-1',
    });
  });

  test(`${capitalizeFirstLetter(provider)}: Gemini reasoning with effort`, async () => {
    await runReasoningTest({
      model: 'google/gemini-3-flash-preview',
      reasoning: { effort: 'low' },
      threadId: 'or-gemini-effort-1',
      runId: 'or-gemini-effort-1',
    });
  });

  test(`${capitalizeFirstLetter(provider)}: Anthropic reasoning with max_tokens`, async () => {
    await runReasoningTest({
      model: 'anthropic/claude-sonnet-4',
      reasoning: { max_tokens: 4000 },
      threadId: 'or-anthropic-reasoning-1',
      runId: 'or-anthropic-reasoning-1',
    });
  });

  test(`${capitalizeFirstLetter(provider)}: Anthropic sonnet-4 reasoning with effort`, async () => {
    await runReasoningTest({
      model: 'anthropic/claude-sonnet-4',
      reasoning: { effort: 'medium' },
      threadId: 'or-anthropic-effort-s4-1',
      runId: 'or-anthropic-effort-s4-1',
    });
  });

  test(`${capitalizeFirstLetter(provider)}: Anthropic sonnet-4-6 reasoning with effort`, async () => {
    await runReasoningTest({
      model: 'anthropic/claude-sonnet-4-6',
      reasoning: { effort: 'medium' },
      threadId: 'or-anthropic-effort-s46-1',
      runId: 'or-anthropic-effort-s46-1',
    });
  });
});
