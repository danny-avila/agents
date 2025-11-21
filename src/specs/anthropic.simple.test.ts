/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-explicit-any */
// src/scripts/cli.test.ts
import { config } from 'dotenv';
config();
import { Calculator } from '@/tools/Calculator';
import {
  HumanMessage,
  BaseMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import {
  ToolEndHandler,
  ModelEndHandler,
  createMetadataAggregator,
} from '@/events';
import { ContentTypes, GraphEvents, Providers, TitleMethod } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { capitalizeFirstLetter } from './spec.utils';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

const provider = Providers.ANTHROPIC;
describe(`${capitalizeFirstLetter(provider)} Streaming Tests`, () => {
  jest.setTimeout(60000);
  let run: Run<t.IState>;
  let runningHistory: BaseMessage[];
  let collectedUsage: UsageMetadata[];
  let conversationHistory: BaseMessage[];
  let aggregateContent: t.ContentAggregator;
  let contentParts: t.MessageContentComplex[];

  const config = {
    configurable: {
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  beforeEach(async () => {
    conversationHistory = [];
    collectedUsage = [];
    const { contentParts: cp, aggregateContent: ac } =
      createContentAggregator();
    contentParts = cp as t.MessageContentComplex[];
    aggregateContent = ac;
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
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        onRunStepSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        onMessageDeltaSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        _metadata?: Record<string, unknown>
      ): void => {
        // Handle tool start
      },
    },
  });

  test(`${capitalizeFirstLetter(provider)}: should process a simple message, generate title`, async () => {
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-run-id',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()],
        instructions:
          'You are a friendly AI assistant. Always address the user by their name.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      customHandlers,
    });

    const userMessage = 'hi';
    conversationHistory.push(new HumanMessage(userMessage));

    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    expect(finalContentParts).toBeDefined();
    const allTextParts = finalContentParts?.every(
      (part) => part.type === ContentTypes.TEXT
    );
    expect(allTextParts).toBe(true);
    expect(collectedUsage.length).toBeGreaterThan(0);
    expect(collectedUsage[0].input_tokens).toBeGreaterThan(0);
    expect(collectedUsage[0].output_tokens).toBeGreaterThan(0);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    conversationHistory.push(...(finalMessages ?? []));
    expect(conversationHistory.length).toBeGreaterThan(1);
    runningHistory = conversationHistory.slice();

    expect(onMessageDeltaSpy).toHaveBeenCalled();
    expect(onMessageDeltaSpy.mock.calls.length).toBeGreaterThan(1);
    expect(onMessageDeltaSpy.mock.calls[0][3]).toBeDefined(); // Graph exists

    expect(onRunStepSpy).toHaveBeenCalled();
    expect(onRunStepSpy.mock.calls.length).toBeGreaterThan(0);
    expect(onRunStepSpy.mock.calls[0][3]).toBeDefined(); // Graph exists

    const { handleLLMEnd, collected } = createMetadataAggregator();
    const titleResult = await run.generateTitle({
      provider,
      inputText: userMessage,
      titleMethod: TitleMethod.STRUCTURED,
      contentParts,
      clientOptions: {
        ...llmConfig,
        model: 'claude-3-5-haiku-latest',
      },
      chainOptions: {
        callbacks: [
          {
            handleLLMEnd,
          },
        ],
      },
    });

    expect(titleResult).toBeDefined();
    expect(titleResult.title).toBeDefined();
    expect(titleResult.language).toBeDefined();
    expect(collected).toBeDefined();
  });

  test(`${capitalizeFirstLetter(provider)}: should generate title using completion method`, async () => {
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-run-id-completion',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()],
        instructions:
          'You are a friendly AI assistant. Always address the user by their name.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      customHandlers,
    });

    const userMessage =
      'Can you help me calculate the area of a circle with radius 5?';
    conversationHistory = [];
    conversationHistory.push(new HumanMessage(userMessage));

    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    expect(finalContentParts).toBeDefined();

    const { handleLLMEnd, collected } = createMetadataAggregator();
    const titleResult = await run.generateTitle({
      provider,
      inputText: userMessage,
      titleMethod: TitleMethod.COMPLETION, // Using completion method
      contentParts,
      clientOptions: {
        ...llmConfig,
        model: 'claude-3-5-haiku-latest',
      },
      chainOptions: {
        callbacks: [
          {
            handleLLMEnd,
          },
        ],
      },
    });

    expect(titleResult).toBeDefined();
    expect(titleResult.title).toBeDefined();
    expect(titleResult.title).not.toBe('');
    // Completion method doesn't return language
    expect(titleResult.language).toBeUndefined();
    expect(collected).toBeDefined();
    console.log(`Completion method generated title: "${titleResult.title}"`);
  });

  test(`${capitalizeFirstLetter(provider)}: should follow-up`, async () => {
    console.log('Previous conversation length:', runningHistory.length);
    console.log(
      'Last message:',
      runningHistory[runningHistory.length - 1].content
    );
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-run-id',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()],
        instructions:
          'You are a friendly AI assistant. Always address the user by their name.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      customHandlers,
    });

    conversationHistory = runningHistory.slice();
    conversationHistory.push(new HumanMessage('how are you?'));

    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    expect(finalContentParts).toBeDefined();
    const allTextParts = finalContentParts?.every(
      (part) => part.type === ContentTypes.TEXT
    );
    expect(allTextParts).toBe(true);
    expect(collectedUsage.length).toBeGreaterThan(0);
    expect(collectedUsage[0].input_tokens).toBeGreaterThan(0);
    expect(collectedUsage[0].output_tokens).toBeGreaterThan(0);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    expect(finalMessages?.length).toBeGreaterThan(0);
    console.log(
      `${capitalizeFirstLetter(provider)} follow-up message:`,
      finalMessages?.[finalMessages.length - 1]?.content
    );

    expect(onMessageDeltaSpy).toHaveBeenCalled();
    expect(onMessageDeltaSpy.mock.calls.length).toBeGreaterThan(1);

    expect(onRunStepSpy).toHaveBeenCalled();
    expect(onRunStepSpy.mock.calls.length).toBeGreaterThan(0);
  });

  test(`${capitalizeFirstLetter(provider)}: should handle parallel tool usage (web search + calculator)`, async () => {
    const llmConfig = getLLMConfig(provider);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-parallel-tools',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [
          {
            type: 'web_search_20250305',
            name: 'web_search',
            max_uses: 5,
          },
          new Calculator(),
        ],
        instructions: 'You are a helpful AI assistant.',
      },
      returnContent: true,
      customHandlers,
    });

    // Use the same query as the edge case script to test actual parallel tool usage
    const userMessage =
      'Can you search the web for the current population of Tokyo, and also calculate what 15% of that population would be? Do both at the same time.';
    conversationHistory = [];
    conversationHistory.push(new HumanMessage(userMessage));

    const inputs = {
      messages: conversationHistory,
    };

    // This should complete without errors despite using both server tools and regular tools in parallel
    const finalContentParts = await run.processStream(inputs, config);
    expect(finalContentParts).toBeDefined();

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    expect(finalMessages?.length).toBeGreaterThan(0);

    const hasWebSearch = contentParts.some(
      (part) =>
        !!(
          part.type === 'tool_call' &&
          part.tool_call?.name === 'web_search' &&
          part.tool_call?.id?.startsWith('srvtoolu_') === true
        )
    );
    const hasCalculator = contentParts.some(
      (part) =>
        !!(
          part.type === 'tool_call' &&
          part.tool_call?.name === 'calculator' &&
          part.tool_call?.id?.startsWith('toolu_') === true
        )
    );

    // Both tools should have been used for this query
    expect(hasWebSearch).toBe(true);
    expect(hasCalculator).toBe(true);

    console.log(
      `${capitalizeFirstLetter(provider)} parallel tools test: web_search (server tool) + calculator (regular tool) both used successfully`
    );
  });

  test('should handle errors appropriately', async () => {
    // Test error scenarios
    await expect(async () => {
      await run.processStream(
        {
          messages: [],
        },
        {} as any
      );
    }).rejects.toThrow();
  });
});
