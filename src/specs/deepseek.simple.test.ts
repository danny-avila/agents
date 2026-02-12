/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { config } from 'dotenv';
config();
import { Calculator } from '@/tools/Calculator';
import {
  HumanMessage,
  BaseMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { ContentTypes, GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { capitalizeFirstLetter } from './spec.utils';
import { getLLMConfig } from '@/utils/llmConfig';
import { Run } from '@/run';

const provider = Providers.DEEPSEEK;
const llmConfig = getLLMConfig(provider);

const skipTests = process.env.DEEPSEEK_API_KEY == null;

(skipTests ? describe.skip : describe)(
  `${capitalizeFirstLetter(provider)} Streaming Tests`,
  () => {
    jest.setTimeout(120000);
    let run: Run<t.IState>;
    let collectedUsage: UsageMetadata[];
    let conversationHistory: BaseMessage[];
    let aggregateContent: t.ContentAggregator;
    let _contentParts: t.MessageContentComplex[];

    const testConfig = {
      configurable: {
        thread_id: 'deepseek-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    beforeEach(async () => {
      conversationHistory = [];
      collectedUsage = [];
      const { contentParts: cp, aggregateContent: ac } =
        createContentAggregator();
      _contentParts = cp as t.MessageContentComplex[];
      aggregateContent = ac;
    });

    const onMessageDeltaSpy = jest.fn();
    const onReasoningDeltaSpy = jest.fn();
    const onRunStepSpy = jest.fn();

    afterAll(() => {
      onMessageDeltaSpy.mockReset();
      onReasoningDeltaSpy.mockReset();
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
      [GraphEvents.ON_REASONING_DELTA]: {
        handle: (
          event: GraphEvents.ON_REASONING_DELTA,
          data: t.StreamEventData
        ): void => {
          onReasoningDeltaSpy(event, data);
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

    test(`${capitalizeFirstLetter(provider)}: should handle tool calls with reasoning_content preservation (streaming)`, async () => {
      const customHandlers = setupCustomHandlers();

      run = await Run.create<t.IState>({
        runId: 'deepseek-tool-test',
        graphConfig: {
          type: 'standard',
          llmConfig,
          tools: [new Calculator()],
          instructions:
            'You are a helpful math assistant. Use the calculator tool to solve math problems.',
        },
        returnContent: true,
        customHandlers,
      });

      const userMessage = 'What is 127 * 453?';
      conversationHistory.push(new HumanMessage(userMessage));

      const inputs = {
        messages: conversationHistory,
      };

      console.log('Starting DeepSeek streaming tool call test...');
      const finalContentParts = await run.processStream(inputs, testConfig);

      expect(finalContentParts).toBeDefined();
      console.log('Final content parts:', finalContentParts);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      expect(finalMessages?.length).toBeGreaterThan(0);

      const hasToolCall = finalMessages?.some(
        (msg) =>
          msg.getType() === 'ai' &&
          Array.isArray((msg as any).tool_calls) &&
          (msg as any).tool_calls.length > 0
      );
      expect(hasToolCall).toBe(true);

      const hasToolResult = finalMessages?.some(
        (msg) => msg.getType() === 'tool'
      );
      expect(hasToolResult).toBe(true);

      console.log(
        'Streaming tool call test passed - reasoning_content was preserved'
      );
      console.log(
        'Final response:',
        finalMessages?.[finalMessages.length - 1]?.content
      );
    });

    test(`${capitalizeFirstLetter(provider)}: should handle tool calls with disableStreaming`, async () => {
      const customHandlers = setupCustomHandlers();

      const nonStreamingLlmConfig: t.LLMConfig = {
        ...llmConfig,
        disableStreaming: true,
      };

      run = await Run.create<t.IState>({
        runId: 'deepseek-non-streaming-tool-test',
        graphConfig: {
          type: 'standard',
          llmConfig: nonStreamingLlmConfig,
          tools: [new Calculator()],
          instructions:
            'You are a helpful math assistant. Use the calculator tool to solve math problems.',
        },
        returnContent: true,
        customHandlers,
      });

      const userMessage = 'What is 99 * 77?';
      conversationHistory.push(new HumanMessage(userMessage));

      const inputs = {
        messages: conversationHistory,
      };

      console.log('Starting DeepSeek non-streaming tool call test...');
      const finalContentParts = await run.processStream(inputs, testConfig);

      expect(finalContentParts).toBeDefined();
      console.log('Final content parts (non-streaming):', finalContentParts);

      const finalMessages = run.getRunMessages();
      expect(finalMessages).toBeDefined();
      expect(finalMessages?.length).toBeGreaterThan(0);

      const hasToolCall = finalMessages?.some(
        (msg) =>
          msg.getType() === 'ai' &&
          Array.isArray((msg as any).tool_calls) &&
          (msg as any).tool_calls.length > 0
      );
      expect(hasToolCall).toBe(true);

      const hasToolResult = finalMessages?.some(
        (msg) => msg.getType() === 'tool'
      );
      expect(hasToolResult).toBe(true);

      console.log('Non-streaming tool call test passed');
      console.log(
        'Final response:',
        finalMessages?.[finalMessages.length - 1]?.content
      );
    });

    test(`${capitalizeFirstLetter(provider)}: should process simple message without tools`, async () => {
      const customHandlers = setupCustomHandlers();

      run = await Run.create<t.IState>({
        runId: 'deepseek-simple-test',
        graphConfig: {
          type: 'standard',
          llmConfig,
          tools: [],
          instructions: 'You are a friendly AI assistant.',
        },
        returnContent: true,
        customHandlers,
      });

      const userMessage = 'Hello! How are you today?';
      conversationHistory.push(new HumanMessage(userMessage));

      const inputs = {
        messages: conversationHistory,
      };

      const finalContentParts = await run.processStream(inputs, testConfig);
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
      console.log(
        `${capitalizeFirstLetter(provider)} response:`,
        finalMessages?.[finalMessages.length - 1]?.content
      );
    });
  }
);
