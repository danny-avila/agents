/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-explicit-any */
// src/specs/azure.simple.test.ts
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
import { capitalizeFirstLetter } from './spec.utils';
import { createContentAggregator } from '@/stream';
import { getLLMConfig } from '@/utils/llmConfig';
import { Run } from '@/run';

const requiredAzureEnv = [
  'AZURE_OPENAI_API_KEY',
  'AZURE_OPENAI_API_INSTANCE',
  'AZURE_OPENAI_API_DEPLOYMENT',
  'AZURE_OPENAI_API_VERSION',
];
const hasAzure = requiredAzureEnv.every(
  (k) => (process.env[k] ?? '').trim() !== ''
);
const describeIfAzure = hasAzure ? describe : describe.skip;

const isContentFilterError = (error: unknown): boolean => {
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes('content management policy') ||
    message.includes('content filtering')
  );
};

const provider = Providers.AZURE;
let contentFilterTriggered = false;
describeIfAzure(`${capitalizeFirstLetter(provider)} Streaming Tests`, () => {
  jest.setTimeout(30000);
  let run: Run<t.IState>;
  let collectedUsage: UsageMetadata[];
  let conversationHistory: BaseMessage[];
  let aggregateContent: t.ContentAggregator;
  let contentParts: t.MessageContentComplex[];
  let runningHistory: BaseMessage[] | null = null;

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
    try {
      const llmConfig = getLLMConfig(provider);
      const customHandlers = setupCustomHandlers();

      run = await Run.create<t.IState>({
        runId: 'test-run-id',
        graphConfig: {
          type: 'standard',
          llmConfig,
          tools: [new Calculator()],
          instructions:
            'You are a helpful AI assistant. Keep responses concise and friendly.',
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
      });

      const userMessage = 'Hello, how are you today?';
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
        clientOptions: llmConfig,
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
    } catch (error) {
      if (isContentFilterError(error)) {
        contentFilterTriggered = true;
        console.warn('Skipping test: Azure content filter triggered');
        return;
      }
      throw error;
    }
  });

  test(`${capitalizeFirstLetter(provider)}: should generate title using completion method`, async () => {
    if (contentFilterTriggered) {
      console.warn(
        'Skipping test: Azure content filter was triggered in previous test'
      );
      return;
    }
    try {
      const llmConfig = getLLMConfig(provider);
      const customHandlers = setupCustomHandlers();

      run = await Run.create<t.IState>({
        runId: 'test-run-id-completion',
        graphConfig: {
          type: 'standard',
          llmConfig,
          tools: [new Calculator()],
          instructions:
            'You are a helpful AI assistant. Keep responses concise and friendly.',
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
      });

      const userMessage = 'What can you help me with today?';
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
        titleMethod: TitleMethod.COMPLETION,
        contentParts,
        clientOptions: llmConfig,
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
      expect(titleResult.language).toBeUndefined();
      expect(collected).toBeDefined();
      console.log(`Completion method generated title: "${titleResult.title}"`);
    } catch (error) {
      if (isContentFilterError(error)) {
        contentFilterTriggered = true;
        console.warn('Skipping test: Azure content filter triggered');
        return;
      }
      throw error;
    }
  });

  test(`${capitalizeFirstLetter(provider)}: should follow-up`, async () => {
    if (contentFilterTriggered || runningHistory == null) {
      console.warn(
        'Skipping test: Azure content filter was triggered or no conversation history'
      );
      return;
    }
    try {
      console.log('Previous conversation length:', runningHistory.length);
      console.log(
        'Last message:',
        runningHistory[runningHistory.length - 1].content
      );
      const llmConfig = getLLMConfig(provider);
      const customHandlers = setupCustomHandlers();

      run = await Run.create<t.IState>({
        runId: 'test-run-id',
        graphConfig: {
          type: 'standard',
          llmConfig,
          tools: [new Calculator()],
          instructions:
            'You are a helpful AI assistant. Keep responses concise and friendly.',
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
      });

      conversationHistory = runningHistory.slice();
      conversationHistory.push(new HumanMessage('What else can you tell me?'));

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
    } catch (error) {
      if (isContentFilterError(error)) {
        console.warn('Skipping test: Azure content filter triggered');
        return;
      }
      throw error;
    }
  });

  test(`${capitalizeFirstLetter(provider)}: disableStreaming should not duplicate message content`, async () => {
    if (contentFilterTriggered) {
      console.warn(
        'Skipping test: Azure content filter was triggered in previous test'
      );
      return;
    }
    try {
      const llmConfig = getLLMConfig(provider);
      const nonStreamingConfig: t.LLMConfig = {
        ...llmConfig,
        disableStreaming: true,
      };

      const messageDeltaPayloads: t.MessageDeltaEvent[] = [];
      const localRunStepSpy = jest.fn();
      const localAggregateContent = createContentAggregator();
      const localContentParts =
        localAggregateContent.contentParts as t.MessageContentComplex[];
      const localAggregate = localAggregateContent.aggregateContent;

      const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
        [GraphEvents.TOOL_END]: new ToolEndHandler(),
        [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
        [GraphEvents.ON_RUN_STEP_COMPLETED]: {
          handle: (
            event: GraphEvents.ON_RUN_STEP_COMPLETED,
            data: t.StreamEventData
          ): void => {
            localAggregate({
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
            localRunStepSpy(event, data, metadata, graph);
            localAggregate({ event, data: data as t.RunStep });
          },
        },
        [GraphEvents.ON_RUN_STEP_DELTA]: {
          handle: (
            event: GraphEvents.ON_RUN_STEP_DELTA,
            data: t.StreamEventData
          ): void => {
            localAggregate({ event, data: data as t.RunStepDeltaEvent });
          },
        },
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (
            event: GraphEvents.ON_MESSAGE_DELTA,
            data: t.StreamEventData
          ): void => {
            messageDeltaPayloads.push(data as t.MessageDeltaEvent);
            localAggregate({ event, data: data as t.MessageDeltaEvent });
          },
        },
      };

      run = await Run.create<t.IState>({
        runId: 'azure-disable-streaming-dedup-test',
        graphConfig: {
          type: 'standard',
          llmConfig: nonStreamingConfig,
          tools: [],
          instructions:
            'You are a helpful AI assistant. Respond with exactly one sentence.',
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers,
      });

      conversationHistory.push(new HumanMessage('Hello'));

      const finalContentParts = await run.processStream(
        { messages: conversationHistory },
        config
      );

      expect(finalContentParts).toBeDefined();
      expect(finalContentParts!.length).toBeGreaterThan(0);
      expect(messageDeltaPayloads.length).toBeGreaterThan(0);

      const allTextDeltas = messageDeltaPayloads
        .flatMap((p) => p.delta.content ?? [])
        .filter((c) => c.type === ContentTypes.TEXT)
        .map((c) => ('text' in c ? c.text : ''));

      const combinedText = allTextDeltas.join('');

      /**
       * When model.stream() is available (the common path even with
       * disableStreaming), ChatModelStreamHandler already dispatches the full
       * text as a single MESSAGE_DELTA. The disableStreaming fallback block in
       * createCallModel must NOT dispatch the same content a second time.
       *
       * If the bug is present, the text is emitted twice and localContentParts
       * will contain duplicated text.
       */
      const aggregatedText = localContentParts
        .filter((p) => p.type === ContentTypes.TEXT)
        .map((p) => ('text' in p ? p.text : ''))
        .join('');

      console.log('Message delta count:', messageDeltaPayloads.length);
      console.log('Combined delta text length:', combinedText.length);
      console.log('Aggregated text length:', aggregatedText.length);

      /**
       * Each delta payload contains the FULL text (non-streaming returns a
       * single chunk). If the bug is present, we get >=2 identical payloads
       * and the aggregated text will be 2x the actual response.
       */
      const uniqueTexts = [...new Set(allTextDeltas)];
      expect(uniqueTexts.length).toBe(1);
      expect(uniqueTexts[0].length).toBeGreaterThan(0);

      const singleResponseText = uniqueTexts[0];
      expect(aggregatedText).toBe(singleResponseText);
      expect(combinedText).toBe(singleResponseText);

      console.log('disableStreaming dedup test passed — no duplicate content');
    } catch (error) {
      if (isContentFilterError(error)) {
        console.warn('Skipping test: Azure content filter triggered');
        return;
      }
      throw error;
    }
  });

  test('should handle errors appropriately', async () => {
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
