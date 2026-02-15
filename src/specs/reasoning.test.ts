/* eslint-disable no-console */

// src/scripts/cli.test.ts
import { config } from 'dotenv';
config();
import {
  HumanMessage,
  BaseMessage,
  MessageContentText,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { capitalizeFirstLetter } from './spec.utils';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

const reasoningText = `<think>
Okay, the user is Jo from New York. I should start by greeting them by name. Let's keep it friendly and open-ended. Maybe mention the weather in New York to make it personal. Then offer help with something specific like plans or questions. Need to keep it concise and welcoming. Check for any typos. Alright, that should work.
</think>
Hi Jo! 🌆 How's everything in New York today? Whether you need recommendations for the city, help with a task, or just want to chat, I'm here for it. What's on your mind? 😊`;

const provider = 'Reasoning LLM';
describe(`${capitalizeFirstLetter(provider)} Streaming Tests`, () => {
  jest.setTimeout(30000);
  let run: Run<t.IState>;
  let contentParts: t.MessageContentComplex[];
  let conversationHistory: BaseMessage[];
  let aggregateContent: t.ContentAggregator;
  let runSteps: Set<string>;

  const config: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    run_id?: string;
    streamMode: string;
  } = {
    configurable: {
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
    callbacks: [
      {
        async handleCustomEvent(event, data, metadata): Promise<void> {
          if (event !== GraphEvents.ON_MESSAGE_DELTA) {
            return;
          }
          const messageDeltaData = data as t.MessageDeltaEvent;

          // Wait until we see the run step (with timeout for safety)
          const maxAttempts = 50; // 5 seconds total
          let attempts = 0;
          while (!runSteps.has(messageDeltaData.id) && attempts < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, 100));
            attempts++;
          }

          if (!runSteps.has(messageDeltaData.id)) {
            console.warn(
              `Timeout waiting for run step: ${messageDeltaData.id}`
            );
          }

          onMessageDeltaSpy(event, data, metadata, run.Graph);
          aggregateContent({ event, data: messageDeltaData });
        },
      },
    ],
  };

  beforeEach(async () => {
    conversationHistory = [];
    const { contentParts: parts, aggregateContent: ac } =
      createContentAggregator();
    aggregateContent = ac;
    runSteps = new Set();
    contentParts = parts as t.MessageContentComplex[];
  });

  afterEach(() => {
    runSteps.clear();
  });

  const onReasoningDeltaSpy = jest.fn();
  const onMessageDeltaSpy = jest.fn();
  const onRunStepSpy = jest.fn();

  afterAll(() => {
    onReasoningDeltaSpy.mockReset();
    onMessageDeltaSpy.mockReset();
    onRunStepSpy.mockReset();
  });

  const setupCustomHandlers = (): Record<
    string | GraphEvents,
    t.EventHandler
  > => ({
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
        const runStepData = data as t.RunStep;
        runSteps.add(runStepData.id);

        onRunStepSpy(event, runStepData, metadata, graph);
        aggregateContent({ event, data: runStepData });
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
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData,
        metadata,
        graph
      ): void => {
        onReasoningDeltaSpy(event, data, metadata, graph);
        aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
      },
    },
  });

  test(`${capitalizeFirstLetter(provider)}: should process a simple reasoning message`, async () => {
    const { userName, location } = await getArgs();
    const llmConfig = getLLMConfig(Providers.OPENAI);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'test-run-id',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions:
          'You are a friendly AI assistant. Always address the user by their name.',
        additional_instructions: `The user's name is ${userName} and they are located in ${location}.`,
      },
      returnContent: true,
      customHandlers,
    });

    run.Graph?.overrideTestModel([reasoningText], 2);

    const userMessage = 'hi';
    conversationHistory.push(new HumanMessage(userMessage));

    const inputs = {
      messages: conversationHistory,
    };

    await run.processStream(inputs, config);
    expect(contentParts).toBeDefined();
    expect(contentParts.length).toBe(2);
    // Tags are stripped from thinking content by the state machine
    const reasoningContent = reasoningText.match(/<think>(.*)<\/think>/s)?.[1];
    const content = reasoningText.split(/<\/think>/)[1];
    expect((contentParts[0] as t.ReasoningContentText).think).toBe(
      reasoningContent
    );
    expect((contentParts[1] as MessageContentText).text).toBe(content);

    const finalMessages = run.getRunMessages();
    expect(finalMessages).toBeDefined();
    conversationHistory.push(...(finalMessages ?? []));
    expect(conversationHistory.length).toBeGreaterThan(1);

    expect(onMessageDeltaSpy).toHaveBeenCalled();
    expect(onMessageDeltaSpy.mock.calls.length).toBeGreaterThan(1);
    expect(onMessageDeltaSpy.mock.calls[0][3]).toBeDefined(); // Graph exists

    expect(onReasoningDeltaSpy).toHaveBeenCalled();
    expect(onReasoningDeltaSpy.mock.calls.length).toBeGreaterThan(1);
    expect(onReasoningDeltaSpy.mock.calls[0][3]).toBeDefined(); // Graph exists

    expect(onRunStepSpy).toHaveBeenCalled();
    expect(onRunStepSpy.mock.calls.length).toBeGreaterThan(0);
    expect(onRunStepSpy.mock.calls[0][3]).toBeDefined(); // Graph exists
  });
});
