// src/specs/fragmented-thinking.test.ts
// Tests for fragmented <thinking> tag handling in streamed content
// This tests the state machine that detects thinking tags split across chunks

import { HumanMessage, MessageContentText } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { Run } from '@/run';

describe('Fragmented Thinking Tags Tests', () => {
  jest.setTimeout(30000);
  let run: Run<t.IState>;
  let contentParts: t.MessageContentComplex[];
  let aggregateContent: t.ContentAggregator;
  let runSteps: Set<string>;

  const config: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    run_id?: string;
    streamMode: string;
  } = {
    configurable: {
      thread_id: 'fragmented-thinking-test',
    },
    streamMode: 'values',
    version: 'v2' as const,
    callbacks: [
      {
        async handleCustomEvent(event, data): Promise<void> {
          if (event !== GraphEvents.ON_MESSAGE_DELTA) {
            return;
          }
          const messageDeltaData = data as t.MessageDeltaEvent;

          // Wait until we see the run step
          const maxAttempts = 50;
          let attempts = 0;
          while (!runSteps.has(messageDeltaData.id) && attempts < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, 100));
            attempts++;
          }

          aggregateContent({ event, data: messageDeltaData });
        },
      },
    ],
  };

  beforeEach(async () => {
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

  afterAll(() => {
    onReasoningDeltaSpy.mockReset();
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
        data: t.StreamEventData
      ): void => {
        const runStepData = data as t.RunStep;
        runSteps.add(runStepData.id);
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
        data: t.StreamEventData
      ): void => {
        onReasoningDeltaSpy(event, data);
        aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
      },
    },
  });

  // Test with a complete thinking block that will be split by whitespace
  // The fake model splits on whitespace by default, so this simulates
  // receiving "<thinking>" and "</thinking>" as separate chunks
  const thinkingResponse =
    '<thinking> Let me think about this. </thinking> The answer is 42.';

  test('should handle thinking tags in streamed content', async () => {
    const llmConfig = getLLMConfig(Providers.BEDROCK);
    const customHandlers = setupCustomHandlers();

    run = await Run.create<t.IState>({
      runId: 'fragmented-thinking-test-run',
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
      },
      returnContent: true,
      customHandlers,
    });

    // Pass the full response - the fake model will split it on whitespace
    run.Graph?.overrideTestModel([thinkingResponse], 2);

    const inputs = {
      messages: [new HumanMessage('What is the meaning of life?')],
    };

    await run.processStream(inputs, config);

    expect(contentParts).toBeDefined();
    expect(contentParts.length).toBe(2);

    // Find the thinking and text parts (order may vary based on event handling)
    const thinkingPart = contentParts.find(
      (p) => (p as t.ReasoningContentText).think !== undefined
    ) as t.ReasoningContentText;
    const textPart = contentParts.find(
      (p) => (p as MessageContentText).text !== undefined
    ) as MessageContentText;

    // Thinking content should not contain the tags
    expect(thinkingPart).toBeDefined();
    expect(thinkingPart.think).toContain('Let me think about this.');
    expect(thinkingPart.think).not.toContain('<thinking>');
    expect(thinkingPart.think).not.toContain('</thinking>');

    // Text content should be the response after thinking
    expect(textPart).toBeDefined();
    expect(textPart.text).toContain('The answer is 42.');
    expect(textPart.text).not.toContain('<thinking>');

    // Verify reasoning delta was called
    expect(onReasoningDeltaSpy).toHaveBeenCalled();
  });
});
