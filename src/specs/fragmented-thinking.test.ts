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

  // Helper to create a fresh run for each test
  const createTestRun = async (
    customHandlers: Record<string | GraphEvents, t.EventHandler>
  ): Promise<Run<t.IState>> => {
    const llmConfig = getLLMConfig(Providers.BEDROCK);
    return Run.create<t.IState>({
      runId: `fragmented-thinking-test-run-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        llmConfig,
        instructions: 'You are a helpful assistant.',
      },
      returnContent: true,
      customHandlers,
    });
  };

  // Test with <thinking> tags
  test('should handle <thinking> tags in streamed content', async () => {
    const customHandlers = setupCustomHandlers();
    run = await createTestRun(customHandlers);

    const responseWithThinkingTag =
      '<thinking> Let me think about this. </thinking> The answer is 42.';
    run.Graph?.overrideTestModel([responseWithThinkingTag], 2);

    const inputs = {
      messages: [new HumanMessage('What is the meaning of life?')],
    };

    await run.processStream(inputs, config);

    expect(contentParts).toBeDefined();
    expect(contentParts.length).toBe(2);

    const thinkingPart = contentParts.find(
      (p) => (p as t.ReasoningContentText).think !== undefined
    ) as t.ReasoningContentText;
    const textPart = contentParts.find(
      (p) => (p as MessageContentText).text !== undefined
    ) as MessageContentText;

    expect(thinkingPart).toBeDefined();
    expect(thinkingPart.think).toContain('Let me think about this.');
    expect(thinkingPart.think).not.toContain('<thinking>');
    expect(thinkingPart.think).not.toContain('</thinking>');

    expect(textPart).toBeDefined();
    expect(textPart.text).toContain('The answer is 42.');
    expect(textPart.text).not.toContain('<thinking>');

    expect(onReasoningDeltaSpy).toHaveBeenCalled();
  });

  // Test with <think> tags (shorter variant)
  test('should handle <think> tags in streamed content', async () => {
    onReasoningDeltaSpy.mockClear();
    const customHandlers = setupCustomHandlers();
    run = await createTestRun(customHandlers);

    const responseWithThinkTag =
      '<think> Processing the question... </think> Here is my response.';
    run.Graph?.overrideTestModel([responseWithThinkTag], 2);

    const inputs = {
      messages: [new HumanMessage('Tell me something.')],
    };

    await run.processStream(inputs, config);

    expect(contentParts).toBeDefined();
    expect(contentParts.length).toBe(2);

    const thinkingPart = contentParts.find(
      (p) => (p as t.ReasoningContentText).think !== undefined
    ) as t.ReasoningContentText;
    const textPart = contentParts.find(
      (p) => (p as MessageContentText).text !== undefined
    ) as MessageContentText;

    expect(thinkingPart).toBeDefined();
    expect(thinkingPart.think).toContain('Processing the question...');
    expect(thinkingPart.think).not.toContain('<think>');
    expect(thinkingPart.think).not.toContain('</think>');

    expect(textPart).toBeDefined();
    expect(textPart.text).toContain('Here is my response.');
    expect(textPart.text).not.toContain('<think>');

    expect(onReasoningDeltaSpy).toHaveBeenCalled();
  });

  // Test with plain text (no thinking tags)
  test('should handle plain text without thinking tags', async () => {
    onReasoningDeltaSpy.mockClear();
    const customHandlers = setupCustomHandlers();
    run = await createTestRun(customHandlers);

    const responseWithoutTags =
      'This is a simple response without any thinking.';
    run.Graph?.overrideTestModel([responseWithoutTags], 2);

    const inputs = {
      messages: [new HumanMessage('Say something simple.')],
    };

    await run.processStream(inputs, config);

    expect(contentParts).toBeDefined();
    expect(contentParts.length).toBe(1);

    const textPart = contentParts[0] as MessageContentText;
    expect(textPart.text).toBe(
      'This is a simple response without any thinking.'
    );

    // No reasoning delta should be called for plain text
    expect(onReasoningDeltaSpy).not.toHaveBeenCalled();
  });

  // Test with multiple thinking blocks in sequence
  test('should handle multiple thinking blocks in sequence', async () => {
    onReasoningDeltaSpy.mockClear();
    const customHandlers = setupCustomHandlers();
    run = await createTestRun(customHandlers);

    const responseWithMultipleThinkingTags =
      '<thinking> First thought. </thinking> Response one. <thinking> Second thought. </thinking> Response two.';
    run.Graph?.overrideTestModel([responseWithMultipleThinkingTags], 2);

    const inputs = {
      messages: [new HumanMessage('Give me a complex response.')],
    };

    await run.processStream(inputs, config);

    expect(contentParts).toBeDefined();
    // Should have thinking and text parts (exact count depends on aggregation)
    expect(contentParts.length).toBeGreaterThanOrEqual(2);

    const thinkingPart = contentParts.find(
      (p) => (p as t.ReasoningContentText).think !== undefined
    ) as t.ReasoningContentText;
    const textPart = contentParts.find(
      (p) => (p as MessageContentText).text !== undefined
    ) as MessageContentText;

    // Verify thinking content contains both thoughts (accumulated)
    expect(thinkingPart).toBeDefined();
    expect(thinkingPart.think).toContain('First thought.');
    expect(thinkingPart.think).toContain('Second thought.');
    expect(thinkingPart.think).not.toContain('<thinking>');
    expect(thinkingPart.think).not.toContain('</thinking>');

    // Verify text content contains both responses
    expect(textPart).toBeDefined();
    expect(textPart.text).toContain('Response one.');
    expect(textPart.text).toContain('Response two.');
    expect(textPart.text).not.toContain('<thinking>');

    expect(onReasoningDeltaSpy).toHaveBeenCalled();
  });
});
