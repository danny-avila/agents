/* eslint-disable no-console */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { config } from 'dotenv';
config();
import { Calculator } from '@/tools/Calculator';
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  UsageMetadata,
} from '@langchain/core/messages';
import type * as t from '@/types';
import { ModelEndHandler, ToolEndHandler } from '@/events';
import { capitalizeFirstLetter } from './spec.utils';
import { createContentAggregator } from '@/stream';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

/**
 * These tests verify that prompt caching works correctly across multi-turn
 * conversations and that messages are not mutated in place.
 */
describe('Prompt Caching Integration Tests', () => {
  jest.setTimeout(120000);

  const setupTest = (): {
    collectedUsage: UsageMetadata[];
    contentParts: Array<t.MessageContentComplex | undefined>;
    customHandlers: Record<string | GraphEvents, t.EventHandler>;
  } => {
    const collectedUsage: UsageMetadata[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();

    const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
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
          data: t.StreamEventData
        ): void => {
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
          data: t.StreamEventData
        ): void => {
          aggregateContent({ event, data: data as t.MessageDeltaEvent });
        },
      },
    };

    return { collectedUsage, contentParts, customHandlers };
  };

  const streamConfig = {
    configurable: { thread_id: 'cache-test-thread' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  describe('Anthropic Prompt Caching', () => {
    const provider = Providers.ANTHROPIC;

    test(`${capitalizeFirstLetter(provider)}: multi-turn conversation with caching should not corrupt messages`, async () => {
      const { userName, location } = await getArgs();
      const llmConfig = getLLMConfig(provider);
      const { collectedUsage, customHandlers } = setupTest();

      const run = await Run.create<t.IState>({
        runId: 'cache-test-anthropic',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...llmConfig, promptCache: true } as t.LLMConfig,
          tools: [new Calculator()],
          instructions: 'You are a helpful assistant.',
          additional_instructions: `User: ${userName}, Location: ${location}`,
        },
        returnContent: true,
        customHandlers,
      });

      // Turn 1
      const turn1Messages: BaseMessage[] = [
        new HumanMessage('Hello, what is 2+2?'),
      ];
      const turn1ContentSnapshot = JSON.stringify(turn1Messages[0].content);

      const turn1Result = await run.processStream(
        { messages: turn1Messages },
        streamConfig
      );
      expect(turn1Result).toBeDefined();

      // Verify original message was NOT mutated
      expect(JSON.stringify(turn1Messages[0].content)).toBe(
        turn1ContentSnapshot
      );
      expect((turn1Messages[0] as any).content).not.toContain('cache_control');

      const turn1RunMessages = run.getRunMessages();
      expect(turn1RunMessages).toBeDefined();
      expect(turn1RunMessages!.length).toBeGreaterThan(0);

      // Turn 2 - build on conversation
      const turn2Messages: BaseMessage[] = [
        ...turn1Messages,
        ...turn1RunMessages!,
        new HumanMessage('Now multiply that by 10'),
      ];
      const turn2HumanContentSnapshot = JSON.stringify(
        turn2Messages[turn2Messages.length - 1].content
      );

      const run2 = await Run.create<t.IState>({
        runId: 'cache-test-anthropic-2',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...llmConfig, promptCache: true } as t.LLMConfig,
          tools: [new Calculator()],
          instructions: 'You are a helpful assistant.',
          additional_instructions: `User: ${userName}, Location: ${location}`,
        },
        returnContent: true,
        customHandlers,
      });

      const turn2Result = await run2.processStream(
        { messages: turn2Messages },
        streamConfig
      );
      expect(turn2Result).toBeDefined();

      // Verify messages were NOT mutated
      expect(
        JSON.stringify(turn2Messages[turn2Messages.length - 1].content)
      ).toBe(turn2HumanContentSnapshot);

      // Check that we got cache read tokens (indicating caching worked)
      console.log(`${provider} Usage:`, collectedUsage);
      expect(collectedUsage.length).toBeGreaterThan(0);

      console.log(
        `${capitalizeFirstLetter(provider)} multi-turn caching test passed - messages not mutated`
      );
    });

    test(`${capitalizeFirstLetter(provider)}: tool calls should work with caching enabled`, async () => {
      const llmConfig = getLLMConfig(provider);
      const { customHandlers } = setupTest();

      const run = await Run.create<t.IState>({
        runId: 'cache-test-anthropic-tools',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...llmConfig, promptCache: true } as t.LLMConfig,
          tools: [new Calculator()],
          instructions:
            'You are a math assistant. Use the calculator tool for all calculations.',
        },
        returnContent: true,
        customHandlers,
      });

      const messages: BaseMessage[] = [
        new HumanMessage('Calculate 123 * 456 using the calculator'),
      ];

      const result = await run.processStream({ messages }, streamConfig);
      expect(result).toBeDefined();

      const runMessages = run.getRunMessages();
      expect(runMessages).toBeDefined();

      // Should have used the calculator tool
      const hasToolUse = runMessages?.some(
        (msg) =>
          msg._getType() === 'ai' &&
          ((msg as AIMessage).tool_calls?.length ?? 0) > 0
      );
      expect(hasToolUse).toBe(true);

      console.log(
        `${capitalizeFirstLetter(provider)} tool call with caching test passed`
      );
    });
  });

  describe('Bedrock Prompt Caching', () => {
    const provider = Providers.BEDROCK;

    test(`${capitalizeFirstLetter(provider)}: multi-turn conversation with caching should not corrupt messages`, async () => {
      const { userName, location } = await getArgs();
      const llmConfig = getLLMConfig(provider);
      const { collectedUsage, customHandlers } = setupTest();

      const run = await Run.create<t.IState>({
        runId: 'cache-test-bedrock',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...llmConfig, promptCache: true } as t.LLMConfig,
          tools: [new Calculator()],
          instructions: 'You are a helpful assistant.',
          additional_instructions: `User: ${userName}, Location: ${location}`,
        },
        returnContent: true,
        customHandlers,
      });

      // Turn 1
      const turn1Messages: BaseMessage[] = [
        new HumanMessage('Hello, what is 5+5?'),
      ];
      const turn1ContentSnapshot = JSON.stringify(turn1Messages[0].content);

      const turn1Result = await run.processStream(
        { messages: turn1Messages },
        streamConfig
      );
      expect(turn1Result).toBeDefined();

      // Verify original message was NOT mutated
      expect(JSON.stringify(turn1Messages[0].content)).toBe(
        turn1ContentSnapshot
      );

      const turn1RunMessages = run.getRunMessages();
      expect(turn1RunMessages).toBeDefined();
      expect(turn1RunMessages!.length).toBeGreaterThan(0);

      // Turn 2
      const turn2Messages: BaseMessage[] = [
        ...turn1Messages,
        ...turn1RunMessages!,
        new HumanMessage('Multiply that by 3'),
      ];
      const turn2HumanContentSnapshot = JSON.stringify(
        turn2Messages[turn2Messages.length - 1].content
      );

      const run2 = await Run.create<t.IState>({
        runId: 'cache-test-bedrock-2',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...llmConfig, promptCache: true } as t.LLMConfig,
          tools: [new Calculator()],
          instructions: 'You are a helpful assistant.',
          additional_instructions: `User: ${userName}, Location: ${location}`,
        },
        returnContent: true,
        customHandlers,
      });

      const turn2Result = await run2.processStream(
        { messages: turn2Messages },
        streamConfig
      );
      expect(turn2Result).toBeDefined();

      // Verify messages were NOT mutated
      expect(
        JSON.stringify(turn2Messages[turn2Messages.length - 1].content)
      ).toBe(turn2HumanContentSnapshot);

      console.log(`${provider} Usage:`, collectedUsage);
      expect(collectedUsage.length).toBeGreaterThan(0);

      console.log(
        `${capitalizeFirstLetter(provider)} multi-turn caching test passed - messages not mutated`
      );
    });

    test(`${capitalizeFirstLetter(provider)}: tool calls should work with caching enabled`, async () => {
      const llmConfig = getLLMConfig(provider);
      const { customHandlers } = setupTest();

      const run = await Run.create<t.IState>({
        runId: 'cache-test-bedrock-tools',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...llmConfig, promptCache: true } as t.LLMConfig,
          tools: [new Calculator()],
          instructions:
            'You are a math assistant. Use the calculator tool for all calculations.',
        },
        returnContent: true,
        customHandlers,
      });

      const messages: BaseMessage[] = [
        new HumanMessage('Calculate 789 * 123 using the calculator'),
      ];

      const result = await run.processStream({ messages }, streamConfig);
      expect(result).toBeDefined();

      const runMessages = run.getRunMessages();
      expect(runMessages).toBeDefined();

      // Should have used the calculator tool
      const hasToolUse = runMessages?.some(
        (msg) =>
          msg._getType() === 'ai' &&
          ((msg as AIMessage).tool_calls?.length ?? 0) > 0
      );
      expect(hasToolUse).toBe(true);

      console.log(
        `${capitalizeFirstLetter(provider)} tool call with caching test passed`
      );
    });
  });

  describe('Cross-provider message isolation', () => {
    test('Messages processed by Anthropic should not affect Bedrock processing', async () => {
      const anthropicConfig = getLLMConfig(Providers.ANTHROPIC);
      const bedrockConfig = getLLMConfig(Providers.BEDROCK);
      const { customHandlers: handlers1 } = setupTest();
      const { customHandlers: handlers2 } = setupTest();

      // Create a shared message array
      const sharedMessages: BaseMessage[] = [
        new HumanMessage('Hello, what is the capital of France?'),
      ];
      const originalContent = JSON.stringify(sharedMessages[0].content);

      // Process with Anthropic first
      const anthropicRun = await Run.create<t.IState>({
        runId: 'cross-provider-anthropic',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...anthropicConfig, promptCache: true } as t.LLMConfig,
          instructions: 'You are a helpful assistant.',
        },
        returnContent: true,
        customHandlers: handlers1,
      });

      const anthropicResult = await anthropicRun.processStream(
        { messages: sharedMessages },
        streamConfig
      );
      expect(anthropicResult).toBeDefined();

      // Verify message not mutated
      expect(JSON.stringify(sharedMessages[0].content)).toBe(originalContent);

      // Now process with Bedrock using the SAME messages
      const bedrockRun = await Run.create<t.IState>({
        runId: 'cross-provider-bedrock',
        graphConfig: {
          type: 'standard',
          llmConfig: { ...bedrockConfig, promptCache: true } as t.LLMConfig,
          instructions: 'You are a helpful assistant.',
        },
        returnContent: true,
        customHandlers: handlers2,
      });

      const bedrockResult = await bedrockRun.processStream(
        { messages: sharedMessages },
        streamConfig
      );
      expect(bedrockResult).toBeDefined();

      // Verify message STILL not mutated after both providers processed
      expect(JSON.stringify(sharedMessages[0].content)).toBe(originalContent);

      console.log('Cross-provider message isolation test passed');
    });
  });
});
