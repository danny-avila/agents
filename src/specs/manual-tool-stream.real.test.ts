/* eslint-disable no-console */
import { config } from 'dotenv';
config();
import { Calculator } from '@langchain/community/tools/calculator';
import { HumanMessage } from '@langchain/core/messages';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { ModelEndHandler } from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { Run } from '@/run';
import type * as t from '@/types';

describe('Manual tool-stream suppression (real Anthropic)', () => {
  jest.setTimeout(60000);

  test('CHAT_MODEL_STREAM is suppressed when tools present and provider is in manual set', async () => {
    const provider = Providers.ANTHROPIC;
    const llmConfig = getLLMConfig(provider);

    let streamCount = 0;
    let modelEndCount = 0;

    const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
      // Count only CHAT_MODEL_STREAM events (which are the ones suppressed for manual providers with tools)
      [GraphEvents.CHAT_MODEL_STREAM]: {
        handle: () => {
          streamCount += 1;
        },
      },
      [GraphEvents.CHAT_MODEL_END]: {
        handle: () => {
          modelEndCount += 1;
        },
      },
    } as unknown as Record<string | GraphEvents, t.EventHandler>;

    const run = await Run.create<t.IState>({
      runId: 'manual-stream-suppress',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [new Calculator()], // presence of tools triggers streaming branch for Anthropic
        instructions: 'Be concise.',
      },
      returnContent: true,
      customHandlers,
    });

    const configV2 = {
      configurable: { thread_id: 'manual-stream-thread' },
      streamMode: 'values',
      version: 'v2' as const,
    };

    await run.processStream({ messages: [new HumanMessage('hi')] }, configV2);
    const finalMessages = run.getRunMessages();

    expect(finalMessages).toBeDefined();
    expect(modelEndCount).toBeGreaterThan(0);
    // Key assertion: stream events from model are filtered out by Run for manual providers when tools are present
    expect(streamCount).toBe(0);
  });
});


