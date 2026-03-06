// src/scripts/cli.ts
import { config } from 'dotenv';
config();

import z from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { getArgs } from '@/scripts/args';
import { Providers } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

const memoryKv: Record<string, string> = {};

const setMemory = tool(
  async ({ key, value }) => {
    if (!/^[a-z_]+$/.test(key)) {
      throw new Error(
        'Key must only contain lowercase letters and underscores'
      );
    }

    memoryKv[key] = value;

    return { ok: true };
  },
  {
    name: 'set_memory',
    description: 'Saves important data about the user into memory.',
    schema: z.object({
      key: z
        .string()
        .describe(
          'The key of the memory value. Always use lowercase and underscores, no other characters.'
        ),
      value: z
        .string()
        .describe('Value can be anything represented as a string'),
    }),
  }
);

async function testStandardStreaming(): Promise<void> {
  const { userName, provider } = await getArgs();

  const run = await Run.create<t.IState>({
    runId: 'memory-run',
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        model: 'gpt-4o-mini',
        temperature: 0.5,
        streaming: false,
      },
      tools: [setMemory],
      instructions:
        'You can use the `set_memory` tool to save important data about the user into memory. If there is nothing to note about the user specifically, respond with `nothing`.',
      toolEnd: true,
    },
    returnContent: true,
    skipCleanup: true,
  });

  const config = {
    configurable: {
      provider,
      thread_id: 'conversation-num-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  console.log('Test 1: Simple message test');

  const userMessage = `hi`;

  conversationHistory.push(new HumanMessage(userMessage));

  const inputs = {
    messages: conversationHistory,
  };
  await run.processStream(inputs, config);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
    console.dir(conversationHistory, { depth: null });
  }

  console.log('\n\n====================\n\n');
  console.dir(memoryKv, { depth: null });
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  process.exit(1);
});

testStandardStreaming().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
