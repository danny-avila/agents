// src/specs/agent-handoffs.live.test.ts
/**
 * Live handoff integration verification.
 *
 * Run with:
 * RUN_HANDOFF_LIVE_TESTS=1 ANTHROPIC_API_KEY=... npm test -- agent-handoffs.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type { BaseMessage, ToolMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { Constants, Providers } from '@/common';
import { Run } from '@/run';

const shouldRunLive =
  process.env.RUN_HANDOFF_LIVE_TESTS === '1' &&
  process.env.ANTHROPIC_API_KEY != null &&
  process.env.ANTHROPIC_API_KEY !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;
const modelName =
  process.env.ANTHROPIC_HANDOFF_LIVE_MODEL ?? 'claude-sonnet-4-6';

function createAnthropicAgent(
  agentId: string,
  instructions: string
): t.AgentInputs {
  return {
    agentId,
    provider: Providers.ANTHROPIC,
    clientOptions: {
      modelName,
      apiKey: process.env.ANTHROPIC_API_KEY,
      temperature: 0,
      maxTokens: 256,
      streaming: true,
    },
    instructions,
    maxContextTokens: 8000,
  };
}

function createStreamConfig(threadId: string): Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} {
  return {
    configurable: { thread_id: threadId },
    streamMode: 'values',
    version: 'v2',
  };
}

function contentToText(message: BaseMessage): string {
  if (typeof message.content === 'string') {
    return message.content;
  }
  if (!Array.isArray(message.content)) {
    return '';
  }
  return message.content
    .map((part) => {
      if (typeof part === 'string') {
        return part;
      }
      if ('text' in part && typeof part.text === 'string') {
        return part.text;
      }
      return '';
    })
    .join('');
}

describeIfLive('Agent handoffs live integration', () => {
  jest.setTimeout(120_000);

  it('routes through a real Anthropic handoff and preserves instructions', async () => {
    const nonce = `live-handoff-${Date.now()}`;
    const expectedReply = `${nonce}-specialist-confirmed`;
    const handoffToolName = `${Constants.LC_TRANSFER_TO_}specialist`;
    const agents: t.AgentInputs[] = [
      createAnthropicAgent(
        'router',
        `You are a routing agent. For every user request, your only valid action is to call the handoff tool named ${handoffToolName}. Do not answer directly.

When you call the handoff tool, include instructions telling the specialist to reply exactly with this marker and no extra words: ${expectedReply}`
      ),
      createAnthropicAgent(
        'specialist',
        'You are the specialist. When you receive handoff instructions with a marker, reply exactly with that marker and no extra words.'
      ),
    ];
    const edges: t.GraphEdge[] = [
      {
        from: 'router',
        to: 'specialist',
        edgeType: 'handoff',
        description: 'Transfer to the specialist for the final response',
        prompt:
          'Instructions for the specialist. Include any exact marker that must be returned.',
        promptKey: 'instructions',
      },
    ];
    const run = await Run.create({
      runId: `${nonce}-run`,
      graphConfig: { type: 'multi-agent', agents, edges },
      returnContent: true,
      skipCleanup: true,
    });

    await run.processStream(
      {
        messages: [
          new HumanMessage(
            `Please delegate this to the specialist. The final answer must be exactly: ${expectedReply}`
          ),
        ],
      },
      createStreamConfig(`${nonce}-thread`)
    );

    const messages = run.getRunMessages() ?? [];
    const handoffMessage = messages.find(
      (message): message is ToolMessage =>
        message.getType() === 'tool' &&
        (message as ToolMessage).name === handoffToolName
    );
    const finalText = messages
      .filter((message) => message.getType() === 'ai')
      .map(contentToText)
      .join('\n');

    expect(handoffMessage).toBeDefined();
    expect(finalText).toContain(expectedReply);
  });
});
