// src/specs/context-usage.live.test.ts
/**
 * Live ON_CONTEXT_USAGE / usage accounting verification with real Anthropic
 * calls — single agent, multi-agent handoff, and subagent isolation.
 *
 * Run with:
 * RUN_CONTEXT_USAGE_LIVE_TESTS=1 ANTHROPIC_API_KEY=... npm test -- context-usage.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { createTokenCounter, TokenEncoderManager } from '@/utils/tokens';
import { Constants, GraphEvents, Providers } from '@/common';
import { ModelEndHandler } from '@/events';
import { Run } from '@/run';

const shouldRunLive =
  process.env.RUN_CONTEXT_USAGE_LIVE_TESTS === '1' &&
  process.env.ANTHROPIC_API_KEY != null &&
  process.env.ANTHROPIC_API_KEY !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;
const modelName =
  process.env.ANTHROPIC_CONTEXT_LIVE_MODEL ?? 'claude-haiku-4-5';

const MAX_CONTEXT_TOKENS = 8000;

function createAnthropicAgent(
  agentId: string,
  instructions: string,
  extras: Partial<t.AgentInputs> = {}
): t.AgentInputs {
  return {
    agentId,
    provider: Providers.ANTHROPIC,
    clientOptions: {
      modelName,
      apiKey: process.env.ANTHROPIC_API_KEY,
      temperature: 0,
      maxTokens: 128,
      streaming: true,
      streamUsage: true,
    },
    instructions,
    maxContextTokens: MAX_CONTEXT_TOKENS,
    ...extras,
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

interface CapturedEvents {
  contextEvents: t.ContextUsageEvent[];
  subagentUpdates: unknown[];
  collectedUsage: Array<Record<string, number | undefined>>;
  handlers: Record<string, t.EventHandler>;
}

function createCapture(): CapturedEvents {
  const contextEvents: t.ContextUsageEvent[] = [];
  const subagentUpdates: unknown[] = [];
  const collectedUsage: Array<Record<string, number | undefined>> = [];
  const handlers: Record<string, t.EventHandler> = {
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage as never),
    [GraphEvents.ON_CONTEXT_USAGE]: {
      handle: (_event, data): void => {
        contextEvents.push(data as unknown as t.ContextUsageEvent);
      },
    },
    [GraphEvents.ON_SUBAGENT_UPDATE]: {
      handle: (_event, data): void => {
        subagentUpdates.push(data);
      },
    },
  };
  return { contextEvents, subagentUpdates, collectedUsage, handlers };
}

describeIfLive('Context usage live integration', () => {
  jest.setTimeout(180_000);

  let tokenCounter: t.TokenCounter;

  beforeAll(async () => {
    tokenCounter = await createTokenCounter();
  });

  afterAll(() => {
    TokenEncoderManager.reset();
  });

  it('emits a snapshot whose estimate tracks real provider input tokens', async () => {
    const capture = createCapture();
    const run = await Run.create<t.IState>({
      runId: `ctx-live-single-${Date.now()}`,
      graphConfig: {
        type: 'standard',
        agents: [
          createAnthropicAgent(
            'solo',
            'You are concise. Reply with one short sentence.'
          ),
        ],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: capture.handlers,
      tokenCounter,
      indexTokenCountMap: {},
    });

    await run.processStream(
      { messages: [new HumanMessage('Say hello in five words or fewer.')] },
      createStreamConfig(`ctx-live-single-${Date.now()}`)
    );

    expect(capture.contextEvents).toHaveLength(1);
    const event = capture.contextEvents[0];
    expect(event.agentId).toBe('solo');
    expect(event.breakdown.maxContextTokens).toBe(MAX_CONTEXT_TOKENS);
    expect(event.contextBudget).toBeLessThanOrEqual(MAX_CONTEXT_TOKENS);

    expect(capture.collectedUsage).toHaveLength(1);
    const usage = capture.collectedUsage[0];
    expect(usage.input_tokens ?? 0).toBeGreaterThan(0);
    expect(usage.output_tokens ?? 0).toBeGreaterThan(0);

    /** The gauge shows `contextBudget - remaining` as occupancy; with a real
     *  tokenizer it should land in the same ballpark as the provider count */
    const estimatedUsed =
      (event.contextBudget ?? 0) - (event.remainingContextTokens ?? 0);
    const providerInput = usage.input_tokens ?? 0;
    expect(estimatedUsed).toBeGreaterThan(0);
    expect(estimatedUsed / providerInput).toBeGreaterThan(0.3);
    expect(estimatedUsed / providerInput).toBeLessThan(3);
  });

  it('emits per-agent snapshots and usage across a real handoff', async () => {
    const capture = createCapture();
    const nonce = `ctx-live-handoff-${Date.now()}`;
    const expectedReply = `${nonce}-confirmed`;
    const handoffToolName = `${Constants.LC_TRANSFER_TO_}specialist`;

    const run = await Run.create<t.IState>({
      runId: `${nonce}-run`,
      graphConfig: {
        type: 'multi-agent',
        agents: [
          createAnthropicAgent(
            'router',
            `You are a routing agent. For every user request, your only valid action is to call the handoff tool named ${handoffToolName}. Do not answer directly.

When you call the handoff tool, include instructions telling the specialist to reply exactly with this marker and no extra words: ${expectedReply}`
          ),
          createAnthropicAgent(
            'specialist',
            'You are the specialist. When you receive handoff instructions with a marker, reply exactly with that marker and no extra words.'
          ),
        ],
        edges: [
          {
            from: 'router',
            to: 'specialist',
            edgeType: 'handoff',
            description: 'Transfer to the specialist for the final response',
            prompt:
              'Instructions for the specialist. Include any exact marker that must be returned.',
            promptKey: 'instructions',
          },
        ],
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: capture.handlers,
      tokenCounter,
      indexTokenCountMap: {},
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

    const agentIds = new Set(
      capture.contextEvents.map((event) => event.agentId)
    );
    expect(agentIds.has('router')).toBe(true);
    expect(agentIds.has('specialist')).toBe(true);

    for (const event of capture.contextEvents) {
      expect(event.breakdown.maxContextTokens).toBe(MAX_CONTEXT_TOKENS);
      expect(event.contextBudget).toBeLessThanOrEqual(MAX_CONTEXT_TOKENS);
      expect(event.remainingContextTokens).toBeGreaterThan(0);
    }

    /** One snapshot per real model call — no ghost snapshots */
    expect(capture.collectedUsage.length).toBe(capture.contextEvents.length);
    expect(capture.collectedUsage.length).toBeGreaterThanOrEqual(2);
  });

  it('keeps subagent runs isolated from parent context/usage events', async () => {
    const capture = createCapture();
    const parent = createAnthropicAgent(
      'parent',
      'You are a supervisor. Delegate research tasks using the subagent tool.',
      {
        subagentConfigs: [
          {
            type: 'researcher',
            name: 'Research Agent',
            description: 'Researches and summarizes information',
            agentInputs: createAnthropicAgent(
              'researcher',
              'You are a research agent. Answer in one short sentence.'
            ),
          },
        ],
      }
    );

    const run = await Run.create<t.IState>({
      runId: `ctx-live-subagent-${Date.now()}`,
      graphConfig: { type: 'standard', agents: [parent] },
      returnContent: true,
      skipCleanup: true,
      customHandlers: capture.handlers,
      tokenCounter,
      indexTokenCountMap: {},
    });

    /** Parent is a fake forced to call the subagent tool — the child run
     *  executes on the real provider, exercising real isolation */
    const subagentToolCall: ToolCall = {
      id: 'call_subagent_live',
      name: Constants.SUBAGENT,
      args: {
        description: 'What is the capital of France? One short sentence.',
        subagent_type: 'researcher',
      },
      type: 'tool_call',
    };
    run.Graph?.overrideTestModel(
      ['Delegating to the researcher.', 'The researcher confirmed the answer.'],
      10,
      [subagentToolCall]
    );

    await run.processStream(
      { messages: [new HumanMessage('What is the capital of France?')] },
      createStreamConfig(`ctx-live-subagent-${Date.now()}`)
    );

    /** Child progress arrives only as wrapped subagent updates */
    expect(capture.subagentUpdates.length).toBeGreaterThan(0);

    /** No raw child snapshots leak into the parent handler registry */
    const childContextEvents = capture.contextEvents.filter(
      (event) => event.agentId !== 'parent'
    );
    expect(childContextEvents).toHaveLength(0);
    for (const event of capture.contextEvents) {
      expect(event.agentId).toBe('parent');
    }

    /** Documented isolation: child model-call usage does not reach the
     *  parent's collected usage (fake parent emits no usage_metadata) */
    expect(capture.collectedUsage).toHaveLength(0);

    const toolMessage = (run.getRunMessages() ?? []).find(
      (message) =>
        message.getType() === 'tool' &&
        (message as { name?: string }).name === Constants.SUBAGENT
    );
    expect(toolMessage).toBeDefined();
    expect(String(toolMessage?.content ?? '').toLowerCase()).toContain('paris');
  });
});
