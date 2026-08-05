/**
 * `attemptInvoke`'s local stream branch (no registered SDK dispatcher) must
 * charge the event budget even for chunks that produce no handling chunk:
 * a pure OpenRouter reasoning-replay chunk is skipped for content handling,
 * and in this branch no `streamEvents` consumer counts the wire event
 * either, so a looping replay stream would otherwise bypass an enabled
 * `maxDeltaEventsPerTurn` indefinitely.
 */
import { AIMessageChunk, HumanMessage } from '@langchain/core/messages';
import { describe, it, expect, jest } from '@jest/globals';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs';
import type * as t from '@/types';
import {
  StreamLimitExceededError,
  resolveStreamLimits,
} from '@/llm/streamLimits';
import { attemptInvoke } from '@/llm/invoke';
import { ContentTypes, Providers } from '@/common';

function createContext(
  overrides: Partial<StandardGraph> = {}
): StandardGraph {
  const runSteps = new Map<string, t.RunStep>();
  const stepIdsByKey = new Map<string, string>();
  let stepCounter = 0;

  const graph = {
    config: { configurable: { user_id: 'user_1' }, metadata: { run_id: 'r1' } },
    eagerEventToolExecution: undefined,
    eagerEventToolExecutions: new Map(),
    eagerEventToolCallChunks: new Map(),
    eagerEventToolSuppressions: new Set<string>(),
    handlerRegistry: undefined,
    hookRegistry: undefined,
    humanInTheLoop: undefined,
    toolOutputReferences: undefined,
    sessions: new Map(),
    toolCallStepIds: new Map(),
    messageIdsByStepKey: new Map(),
    messageStepHasToolCalls: new Map(),
    prelimMessageIdsByStepKey: new Map([['step-key', 'msg_1']]),
    getAgentContext: jest.fn(
      (): Partial<AgentContext> => ({
        provider: Providers.OPENAI,
        reasoningKey: 'reasoning_content',
        currentTokenType: ContentTypes.TEXT,
        toolDefinitions: [],
        graphTools: [],
        agentId: 'agent_1',
      })
    ),
    getOrCreateToolOutputRegistry: jest.fn(() => undefined),
    shouldPreemptStream: jest.fn(() => false),
    claimPreemptSeal: jest.fn(() => false),
    getStepKey: jest.fn(() => 'step-key'),
    getStepIdByKey: jest.fn((stepKey: string) => {
      const stepId = stepIdsByKey.get(stepKey);
      if (stepId == null) {
        throw new Error('no current step');
      }
      return stepId;
    }),
    getRunStep: jest.fn((stepId: string) => runSteps.get(stepId)),
    dispatchRunStep: jest.fn(async (stepKey: string, details: unknown) => {
      const id = `step_${++stepCounter}`;
      stepIdsByKey.set(stepKey, id);
      runSteps.set(id, {
        id,
        type: (details as { type: t.RunStep['type'] }).type,
        stepDetails: details as t.RunStep['stepDetails'],
      } as t.RunStep);
      return id;
    }),
    dispatchRunStepDelta: jest.fn(async () => undefined),
    dispatchMessageDelta: jest.fn(async () => undefined),
    dispatchReasoningDelta: jest.fn(async () => undefined),
    ...overrides,
  };
  return graph as unknown as StandardGraph;
}

const replayChunk = (): AIMessageChunk =>
  new AIMessageChunk({
    content: 'abc',
    additional_kwargs: { reasoning_details: [{ type: 'reasoning.text' }] },
  });

function createReplayModel(replays: number): t.ChatModel {
  return {
    stream: async function* stream(): AsyncGenerator<AIMessageChunk> {
      yield new AIMessageChunk({ content: 'abc' });
      for (let i = 0; i < replays; i++) {
        yield replayChunk();
      }
    },
  } as unknown as t.ChatModel;
}

describe('attemptInvoke local branch stream limits', () => {
  it('charges skipped OpenRouter replay chunks against the event budget', async () => {
    const context = createContext({
      streamLimits: resolveStreamLimits({ maxDeltaEventsPerTurn: 3 }),
    });

    let caught: unknown;
    try {
      await attemptInvoke(
        {
          model: createReplayModel(10),
          messages: [new HumanMessage('hi')],
          provider: Providers.OPENROUTER,
          context: context as unknown as Parameters<
            typeof attemptInvoke
          >[0]['context'],
        },
        { metadata: { langgraph_node: 'agent', langgraph_step: 1 } }
      );
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(StreamLimitExceededError);
    expect((caught as StreamLimitExceededError).kind).toBe('delta_events');
  });

  it('leaves replay streams unbounded when the cap is disabled', async () => {
    const context = createContext({ streamLimits: resolveStreamLimits() });

    const result = await attemptInvoke(
      {
        model: createReplayModel(10),
        messages: [new HumanMessage('hi')],
        provider: Providers.OPENROUTER,
        context: context as unknown as Parameters<
          typeof attemptInvoke
        >[0]['context'],
      },
      { metadata: { langgraph_node: 'agent', langgraph_step: 1 } }
    );
    expect(result.messages).toHaveLength(1);
  });
});
