import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { HookRegistry } from '@/hooks/HookRegistry';
import { Providers } from '@/common';
import { AgentContext } from '@/agents/AgentContext';
import type { AgentInputs, ResolvedSubagentConfig } from '@/types';
import {
  SubagentExecutor,
  filterSubagentResult,
  resolveSubagentConfigs,
} from '../subagent';
import type { StandardGraph } from '@/graphs/Graph';

jest.setTimeout(15000);

const makeChildInputs = (agentId = 'child-agent'): AgentInputs => ({
  agentId,
  provider: Providers.OPENAI,
  clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
  instructions: 'You are a helper agent.',
  maxContextTokens: 8000,
});

const makeConfig = (
  type = 'researcher',
  overrides: Partial<ResolvedSubagentConfig> = {}
): ResolvedSubagentConfig => ({
  type,
  name: 'Test Researcher',
  description: 'Researches things',
  agentInputs: makeChildInputs(),
  ...overrides,
});

describe('filterSubagentResult', () => {
  it('extracts text from last AIMessage string content', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('task'),
      new AIMessage('Here is the result'),
    ];
    expect(filterSubagentResult(messages)).toBe('Here is the result');
  });

  it('extracts text blocks from array content', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'text', text: 'First part.' },
          { type: 'text', text: 'Second part.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('First part.\nSecond part.');
  });

  it('strips tool_use blocks from array content', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'tool_use', id: 'call_1', name: 'search', input: {} },
          { type: 'text', text: 'Final answer.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Final answer.');
  });

  it('strips thinking blocks from array content', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'thinking', thinking: 'Let me think...' },
          { type: 'text', text: 'The result.' },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('The result.');
  });

  it('returns "Task completed" when no text blocks remain', () => {
    const messages: BaseMessage[] = [
      new AIMessage({
        content: [
          { type: 'tool_use', id: 'call_1', name: 'do_thing', input: {} },
        ],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Task completed');
  });

  it('returns "Task completed" for empty string content', () => {
    const messages: BaseMessage[] = [new AIMessage('')];
    expect(filterSubagentResult(messages)).toBe('Task completed');
  });

  it('returns "Task completed" when no messages', () => {
    expect(filterSubagentResult([])).toBe('Task completed');
  });

  it('returns "Task completed" when no AIMessage found', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('task'),
      new ToolMessage({ content: 'result', tool_call_id: 'x' }),
    ];
    expect(filterSubagentResult(messages)).toBe('Task completed');
  });

  it('uses last AIMessage, not first', () => {
    const messages: BaseMessage[] = [
      new AIMessage('First response'),
      new ToolMessage({ content: 'tool output', tool_call_id: 'x' }),
      new AIMessage('Final response'),
    ];
    expect(filterSubagentResult(messages)).toBe('Final response');
  });
});

describe('resolveSubagentConfigs', () => {
  const parentInputs: AgentInputs = {
    agentId: 'parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o', apiKey: 'test' },
    instructions: 'You are a parent agent.',
    maxContextTokens: 16000,
  };

  it('passes through configs with explicit agentInputs', () => {
    const config = makeConfig();
    const parentContext = AgentContext.fromConfig(parentInputs);
    const resolved = resolveSubagentConfigs([config], parentContext);
    expect(resolved).toHaveLength(1);
    expect(resolved[0].agentInputs.agentId).toBe('child-agent');
  });

  it('resolves self-spawn from parent _sourceInputs', () => {
    const selfConfig = {
      type: 'self',
      name: 'Self Spawn',
      description: 'Context isolation only',
      self: true,
    };
    const parentContext = AgentContext.fromConfig(parentInputs);
    const resolved = resolveSubagentConfigs([selfConfig], parentContext);
    expect(resolved).toHaveLength(1);
    expect(resolved[0].agentInputs.provider).toBe(Providers.OPENAI);
    expect(resolved[0].agentInputs.instructions).toBe(
      'You are a parent agent.'
    );
  });

  it('filters out configs with self=true when _sourceInputs is missing', () => {
    const selfConfig = {
      type: 'self',
      name: 'Self Spawn',
      description: 'Context isolation only',
      self: true,
    };
    const parentContext = new AgentContext({
      agentId: 'bare',
      provider: Providers.OPENAI,
      instructionTokens: 0,
    });
    const resolved = resolveSubagentConfigs([selfConfig], parentContext);
    expect(resolved).toHaveLength(0);
  });

  it('filters out configs without agentInputs and self=false', () => {
    const badConfig = {
      type: 'broken',
      name: 'Broken',
      description: 'Missing inputs',
    };
    const parentContext = AgentContext.fromConfig(parentInputs);
    const resolved = resolveSubagentConfigs([badConfig], parentContext);
    expect(resolved).toHaveLength(0);
  });
});

describe('SubagentExecutor', () => {
  const config = makeConfig();

  function createExecutor(
    overrides: Partial<ConstructorParameters<typeof SubagentExecutor>[0]> = {}
  ): SubagentExecutor {
    return new SubagentExecutor({
      configs: new Map([[config.type, config]]),
      parentRunId: 'test-run',
      parentAgentId: 'parent-agent',
      ...overrides,
    });
  }

  /** Spy handle for Graph module mock — set per test, restored in afterEach */
  let graphSpy: jest.SpyInstance | undefined;

  function mockStandardGraph(
    invokeResult: { messages: BaseMessage[] },
    clearSpy?: jest.Mock
  ): { clearHeavyState: jest.Mock } {
    const mockClear = clearSpy ?? jest.fn();
    const GraphModule =
      jest.requireActual<typeof import('@/graphs/Graph')>('@/graphs/Graph');
    graphSpy = jest.spyOn(GraphModule, 'StandardGraph').mockImplementation(
      () =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockResolvedValue(invokeResult),
          }),
          clearHeavyState: mockClear,
        }) as unknown as StandardGraph
    );
    return { clearHeavyState: mockClear };
  }

  function mockStandardGraphError(error: Error): void {
    const GraphModule =
      jest.requireActual<typeof import('@/graphs/Graph')>('@/graphs/Graph');
    graphSpy = jest.spyOn(GraphModule, 'StandardGraph').mockImplementation(
      () =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockRejectedValue(error),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph
    );
  }

  afterEach(() => {
    graphSpy?.mockRestore();
    graphSpy = undefined;
  });

  it('returns error for unknown subagent type', async () => {
    const executor = createExecutor();
    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'nonexistent',
    });
    expect(result.content).toContain('Unknown subagent type');
    expect(result.content).toContain('nonexistent');
    expect(result.content).toContain('researcher');
    expect(result.messages).toEqual([]);
  });

  it('returns error when depth exceeds maxDepth', async () => {
    const executor = createExecutor({ depth: 2, maxDepth: 2 });
    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });
    expect(result.content).toContain('Maximum subagent nesting depth');
    expect(result.messages).toEqual([]);
  });

  it('executes child graph and returns filtered content', async () => {
    const executor = createExecutor();
    const { clearHeavyState } = mockStandardGraph({
      messages: [
        new HumanMessage('research this topic'),
        new AIMessage('Here is my research summary.'),
      ],
    });

    const result = await executor.execute({
      description: 'Research this topic',
      subagentType: 'researcher',
    });

    expect(result.content).toBe('Here is my research summary.');
    expect(result.messages).toHaveLength(2);
    expect(clearHeavyState).toHaveBeenCalled();
  });

  it('returns error message when child graph throws', async () => {
    const executor = createExecutor();
    mockStandardGraphError(new Error('Graph recursion limit reached'));

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });

    expect(result.content).toContain('Subagent error');
    expect(result.content).toContain('Graph recursion limit reached');
    expect(result.messages).toEqual([]);
  });

  describe('hooks', () => {
    let capturedStart: unknown;
    let capturedStop: unknown;

    beforeEach(() => {
      capturedStart = undefined;
      capturedStop = undefined;
    });

    it('fires SubagentStart before execution', async () => {
      const registry = new HookRegistry();
      registry.register('SubagentStart', {
        hooks: [
          async (input): Promise<Record<string, never>> => {
            capturedStart = input;
            return {};
          },
        ],
      });

      const executor = createExecutor({ hookRegistry: registry });
      mockStandardGraph({ messages: [new AIMessage('done')] });

      await executor.execute({
        description: 'Test task',
        subagentType: 'researcher',
      });

      expect(capturedStart).toBeDefined();
      const input = capturedStart as Record<string, unknown>;
      expect(input.hook_event_name).toBe('SubagentStart');
      expect(input.parentAgentId).toBe('parent-agent');
      expect(input.agentType).toBe('researcher');
    });

    it('fires SubagentStop after execution', async () => {
      const registry = new HookRegistry();
      registry.register('SubagentStop', {
        hooks: [
          async (input): Promise<Record<string, never>> => {
            capturedStop = input;
            return {};
          },
        ],
      });

      const executor = createExecutor({ hookRegistry: registry });
      mockStandardGraph({ messages: [new AIMessage('done')] });

      await executor.execute({
        description: 'Test task',
        subagentType: 'researcher',
      });

      /** SubagentStop fires fire-and-forget; give microtask time to settle */
      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(capturedStop).toBeDefined();
      const input = capturedStop as Record<string, unknown>;
      expect(input.hook_event_name).toBe('SubagentStop');
      expect(input.agentType).toBe('researcher');
    });

    it('SubagentStart deny blocks execution', async () => {
      const registry = new HookRegistry();
      registry.register('SubagentStart', {
        hooks: [
          async (): Promise<{ decision: 'deny'; reason: string }> => ({
            decision: 'deny',
            reason: 'Not authorized',
          }),
        ],
      });

      const executor = createExecutor({ hookRegistry: registry });
      const result = await executor.execute({
        description: 'Blocked task',
        subagentType: 'researcher',
      });

      expect(result.content).toBe('Blocked: Not authorized');
      expect(result.messages).toEqual([]);
    });
  });
});
