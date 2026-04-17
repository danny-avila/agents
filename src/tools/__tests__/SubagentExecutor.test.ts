import { describe, it, expect, beforeEach } from '@jest/globals';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { HookRegistry } from '@/hooks/HookRegistry';
import { Providers, GraphEvents } from '@/common';
import { HandlerRegistry } from '@/events';
import { AgentContext } from '@/agents/AgentContext';
import type { AgentInputs, ResolvedSubagentConfig } from '@/types';
import {
  SubagentExecutor,
  filterSubagentResult,
  resolveSubagentConfigs,
  buildChildInputs,
  summarizeEvent,
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

  it('salvages text from an earlier AIMessage when the last has only tool_use', () => {
    /**
     * Scenario: subagent hit maxTurns mid-tool-call. The last AIMessage is
     * pure tool_use with no text. Partial progress from an earlier turn
     * should still be returned instead of "Task completed".
     */
    const messages: BaseMessage[] = [
      new HumanMessage('task'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me search.' },
          { type: 'tool_use', id: 'c1', name: 'search', input: {} },
        ],
      }),
      new ToolMessage({ content: 'Paris.', tool_call_id: 'c1' }),
      new AIMessage({
        content: [{ type: 'tool_use', id: 'c2', name: 'search', input: {} }],
      }),
    ];
    expect(filterSubagentResult(messages)).toBe('Let me search.');
  });

  it('salvages from earlier AIMessage when last has empty string content', () => {
    const messages: BaseMessage[] = [
      new AIMessage('Partial answer.'),
      new ToolMessage({ content: 'tool out', tool_call_id: 'x' }),
      new AIMessage(''),
    ];
    expect(filterSubagentResult(messages)).toBe('Partial answer.');
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

  it('throws on duplicate subagent types', () => {
    const parentContext = AgentContext.fromConfig(parentInputs);
    const dup1 = makeConfig('researcher');
    const dup2 = makeConfig('researcher');
    expect(() => resolveSubagentConfigs([dup1, dup2], parentContext)).toThrow(
      /Duplicate subagent type "researcher"/
    );
  });
});

describe('buildChildInputs', () => {
  const parentAgentInputs: AgentInputs = {
    agentId: 'parent',
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test' },
    instructions: 'parent',
    maxContextTokens: 8000,
    subagentConfigs: [{ type: 'researcher', name: 'R', description: 'd' }],
    maxSubagentDepth: 3,
  };

  it('strips subagentConfigs and maxSubagentDepth when allowNested is false', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.subagentConfigs).toBeUndefined();
    expect(result.maxSubagentDepth).toBeUndefined();
  });

  it('decrements maxSubagentDepth when allowNested is true', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
      allowNested: true,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.maxSubagentDepth).toBe(2);
    expect(result.subagentConfigs).toEqual(parentAgentInputs.subagentConfigs);
  });

  it('clamps decremented depth to 0 (never negative)', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
      allowNested: true,
    };
    const result = buildChildInputs(config, 'child', 0);
    expect(result.maxSubagentDepth).toBe(0);
  });

  it('always strips toolDefinitions (forces traditional mode)', () => {
    const inputsWithToolDefs: AgentInputs = {
      ...parentAgentInputs,
      toolDefinitions: [{ name: 't', description: 'x' }],
    };
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: inputsWithToolDefs,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.toolDefinitions).toBeUndefined();
  });

  it('strips parent-run-scoped initialSummary and discoveredTools from child inputs', () => {
    /**
     * Codex P1: a child inheriting `initialSummary` or `discoveredTools` from
     * the parent's shallow-spread AgentInputs leaks unrelated conversation
     * context / prior tool-search state into an isolated subagent run,
     * defeating the context-isolation contract. Both fields must be cleared.
     */
    const inputsWithRunContext: AgentInputs = {
      ...parentAgentInputs,
      initialSummary: { text: 'prior conversation summary', tokenCount: 42 },
      discoveredTools: ['prior_tool_a', 'prior_tool_b'],
    };
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: inputsWithRunContext,
    };
    const result = buildChildInputs(config, 'child', 3);
    expect(result.initialSummary).toBeUndefined();
    expect(result.discoveredTools).toBeUndefined();
  });

  it('overrides agentId with the passed childAgentId', () => {
    const config: ResolvedSubagentConfig = {
      type: 'researcher',
      name: 'R',
      description: 'd',
      agentInputs: parentAgentInputs,
    };
    const result = buildChildInputs(config, 'my-child', 3);
    expect(result.agentId).toBe('my-child');
  });
});

describe('SubagentExecutor', () => {
  const config = makeConfig();

  /**
   * Build a stub `createChildGraph` factory that returns a minimal
   * `StandardGraph`-shaped object whose `createWorkflow().invoke()`
   * resolves to `invokeResult`. Avoids `jest.spyOn(StandardGraph)` so
   * that SubagentExecutor does not need a runtime dep on the graphs
   * module (circular-dep-safe).
   */
  function makeStubGraphFactory(
    invokeResult: { messages: BaseMessage[] },
    clearSpy?: jest.Mock
  ): { factory: () => StandardGraph; clearHeavyState: jest.Mock } {
    const mockClear = clearSpy ?? jest.fn();
    const factory = (): StandardGraph =>
      ({
        createWorkflow: (): { invoke: jest.Mock } => ({
          invoke: jest.fn().mockResolvedValue(invokeResult),
        }),
        clearHeavyState: mockClear,
      }) as unknown as StandardGraph;
    return { factory, clearHeavyState: mockClear };
  }

  function makeThrowingGraphFactory(error: Error): () => StandardGraph {
    return (): StandardGraph =>
      ({
        createWorkflow: (): { invoke: jest.Mock } => ({
          invoke: jest.fn().mockRejectedValue(error),
        }),
        clearHeavyState: jest.fn(),
      }) as unknown as StandardGraph;
  }

  /** No-op factory for tests that never reach child graph construction. */
  function makeNoopGraphFactory(): () => StandardGraph {
    return (): StandardGraph =>
      ({
        createWorkflow: (): { invoke: jest.Mock } => ({
          invoke: jest.fn().mockResolvedValue({ messages: [] }),
        }),
        clearHeavyState: jest.fn(),
      }) as unknown as StandardGraph;
  }

  function createExecutor(
    overrides: Partial<ConstructorParameters<typeof SubagentExecutor>[0]> = {}
  ): SubagentExecutor {
    return new SubagentExecutor({
      configs: new Map([[config.type, config]]),
      parentRunId: 'test-run',
      parentAgentId: 'parent-agent',
      createChildGraph: makeNoopGraphFactory(),
      ...overrides,
    });
  }

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

  it('returns error when maxDepth is 0 (nesting budget exhausted)', async () => {
    const executor = createExecutor({ maxDepth: 0 });
    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });
    expect(result.content).toContain('Maximum subagent nesting depth');
    expect(result.messages).toEqual([]);
  });

  it('executes child graph and returns filtered content', async () => {
    const { factory, clearHeavyState } = makeStubGraphFactory({
      messages: [
        new HumanMessage('research this topic'),
        new AIMessage('Here is my research summary.'),
      ],
    });
    const executor = createExecutor({ createChildGraph: factory });

    const result = await executor.execute({
      description: 'Research this topic',
      subagentType: 'researcher',
    });

    expect(result.content).toBe('Here is my research summary.');
    expect(result.messages).toHaveLength(2);
    expect(clearHeavyState).toHaveBeenCalled();
  });

  it('returns error message when child graph throws', async () => {
    const executor = createExecutor({
      createChildGraph: makeThrowingGraphFactory(
        new Error('Graph recursion limit reached')
      ),
    });

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });

    expect(result.content).toContain('Subagent error');
    expect(result.content).toContain('Graph recursion limit reached');
    expect(result.messages).toEqual([]);
  });

  it('truncates long error messages to 200 chars', async () => {
    const longMessage = 'x'.repeat(500);
    const executor = createExecutor({
      createChildGraph: makeThrowingGraphFactory(new Error(longMessage)),
    });

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });

    /**
     * Expected composition: "Subagent error: " (16) + 200 truncated chars + "..." (3) = 219.
     * Assert the exact envelope to catch regressions in the truncation constant.
     */
    const MAX_TRUNCATED_LENGTH = 'Subagent error: '.length + 200 + '...'.length;
    expect(result.content.length).toBe(MAX_TRUNCATED_LENGTH);
    expect(result.content.startsWith('Subagent error: ')).toBe(true);
    expect(result.content.endsWith('...')).toBe(true);
  });

  it('does not truncate short error messages', async () => {
    const shortMessage = 'brief error detail';
    const executor = createExecutor({
      createChildGraph: makeThrowingGraphFactory(new Error(shortMessage)),
    });

    const result = await executor.execute({
      description: 'Do something',
      subagentType: 'researcher',
    });

    expect(result.content).toBe(`Subagent error: ${shortMessage}`);
    expect(result.content.endsWith('...')).toBe(false);
  });

  it('builds child with decremented maxSubagentDepth when allowNested=true', async () => {
    const nestedConfig: ResolvedSubagentConfig = {
      type: 'nested',
      name: 'Nested',
      description: 'allows nesting',
      allowNested: true,
      agentInputs: {
        ...makeChildInputs('nested-child'),
        subagentConfigs: [
          {
            type: 'nested',
            name: 'Nested',
            description: 'allows nesting',
            allowNested: true,
          },
        ],
        maxSubagentDepth: 3,
      },
    };

    let observedChildInputs: AgentInputs | undefined;
    const executor = new SubagentExecutor({
      configs: new Map([[nestedConfig.type, nestedConfig]]),
      parentRunId: 'test-run',
      parentAgentId: 'parent',
      maxDepth: 3,
      createChildGraph: (input): StandardGraph => {
        observedChildInputs = input.agents[0];
        return {
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockResolvedValue({
              messages: [new AIMessage('nested done')],
            }),
          }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      },
    });

    await executor.execute({
      description: 'nested task',
      subagentType: 'nested',
    });

    expect(observedChildInputs).toBeDefined();
    expect(observedChildInputs!.maxSubagentDepth).toBe(2);
    expect(observedChildInputs!.subagentConfigs).toBeDefined();
  });

  it('strips subagentConfigs from child when allowNested is not set', async () => {
    let observedChildInputs: AgentInputs | undefined;
    const executor = createExecutor({
      maxDepth: 3,
      createChildGraph: (input): StandardGraph => {
        observedChildInputs = input.agents[0];
        return {
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockResolvedValue({
              messages: [new AIMessage('done')],
            }),
          }),
          clearHeavyState: jest.fn(),
        } as unknown as StandardGraph;
      },
    });

    await executor.execute({
      description: 'task',
      subagentType: 'researcher',
    });

    expect(observedChildInputs).toBeDefined();
    expect(observedChildInputs!.subagentConfigs).toBeUndefined();
    expect(observedChildInputs!.maxSubagentDepth).toBeUndefined();
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

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        hookRegistry: registry,
        createChildGraph: factory,
      });

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

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        hookRegistry: registry,
        createChildGraph: factory,
      });

      await executor.execute({
        description: 'Test task',
        subagentType: 'researcher',
      });

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

  describe('event forwarding', () => {
    it('emits start/stop ON_SUBAGENT_UPDATE envelopes when parentHandlerRegistry is provided', async () => {
      const events: unknown[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Test task',
        subagentType: 'researcher',
      });

      const phases = events.map((e) => (e as { phase: string }).phase);
      expect(phases[0]).toBe('start');
      expect(phases[phases.length - 1]).toBe('stop');
    });

    it('keeps toolDefinitions on child when registry has ON_TOOL_EXECUTE handler', async () => {
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_TOOL_EXECUTE, {
        handle: (): void => {},
      });
      let observedChildInputs: AgentInputs | undefined;
      const configWithDefs: ResolvedSubagentConfig = {
        type: 'researcher',
        name: 'Research Specialist',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher',
          provider: Providers.OPENAI,
          toolDefinitions: [
            { name: 'web', description: 'search', parameters: {} },
          ],
        } as AgentInputs,
      };

      const executor = new SubagentExecutor({
        configs: new Map([[configWithDefs.type, configWithDefs]]),
        parentRunId: 'run',
        parentAgentId: 'parent',
        parentHandlerRegistry: registry,
        createChildGraph: (input): StandardGraph => {
          observedChildInputs = input.agents[0];
          return {
            createWorkflow: (): { invoke: jest.Mock } => ({
              invoke: jest.fn().mockResolvedValue({
                messages: [new AIMessage('ok')],
              }),
            }),
            clearHeavyState: jest.fn(),
          } as unknown as StandardGraph;
        },
      });

      await executor.execute({
        description: 'find weather',
        subagentType: 'researcher',
      });

      expect(observedChildInputs?.toolDefinitions).toHaveLength(1);
      expect(observedChildInputs?.toolDefinitions?.[0]?.name).toBe('web');
    });

    it('strips toolDefinitions when registry is present but ON_TOOL_EXECUTE handler is absent', async () => {
      const registry = new HandlerRegistry();
      let observedChildInputs: AgentInputs | undefined;
      const configWithDefs: ResolvedSubagentConfig = {
        type: 'researcher',
        name: 'Research Specialist',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher',
          provider: Providers.OPENAI,
          toolDefinitions: [
            { name: 'web', description: 'search', parameters: {} },
          ],
        } as AgentInputs,
      };

      const executor = new SubagentExecutor({
        configs: new Map([[configWithDefs.type, configWithDefs]]),
        parentRunId: 'run',
        parentAgentId: 'parent',
        parentHandlerRegistry: registry,
        createChildGraph: (input): StandardGraph => {
          observedChildInputs = input.agents[0];
          return {
            createWorkflow: (): { invoke: jest.Mock } => ({
              invoke: jest.fn().mockResolvedValue({
                messages: [new AIMessage('ok')],
              }),
            }),
            clearHeavyState: jest.fn(),
          } as unknown as StandardGraph;
        },
      });

      await executor.execute({
        description: 'find weather',
        subagentType: 'researcher',
      });

      expect(observedChildInputs?.toolDefinitions).toBeUndefined();
    });

    it('forwards parentToolCallId from execute params to SubagentUpdateEvent envelopes', async () => {
      const events: unknown[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
        parentToolCallId: 'call_abc123',
      });

      expect(events.length).toBeGreaterThan(0);
      for (const e of events) {
        expect((e as { parentToolCallId?: string }).parentToolCallId).toBe(
          'call_abc123'
        );
      }
    });

    it('still strips toolDefinitions when no parentHandlerRegistry is provided (legacy isolation)', async () => {
      let observedChildInputs: AgentInputs | undefined;
      const configWithDefs: ResolvedSubagentConfig = {
        type: 'researcher',
        name: 'Research Specialist',
        description: 'Researches topics',
        agentInputs: {
          agentId: 'researcher',
          provider: Providers.OPENAI,
          toolDefinitions: [
            { name: 'web', description: 'search', parameters: {} },
          ],
        } as AgentInputs,
      };

      const executor = new SubagentExecutor({
        configs: new Map([[configWithDefs.type, configWithDefs]]),
        parentRunId: 'run',
        parentAgentId: 'parent',
        createChildGraph: (input): StandardGraph => {
          observedChildInputs = input.agents[0];
          return {
            createWorkflow: (): { invoke: jest.Mock } => ({
              invoke: jest.fn().mockResolvedValue({
                messages: [new AIMessage('ok')],
              }),
            }),
            clearHeavyState: jest.fn(),
          } as unknown as StandardGraph;
        },
      });

      await executor.execute({
        description: 'find weather',
        subagentType: 'researcher',
      });

      expect(observedChildInputs?.toolDefinitions).toBeUndefined();
    });

    it('accepts parentHandlerRegistry as a lazy getter', async () => {
      const lazyHolder: { registry?: InstanceType<typeof HandlerRegistry> } =
        {};
      const events: unknown[] = [];
      const { factory } = makeStubGraphFactory({
        messages: [new AIMessage('done')],
      });
      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: () => lazyHolder.registry,
      });

      lazyHolder.registry = new HandlerRegistry();
      lazyHolder.registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
      });

      expect(events.length).toBeGreaterThan(0);
      expect((events[0] as { phase: string }).phase).toBe('start');
    });

    it('routes child ON_TOOL_EXECUTE dispatches through the parent registry', async () => {
      /**
       * Drives the forwarder callback the executor installs on the child's
       * `workflow.invoke({ callbacks: [forwarder] })`. We capture that
       * callback when the child workflow runs, then synthesize the same
       * `handleCustomEvent` call that a real `ToolNode` would make when
       * the child LLM emits a tool_call. If the forwarder routes correctly,
       * the parent's `ON_TOOL_EXECUTE` handler receives the batch and
       * resolves the promise with our canned results.
       */

      const parentToolHandler = jest.fn(
        async (_event: string, rawData: unknown): Promise<void> => {
          const req = rawData as {
            toolCalls: Array<{ id: string; name: string }>;
            resolve: (results: unknown[]) => void;
          };
          req.resolve(
            req.toolCalls.map((tc) => ({
              toolCallId: tc.id,
              status: 'success',
              content: `ran ${tc.name}`,
            }))
          );
        }
      );

      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_TOOL_EXECUTE, {
        handle: parentToolHandler,
      });

      let capturedInvokeOptions: unknown;
      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              capturedInvokeOptions = options;
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
        parentToolCallId: 'call_parent_123',
      });

      const opts = capturedInvokeOptions as
        | { callbacks?: unknown[] }
        | undefined;
      expect(opts?.callbacks).toBeDefined();
      const forwarder = (opts?.callbacks ?? [])[0] as {
        handleCustomEvent?: (
          eventName: string,
          data: unknown,
          runId: string,
          tags?: string[],
          metadata?: Record<string, unknown>
        ) => Promise<void> | void;
      };
      expect(typeof forwarder.handleCustomEvent).toBe('function');

      /** Simulate the child's ToolNode emitting a real batch request. */
      const resolvePromise = new Promise<
        Array<{ toolCallId: string; status: string; content: string }>
      >((resolve, reject) => {
        const batchRequest = {
          toolCalls: [{ id: 'call_child_xyz', name: 'calculator', args: {} }],
          agentId: 'researcher',
          resolve,
          reject,
        };
        forwarder.handleCustomEvent?.(
          GraphEvents.ON_TOOL_EXECUTE,
          batchRequest,
          'child-run-id'
        );
      });

      const results = await resolvePromise;
      expect(parentToolHandler).toHaveBeenCalledTimes(1);
      expect(results).toEqual([
        {
          toolCallId: 'call_child_xyz',
          status: 'success',
          content: 'ran calculator',
        },
      ]);
    });

    it('does NOT forward ON_TOOL_EXECUTE when the parent registry has no handler (safe fallback)', async () => {
      /**
       * The executor strips `toolDefinitions` when the parent registry has
       * no `ON_TOOL_EXECUTE` handler (see the companion strip-on-no-handler
       * test). Defence-in-depth: if the LLM somehow still dispatches a tool
       * call, the forwarder must not silently consume it without resolving;
       * reject would be better than hang. This test confirms no handler
       * is invoked on the parent side so it's clear a forwarded request
       * would need separate treatment.
       */

      const registry = new HandlerRegistry();
      /** Only ON_SUBAGENT_UPDATE registered — no ON_TOOL_EXECUTE. */
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, { handle: jest.fn() });

      let capturedInvokeOptions: unknown;
      const factory: () => StandardGraph = (): StandardGraph =>
        ({
          createWorkflow: (): { invoke: jest.Mock } => ({
            invoke: jest.fn().mockImplementation(async (_state, options) => {
              capturedInvokeOptions = options;
              return { messages: [new AIMessage('ok')] };
            }),
          }),
          clearHeavyState: jest.fn(),
        }) as unknown as StandardGraph;

      const executor = createExecutor({
        createChildGraph: factory,
        parentHandlerRegistry: registry,
      });

      await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
      });

      const opts = capturedInvokeOptions as { callbacks?: unknown[] };
      const forwarder = (opts.callbacks ?? [])[0] as {
        handleCustomEvent?: (
          eventName: string,
          data: unknown
        ) => Promise<void> | void;
      };

      let resolved = false;
      const batchRequest = {
        toolCalls: [{ id: 'call_x', name: 'calculator', args: {} }],
        agentId: 'researcher',
        resolve: (): void => {
          resolved = true;
        },
        reject: (): void => {},
      };
      await forwarder.handleCustomEvent?.(
        GraphEvents.ON_TOOL_EXECUTE,
        batchRequest
      );

      /** No handler exists → nothing resolves the promise. This is the
       *  state that justifies the `keepToolDefinitions` gate: without the
       *  gate we'd deadlock here. The gate ensures the LLM never sees
       *  tools in the first place, making this scenario unreachable in
       *  practice — the test just documents the fallback. */
      expect(resolved).toBe(false);
    });

    it('emits an `error` phase envelope when the child graph throws', async () => {
      const events: unknown[] = [];
      const registry = new HandlerRegistry();
      registry.register(GraphEvents.ON_SUBAGENT_UPDATE, {
        handle: (_event, data): void => {
          events.push(data);
        },
      });

      const executor = createExecutor({
        createChildGraph: makeThrowingGraphFactory(
          new Error('recursion limit')
        ),
        parentHandlerRegistry: registry,
      });

      const result = await executor.execute({
        description: 'Task',
        subagentType: 'researcher',
        parentToolCallId: 'call_err',
      });

      expect(result.content).toContain('Subagent error: recursion limit');
      const phases = events.map((e) => (e as { phase: string }).phase);
      expect(phases).toContain('start');
      expect(phases).toContain('error');
      const errEvent = events.find(
        (e) => (e as { phase: string }).phase === 'error'
      ) as { data?: { message?: string }; parentToolCallId?: string };
      expect(errEvent.data?.message).toContain('recursion limit');
      expect(errEvent.parentToolCallId).toBe('call_err');
    });
  });
});

describe('summarizeEvent', () => {
  it('labels a run step tool_calls stepDetails by tool name', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ name: 'calculator', id: 'c1' }],
      },
    });
    expect(label).toBe('Using tool: calculator');
  });

  it('joins multiple tool names on a single run step', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: {
        type: 'tool_calls',
        tool_calls: [{ name: 'web' }, { name: 'calculator' }],
      },
    });
    expect(label).toBe('Using tool: web, calculator');
  });

  it('falls back to "Planning tool call" when tool_calls is empty', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: { type: 'tool_calls', tool_calls: [] },
    });
    expect(label).toBe('Planning tool call');
  });

  it('labels message_creation steps as "Thinking…"', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP, {
      stepDetails: { type: 'message_creation' },
    });
    expect(label).toBe('Thinking…');
  });

  it('labels ON_TOOL_EXECUTE with the batch of tool names', () => {
    const label = summarizeEvent(GraphEvents.ON_TOOL_EXECUTE, {
      toolCalls: [{ name: 'web' }, { name: 'calculator' }],
    });
    expect(label).toBe('Calling web, calculator');
  });

  it('falls back to a generic "Calling tool" when toolCalls is empty', () => {
    const label = summarizeEvent(GraphEvents.ON_TOOL_EXECUTE, {
      toolCalls: [],
    });
    expect(label).toBe('Calling tool');
  });

  it('labels completed run steps by completed tool name', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP_COMPLETED, {
      result: { type: 'tool_call', tool_call: { name: 'calculator' } },
    });
    expect(label).toBe('Tool calculator complete');
  });

  it('labels completed steps without a tool name as "Step complete"', () => {
    const label = summarizeEvent(GraphEvents.ON_RUN_STEP_COMPLETED, {
      result: { type: 'message_creation' },
    });
    expect(label).toBe('Step complete');
  });

  it('labels ON_MESSAGE_DELTA as "Streaming…"', () => {
    expect(summarizeEvent(GraphEvents.ON_MESSAGE_DELTA, {})).toBe('Streaming…');
  });

  it('falls back to top-level `step.type` when `stepDetails` is absent', () => {
    /**
     * Covers the `step.stepDetails?.type ?? step.type ?? 'step'` chain
     * when the payload uses the top-level form (no `stepDetails` wrapper).
     * Exercises the second clause of the fallback so future changes to
     * the resolution order fail fast.
     */
    expect(
      summarizeEvent(GraphEvents.ON_RUN_STEP, { type: 'tool_calls' })
    ).toBe('Planning tool call');
    expect(
      summarizeEvent(GraphEvents.ON_RUN_STEP, { type: 'message_creation' })
    ).toBe('Thinking…');
  });

  it('falls back to "Step: step" when neither `stepDetails.type` nor `step.type` is present', () => {
    /** Exercises the final `?? 'step'` default plus the generic
     *  `Step: <detailType>` branch when a run step arrives with an
     *  unrecognized shape. */
    expect(summarizeEvent(GraphEvents.ON_RUN_STEP, {})).toBe('Step: step');
  });

  it('returns the event name for unknown events', () => {
    expect(summarizeEvent('on_unknown_event', {})).toBe('on_unknown_event');
  });
});
