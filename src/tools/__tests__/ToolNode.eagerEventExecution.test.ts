import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { PreToolUseHookInput, PreToolUseHookOutput } from '@/hooks';
import type * as t from '@/types';
import { GraphEvents } from '@/common';
import { HookRegistry } from '@/hooks';
import * as events from '@/utils/events';
import { ToolNode } from '../ToolNode';

function createDummyTool(name: string): StructuredToolInterface {
  return tool(async () => 'direct should not run', {
    name,
    description: 'dummy',
    schema: z.object({ city: z.string() }),
  }) as unknown as StructuredToolInterface;
}

function createAIMessage(
  id: string,
  name: string,
  args: Record<string, unknown>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [{ id, name, args }],
  });
}

function createAIMessageWithToolCalls(
  toolCalls: Array<{
    id: string;
    name: string;
    args: Record<string, unknown>;
  }>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: toolCalls,
  });
}

function installToolExecuteResponder(content: string): {
  toolExecuteCalls: t.ToolExecuteBatchRequest[];
} {
  const toolExecuteCalls: t.ToolExecuteBatchRequest[] = [];
  jest
    .spyOn(events, 'safeDispatchCustomEvent')
    .mockImplementation(async (event, data): Promise<void> => {
      if (event !== GraphEvents.ON_TOOL_EXECUTE) {
        return;
      }
      const batch = data as t.ToolExecuteBatchRequest;
      toolExecuteCalls.push(batch);
      batch.resolve(
        batch.toolCalls.map((request) => ({
          toolCallId: request.id,
          status: 'success' as const,
          content,
        }))
      );
    });
  return { toolExecuteCalls };
}

describe('ToolNode eager event tool execution', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('uses a matching prestarted event result without redispatching the tool', async () => {
    const { toolExecuteCalls } = installToolExecuteResponder('redispatched');
    const eagerExecutions = new Map<string, t.EagerEventToolExecution>();
    const eagerUsageCount = new Map<string, number>([['weather', 1]]);
    const request: t.ToolCallRequest = {
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: 'step_weather',
      turn: 0,
    };
    eagerExecutions.set('call_weather', {
      toolCallId: 'call_weather',
      toolName: 'weather',
      args: { city: 'NYC' },
      request,
      promise: Promise.resolve({
        results: [
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'eager result',
          },
        ],
      }),
    });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      eagerEventToolExecution: { enabled: true },
      eagerEventToolExecutions: eagerExecutions,
      eagerEventToolUsageCount: eagerUsageCount,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    const result = (await toolNode.invoke({
      messages: [createAIMessage('call_weather', 'weather', { city: 'NYC' })],
    })) as { messages: ToolMessage[] };

    expect(toolExecuteCalls).toHaveLength(0);
    expect(eagerUsageCount.get('weather')).toBe(1);
    expect(eagerExecutions.has('call_weather')).toBe(false);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].content).toBe('eager result');
  });

  it('fails closed without redispatching when final args differ from the prestarted call', async () => {
    const { toolExecuteCalls } = installToolExecuteResponder('fresh result');
    const eagerExecutions = new Map<string, t.EagerEventToolExecution>();
    const request: t.ToolCallRequest = {
      id: 'call_weather',
      name: 'weather',
      args: { city: 'NYC' },
      stepId: 'step_weather',
      turn: 0,
    };
    eagerExecutions.set('call_weather', {
      toolCallId: 'call_weather',
      toolName: 'weather',
      args: { city: 'NYC' },
      request,
      promise: Promise.resolve({
        results: [
          {
            toolCallId: 'call_weather',
            status: 'success',
            content: 'stale eager result',
          },
        ],
      }),
    });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      eagerEventToolExecution: { enabled: true },
      eagerEventToolExecutions: eagerExecutions,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    const result = (await toolNode.invoke({
      messages: [
        createAIMessage('call_weather', 'weather', { city: 'Boston' }),
      ],
    })) as { messages: ToolMessage[] };

    expect(toolExecuteCalls).toHaveLength(0);
    expect(eagerExecutions.has('call_weather')).toBe(false);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].status).toBe('error');
    expect(result.messages[0].content).toContain('refusing to re-run');
  });

  it('increments the shared eager turn counter for normal event dispatch', async () => {
    const { toolExecuteCalls } = installToolExecuteResponder('normal result');
    const eagerUsageCount = new Map<string, number>();

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      eagerEventToolExecution: { enabled: true },
      eagerEventToolExecutions: new Map(),
      eagerEventToolUsageCount: eagerUsageCount,
      toolCallStepIds: new Map([['call_weather', 'step_weather']]),
    });

    const result = (await toolNode.invoke({
      messages: [createAIMessage('call_weather', 'weather', { city: 'NYC' })],
    })) as { messages: ToolMessage[] };

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls[0]).toMatchObject({
      id: 'call_weather',
      name: 'weather',
      turn: 0,
    });
    expect(eagerUsageCount.get('weather')).toBe(1);
    expect(result.messages).toHaveLength(1);
    expect(result.messages[0].content).toBe('normal result');
  });

  it('preserves final turn order when only part of a batch was prestarted', async () => {
    const { toolExecuteCalls } = installToolExecuteResponder('normal result');
    const eagerExecutions = new Map<string, t.EagerEventToolExecution>();
    const eagerUsageCount = new Map<string, number>([['weather', 1]]);
    const staleRequest: t.ToolCallRequest = {
      id: 'call_weather_2',
      name: 'weather',
      args: { city: 'Boston' },
      stepId: 'step_weather_2',
      turn: 0,
    };
    eagerExecutions.set('call_weather_2', {
      toolCallId: 'call_weather_2',
      toolName: 'weather',
      args: { city: 'Boston' },
      request: staleRequest,
      promise: Promise.resolve({
        results: [
          {
            toolCallId: 'call_weather_2',
            status: 'success',
            content: 'stale eager result',
          },
        ],
      }),
    });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      eagerEventToolExecution: { enabled: true },
      eagerEventToolExecutions: eagerExecutions,
      eagerEventToolUsageCount: eagerUsageCount,
      toolCallStepIds: new Map([
        ['call_weather_1', 'step_weather_1'],
        ['call_weather_2', 'step_weather_2'],
      ]),
    });

    const result = (await toolNode.invoke({
      messages: [
        createAIMessageWithToolCalls([
          {
            id: 'call_weather_1',
            name: 'weather',
            args: { city: 'NYC' },
          },
          {
            id: 'call_weather_2',
            name: 'weather',
            args: { city: 'Boston' },
          },
        ]),
      ],
    })) as { messages: ToolMessage[] };

    expect(toolExecuteCalls).toHaveLength(1);
    expect(toolExecuteCalls[0].toolCalls).toEqual([
      expect.objectContaining({
        id: 'call_weather_1',
        name: 'weather',
        turn: 0,
      }),
    ]);
    expect(eagerExecutions.has('call_weather_2')).toBe(false);
    expect(eagerUsageCount.get('weather')).toBe(2);
    expect(result.messages.map((message) => message.tool_call_id)).toEqual([
      'call_weather_1',
      'call_weather_2',
    ]);
    expect(result.messages[0].status).toBe('success');
    expect(result.messages[1].status).toBe('error');
    expect(result.messages[1].content).toContain('refusing to re-run');
  });

  it('uses the legacy turn counter when eager mode cannot be consumed', async () => {
    const { toolExecuteCalls } = installToolExecuteResponder('normal result');
    const eagerUsageCount = new Map<string, number>();
    const preToolTurns: Array<number | undefined> = [];
    const hookRegistry = new HookRegistry();
    hookRegistry.register('PreToolUse', {
      hooks: [
        async (
          input: PreToolUseHookInput
        ): Promise<PreToolUseHookOutput> => {
          preToolTurns.push(input.turn);
          return { decision: 'allow' };
        },
      ],
    });

    const toolNode = new ToolNode({
      tools: [createDummyTool('weather')],
      eventDrivenMode: true,
      eagerEventToolExecution: { enabled: true },
      eagerEventToolExecutions: new Map(),
      eagerEventToolUsageCount: eagerUsageCount,
      hookRegistry,
      toolCallStepIds: new Map([
        ['call_weather_1', 'step_weather_1'],
        ['call_weather_2', 'step_weather_2'],
      ]),
    });

    await toolNode.invoke({
      messages: [
        createAIMessage('call_weather_1', 'weather', { city: 'NYC' }),
      ],
    });

    eagerUsageCount.clear();

    await toolNode.invoke({
      messages: [
        createAIMessage('call_weather_2', 'weather', { city: 'Boston' }),
      ],
    });

    expect(preToolTurns).toEqual([0, 1]);
    expect(toolExecuteCalls.map((call) => call.toolCalls[0].turn)).toEqual([
      0, 1,
    ]);
    expect(eagerUsageCount.size).toBe(0);
  });
});
