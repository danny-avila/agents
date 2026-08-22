import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { ToolNode } from '../ToolNode';
import * as events from '@/utils/events';
import { Constants, GraphEvents } from '@/common';

describe('ToolNode programmatic caller policy', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('separates programmatic tools from direct-only preflight definitions', async () => {
    const capturedConfigs: Array<ToolCall & Partial<t.ProgrammaticCache>> = [];
    const ptcTool = tool(
      async (_args, config) => {
        capturedConfigs.push(
          config.toolCall as ToolCall & Partial<t.ProgrammaticCache>
        );
        return 'done';
      },
      {
        name: Constants.PROGRAMMATIC_TOOL_CALLING,
        description: 'Run tools with code',
        schema: z.object({ code: z.string() }),
      }
    );
    const programmaticTool = tool(async () => 'programmatic', {
      name: 'programmatic_tool',
      description: 'Programmatic tool',
      schema: z.object({}),
    });
    const directTool = tool(async () => 'direct', {
      name: 'direct_tool',
      description: 'Direct tool',
      schema: z.object({}),
    });
    const toolRegistry: t.LCToolRegistry = new Map([
      [
        programmaticTool.name,
        {
          name: programmaticTool.name,
          allowed_callers: ['code_execution'],
        },
      ],
      [directTool.name, { name: directTool.name, allowed_callers: ['direct'] }],
    ]);
    const node = new ToolNode({
      tools: [ptcTool, programmaticTool, directTool],
      toolRegistry,
    });

    await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'ptc-call',
              name: Constants.PROGRAMMATIC_TOOL_CALLING,
              args: { code: 'print("done")' },
            },
          ],
        }),
      ],
    });

    expect(capturedConfigs).toHaveLength(1);
    expect(capturedConfigs[0].toolDefs).toEqual([
      { name: 'programmatic_tool', allowed_callers: ['code_execution'] },
    ]);
    expect(capturedConfigs[0].disallowedToolDefs).toEqual([
      { name: 'direct_tool' },
    ]);
    expect(capturedConfigs[0].toolMap).toEqual(
      new Map([['programmatic_tool', programmaticTool]])
    );
    expect(capturedConfigs[0].programmaticToolName).toBe(
      Constants.PROGRAMMATIC_TOOL_CALLING
    );
  });

  it('projects deferred policy definitions only after discovery', async () => {
    const capturedConfigs: Array<ToolCall & Partial<t.ProgrammaticCache>> = [];
    const discovered = new Set<string>();
    const ptcTool = tool(
      async (_args, config) => {
        capturedConfigs.push(
          config.toolCall as ToolCall & Partial<t.ProgrammaticCache>
        );
        return 'done';
      },
      {
        name: Constants.PROGRAMMATIC_TOOL_CALLING,
        description: 'Run tools with code',
        schema: z.object({ code: z.string() }),
      }
    );
    const directTool = tool(async () => 'direct', {
      name: 'deferred_direct_tool',
      description: 'Deferred direct tool',
      schema: z.object({}),
    });
    const node = new ToolNode({
      tools: [ptcTool, directTool],
      toolRegistry: new Map([
        [
          directTool.name,
          {
            name: directTool.name,
            allowed_callers: ['direct'],
            defer_loading: true,
          },
        ],
      ]),
      getDiscoveredToolNames: () => [...discovered],
    });
    const invoke = async (id: string): Promise<void> => {
      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [
              {
                id,
                name: Constants.PROGRAMMATIC_TOOL_CALLING,
                args: { code: 'print("done")' },
              },
            ],
          }),
        ],
      });
    };

    await invoke('before-discovery');
    discovered.add(directTool.name);
    await invoke('after-discovery');

    expect(capturedConfigs[0].disallowedToolDefs).toEqual([]);
    expect(capturedConfigs[1].disallowedToolDefs).toEqual([
      { name: directTool.name },
    ]);
  });

  it('dispatches the same live caller projection in event-driven mode', async () => {
    const discovered = new Set<string>();
    const snapshots: t.CallerCapabilityProjectionSnapshot[] = [];
    jest
      .spyOn(events, 'safeDispatchCustomEvent')
      .mockImplementation(async (event, data): Promise<void> => {
        if (event !== GraphEvents.ON_TOOL_EXECUTE) {
          return;
        }
        const batch = data as t.ToolExecuteBatchRequest;
        snapshots.push(batch.callerCapabilityProjection!);
        batch.resolve(
          batch.toolCalls.map((toolCall) => ({
            toolCallId: toolCall.id,
            status: 'success',
            content: 'done',
          }))
        );
      });

    const eventTool = tool(async () => 'should dispatch', {
      name: 'event_tool',
      description: 'Event tool',
      schema: z.object({}),
    });
    const toolDefinitions: t.LCToolRegistry = new Map([
      [eventTool.name, { name: eventTool.name }],
      [
        'deferred_programmatic_tool',
        {
          name: 'deferred_programmatic_tool',
          allowed_callers: ['code_execution'],
          defer_loading: true,
        },
      ],
    ]);
    const toolRegistry: t.LCToolRegistry = new Map([
      [
        'deferred_direct_tool',
        {
          name: 'deferred_direct_tool',
          allowed_callers: ['direct'],
          defer_loading: true,
        },
      ],
    ]);
    const node = new ToolNode({
      tools: [eventTool],
      toolRegistry,
      toolDefinitions,
      eventDrivenMode: true,
      getDiscoveredToolNames: () => [...discovered],
      toolCallStepIds: new Map([
        ['before-discovery', 'step-before'],
        ['after-discovery', 'step-after'],
      ]),
    });
    const invoke = async (id: string): Promise<void> => {
      await node.invoke({
        messages: [
          new AIMessage({
            content: '',
            tool_calls: [{ id, name: eventTool.name, args: {} }],
          }),
        ],
      });
    };

    await invoke('before-discovery');
    discovered.add('deferred_programmatic_tool');
    discovered.add('deferred_direct_tool');
    await invoke('after-discovery');

    expect(snapshots).toEqual([
      {
        version: 1,
        directToolNames: ['event_tool'],
        codeExecutionToolNames: [],
        directOnlyToolNames: ['event_tool'],
        codeExecutionOnlyToolNames: [],
      },
      {
        version: 1,
        directToolNames: ['event_tool', 'deferred_direct_tool'],
        codeExecutionToolNames: ['deferred_programmatic_tool'],
        directOnlyToolNames: ['event_tool', 'deferred_direct_tool'],
        codeExecutionOnlyToolNames: ['deferred_programmatic_tool'],
      },
    ]);
  });

  it('keeps schema-only event definitions out of a direct runner cache', async () => {
    const capturedConfigs: Array<ToolCall & Partial<t.ProgrammaticCache>> = [];
    const ptcTool = tool(
      async (_args, config) => {
        capturedConfigs.push(
          config.toolCall as ToolCall & Partial<t.ProgrammaticCache>
        );
        return 'done';
      },
      {
        name: Constants.PROGRAMMATIC_TOOL_CALLING,
        description: 'Run tools with code',
        schema: z.object({ code: z.string() }),
      }
    );
    const eventStub = tool(async () => 'schema stub should not run', {
      name: 'event_programmatic_tool',
      description: 'Event programmatic tool',
      schema: z.object({}),
    });
    const node = new ToolNode({
      tools: [ptcTool, eventStub],
      eventDrivenMode: true,
      directToolNames: new Set([Constants.PROGRAMMATIC_TOOL_CALLING]),
      toolDefinitions: new Map([
        [
          eventStub.name,
          {
            name: eventStub.name,
            allowed_callers: ['code_execution'],
          },
        ],
      ]),
    });

    await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            {
              id: 'local-ptc',
              name: Constants.PROGRAMMATIC_TOOL_CALLING,
              args: { code: 'event_programmatic_tool "{}"' },
            },
          ],
        }),
      ],
    });

    expect(capturedConfigs).toHaveLength(1);
    expect(capturedConfigs[0].toolDefs).toEqual([]);
    expect(capturedConfigs[0].toolMap).toEqual(new Map());
  });
});
