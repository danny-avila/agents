import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import type { ToolCall } from '@langchain/core/messages/tool';
import type * as t from '@/types';
import { ToolNode } from '../ToolNode';
import { Constants } from '@/common';

describe('ToolNode programmatic caller policy', () => {
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
});
