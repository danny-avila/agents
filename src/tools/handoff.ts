// src/tools/handoff.ts
import { z } from 'zod';
import { tool, type DynamicStructuredTool } from '@langchain/core/tools';
import { Command } from '@langchain/langgraph';

/**
 * Minimal supervised handoff tool that emits LangGraph Commands.
 * 
 * Note: Current graph defines nodes: 'router', 'agent', 'tools'.
 * Returning a Command with { goto: target } will route to that node.
 */
const handoffSchema = z.object({
  target: z.enum(['router', 'agent', 'tools']).describe('Destination node.'),
  mode: z
    .enum(['one_way', 'call_return', 'fan_out'])
    .describe('Handoff mode. For now, all modes route to the target.'),
  args: z.unknown().optional().describe('Optional payload for future routing logic.'),
});

export type HandoffParams = z.infer<typeof handoffSchema>;

export function createHandoffTool(): DynamicStructuredTool<typeof handoffSchema> {
  return tool<typeof handoffSchema>(
    async (params) => {
      const { target } = params;
      // For initial implementation, simply jump to the target node
      return new Command({ goto: target });
    },
    {
      name: 'handoff',
      description:
        'Route control flow to another node within the supervised graph (router/agent/tools). Returns a LangGraph Command.',
      schema: handoffSchema,
    }
  );
}

 