// src/tools/handoff.ts
import { z } from 'zod';
import { tool, type DynamicStructuredTool } from '@langchain/core/tools';
import { ToolMessage } from '@langchain/core/messages';
import { Command, getCurrentTaskInput } from '@langchain/langgraph';
import type { MessagesAnnotation } from '@langchain/langgraph';

interface CreateHandoffToolParams {
  agentName: string;
  description?: string;
}

/**
 * Creates a handoff tool that properly routes to another agent using LangGraph Commands
 */
export function createHandoffTool({
  agentName,
  description,
}: CreateHandoffToolParams): DynamicStructuredTool {
  const toolName = `transfer_to_${agentName}`;
  const toolDescription =
    description ?? `Transfer control to the ${agentName} agent`;

  return tool(
    async (_, config) => {
      const toolMessage = new ToolMessage({
        content: `Successfully transferred to ${agentName}`,
        name: toolName,
        tool_call_id: config.toolCall.id,
      });

      // Get current state from the graph context
      const state = getCurrentTaskInput() as typeof MessagesAnnotation.State;

      // Return a Command to navigate to the target agent
      return new Command({
        goto: agentName,
        update: {
          messages: state.messages.concat(toolMessage),
        },
        // Indicate we're navigating within the parent graph
        graph: Command.PARENT,
      });
    },
    {
      name: toolName,
      schema: z.object({}),
      description: toolDescription,
    }
  );
}
