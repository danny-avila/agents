import { Constants } from '@/common';
import type { SubagentConfig } from '@/types';
import type { JsonSchemaType, LCTool } from '@/types/tools';

export const SubagentToolName = Constants.SUBAGENT;

export const SubagentToolDescription = `Delegate a task to a specialized subagent that runs in an isolated context window. The subagent executes independently and returns only its final text result — all intermediate tool calls, reasoning, and context stay isolated.

WHEN TO USE:
- The task is self-contained and can be described in a single prompt.
- You want to offload verbose or exploratory work without bloating your own context.
- A specialized subagent is available for the task domain.

WHAT HAPPENS:
- A fresh agent is created with the task description as its only input.
- The subagent runs to completion using its own tools and context.
- Only the final text response is returned to you.

CONSTRAINTS:
- subagent_type must match one of the available types listed below.
- The subagent cannot see your conversation history.`;

export const SubagentToolSchema = {
  type: 'object',
  properties: {
    description: {
      type: 'string',
      description:
        'Complete task description for the subagent. This is the ONLY information it receives — include all necessary context, requirements, and constraints.',
    },
    subagent_type: {
      type: 'string',
      description:
        'Which subagent type to delegate to. Must be one of the available types.',
    },
  },
  required: ['description', 'subagent_type'] as string[],
} as const;

export const SubagentToolDefinition: LCTool = {
  name: SubagentToolName,
  description: SubagentToolDescription,
  parameters: SubagentToolSchema,
};

/**
 * Build the name, schema, and description params for `tool()` from available configs.
 * Used by `Graph.createAgentNode()` when constructing the runtime tool instance.
 */
export function buildSubagentToolParams(configs: SubagentConfig[]): {
  name: string;
  schema: JsonSchemaType;
  description: string;
} {
  const types = configs.map((c) => c.type);
  const typeDescriptions = configs
    .map((c) => `- "${c.type}" (${c.name}): ${c.description}`)
    .join('\n');

  return {
    name: SubagentToolName,
    schema: {
      type: 'object',
      properties: {
        description: {
          type: 'string',
          description:
            'Complete task description for the subagent. This is the ONLY information it receives — include all necessary context, requirements, and constraints.',
        },
        subagent_type: {
          type: 'string',
          enum: types,
          description: `Which subagent type to delegate to. Available: ${types.join(', ')}.`,
        },
      },
      required: ['description', 'subagent_type'],
    },
    description: `${SubagentToolDescription}\n\nAvailable types:\n${typeDescriptions}`,
  };
}

/**
 * Create a SubagentTool LCTool definition with dynamic enum and description
 * populated from the available subagent configs.
 * Used for the tool registry in event-driven mode.
 */
export function createSubagentToolDefinition(
  configs: SubagentConfig[]
): LCTool {
  const params = buildSubagentToolParams(configs);
  return {
    name: params.name,
    description: params.description,
    parameters: params.schema,
  };
}
