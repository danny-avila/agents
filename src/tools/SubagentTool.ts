import type { JsonSchemaType, LCTool } from '@/types/tools';
import type { SubagentConfig } from '@/types';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { Constants } from '@/common';

export const SubagentToolName = Constants.SUBAGENT;

export const SubagentToolDescription = `Delegate a task to a specialized subagent or bounded agent team that runs in an isolated context window. The delegated execution returns only its designated final text result — all intermediate tool calls, reasoning, and context stay isolated.

WHEN TO USE:
- The task is self-contained and can be described in a single prompt.
- You want to offload verbose or exploratory work without bloating your own context.
- A specialized subagent is available for the task domain.

WHAT HAPPENS:
- A fresh agent or configured agent graph is created with the task description as its only input.
- The delegated agent or team runs to completion using isolated tools and context.
- Only the single agent's final response or the graph's designated result-agent response is returned to you.

CONSTRAINTS:
- subagent_type must match one of the available types listed below.
- The subagent cannot see your conversation history.`;

const DESCRIPTION_PROP_DESCRIPTION =
  'Complete task description for the subagent. This is the ONLY information it receives — include all necessary context, requirements, and constraints.';

const SUBAGENT_TYPE_PROP_DESCRIPTION =
  'Which subagent type to delegate to. Must be one of the available types.';

const RUN_IN_BACKGROUND_PROP_DESCRIPTION =
  'Set true to start the subagent as a detached process-local task and return a background_task_id immediately. Poll the host background-task tool to collect its result. The task can outlive this turn but does not survive a process restart.';

export const SubagentToolSchema = {
  type: 'object',
  properties: {
    intent: { ...INTENT_PROPERTY },
    description: {
      type: 'string',
      description: DESCRIPTION_PROP_DESCRIPTION,
    },
    subagent_type: {
      type: 'string',
      description: SUBAGENT_TYPE_PROP_DESCRIPTION,
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
 * Extends `SubagentToolSchema` by populating `subagent_type.enum` dynamically.
 */
export function buildSubagentToolParams(
  configs: SubagentConfig[],
  options: { background?: boolean } = {}
): {
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
        intent: { ...INTENT_PROPERTY },
        description: {
          type: 'string',
          description: DESCRIPTION_PROP_DESCRIPTION,
        },
        subagent_type: {
          type: 'string',
          enum: types,
          description: `${SUBAGENT_TYPE_PROP_DESCRIPTION} Available: ${types.join(', ')}.`,
        },
        ...(options.background === true
          ? {
            run_in_background: {
              type: 'boolean',
              description: RUN_IN_BACKGROUND_PROP_DESCRIPTION,
            },
          }
          : {}),
      },
      required: ['description', 'subagent_type'],
    },
    description: `${SubagentToolDescription}${
      options.background === true
        ? '\n\nBACKGROUND EXECUTION:\n- Set run_in_background to true when you do not need the result immediately. The call returns a background_task_id; use the host background-task tools to poll, steer, queue, interrupt, or cancel it.'
        : ''
    }\n\nAvailable types:\n${typeDescriptions}`,
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
