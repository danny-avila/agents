import { tool, type StructuredToolInterface } from '@langchain/core/tools';
import type { LCTool } from '@/types';

export function getToolBindingName(toolBinding: unknown): string | undefined {
  if (
    toolBinding == null ||
    (typeof toolBinding !== 'object' && typeof toolBinding !== 'function')
  ) {
    return undefined;
  }
  const candidate = toolBinding as {
    name?: unknown;
    function?: { name?: unknown };
    toolSpec?: { name?: unknown };
  };
  if (typeof candidate.name === 'string') {
    return candidate.name;
  }
  if (typeof candidate.function?.name === 'string') {
    return candidate.function.name;
  }
  return typeof candidate.toolSpec?.name === 'string'
    ? candidate.toolSpec.name
    : undefined;
}

/**
 * Creates a schema-only tool for LLM binding in event-driven mode.
 * These tools have valid schemas for the LLM to understand but should
 * never be invoked directly - ToolNode handles execution via events.
 */
export function createSchemaOnlyTool(
  definition: LCTool
): StructuredToolInterface {
  const { name, description, parameters, responseFormat } = definition;

  return tool(
    async () => {
      throw new Error(
        `Tool "${name}" should not be invoked directly in event-driven mode. ` +
          'ToolNode should dispatch ON_TOOL_EXECUTE events instead.'
      );
    },
    {
      name,
      description: description ?? '',
      schema: parameters ?? { type: 'object', properties: {} },
      responseFormat: responseFormat ?? 'content_and_artifact',
    }
  );
}

/**
 * Creates schema-only tools for all definitions in an array.
 */
export function createSchemaOnlyTools(
  definitions: LCTool[]
): StructuredToolInterface[] {
  return definitions.map((def) => createSchemaOnlyTool(def));
}
