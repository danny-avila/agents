import { tool, type StructuredToolInterface } from '@langchain/core/tools';
import type { LCTool } from '@/types';

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
