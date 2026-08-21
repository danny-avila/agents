/** Provider/SDK block types whose payload bytes are tool-produced. Keep this
 * explicit: suffix matching would let untrusted content invent tool authority. */
const PROVIDER_TOOL_RESULT_TYPES: ReadonlySet<string> = new Set([
  'advisor_tool_result',
  'bash_code_execution_tool_result',
  'codeExecutionResult',
  'code_execution_tool_result',
  'mcp_tool_result',
  'search_result',
  'server_tool_call_result',
  'server_tool_result',
  'text_editor_code_execution_tool_result',
  'tool_result',
  'tool_search_tool_result',
  'toolResponse',
  'toolResult',
  'web_fetch_tool_result',
  'web_search_result',
  'web_search_tool_result',
]);

export function isProviderToolResultType(type: unknown): boolean {
  return typeof type === 'string' && PROVIDER_TOOL_RESULT_TYPES.has(type);
}

export function isProviderToolResultPart(part: unknown): boolean {
  return (
    part != null &&
    typeof part === 'object' &&
    'type' in part &&
    isProviderToolResultType(part.type)
  );
}
