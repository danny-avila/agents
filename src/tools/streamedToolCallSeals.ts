export const STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY =
  'lc_streamed_tool_call_adapter';
export const STREAMED_TOOL_CALL_SEAL_METADATA_KEY =
  'lc_streamed_tool_call_seal';
export const OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER = 'openai_responses';
export const BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER = 'bedrock_converse';
export const GOOGLE_STREAMED_TOOL_CALL_ADAPTER = 'google_genai';

export type StreamedToolCallAdapter =
  | typeof OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER
  | typeof BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER
  | typeof GOOGLE_STREAMED_TOOL_CALL_ADAPTER;

const STREAMED_TOOL_CALL_ADAPTERS: ReadonlySet<string> = new Set([
  OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER,
  BEDROCK_CONVERSE_STREAMED_TOOL_CALL_ADAPTER,
  GOOGLE_STREAMED_TOOL_CALL_ADAPTER,
]);

export type StreamedToolCallSeal =
  | {
      kind: 'single';
      id?: string;
      index?: number;
    }
  | {
      kind: 'all';
    };

export function getStreamedToolCallAdapter(
  metadata: Record<string, unknown> | undefined
): StreamedToolCallAdapter | undefined {
  const adapter = metadata?.[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY];
  if (typeof adapter === 'string' && STREAMED_TOOL_CALL_ADAPTERS.has(adapter)) {
    return adapter as StreamedToolCallAdapter;
  }
  return undefined;
}

export function getStreamedToolCallSeal(
  metadata: Record<string, unknown> | undefined
): StreamedToolCallSeal | undefined {
  const seal = metadata?.[STREAMED_TOOL_CALL_SEAL_METADATA_KEY];
  if (seal == null || typeof seal !== 'object') {
    return undefined;
  }
  if (!('kind' in seal)) {
    return undefined;
  }
  if (seal.kind === 'all') {
    return { kind: 'all' };
  }
  if (seal.kind !== 'single') {
    return undefined;
  }
  const id = 'id' in seal && typeof seal.id === 'string' ? seal.id : undefined;
  const index =
    'index' in seal && typeof seal.index === 'number' ? seal.index : undefined;
  if (id == null && index == null) {
    return undefined;
  }
  return { kind: 'single', id, index };
}
