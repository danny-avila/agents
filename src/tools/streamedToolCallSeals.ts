export const STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY =
  'lc_streamed_tool_call_adapter';
export const STREAMED_TOOL_CALL_SEAL_METADATA_KEY =
  'lc_streamed_tool_call_seal';
export const OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER = 'openai_responses';

export type StreamedToolCallAdapter =
  typeof OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER;

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
  if (
    metadata?.[STREAMED_TOOL_CALL_ADAPTER_METADATA_KEY] ===
    OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER
  ) {
    return OPENAI_RESPONSES_STREAMED_TOOL_CALL_ADAPTER;
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
    'index' in seal && typeof seal.index === 'number'
      ? seal.index
      : undefined;
  if (id == null && index == null) {
    return undefined;
  }
  return { kind: 'single', id, index };
}
