import type Anthropic from '@anthropic-ai/sdk';
import type { BaseMessage } from '@langchain/core/messages';
export type AnthropicMessages = Array<AnthropicMessage | BaseMessage>;
export type AnthropicMessage = Anthropic.MessageParam;

/**
 * Per-message ref metadata stamped onto a `ToolMessage` at execution
 * time. Read by `annotateMessagesForLLM` to apply transient annotation
 * to a copy of the message right before it goes on the wire to the
 * provider. Never read after the run-scoped registry has been cleared.
 *
 * Lives in `ToolMessage.additional_kwargs`. LangChain's provider
 * serializers don't transmit `additional_kwargs` to provider APIs, so
 * the metadata never leaks even if you forget to clean it.
 */
export interface ToolMessageRefMetadata {
  /** Key under which this message's selected output snapshot was registered. */
  _refKey?: string;
  /**
   * Registry bucket scope under which `_refKey` was stored. For named
   * runs this equals `config.configurable.run_id`; for anonymous
   * invocations (no `run_id`) ToolNode mints a per-batch synthetic
   * scope (`\0anon-<n>`) so concurrent batches don't collide. Stamping
   * the scope on the message itself lets `annotateMessagesForLLM`
   * recover it without re-deriving from config — which is impossible
   * for the anonymous case, since the scope is internal to ToolNode.
   */
  _refScope?: string;
  /** Placeholders the model used that could not be resolved this batch. */
  _unresolvedRefs?: string[];
}
