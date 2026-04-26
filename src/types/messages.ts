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
  /** Key under which this message's untruncated output was registered. */
  _refKey?: string;
  /** Placeholders the model used that could not be resolved this batch. */
  _unresolvedRefs?: string[];
}
