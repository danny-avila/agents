// src/messages/injected.ts
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { InjectedMessage } from '@/types/tools';
import { toLangChainContent } from './langchain';

/**
 * Converts `InjectedMessage` instances to LangChain `HumanMessage` objects.
 * Both 'user' and 'system' roles become `HumanMessage` to avoid provider
 * rejections (Anthropic/Google reject non-leading SystemMessages). The
 * original role is preserved in `additional_kwargs` for downstream consumers.
 *
 * Shared by both injection boundaries — `ToolNode`'s tool-batch dispatch and
 * `StandardGraph`'s preempt-boundary dispatch. Keeping one implementation is
 * load-bearing rather than tidy: the provider-safety argument for sealing a
 * stream mid-generation rests on the injected turn having byte-identical
 * shape to the one the already-shipped tool boundary emits, so the two sites
 * must not be able to drift.
 */
export function convertInjectedMessages(
  messages: InjectedMessage[]
): BaseMessage[] {
  const converted: BaseMessage[] = [];
  for (const msg of messages) {
    const additional_kwargs: Record<string, unknown> = {
      role: msg.role,
    };
    if (msg.isMeta != null) additional_kwargs.isMeta = msg.isMeta;
    if (msg.source != null) additional_kwargs.source = msg.source;
    if (msg.skillName != null) additional_kwargs.skillName = msg.skillName;

    converted.push(
      new HumanMessage({
        content: toLangChainContent(msg.content),
        additional_kwargs,
      })
    );
  }
  return converted;
}
