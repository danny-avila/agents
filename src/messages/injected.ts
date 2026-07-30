// src/messages/injected.ts
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { InjectedMessage } from '@/types/tools';
import { toLangChainContent } from './langchain';
import { ContentTypes } from '@/common';

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
/**
 * True when an entry carries nothing a provider will accept as a turn.
 *
 * The public `InjectedMessage` type admits `content: ''` and `content: []`,
 * and a host hook is free to return one. Converting it anyway produces a
 * `HumanMessage` that passes the caller's `length > 0` test, so the graph
 * resumes and sends a trailing EMPTY user turn — which Anthropic and other
 * strict providers reject outright, turning a cooperative seal into a failed
 * run. Permissive providers merely burn a model call.
 *
 * Whitespace counts as empty, matching the standard `canSealPreempt` applies
 * to the assistant side of the same pair. Non-text blocks count as content: a
 * media-only steer is a real turn and must survive.
 */
function isEmptyInjectedContent(content: InjectedMessage['content']): boolean {
  if (typeof content === 'string') {
    return content.trim() === '';
  }
  if (content.length === 0) {
    return true;
  }
  return content.every((block) => {
    if (block.type !== ContentTypes.TEXT) {
      return false;
    }
    const text = block[ContentTypes.TEXT];
    return typeof text !== 'string' || text.trim() === '';
  });
}

export function convertInjectedMessages(
  messages: InjectedMessage[]
): BaseMessage[] {
  const converted: BaseMessage[] = [];
  for (const msg of messages) {
    if (isEmptyInjectedContent(msg.content)) {
      continue;
    }
    /** Provenance, recorded here because this is the only place that knows it.
     *  `isMeta` and `source` are both optional on `InjectedMessage`, so a bare
     *  entry is otherwise indistinguishable from a message replayed out of the
     *  payload — and downstream consumers such as compaction coverage need to
     *  know that this message has no persisted source ID to name. Kept separate
     *  from `isMeta`, which carries UI and cache meaning of its own. */
    const additional_kwargs: Record<string, unknown> = {
      role: msg.role,
      injected: true,
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
