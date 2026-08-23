import { ChatVertexAI } from '@langchain/google-vertexai';
import type { Runnable } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatOpenAI, AzureChatOpenAI } from '@/llm/openai';
import { getChatModelClass } from '@/llm/providers';
import { isOpenAILike } from '@/utils';
import { Providers } from '@/common';

/**
 * Creates a chat model instance for a given built-in or host-registered
 * provider, applies provider-specific field assignments, and optionally binds
 * tools.
 */
export function initializeModel<P extends t.ProviderName>({
  provider,
  clientOptions,
  tools,
  override,
}: {
  provider: P;
  clientOptions?: t.ProviderOptionsFor<P>;
  tools?: t.GraphTools;
  override?: t.ChatModelInstance;
}): Runnable {
  const model =
    override ??
    new (getChatModelClass(provider))(
      clientOptions ?? ({} as t.ProviderOptionsFor<P>)
    );

  if (
    isOpenAILike(provider) &&
    (model instanceof ChatOpenAI || model instanceof AzureChatOpenAI)
  ) {
    const opts = clientOptions as t.OpenAIClientOptions | undefined;
    if (opts) {
      model.temperature = opts.temperature as number;
      model.topP = opts.topP as number;
      model.frequencyPenalty = opts.frequencyPenalty as number;
      model.presencePenalty = opts.presencePenalty as number;
      model.n = opts.n as number;
    }
  } else if (provider === Providers.VERTEXAI && model instanceof ChatVertexAI) {
    const opts = clientOptions as t.VertexAIClientOptions | undefined;
    if (opts) {
      model.temperature = opts.temperature as number;
      model.topP = opts.topP as number;
      model.topK = opts.topK as number;
      model.topLogprobs = opts.topLogprobs as number;
      model.frequencyPenalty = opts.frequencyPenalty as number;
      model.presencePenalty = opts.presencePenalty as number;
      model.maxOutputTokens = opts.maxOutputTokens as number;
    }
  }

  if (!tools || tools.length === 0) {
    return model;
  }

  if (!('bindTools' in model) || typeof model.bindTools !== 'function') {
    throw new TypeError(
      `LLM provider does not support tool binding: ${provider}`
    );
  }

  return model.bindTools(tools);
}
