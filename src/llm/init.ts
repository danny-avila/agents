import type { Runnable } from '@langchain/core/runnables';
import type * as t from '@/types';
import { getChatModelClass } from '@/llm/providers';
import { requireInternalModule, requireLazyModule } from '@/lazyRequire';
import { isOpenAILike } from '@/utils';
import { Providers } from '@/common';

type InitializeModelParams<P extends t.ProviderName> = {
  provider: P;
  tools?: t.GraphTools;
} & (
  | {
      override: t.ChatModelInstance;
      clientOptions?: t.ProviderOptionsFor<P>;
    }
  | ([P] extends [keyof t.ProviderOptionsMap]
      ? {
          override?: t.ChatModelInstance;
          clientOptions?: t.ProviderOptionsFor<P>;
        }
      : object extends t.ProviderOptionsFor<P>
        ? {
            override?: t.ChatModelInstance;
            clientOptions?: t.ProviderOptionsFor<P>;
          }
        : {
            override?: undefined;
            clientOptions: t.ProviderOptionsFor<P>;
          })
);

/** These guards run only for their own provider families, so the class they test
 *  against loads with the family's first request instead of at import time. */
function isOpenAIChatModel(
  model: unknown
): model is import('@/llm/openai').ChatOpenAI | import('@/llm/openai').AzureChatOpenAI {
  const { ChatOpenAI, AzureChatOpenAI } =
    requireInternalModule<typeof import('@/llm/openai')>('llm/openai/index');
  return model instanceof ChatOpenAI || model instanceof AzureChatOpenAI;
}

function isLangchainVertexModel(
  model: unknown
): model is import('@langchain/google-vertexai').ChatVertexAI {
  const { ChatVertexAI } = requireLazyModule<typeof import('@langchain/google-vertexai')>(
    '@langchain/google-vertexai'
  );
  return model instanceof ChatVertexAI;
}

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
}: InitializeModelParams<P>): Runnable {
  const model =
    override ??
    new (getChatModelClass(provider))(
      (clientOptions ?? {}) as t.ProviderOptionsFor<P>
    );

  if (isOpenAILike(provider) && isOpenAIChatModel(model)) {
    const opts = clientOptions as t.OpenAIClientOptions | undefined;
    if (opts) {
      model.temperature = opts.temperature as number;
      model.topP = opts.topP as number;
      model.frequencyPenalty = opts.frequencyPenalty as number;
      model.presencePenalty = opts.presencePenalty as number;
      model.n = opts.n as number;
    }
  } else if (provider === Providers.VERTEXAI && isLangchainVertexModel(model)) {
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
