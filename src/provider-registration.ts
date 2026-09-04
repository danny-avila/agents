import type { BaseChatModel } from '@langchain/core/language_models/chat_models';

declare const CUSTOM_PROVIDER_OPTIONS_TYPE: unique symbol;

/** Declaration-merge this map to type host-registered provider options. */
export interface CustomProviderOptionsMap {
  readonly [CUSTOM_PROVIDER_OPTIONS_TYPE]?: never;
}

export type ProviderFamily =
  | 'openai'
  | 'anthropic'
  | 'bedrock'
  | 'google'
  | 'mistral'
  | 'generic';

export interface ProviderRegistrationOptions<
  TOptions extends object,
  TModel extends BaseChatModel,
> {
  provider: string;
  model: new (config: TOptions) => TModel;
  family?: ProviderFamily;
  manualToolStream?: boolean;
  strictAlternation?: boolean;
}

export { registerProvider } from './llm/providers';
