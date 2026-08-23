import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type {
  ProviderFamily,
  ProviderRegistrationOptions,
} from '../provider-registration';
import type { ProviderModelConstructor, ProviderName } from '@/types';

interface StoredProviderRegistration {
  model: new (config: never) => BaseChatModel;
  family: ProviderFamily;
  manualToolStream: boolean;
  strictAlternation: boolean;
  owner: symbol;
}

const providers = new Map<ProviderName, StoredProviderRegistration>();

function normalizeProvider(provider: ProviderName): ProviderName {
  if (typeof provider !== 'string' || provider.trim() === '') {
    throw new TypeError('LLM provider name must be a non-empty string');
  }
  if (provider !== provider.trim()) {
    throw new TypeError(
      'LLM provider name must not have surrounding whitespace'
    );
  }
  return provider;
}

function isConstructible<T extends abstract new (...args: never[]) => object>(
  value: T
): boolean {
  try {
    Reflect.construct(String, [], value);
    return true;
  } catch {
    return false;
  }
}

/** Registers one host provider until the returned disposer is called. */
export function registerProvider<
  TOptions extends object,
  TModel extends BaseChatModel,
>(options: ProviderRegistrationOptions<TOptions, TModel>): () => void {
  const provider = normalizeProvider(options.provider);
  if (typeof options.model !== 'function' || !isConstructible(options.model)) {
    throw new TypeError(`LLM provider constructor is invalid: ${provider}`);
  }
  if (providers.has(provider)) {
    throw new Error(`LLM provider already registered: ${provider}`);
  }

  const owner = Symbol(provider);
  providers.set(provider, {
    model: options.model,
    family: options.family ?? 'generic',
    manualToolStream: options.manualToolStream ?? false,
    strictAlternation: options.strictAlternation ?? false,
    owner,
  });

  return (): void => {
    if (providers.get(provider)?.owner === owner) {
      providers.delete(provider);
    }
  };
}

export function getRegisteredChatModelClass<P extends ProviderName>(
  provider: P
): ProviderModelConstructor<P> {
  const registration = providers.get(provider);
  if (!registration) {
    throw new Error(`Unsupported LLM provider: ${provider}`);
  }
  return registration.model as ProviderModelConstructor<P>;
}

export function getProviderFamily(
  provider: ProviderName
): ProviderFamily | undefined {
  return providers.get(provider)?.family;
}

export function providerUsesManualToolStream(provider: ProviderName): boolean {
  return providers.get(provider)?.manualToolStream ?? false;
}

export function providerRequiresStrictAlternation(
  provider: ProviderName
): boolean {
  return providers.get(provider)?.strictAlternation ?? false;
}
