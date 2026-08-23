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

interface ProviderRegistryGlobal {
  [key: symbol]: Map<ProviderName, StoredProviderRegistration> | undefined;
}

const PROVIDER_REGISTRY_KEY = Symbol.for(
  '@librechat/agents:providerRegistry:v1'
);
const providerRegistryGlobal = globalThis as ProviderRegistryGlobal;
const registeredProviders =
  providerRegistryGlobal[PROVIDER_REGISTRY_KEY] ??
  new Map<ProviderName, StoredProviderRegistration>();
providerRegistryGlobal[PROVIDER_REGISTRY_KEY] = registeredProviders;
const builtInProviders = new Map<ProviderName, StoredProviderRegistration>();

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

function createRegistration<
  TOptions extends object,
  TModel extends BaseChatModel,
>(
  provider: ProviderName,
  options: ProviderRegistrationOptions<TOptions, TModel>
): StoredProviderRegistration {
  if (typeof options.model !== 'function' || !isConstructible(options.model)) {
    throw new TypeError(`LLM provider constructor is invalid: ${provider}`);
  }
  return {
    model: options.model,
    family: options.family ?? 'generic',
    manualToolStream: options.manualToolStream ?? false,
    strictAlternation: options.strictAlternation ?? false,
    owner: Symbol(provider),
  };
}

function getRegistration(
  provider: ProviderName
): StoredProviderRegistration | undefined {
  return registeredProviders.get(provider) ?? builtInProviders.get(provider);
}

/** Registers one host provider until the returned disposer is called. */
export function registerProvider<
  TOptions extends object,
  TModel extends BaseChatModel,
>(options: ProviderRegistrationOptions<TOptions, TModel>): () => void {
  const provider = normalizeProvider(options.provider);
  if (getRegistration(provider) != null) {
    throw new Error(`LLM provider already registered: ${provider}`);
  }

  const registration = createRegistration(provider, options);
  registeredProviders.set(provider, registration);

  return (): void => {
    if (registeredProviders.get(provider)?.owner === registration.owner) {
      registeredProviders.delete(provider);
    }
  };
}

/** Initializes one built-in for the current package module graph. */
export function registerBuiltInProvider<
  TOptions extends object,
  TModel extends BaseChatModel,
>(options: ProviderRegistrationOptions<TOptions, TModel>): void {
  const provider = normalizeProvider(options.provider);
  if (getRegistration(provider) != null) {
    throw new Error(`LLM provider already registered: ${provider}`);
  }
  builtInProviders.set(provider, createRegistration(provider, options));
}

export function getRegisteredChatModelClass<P extends ProviderName>(
  provider: P
): ProviderModelConstructor<P> {
  const registration = getRegistration(provider);
  if (!registration) {
    throw new Error(`Unsupported LLM provider: ${provider}`);
  }
  return registration.model as ProviderModelConstructor<P>;
}

export function getProviderFamily(
  provider: ProviderName
): ProviderFamily | undefined {
  return getRegistration(provider)?.family;
}

export function providerUsesManualToolStream(provider: ProviderName): boolean {
  return getRegistration(provider)?.manualToolStream ?? false;
}

export function providerRequiresStrictAlternation(
  provider: ProviderName
): boolean {
  return getRegistration(provider)?.strictAlternation ?? false;
}
