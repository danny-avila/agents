import {
  getChatModelClass,
  getRegisteredChatModelClass,
  registerChatModelProvider,
} from './providers';

class HostProvider {
  constructor(public readonly config: unknown) {}
}

describe('host provider registry', () => {
  it('registers and resolves a host provider', () => {
    const provider = `test-provider-${Date.now()}`;
    registerChatModelProvider(provider, HostProvider as never);
    expect(getRegisteredChatModelClass(provider)).toBe(HostProvider);
  });

  it('rejects duplicate registration', () => {
    const provider = `duplicate-provider-${Date.now()}`;
    registerChatModelProvider(provider, HostProvider as never);
    expect(() => registerChatModelProvider(provider, HostProvider as never)).toThrow(
      `Provider already registered: ${provider}`,
    );
  });

  it('rejects invalid names and constructors', () => {
    expect(() => registerChatModelProvider('', HostProvider as never)).toThrow(
      'Provider name must be a non-empty string',
    );
    expect(() => registerChatModelProvider('invalid-provider', null as never)).toThrow(
      'Provider constructor is invalid: invalid-provider',
    );
  });

  it('preserves built-in lookup and unsupported-provider errors', () => {
    expect(getChatModelClass('openAI')).toBeDefined();
    expect(() => getRegisteredChatModelClass(`missing-${Date.now()}`)).toThrow(
      /Unsupported LLM provider/,
    );
  });
});
