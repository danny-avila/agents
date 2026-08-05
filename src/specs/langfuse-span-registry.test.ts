import type { Span } from '@opentelemetry/api';
import type * as t from '@/types';
import {
  getLangfuseSpanProcessorParams,
  getLangfuseManagedSpanDestination,
  registerLangfuseManagedSpan,
  resolveLangfuseDestinationKey,
} from '@/langfuseSpanRegistry';

const ORIGINAL_ENV = { ...process.env };

function tenantConfig(
  overrides: Partial<t.LangfuseConfig> = {}
): t.LangfuseConfig {
  return {
    enabled: true,
    publicKey: 'pk-registry',
    secretKey: 'sk-registry',
    baseUrl: 'https://langfuse.registry',
    ...overrides,
  };
}

describe('Langfuse span registry', () => {
  afterEach(() => {
    process.env = { ...ORIGINAL_ENV };
  });

  it('tracks managed spans by destination', () => {
    const span = { name: 'managed' } as unknown as Span;
    const key = resolveLangfuseDestinationKey(tenantConfig());
    expect(key).toBeDefined();
    registerLangfuseManagedSpan(span, key as string);
    expect(getLangfuseManagedSpanDestination(span)).toBe(key);

    const untracked = { name: 'foreign' } as unknown as Span;
    expect(getLangfuseManagedSpanDestination(untracked)).toBeUndefined();
  });

  it('treats processor policy as irrelevant to destination identity', () => {
    const base = resolveLangfuseDestinationKey(tenantConfig());
    const redacting = resolveLangfuseDestinationKey(
      tenantConfig({ toolOutputTracing: { enabled: false } })
    );
    const mediaDisabled = resolveLangfuseDestinationKey(
      tenantConfig({ mediaUploadEnabled: false })
    );
    expect(redacting).toBe(base);
    expect(mediaDisabled).toBe(base);
  });

  it('passes media upload policy to the Langfuse span processor params', () => {
    expect(
      getLangfuseSpanProcessorParams(
        tenantConfig({ mediaUploadEnabled: false })
      )
    ).toEqual(
      expect.objectContaining({
        mediaUploadEnabled: false,
      })
    );
  });

  it('separates destinations by credentials, endpoint, and environment', () => {
    const base = resolveLangfuseDestinationKey(tenantConfig());
    expect(
      resolveLangfuseDestinationKey(tenantConfig({ publicKey: 'pk-other' }))
    ).not.toBe(base);
    expect(
      resolveLangfuseDestinationKey(
        tenantConfig({ baseUrl: 'https://langfuse.other' })
      )
    ).not.toBe(base);
    expect(
      resolveLangfuseDestinationKey(tenantConfig({ environment: 'staging' }))
    ).not.toBe(base);
  });

  it('resolves no destination without credentials', () => {
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    expect(resolveLangfuseDestinationKey(undefined)).toBeUndefined();
    expect(
      resolveLangfuseDestinationKey(tenantConfig({ enabled: false }))
    ).toBeUndefined();
  });
});
