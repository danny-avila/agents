import { context } from '@opentelemetry/api';
import {
  getLangfuseRuntimeConfig,
  getLangfuseRuntimeToolOutputTracingConfig,
  getTraceIdSeed,
  runWithLangfuseRuntimeContext,
  runWithTraceIdSeed,
  traceIdFromSeed,
} from '@/langfuseRuntimeContext';
import {
  resolveTraceIdSeedForSpan,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';

describe('Langfuse runtime context', () => {
  it('exposes tenant config and deterministic trace seed inside the scope', () => {
    const langfuse = {
      publicKey: 'pk-runtime',
      secretKey: 'sk-runtime',
      baseUrl: 'https://langfuse.runtime',
    };

    runWithLangfuseRuntimeContext(
      { langfuse, traceIdSeed: 'run-runtime' },
      () => {
        expect(getLangfuseRuntimeConfig()).toBe(langfuse);
        expect(getTraceIdSeed()).toBe('run-runtime');
      }
    );

    expect(getLangfuseRuntimeConfig()).toBeUndefined();
    expect(getTraceIdSeed()).toBeUndefined();
  });

  it('merges nested scopes without dropping inherited tenant config', () => {
    const langfuse = {
      publicKey: 'pk-parent',
      secretKey: 'sk-parent',
      baseUrl: 'https://langfuse.parent',
    };

    runWithLangfuseRuntimeContext(
      { langfuse, traceIdSeed: 'parent-run' },
      () => {
        runWithTraceIdSeed('child-run', () => {
          expect(getLangfuseRuntimeConfig()).toBe(langfuse);
          expect(getTraceIdSeed()).toBe('child-run');
        });

        expect(getLangfuseRuntimeConfig()).toBe(langfuse);
        expect(getTraceIdSeed()).toBe('parent-run');
      }
    );
  });

  it('carries resolved tool-output tracing config in the same runtime scope', () => {
    const toolOutputTracing = {
      enabled: false,
      redactedToolNames: new Set(['search']),
      redactedToolNameMatchMode: 'exact' as const,
      redactionText: '[redacted]',
    };

    runWithLangfuseRuntimeContext({ toolOutputTracing }, () => {
      expect(getLangfuseRuntimeToolOutputTracingConfig()).toBe(
        toolOutputTracing
      );
    });

    expect(getLangfuseRuntimeToolOutputTracingConfig()).toBeUndefined();
  });

  it('also exposes deterministic trace seeds through OTel context', () => {
    withLangfuseRuntimeScope({ traceIdSeed: 'run-otel' }, () => {
      expect(resolveTraceIdSeedForSpan(context.active())).toBe('run-otel');
    });

    expect(resolveTraceIdSeedForSpan(context.active())).toBeUndefined();
  });

  it('ignores empty trace seeds', () => {
    const result = runWithTraceIdSeed(' ', () => getTraceIdSeed());

    expect(result).toBeUndefined();
  });

  it('derives Langfuse-compatible deterministic trace ids', () => {
    expect(traceIdFromSeed('run-runtime')).toBe(
      '5b8a8af5718b2eba96b83a2b8fbfa7f4'
    );
  });
});
