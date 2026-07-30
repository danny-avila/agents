import { createHash } from 'node:crypto';
import type { LangfuseSpanProcessorParams } from '@langfuse/otel';
import type { Span } from '@opentelemetry/api';
import type * as t from '@/types';
import {
  hasLangfuseConfigCredentials,
  hasLangfuseEnvCredentials,
  hasLangfuseEnvConfig,
} from '@/langfuseConfig';
import { isPresent } from '@/utils/misc';

/**
 * Spans created through the Langfuse tracer provider, keyed by the export
 * destination they were routed to. Callback handlers use this to distinguish
 * safe ambient parents from spans a root observation must not inherit:
 *
 * - Foreign spans (a host's own OpenTelemetry instrumentation, e.g. HTTP
 *   server spans on the global provider) are never exported to Langfuse —
 *   inheriting one orphans the trace root (its root/trace input-output
 *   shaping is skipped), collapses concurrent runs inside one request
 *   context into a single merged trace, and bypasses the seeded
 *   deterministic trace id generator.
 * - Langfuse-managed spans bound to a *different* destination (another
 *   tenant's project) would leave the new trace dangling in its own
 *   destination while inheriting the other tenant's trace id.
 *
 * Only a managed span whose destination matches the starting run's resolved
 * destination is a safe parent — that is the sanctioned way for hosts to
 * group runs under their own Langfuse observations.
 */
const managedSpanDestinations = new WeakMap<Span, string>();

export function registerLangfuseManagedSpan(
  span: Span,
  destinationKey: string
): void {
  managedSpanDestinations.set(span, destinationKey);
}

export function getLangfuseManagedSpanDestination(
  span: Span
): string | undefined {
  return managedSpanDestinations.get(span);
}

function resolveLangfuseEnvironment(
  langfuse?: t.LangfuseConfig
): string | undefined {
  const candidates = [
    langfuse?.environment,
    process.env.LANGFUSE_TRACING_ENVIRONMENT,
    process.env.NODE_ENV,
  ];
  for (const candidate of candidates) {
    if (candidate != null && candidate.trim() !== '') {
      return candidate.trim();
    }
  }
  return undefined;
}

export function getLangfuseSpanProcessorParams(
  langfuse?: t.LangfuseConfig
): LangfuseSpanProcessorParams | undefined {
  if (langfuse?.enabled === false) {
    return undefined;
  }
  const environment = resolveLangfuseEnvironment(langfuse);
  if (hasLangfuseConfigCredentials(langfuse)) {
    return {
      publicKey: langfuse.publicKey,
      secretKey: langfuse.secretKey,
      ...(isPresent(langfuse.baseUrl) ? { baseUrl: langfuse.baseUrl } : {}),
      ...(isPresent(environment) ? { environment } : {}),
    };
  }
  if (hasLangfuseEnvConfig()) {
    const baseUrl =
      langfuse?.baseUrl ??
      process.env.LANGFUSE_BASE_URL ??
      process.env.LANGFUSE_BASEURL;
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      ...(isPresent(baseUrl) ? { baseUrl } : {}),
      ...(isPresent(environment) ? { environment } : {}),
    };
  }
  if (isPresent(langfuse?.baseUrl) && hasLangfuseEnvCredentials()) {
    return {
      publicKey: process.env.LANGFUSE_PUBLIC_KEY as string,
      secretKey: process.env.LANGFUSE_SECRET_KEY as string,
      baseUrl: langfuse.baseUrl,
      ...(isPresent(environment) ? { environment } : {}),
    };
  }
  return undefined;
}

function hashCacheKeyValue(value: string | undefined): string | undefined {
  return isPresent(value)
    ? createHash('sha256').update(value, 'utf8').digest('hex')
    : undefined;
}

export function getLangfuseDestinationKey(
  params: LangfuseSpanProcessorParams,
  langfuse?: t.LangfuseConfig
): string {
  return JSON.stringify({
    publicKey: params.publicKey,
    secretKeyHash: hashCacheKeyValue(params.secretKey),
    baseUrl: params.baseUrl,
    environment: params.environment,
    toolOutputTracing: langfuse?.toolOutputTracing,
  });
}

/** The export destination a run with this config resolves to, or `undefined`
 *  when no Langfuse destination is configured. */
export function resolveLangfuseDestinationKey(
  langfuse?: t.LangfuseConfig
): string | undefined {
  const params = getLangfuseSpanProcessorParams(langfuse);
  return params == null
    ? undefined
    : getLangfuseDestinationKey(params, langfuse);
}
