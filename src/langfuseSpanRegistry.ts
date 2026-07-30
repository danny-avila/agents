import type { Span } from '@opentelemetry/api';

/**
 * Spans created through the Langfuse tracer provider, tracked so callback
 * handlers can distinguish them from ambient spans created by a host's own
 * OpenTelemetry instrumentation (e.g. HTTP server spans on the global
 * provider). Root observations must never inherit trace identity from a
 * foreign span: the foreign parent is never exported to Langfuse, which
 * orphans the trace root (its root/trace input-output shaping is skipped),
 * collapses concurrent runs inside one request context into a single merged
 * trace, and bypasses the seeded deterministic trace id generator. Spans
 * registered here are safe parents — hosts can still group runs under their
 * own Langfuse observations.
 */
const langfuseManagedSpans = new WeakSet<Span>();

export function registerLangfuseManagedSpan(span: Span): void {
  langfuseManagedSpans.add(span);
}

export function isLangfuseManagedSpan(span: Span): boolean {
  return langfuseManagedSpans.has(span);
}
