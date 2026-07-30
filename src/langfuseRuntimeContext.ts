import { createHash } from 'node:crypto';
import { AsyncLocalStorage } from 'node:async_hooks';
import type * as t from '@/types';

export type ResolvedLangfuseToolOutputTracingConfig = {
  enabled: boolean;
  redactedToolNames: Set<string>;
  redactedToolNameMatchMode: 'exact' | 'partial';
  redactionText: string;
};

export type LangfuseRuntimeContext = {
  langfuse?: t.LangfuseConfig;
  traceIdSeed?: string;
  toolOutputTracing?: ResolvedLangfuseToolOutputTracingConfig;
  /**
   * Identity of the run this scope belongs to. LangChain executes
   * non-awaited callbacks on a shared background queue
   * (`@langchain/core` `consumeCallback`, a process-wide `p-queue` with
   * concurrency 1), so a callback can run inside a DIFFERENT concurrent
   * run's async context. Handlers compare this id against their own run to
   * decide whether an ambient scope is theirs to adopt.
   */
  runId?: string;
};

const langfuseRuntimeContextStore =
  new AsyncLocalStorage<LangfuseRuntimeContext>();

function hasText(value: string | undefined): value is string {
  return value != null && value.trim() !== '';
}

export function hasLangfuseRuntimeContextValue(
  context: LangfuseRuntimeContext
): boolean {
  return (
    context.langfuse != null ||
    hasText(context.traceIdSeed) ||
    hasText(context.runId) ||
    context.toolOutputTracing != null
  );
}

/** sha256(seed) -> first 32 hex chars; matches `@langfuse/tracing` `createTraceId`. */
export function traceIdFromSeed(seed: string): string {
  return createHash('sha256').update(seed, 'utf8').digest('hex').slice(0, 32);
}

export function getLangfuseRuntimeContext():
  | LangfuseRuntimeContext
  | undefined {
  return langfuseRuntimeContextStore.getStore();
}

export function getLangfuseRuntimeConfig(): t.LangfuseConfig | undefined {
  return getLangfuseRuntimeContext()?.langfuse;
}

export function getTraceIdSeed(): string | undefined {
  return getLangfuseRuntimeContext()?.traceIdSeed;
}

export function getLangfuseScopeRunId(): string | undefined {
  return getLangfuseRuntimeContext()?.runId;
}

export function getLangfuseRuntimeToolOutputTracingConfig():
  | ResolvedLangfuseToolOutputTracingConfig
  | undefined {
  return getLangfuseRuntimeContext()?.toolOutputTracing;
}

/**
 * Runs `fn` with a merged Langfuse runtime context. Undefined fields inherit
 * from the parent scope; callers intentionally cannot clear parent values by
 * passing `undefined`.
 */
export function runWithLangfuseRuntimeContext<T>(
  context: LangfuseRuntimeContext,
  fn: () => T
): T {
  const current = getLangfuseRuntimeContext();
  const next = {
    ...(current ?? {}),
    ...(context.langfuse !== undefined ? { langfuse: context.langfuse } : {}),
    ...(hasText(context.traceIdSeed)
      ? { traceIdSeed: context.traceIdSeed }
      : {}),
    ...(hasText(context.runId) ? { runId: context.runId } : {}),
    ...(context.toolOutputTracing !== undefined
      ? { toolOutputTracing: context.toolOutputTracing }
      : {}),
  };

  return hasLangfuseRuntimeContextValue(next)
    ? langfuseRuntimeContextStore.run(next, fn)
    : fn();
}

export function runWithTraceIdSeed<T>(
  seed: string | undefined,
  fn: () => T
): T {
  return hasText(seed)
    ? runWithLangfuseRuntimeContext({ traceIdSeed: seed }, fn)
    : fn();
}
