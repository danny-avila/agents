/* eslint-disable no-console */
// src/hooks/executeHooks.ts
import type { Logger } from 'winston';
import type { HookRegistry } from './HookRegistry';
import type {
  HookInput,
  HookEvent,
  HookOutput,
  HookMatcher,
  ToolDecision,
  StopDecision,
  HookCallback,
  AggregatedHookResult,
} from './types';
import { matchesQuery } from './matchers';

/** Default per-hook timeout when a matcher doesn't set its own. */
export const DEFAULT_HOOK_TIMEOUT_MS = 30_000;

/**
 * Options for a single `executeHooks` call. The `input` drives everything —
 * the event name is read from `input.hook_event_name`, matchers are looked
 * up against that event, and each hook receives `input` directly.
 */
export interface ExecuteHooksOptions {
  registry: HookRegistry;
  input: HookInput;
  /** Scope lookup to this session (in addition to global matchers). */
  sessionId?: string;
  /** Query string matched against each matcher's regex (tool name, etc.). */
  matchQuery?: string;
  /** Parent AbortSignal — combined with per-hook timeout into the hook signal. */
  signal?: AbortSignal;
  /** Default per-hook timeout; overridden by `matcher.timeout` when present. */
  timeoutMs?: number;
  /** Optional winston logger for non-internal hook errors. */
  logger?: Logger;
}

type WideMatcher = HookMatcher<HookEvent>;
type WideCallback = HookCallback<HookEvent>;

interface HookOutcome {
  matcher: WideMatcher;
  output: HookOutput | null;
  error: string | null;
  timedOut: boolean;
}

function freshResult(): AggregatedHookResult {
  return {
    additionalContexts: [],
    errors: [],
  };
}

function combineSignals(
  parent: AbortSignal | undefined,
  timeoutMs: number
): AbortSignal {
  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  if (parent === undefined) {
    return timeoutSignal;
  }
  return AbortSignal.any([parent, timeoutSignal]);
}

function isTimeout(err: unknown): boolean {
  if (err instanceof Error) {
    return err.name === 'TimeoutError' || err.name === 'AbortError';
  }
  return false;
}

function describeError(err: unknown): string {
  if (err instanceof Error) {
    return err.message !== '' ? err.message : err.name;
  }
  return String(err);
}

function makeAbortPromise(signal: AbortSignal): {
  promise: Promise<never>;
  cleanup: () => void;
} {
  let onAbort: (() => void) | undefined;
  const promise = new Promise<never>((_resolve, reject) => {
    if (signal.aborted) {
      reject(
        signal.reason instanceof Error ? signal.reason : new Error('aborted')
      );
      return;
    }
    onAbort = (): void => {
      reject(
        signal.reason instanceof Error ? signal.reason : new Error('aborted')
      );
    };
    signal.addEventListener('abort', onAbort, { once: true });
  });
  const cleanup = (): void => {
    if (onAbort !== undefined) {
      signal.removeEventListener('abort', onAbort);
      onAbort = undefined;
    }
  };
  return { promise, cleanup };
}

async function runHook(
  hook: WideCallback,
  input: HookInput,
  signal: AbortSignal,
  matcher: WideMatcher
): Promise<HookOutcome> {
  const hookPromise = Promise.resolve().then(() => hook(input, signal));
  const { promise: abortPromise, cleanup } = makeAbortPromise(signal);
  try {
    const output = await Promise.race([hookPromise, abortPromise]);
    return { matcher, output, error: null, timedOut: false };
  } catch (err) {
    return {
      matcher,
      output: null,
      error: describeError(err),
      timedOut: isTimeout(err),
    };
  } finally {
    cleanup();
  }
}

function reportErrors(
  outcomes: readonly HookOutcome[],
  event: HookEvent,
  logger: Logger | undefined
): void {
  for (const outcome of outcomes) {
    if (outcome.error === null) {
      continue;
    }
    if (outcome.matcher.internal === true) {
      continue;
    }
    const label = outcome.timedOut ? 'timed out' : 'threw an error';
    const message = `Hook for ${event} ${label}: ${outcome.error}`;
    if (logger !== undefined) {
      logger.warn(message);
      continue;
    }
    console.warn(message);
  }
}

function applyToolDecision(
  agg: AggregatedHookResult,
  decision: ToolDecision,
  reason: string | undefined
): void {
  if (decision === 'deny') {
    if (agg.decision === 'deny') {
      return;
    }
    agg.decision = 'deny';
    agg.reason = reason;
    return;
  }
  if (decision === 'ask') {
    if (agg.decision === 'deny' || agg.decision === 'ask') {
      return;
    }
    agg.decision = 'ask';
    agg.reason = reason;
    return;
  }
  if (agg.decision === undefined) {
    agg.decision = 'allow';
    agg.reason = reason;
  }
}

function applyStopDecision(
  agg: AggregatedHookResult,
  decision: StopDecision,
  reason: string | undefined
): void {
  if (decision === 'block') {
    if (agg.stopDecision === 'block') {
      return;
    }
    agg.stopDecision = 'block';
    agg.reason = reason;
    return;
  }
  if (agg.stopDecision === undefined) {
    agg.stopDecision = 'continue';
    if (agg.reason === undefined) {
      agg.reason = reason;
    }
  }
}

function applyDecision(agg: AggregatedHookResult, output: HookOutput): void {
  if (!('decision' in output) || output.decision === undefined) {
    return;
  }
  const decision = output.decision;
  const reason =
    'reason' in output && typeof output.reason === 'string'
      ? output.reason
      : undefined;
  if (decision === 'deny' || decision === 'ask' || decision === 'allow') {
    applyToolDecision(agg, decision, reason);
    return;
  }
  applyStopDecision(agg, decision, reason);
}

function applyContext(agg: AggregatedHookResult, output: HookOutput): void {
  if (
    typeof output.additionalContext === 'string' &&
    output.additionalContext.length > 0
  ) {
    agg.additionalContexts.push(output.additionalContext);
  }
}

function applyStopFlag(agg: AggregatedHookResult, output: HookOutput): void {
  if (output.preventContinuation !== true) {
    return;
  }
  agg.preventContinuation = true;
  if (typeof output.stopReason === 'string' && agg.stopReason === undefined) {
    agg.stopReason = output.stopReason;
  }
}

function applyUpdatedInput(
  agg: AggregatedHookResult,
  output: HookOutput
): void {
  if (!('updatedInput' in output) || output.updatedInput === undefined) {
    return;
  }
  agg.updatedInput = output.updatedInput;
}

function fold(outcomes: readonly HookOutcome[]): AggregatedHookResult {
  const agg = freshResult();
  for (const outcome of outcomes) {
    if (outcome.error !== null) {
      if (outcome.matcher.internal !== true) {
        agg.errors.push(outcome.error);
      }
      continue;
    }
    const output = outcome.output;
    if (output === null) {
      continue;
    }
    applyContext(agg, output);
    applyStopFlag(agg, output);
    applyDecision(agg, output);
    applyUpdatedInput(agg, output);
  }
  return agg;
}

function collectOnceMatchersForRemoval(
  outcomes: readonly HookOutcome[]
): WideMatcher[] {
  const successByMatcher = new Map<WideMatcher, boolean>();
  for (const outcome of outcomes) {
    if (outcome.matcher.once !== true) {
      continue;
    }
    const prior = successByMatcher.get(outcome.matcher) ?? false;
    successByMatcher.set(outcome.matcher, prior || outcome.error === null);
  }
  const removable: WideMatcher[] = [];
  for (const [matcher, hasSuccess] of successByMatcher) {
    if (hasSuccess) {
      removable.push(matcher);
    }
  }
  return removable;
}

/**
 * Fires every matcher registered against `input.hook_event_name`, folding
 * their results per `deny > ask > allow` precedence and accumulating
 * context/errors.
 *
 * ## Parallelism and determinism
 *
 * All matching hooks fire simultaneously via `Promise.all`. Outputs are
 * folded in completion order, so a `PreToolUse` hook that returns
 * `updatedInput` in a multi-hook matcher will race with its siblings —
 * registration-time ordering is not respected. If the caller needs a
 * deterministic input rewrite, it must ensure only one hook per matcher
 * writes `updatedInput`.
 *
 * ## Timeouts and cancellation
 *
 * Each hook receives its own `AbortSignal` derived from (a) the caller's
 * parent signal and (b) a timeout from `matcher.timeout` (falling back to
 * `opts.timeoutMs`, default {@link DEFAULT_HOOK_TIMEOUT_MS}). When either
 * fires, the hook's signal aborts. Timeout/abort errors are swallowed into
 * the aggregated result's `errors` array (non-fatal by default).
 *
 * ## Internal matchers
 *
 * A matcher with `internal: true` is excluded from both the `errors` array
 * and the logger output. Use it for infrastructure hooks whose failures
 * should not pollute user-visible diagnostics.
 *
 * ## Once semantics
 *
 * A matcher with `once: true` is removed from the registry after at least
 * one of its hooks completes without throwing. If every hook in the matcher
 * throws, the matcher is retained so the caller can retry.
 */
export async function executeHooks(
  opts: ExecuteHooksOptions
): Promise<AggregatedHookResult> {
  const {
    registry,
    input,
    sessionId,
    matchQuery,
    signal,
    timeoutMs = DEFAULT_HOOK_TIMEOUT_MS,
    logger,
  } = opts;
  const event = input.hook_event_name;
  const matchers = registry.getMatchers(event, sessionId);
  if (matchers.length === 0) {
    return freshResult();
  }
  const filtered = matchers.filter((m) => matchesQuery(m.matcher, matchQuery));
  if (filtered.length === 0) {
    return freshResult();
  }

  const tasks: Promise<HookOutcome>[] = [];
  for (const matcher of filtered) {
    const perHookTimeout = matcher.timeout ?? timeoutMs;
    for (const hook of matcher.hooks) {
      const hookSignal = combineSignals(signal, perHookTimeout);
      tasks.push(runHook(hook, input, hookSignal, matcher));
    }
  }

  const outcomes = await Promise.all(tasks);
  const toRemove = collectOnceMatchersForRemoval(outcomes);
  for (const matcher of toRemove) {
    registry.removeMatcher(event, matcher, sessionId);
  }
  reportErrors(outcomes, event, logger);
  return fold(outcomes);
}
