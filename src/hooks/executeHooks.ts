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
  /** Query string matched against each matcher's pattern (tool name, etc.). */
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
    // eslint-disable-next-line no-console
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

function applyUpdatedOutput(
  agg: AggregatedHookResult,
  output: HookOutput
): void {
  if (!('updatedOutput' in output) || output.updatedOutput === undefined) {
    return;
  }
  agg.updatedOutput = output.updatedOutput;
}

function applyAllowedDecisions(
  agg: AggregatedHookResult,
  output: HookOutput
): void {
  if (
    !('allowedDecisions' in output) ||
    output.allowedDecisions === undefined
  ) {
    return;
  }
  agg.allowedDecisions = output.allowedDecisions;
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
    /**
     * Skip fire-and-forget outputs entirely: the agent has already
     * moved on, so an async hook cannot influence the run. Background
     * work inside the hook body still runs (we don't cancel it), it
     * just doesn't fold into the aggregate result.
     */
    if (output.async === true) {
      continue;
    }
    applyContext(agg, output);
    applyStopFlag(agg, output);
    applyDecision(agg, output);
    applyUpdatedInput(agg, output);
    applyUpdatedOutput(agg, output);
    applyAllowedDecisions(agg, output);
  }
  return agg;
}

/**
 * Fires every matcher registered against `input.hook_event_name`, folding
 * their results per `deny > ask > allow` precedence and accumulating
 * context/errors.
 *
 * ## Parallelism and determinism
 *
 * All matching hooks fire simultaneously and are awaited via `Promise.all`,
 * which preserves input-array order in its returned results. The fold
 * therefore iterates outcomes in **registration order** — outer loop over
 * matchers as they sit in the registry (global first, then session), inner
 * loop over each matcher's `hooks` array. Last-writer-wins fields
 * (`updatedInput`, `updatedOutput`) are deterministic in that order, even
 * though hooks may complete in arbitrary wall-clock order.
 *
 * Consumers that need a single authoritative rewrite should still scope
 * `updatedInput`/`updatedOutput` to one hook per matcher to avoid subtle
 * precedence bugs when matchers are added in a different order than
 * expected.
 *
 * ## Timeouts and cancellation
 *
 * Each matcher receives **one shared `AbortSignal`** derived from the
 * caller's parent signal combined with `matcher.timeout` (falling back to
 * `opts.timeoutMs`, default {@link DEFAULT_HOOK_TIMEOUT_MS}). Sharing the
 * signal across hooks in a matcher collapses N timer allocations into
 * one, which matters on the PreToolUse hot path where a matcher with
 * several hooks fires on every tool call. Each hook call is raced
 * against the shared signal, so even a hook that ignores the signal is
 * force-unblocked when the timeout fires. Timeout/abort errors are
 * swallowed into the aggregated result's `errors` array (non-fatal by
 * default).
 *
 * ## Internal matchers
 *
 * A matcher with `internal: true` is excluded from both the `errors` array
 * and the logger output. Use it for infrastructure hooks whose failures
 * should not pollute user-visible diagnostics.
 *
 * ## Once semantics — atomic at-most-once
 *
 * A matcher with `once: true` is removed from the registry **before any
 * hook runs**, inside the synchronous prefix of `executeHooks` (between
 * `getMatchers` and the first `await`). Because Node's event loop serialises
 * sync work, two concurrent `executeHooks` calls can never both observe
 * and dispatch the same `once` matcher — whichever call runs its sync
 * prefix first consumes it, and the loser sees an empty bucket.
 *
 * Trade-off: if every hook in a `once` matcher throws, the matcher is
 * still gone. "Once" here means "at most one dispatch, ever", not "at
 * most one successful execution with retry on failure". Hosts that need
 * retry semantics should register a normal matcher and self-unregister
 * via the `unregister` callback returned from `registry.register`.
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

  // --- SYNC CRITICAL SECTION: once-matcher removal must complete before any await ---
  const tasks: Promise<HookOutcome>[] = [];
  for (const matcher of matchers) {
    if (!matchesQuery(matcher.pattern, matchQuery)) {
      continue;
    }
    if (matcher.once === true) {
      registry.removeMatcher(event, matcher, sessionId);
    }
    const perHookTimeout = matcher.timeout ?? timeoutMs;
    const matcherSignal = combineSignals(signal, perHookTimeout);
    for (const hook of matcher.hooks) {
      tasks.push(runHook(hook, input, matcherSignal, matcher));
    }
  }
  // --- END SYNC CRITICAL SECTION ---
  if (tasks.length === 0) {
    return freshResult();
  }

  const outcomes = await Promise.all(tasks);
  reportErrors(outcomes, event, logger);
  const aggregated = fold(outcomes);
  /**
   * Centralized `preventContinuation` propagation: when any hook (across
   * any callsite — RunStart, PreToolUse, PostToolBatch, SubagentStop,
   * etc.) returns `preventContinuation: true`, raise a halt signal on
   * the registry scoped to the run's `sessionId`. `Run.processStream`
   * polls the signal between stream events using its own id and exits
   * cleanly, skipping the `Stop` hook (since the run is being halted,
   * not naturally completing).
   *
   * First-write-wins per session inside the registry — a halt already
   * raised by an earlier hook in the same run is preserved so the
   * original `reason` / `source` are not clobbered. Hooks fired
   * without a `sessionId` cannot raise a halt (there's no run for the
   * loop to poll under), which is fine: every in-tree callsite passes
   * `sessionId: runId`. Pre-stream callsites in `Run.processStream`
   * still read `preventContinuation` directly off the result for an
   * early return because they have not yet entered the stream loop.
   */
  if (aggregated.preventContinuation === true && sessionId !== undefined) {
    registry.haltRun(
      sessionId,
      aggregated.stopReason ?? 'preventContinuation',
      event
    );
  }
  return aggregated;
}
