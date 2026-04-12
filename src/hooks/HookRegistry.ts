// src/hooks/HookRegistry.ts
import type { HookEvent, HookMatcher } from './types';

/**
 * Internal matcher storage type.
 *
 * Matchers registered via the public `register<E>` API are strictly typed
 * to a single `E`, but the storage needs one uniform slot type per event.
 * We store them as `HookMatcher<HookEvent>` and cast once at the variance
 * boundary — see `ensureList` and `snapshot` below. The invariant (every
 * matcher in `bucket[event]` was registered with that exact event) is
 * enforced by the public API; breaking it requires bypassing the types.
 */
type MatcherBucket = Partial<Record<HookEvent, HookMatcher<HookEvent>[]>>;

/**
 * Run-scoped storage for hook matchers with an additional layer for
 * session-scoped matchers that should be cleaned up between sessions.
 *
 * Hosts construct one registry per `Run` (mirroring how `HandlerRegistry` is
 * scoped) and register global matchers + per-session matchers against it.
 * Registration is strictly additive — nothing in this class mutates a
 * matcher's callbacks or flags after insertion.
 *
 * ## Why `Map<sessionId, MatcherBucket>` and not `Record`
 *
 * LibreChat runs thousands of parallel sessions in one Node process, and
 * hook registration happens inside hot paths (tool loading, agent spawning).
 * A `Record<sessionId, ...>` has to be spread on every insertion, which is
 * O(n) per call and O(n²) total for a batch of parallel registrations. A
 * Map mutates in place, keeping insertions O(1). This mirrors the reasoning
 * Claude Code documents at `utils/hooks/sessionHooks.ts:62`.
 */
export class HookRegistry {
  private readonly global: MatcherBucket = {};
  private readonly sessions: Map<string, MatcherBucket> = new Map();

  /**
   * Register a matcher for the lifetime of this registry (= one Run).
   * Returns an unregister function that removes the matcher by reference.
   */
  register<E extends HookEvent>(event: E, matcher: HookMatcher<E>): () => void {
    const list = ensureList(this.global, event);
    list.push(widen(matcher));
    return () => {
      removeFromList(list, matcher);
    };
  }

  /**
   * Register a matcher for a specific session. Cleared automatically when
   * `clearSession(sessionId)` is called, or can be removed directly via the
   * returned unregister function.
   */
  registerSession<E extends HookEvent>(
    sessionId: string,
    event: E,
    matcher: HookMatcher<E>
  ): () => void {
    const bucket = this.ensureSessionBucket(sessionId);
    const list = ensureList(bucket, event);
    list.push(widen(matcher));
    return () => {
      removeFromList(list, matcher);
    };
  }

  /**
   * Returns all matchers registered for `event`, concatenating global first
   * and then session-specific (when `sessionId` is supplied). The caller
   * receives a fresh array, so iterating it is safe even if a matcher is
   * removed mid-iteration (e.g. via `once: true`).
   */
  getMatchers<E extends HookEvent>(
    event: E,
    sessionId?: string
  ): HookMatcher<E>[] {
    const globalList = readList(this.global, event);
    if (sessionId === undefined) {
      return snapshot<E>(globalList);
    }
    const bucket = this.sessions.get(sessionId);
    if (bucket === undefined) {
      return snapshot<E>(globalList);
    }
    const sessionList = readList(bucket, event);
    if (globalList.length === 0) {
      return snapshot<E>(sessionList);
    }
    if (sessionList.length === 0) {
      return snapshot<E>(globalList);
    }
    return snapshot<E>([...globalList, ...sessionList]);
  }

  /**
   * Removes `matcher` by reference from global storage first, falling back
   * to the session bucket when `sessionId` is supplied. Used by
   * `executeHooks` to drop `once: true` matchers after they fire.
   */
  removeMatcher<E extends HookEvent>(
    event: E,
    matcher: HookMatcher<E>,
    sessionId?: string
  ): boolean {
    if (removeFromList(readList(this.global, event), matcher)) {
      return true;
    }
    if (sessionId === undefined) {
      return false;
    }
    const bucket = this.sessions.get(sessionId);
    if (bucket === undefined) {
      return false;
    }
    return removeFromList(readList(bucket, event), matcher);
  }

  /**
   * Drops every session-scoped matcher for `sessionId`. Call this in the
   * `finally` block around a Run so a `once: true` hook that never fired
   * cannot leak into the next session on the same registry.
   */
  clearSession(sessionId: string): void {
    this.sessions.delete(sessionId);
  }

  /** True if at least one matcher exists for `event` (global + session). */
  hasHookFor(event: HookEvent, sessionId?: string): boolean {
    if (readList(this.global, event).length > 0) {
      return true;
    }
    if (sessionId === undefined) {
      return false;
    }
    const bucket = this.sessions.get(sessionId);
    if (bucket === undefined) {
      return false;
    }
    return readList(bucket, event).length > 0;
  }

  private ensureSessionBucket(sessionId: string): MatcherBucket {
    const existing = this.sessions.get(sessionId);
    if (existing !== undefined) {
      return existing;
    }
    const fresh: MatcherBucket = {};
    this.sessions.set(sessionId, fresh);
    return fresh;
  }
}

function ensureList(
  bucket: MatcherBucket,
  event: HookEvent
): HookMatcher<HookEvent>[] {
  const existing = bucket[event];
  if (existing !== undefined) {
    return existing;
  }
  const fresh: HookMatcher<HookEvent>[] = [];
  bucket[event] = fresh;
  return fresh;
}

function readList(
  bucket: MatcherBucket,
  event: HookEvent
): HookMatcher<HookEvent>[] {
  return bucket[event] ?? [];
}

function removeFromList<E extends HookEvent>(
  list: HookMatcher<HookEvent>[],
  matcher: HookMatcher<E>
): boolean {
  const idx = list.indexOf(widen(matcher));
  if (idx < 0) {
    return false;
  }
  list.splice(idx, 1);
  return true;
}

/**
 * Widen a per-event matcher to the storage's uniform slot type. Unsound at
 * the type level (function parameters are contravariant) but safe by
 * construction: `HookRegistry.register<E>` only ever puts matchers into the
 * bucket slot for their own event, and reads go through `snapshot<E>`
 * which is only called with the same `E`.
 */
function widen<E extends HookEvent>(
  matcher: HookMatcher<E>
): HookMatcher<HookEvent> {
  return matcher as unknown as HookMatcher<HookEvent>;
}

/**
 * Narrow a storage list back to a per-event matcher list on the way out.
 * Sound counterpart to `widen`: the list only contains matchers that were
 * registered against `E`, because the public API enforces it on insert.
 */
function snapshot<E extends HookEvent>(
  list: readonly HookMatcher<HookEvent>[]
): HookMatcher<E>[] {
  return list.slice() as unknown as HookMatcher<E>[];
}
