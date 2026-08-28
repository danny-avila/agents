import type { RunnableConfig } from '@langchain/core/runnables';

/** Durable event payload accepted by the actor lifecycle. */
export type EventActorEvent =
  | null
  | boolean
  | number
  | string
  | readonly EventActorEvent[]
  | { readonly [key: string]: EventActorEvent };

/** Stable reference to one persisted LangGraph checkpoint. */
export interface EventActorCheckpointReference {
  threadId: string;
  checkpointId?: string;
  checkpointNs: string;
}

/** Committed logical head read before an event invocation is prepared. */
export interface EventActorHead {
  actorThreadId: string;
  generation: number;
  checkpoint?: EventActorCheckpointReference;
}

/** Invocation-owned checkpoint fork that cannot become authoritative in place. */
export interface EventActorCheckpointFork
  extends EventActorCheckpointReference {
  invocationId: string;
}

export interface EventActorInvocationReference {
  actorThreadId: string;
  invocationId: string;
  depth: number;
  continuation: 'warm' | 'cold';
  base: EventActorHead;
  fork: EventActorCheckpointFork;
}

export interface EventActorInvocation<TEvent extends EventActorEvent>
  extends EventActorInvocationReference {
  event: TEvent;
}

export interface EventActorPreparedInvocation<TEvent extends EventActorEvent>
  extends EventActorInvocation<TEvent> {
  /**
   * Executor-authenticated, time-bounded binding over the complete prepared
   * invocation. Its wire representation is opaque to callers.
   */
  preparationDigest: string;
}

export type EventActorAdapterPreparation<TEvent extends EventActorEvent> =
  | { status: 'ready'; invocation: EventActorInvocation<TEvent> }
  | { status: 'checkpoint_unavailable'; head: EventActorHead };

export type EventActorPreparation<TEvent extends EventActorEvent> =
  | { status: 'ready'; invocation: EventActorPreparedInvocation<TEvent> }
  | {
      status: 'checkpoint_unavailable';
      request: EventActorPrepareRequest<TEvent>;
      head: EventActorHead;
      /** Executor-authenticated binding over this exact request/head pair. */
      preparationDigest: string;
    };

export type EventActorTerminalResult<TResult extends EventActorEvent> =
  | {
      status: 'applied';
      result: TResult;
      checkpoint: EventActorCheckpointFork;
    }
  | { status: 'completed_no_action'; result?: TResult };

/** JSON-safe interrupt descriptor retained with a suspended invocation fork. */
export interface EventActorInterrupt<
  TPayload extends EventActorEvent = EventActorEvent,
> {
  id: string;
  payload: TPayload;
}

/**
 * Nonterminal adapter outcome. The checkpoint must already contain the pause;
 * the SDK publishes its authority through `suspend` before exposing it.
 */
export interface EventActorAdapterSuspendedResult<
  TPayload extends EventActorEvent = EventActorEvent,
> {
  status: 'suspended';
  checkpoint: EventActorCheckpointFork;
  interrupt: EventActorInterrupt<TPayload>;
}

export type EventActorAdapterInvocationResult<
  TResult extends EventActorEvent,
  TPayload extends EventActorEvent = EventActorEvent,
> =
  | EventActorTerminalResult<TResult>
  | EventActorAdapterSuspendedResult<TPayload>;

/**
 * Authenticated, versioned, JSON-safe evidence for one nonterminal invocation.
 * Integrity does not make this evidence one-shot; the host's durable current
 * suspension record is the replay and ownership fence.
 */
export interface EventActorSuspension<
  TPayload extends EventActorEvent = EventActorEvent,
> {
  version: 1;
  suspensionId: string;
  attempt: number;
  issuedAt: number;
  expiresAt: number;
  invocation: EventActorInvocationReference;
  checkpoint: EventActorCheckpointFork;
  interrupt: EventActorInterrupt<TPayload>;
  suspensionDigest: string;
}

export interface EventActorSuspendedResult<
  TPayload extends EventActorEvent = EventActorEvent,
> {
  status: 'suspended';
  suspension: EventActorSuspension<TPayload>;
}

export interface EventActorSuspendRequest {
  suspension: EventActorSuspension;
  /** CAS predecessor required when a claimed resume pauses again. */
  previous?: {
    suspensionId: string;
    attempt: number;
    resumeAttemptId: string;
  };
}

export type EventActorSuspendResult =
  | { status: 'stored' }
  | { status: 'stale' };

export interface EventActorAdapterResumeRequest {
  suspension: EventActorSuspension;
  resumeAttemptId: string;
  value: EventActorEvent;
}

export type EventActorAdapterResumeResult<TResult extends EventActorEvent> =
  | {
      status: 'claimed';
      result: EventActorAdapterInvocationResult<TResult>;
    }
  | {
      /** The host proved no action before returning this claimed failure. */
      status: 'claimed_failed';
      error: Error;
    }
  | { status: 'stale' };

export interface EventActorResumeRequest {
  suspension: EventActorSuspension;
  resumeAttemptId: string;
  value: EventActorEvent;
  signal?: AbortSignal;
}

export interface EventActorCancelSuspensionRequest {
  suspension: EventActorSuspension;
  cancelAttemptId: string;
  reason?: 'cancelled' | 'expired';
  signal?: AbortSignal;
}

export interface EventActorAdapterCancelSuspensionRequest {
  suspension: EventActorSuspension;
  cancelAttemptId: string;
  reason: 'cancelled' | 'expired';
}

export type EventActorCancelSuspensionResult =
  | { status: 'cancelled' }
  | EventActorIndeterminateResult<EventActorEvent>;

export type EventActorAdapterCancelSuspensionResult =
  | { status: 'cancelled' }
  | { status: 'stale' };

export interface EventActorSettleSuspensionRequest {
  suspensionId: string;
  attempt: number;
  resumeAttemptId: string;
  status: 'completed_no_action' | 'failed';
}

export type EventActorSettleSuspensionResult =
  | { status: 'settled' }
  | { status: 'stale' };

export type EventActorAppliedResult<TResult extends EventActorEvent> = Extract<
  EventActorTerminalResult<TResult>,
  { status: 'applied' }
> & {
  /** Executor-issued settlement for the invocation that produced this action. */
  invocation: EventActorInvocationReference;
  /** Authenticated cross-executor authority for a resumed terminal action. */
  settlementAuthority?: EventActorSettlementAuthority;
};

/**
 * Authenticated fence that binds a resumed terminal result to its claimed
 * suspension. The host must consume it atomically with the actor-head CAS.
 */
export interface EventActorSettlementAuthority {
  version: 1;
  suspensionId: string;
  attempt: number;
  resumeAttemptId: string;
  issuedAt: number;
  expiresAt: number;
  settlementDigest: string;
}

export interface EventActorIndeterminateResult<
  TResult extends EventActorEvent,
> {
  /** Applied handling cannot be proven safe to retry; retain its fork. */
  status: 'commit_indeterminate';
  result?: TResult;
  checkpoint: EventActorCheckpointFork;
  error: Error;
}

export type EventActorInvocationResult<TResult extends EventActorEvent> =
  | EventActorAppliedResult<TResult>
  | EventActorIndeterminateResult<TResult>
  | EventActorSuspendedResult
  | Extract<
      EventActorTerminalResult<TResult>,
      { status: 'completed_no_action' }
    >;

export interface EventActorInvocationContext {
  signal: AbortSignal;
  config: RunnableConfig;
}

export interface EventActorPreparationContext {
  /** Explicit task-owned cancellation; parent-run ambient signals are excluded. */
  signal: AbortSignal;
}

export interface EventActorPrepareRequest<TEvent extends EventActorEvent> {
  actorThreadId: string;
  invocationId: string;
  depth: number;
  event: TEvent;
}

export interface EventActorAdapterPrepareRequest<TEvent extends EventActorEvent>
  extends EventActorPrepareRequest<TEvent> {
  /** Unique execution-attempt namespace; invocationId remains the logical idempotency key. */
  checkpointNs: string;
}

export interface EventActorCommitRequest<TResult extends EventActorEvent> {
  invocation: EventActorInvocationReference;
  expectedHead: EventActorHead;
  checkpoint: EventActorCheckpointFork;
  result: TResult;
  /** Host must consume this suspension fence atomically with the head CAS. */
  settlementAuthority?: EventActorSettlementAuthority;
  retention: {
    committedCheckpoints: 2;
    dormantCheckpointTtlMs: number;
  };
}

export type EventActorCommitResult =
  | { status: 'committed'; head: EventActorHead }
  | { status: 'stale'; head?: EventActorHead };

/** Public settlement outcome after an action has already been applied. */
export type EventActorSettlementResult<TResult extends EventActorEvent> =
  | EventActorCommitResult
  | EventActorIndeterminateResult<TResult>;

export type EventActorDiscardReason =
  | 'cancelled'
  | 'completed_no_action'
  | 'failed';

export interface EventActorDiscardRequest {
  invocation: EventActorInvocationReference;
  reason: EventActorDiscardReason;
}

/**
 * Host adapter for durable actor state and the concrete agent invocation.
 * `commit` must compare both the expected generation and checkpoint identity
 * atomically before advancing the logical actor head. When settlement authority
 * is present, that same transaction must also verify and close the exact
 * suspension/resume-attempt fence, including stale-head outcomes. Retrying an
 * ambiguous acknowledgement must return the durable outcome rather than apply
 * the same terminal transition again. The host mailbox
 * deduplicates the logical `invocationId` before entering this seam, while each
 * SDK execution attempt receives a distinct checkpoint namespace. Preparation
 * methods own rollback until they return a ready invocation and must treat the
 * request event as immutable. On cancellation they roll back and reject with
 * `context.signal.reason`; cleanup failures reject with their own error so they
 * remain observable. `invoke` returns only after its provider, stream, timer,
 * and executor resources have been released. Once qualifying action evidence
 * exists, `invoke` must return `applied` even if a later abort or provider
 * failure occurs; a thrown error is therefore a definite no-action failure
 * whose fork is safe to discard. `commit` must not reclaim an applied stale
 * fork: the SDK retains and surfaces it as `commit_conflict` for host
 * reconciliation. `discard` must be idempotent for the same invocation because
 * an ambiguous cleanup failure can be retried through the public lifecycle.
 *
 * Suspension-capable hosts implement all optional suspension methods. `suspend`
 * publishes an initial suspension only while its logical invocation is current;
 * with `previous`, it atomically replaces only the exact claimed predecessor.
 * `resume` atomically claims the current suspension before applying its value to
 * the declared interrupt and rejects duplicate or competing claims as `stale`.
 * It must return `claimed_failed` only when it can prove no qualifying action;
 * any failure after an action returns `applied`. `settleSuspension` atomically
 * discards the claimed fork and closes its fence for definite no-action
 * outcomes. `cancelSuspension`
 * atomically claims, discards, and closes current state, including expired
 * evidence; expiration alone never implies a safe action outcome.
 */
export interface EventActorHostAdapter<
  TEvent extends EventActorEvent,
  TResult extends EventActorEvent,
> {
  prepare(
    request: EventActorAdapterPrepareRequest<TEvent>,
    context: EventActorPreparationContext
  ): Promise<EventActorAdapterPreparation<TEvent>>;
  coldContinue(
    request: EventActorAdapterPrepareRequest<TEvent>,
    head: EventActorHead,
    context: EventActorPreparationContext
  ): Promise<EventActorInvocation<TEvent>>;
  invoke(
    invocation: EventActorInvocation<TEvent>,
    context: EventActorInvocationContext
  ): Promise<EventActorAdapterInvocationResult<TResult>>;
  suspend?(request: EventActorSuspendRequest): Promise<EventActorSuspendResult>;
  resume?(
    request: EventActorAdapterResumeRequest,
    context: EventActorInvocationContext
  ): Promise<EventActorAdapterResumeResult<TResult>>;
  /** Atomically claims, discards, and closes the current suspension. */
  cancelSuspension?(
    request: EventActorAdapterCancelSuspensionRequest,
    context: EventActorPreparationContext
  ): Promise<EventActorAdapterCancelSuspensionResult>;
  settleSuspension?(
    request: EventActorSettleSuspensionRequest
  ): Promise<EventActorSettleSuspensionResult>;
  commit(
    request: EventActorCommitRequest<TResult>
  ): Promise<EventActorCommitResult>;
  discard(request: EventActorDiscardRequest): Promise<void>;
}

export interface EventActorExecutionRequest<TEvent extends EventActorEvent> {
  actorThreadId: string;
  invocationId: string;
  event: TEvent;
  depth?: number;
  /** Explicit task-owned signal. Ambient parent-run signals are ignored. */
  signal?: AbortSignal;
}

export type EventActorExecutionResult<TResult extends EventActorEvent> =
  | {
      status: 'applied';
      result: TResult;
      head: EventActorHead;
      continuation: 'warm' | 'cold';
    }
  | {
      status: 'completed_no_action';
      result?: TResult;
      continuation: 'warm' | 'cold';
    }
  | {
      status: 'cancelled';
      continuation: 'warm' | 'cold';
    }
  | (EventActorSuspendedResult & {
      continuation: 'warm' | 'cold';
    })
  | {
      /** The action happened, but another head won the CAS. Reconcile; do not retry. */
      status: 'commit_conflict';
      result: TResult;
      checkpoint: EventActorCheckpointFork;
      head?: EventActorHead;
      continuation: 'warm' | 'cold';
    }
  | {
      /** Applied handling cannot be proven safe to retry; retain its fork. */
      status: 'commit_indeterminate';
      result?: TResult;
      checkpoint: EventActorCheckpointFork;
      error: Error;
      continuation: 'warm' | 'cold';
    }
  | {
      status: 'failed';
      error: Error;
      continuation: 'warm' | 'cold';
    };

export interface EventActorExecutorOptions {
  maxDepth?: number;
  /** Also bounds signed preparation authority and local terminal fences. */
  dormantCheckpointTtlMs?: number;
  /** Stable private key of at least 32 bytes for cross-lifetime handoffs. */
  preparationSigningKey?: string | Uint8Array;
  /** Maximum UTF-8 byte size of canonical suspension evidence. */
  maxSuspensionPayloadBytes?: number;
}
