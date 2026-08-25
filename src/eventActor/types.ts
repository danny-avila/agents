import type { RunnableConfig } from '@langchain/core/runnables';

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

export interface EventActorInvocation<TEvent>
  extends EventActorInvocationReference {
  event: TEvent;
}

export type EventActorPreparation<TEvent> =
  | { status: 'ready'; invocation: EventActorInvocation<TEvent> }
  | { status: 'checkpoint_unavailable'; head: EventActorHead };

export type EventActorTerminalResult<TResult> =
  | {
      status: 'applied';
      result: TResult;
      checkpoint: EventActorCheckpointFork;
    }
  | { status: 'completed_no_action'; result?: TResult };

export type EventActorAppliedResult<TResult> = Extract<
  EventActorTerminalResult<TResult>,
  { status: 'applied' }
>;

export interface EventActorInvocationContext {
  signal: AbortSignal;
  config: RunnableConfig;
}

export interface EventActorPrepareRequest<TEvent> {
  actorThreadId: string;
  invocationId: string;
  depth: number;
  event: TEvent;
}

export interface EventActorAdapterPrepareRequest<TEvent>
  extends EventActorPrepareRequest<TEvent> {
  /** Unique execution-attempt namespace; invocationId remains the logical idempotency key. */
  checkpointNs: string;
}

export interface EventActorCommitRequest<TResult> {
  invocation: EventActorInvocationReference;
  expectedHead: EventActorHead;
  checkpoint: EventActorCheckpointFork;
  result: TResult;
  retention: {
    committedCheckpoints: 2;
    dormantCheckpointTtlMs: number;
  };
}

export type EventActorCommitResult =
  | { status: 'committed'; head: EventActorHead }
  | { status: 'stale'; head?: EventActorHead };

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
 * atomically before advancing the logical actor head. The host mailbox
 * deduplicates the logical `invocationId` before entering this seam, while each
 * SDK execution attempt receives a distinct checkpoint namespace. Preparation
 * methods own rollback until they return a ready invocation and must treat the
 * request event as immutable. `invoke` returns only after its provider, stream,
 * timer, and executor resources have been released. Once qualifying action
 * evidence exists, `invoke` must return `applied` even if a later abort or
 * provider failure occurs; a thrown error is therefore a definite no-action
 * failure whose fork is safe to discard. `commit` must not reclaim an applied
 * stale fork: the SDK retains and surfaces it as `commit_conflict` for host
 * reconciliation.
 */
export interface EventActorHostAdapter<TEvent, TResult> {
  prepare(
    request: EventActorAdapterPrepareRequest<TEvent>
  ): Promise<EventActorPreparation<TEvent>>;
  coldContinue(
    request: EventActorAdapterPrepareRequest<TEvent>,
    head: EventActorHead
  ): Promise<EventActorInvocation<TEvent>>;
  invoke(
    invocation: EventActorInvocation<TEvent>,
    context: EventActorInvocationContext
  ): Promise<EventActorTerminalResult<TResult>>;
  commit(
    request: EventActorCommitRequest<TResult>
  ): Promise<EventActorCommitResult>;
  discard(request: EventActorDiscardRequest): Promise<void>;
}

export interface EventActorExecutionRequest<TEvent> {
  actorThreadId: string;
  invocationId: string;
  event: TEvent;
  depth?: number;
  /** Explicit task-owned signal. Ambient parent-run signals are ignored. */
  signal?: AbortSignal;
}

export type EventActorExecutionResult<TResult> =
  | {
      status: 'applied';
      result: TResult;
      head: EventActorHead;
      continuation: 'warm' | 'cold';
    }
  | {
      status: 'completed_no_action' | 'cancelled';
      continuation: 'warm' | 'cold';
    }
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
      result: TResult;
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
  dormantCheckpointTtlMs?: number;
}
