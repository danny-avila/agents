import { createHash } from 'node:crypto';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  EventActorAdapterPrepareRequest,
  EventActorAppliedResult,
  EventActorCheckpointFork,
  EventActorCommitResult,
  EventActorDiscardReason,
  EventActorExecutionRequest,
  EventActorExecutionResult,
  EventActorExecutorOptions,
  EventActorHead,
  EventActorHostAdapter,
  EventActorInvocation,
  EventActorInvocationReference,
  EventActorPrepareRequest,
  EventActorPreparation,
  EventActorTerminalResult,
} from './types';

const DEFAULT_MAX_DEPTH = 1;
const DEFAULT_DORMANT_CHECKPOINT_TTL_MS = 24 * 60 * 60 * 1_000;

function createInvocationCheckpointNs(
  request: EventActorPrepareRequest<unknown>
): string {
  return `event-actor/${createHash('sha256')
    .update(request.actorThreadId)
    .update('\0')
    .update(request.invocationId)
    .digest('hex')
    .slice(0, 32)}`;
}

function requireNonEmpty(value: string, name: string): void {
  if (value.trim() === '') {
    throw new Error(`${name} must not be empty`);
  }
}

function validateHead(
  head: EventActorHead,
  actorThreadId: string,
  checkpointRequired = false
): void {
  if (
    head.actorThreadId !== actorThreadId ||
    !Number.isSafeInteger(head.generation) ||
    head.generation < 0
  ) {
    throw new Error('Event actor head is invalid');
  }
  if (head.checkpoint == null) {
    if (checkpointRequired) {
      throw new Error('Committed event actor head has no checkpoint');
    }
    return;
  }
  requireNonEmpty(head.checkpoint.threadId, 'head.checkpoint.threadId');
  requireNonEmpty(head.checkpoint.checkpointNs, 'head.checkpoint.checkpointNs');
  requireNonEmpty(
    head.checkpoint.checkpointId ?? '',
    'head.checkpoint.checkpointId'
  );
}

function validateInvocation<TEvent>(
  request: EventActorPrepareRequest<TEvent>,
  invocation: EventActorInvocation<TEvent>,
  continuation: 'warm' | 'cold',
  checkpointNs: string,
  expectedHead?: EventActorInvocation<TEvent>['base']
): void {
  if (
    invocation.actorThreadId !== request.actorThreadId ||
    invocation.invocationId !== request.invocationId ||
    invocation.depth !== request.depth ||
    invocation.continuation !== continuation
  ) {
    throw new Error('Event actor preparation returned a mismatched invocation');
  }
  if (
    invocation.base.actorThreadId !== request.actorThreadId ||
    invocation.fork.invocationId !== request.invocationId ||
    invocation.fork.checkpointNs !== checkpointNs
  ) {
    throw new Error(
      'Event actor preparation returned mismatched checkpoint ownership'
    );
  }
  validateHead(invocation.base, request.actorThreadId);
  requireNonEmpty(invocation.fork.threadId, 'fork.threadId');
  requireNonEmpty(invocation.fork.checkpointNs, 'fork.checkpointNs');
  if (
    expectedHead != null &&
    (expectedHead.actorThreadId !== request.actorThreadId ||
      invocation.base.generation !== expectedHead.generation ||
      invocation.base.checkpoint?.threadId !==
        expectedHead.checkpoint?.threadId ||
      invocation.base.checkpoint?.checkpointId !==
        expectedHead.checkpoint?.checkpointId ||
      invocation.base.checkpoint?.checkpointNs !==
        expectedHead.checkpoint?.checkpointNs)
  ) {
    throw new Error('Cold continuation did not use the prepared actor head');
  }
}

function validateTerminalCheckpoint(
  invocation: EventActorInvocationReference,
  checkpoint: EventActorCheckpointFork
): void {
  if (
    checkpoint.invocationId !== invocation.invocationId ||
    checkpoint.threadId !== invocation.fork.threadId ||
    checkpoint.checkpointNs !== invocation.fork.checkpointNs ||
    checkpoint.checkpointId == null ||
    checkpoint.checkpointId.trim() === ''
  ) {
    throw new Error(
      'Event actor result escaped its invocation checkpoint fork'
    );
  }
}

function createRunnableConfig(
  invocation: EventActorInvocationReference,
  signal: AbortSignal
): RunnableConfig {
  return {
    signal,
    configurable: {
      thread_id: invocation.fork.threadId,
      checkpoint_ns: invocation.fork.checkpointNs,
      ...(invocation.fork.checkpointId == null
        ? {}
        : { checkpoint_id: invocation.fork.checkpointId }),
      event_actor_thread_id: invocation.actorThreadId,
      event_actor_invocation_id: invocation.invocationId,
      event_actor_generation: invocation.base.generation,
      event_actor_depth: invocation.depth,
      event_actor_continuation: invocation.continuation,
    },
  };
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function isAborted(signal?: AbortSignal): boolean {
  return signal?.aborted === true;
}

function validateCommittedHead(
  invocation: EventActorInvocationReference,
  head: EventActorInvocationReference['base']
): void {
  validateHead(head, invocation.actorThreadId, true);
  if (head.generation !== invocation.base.generation + 1) {
    throw new Error('Event actor commit returned an invalid logical head');
  }
}

/**
 * Runs one event against an isolated checkpoint fork and advances the stable
 * actor head only through the host's atomic commit interface.
 */
export class EventActorExecutor<TEvent, TResult> {
  private readonly maxDepth: number;
  private readonly dormantCheckpointTtlMs: number;

  constructor(
    private readonly adapter: EventActorHostAdapter<TEvent, TResult>,
    options: EventActorExecutorOptions = {}
  ) {
    this.maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH;
    this.dormantCheckpointTtlMs =
      options.dormantCheckpointTtlMs ?? DEFAULT_DORMANT_CHECKPOINT_TTL_MS;
    if (!Number.isSafeInteger(this.maxDepth) || this.maxDepth < 1) {
      throw new Error('maxDepth must be a positive safe integer');
    }
    if (
      !Number.isSafeInteger(this.dormantCheckpointTtlMs) ||
      this.dormantCheckpointTtlMs < 1
    ) {
      throw new Error('dormantCheckpointTtlMs must be a positive safe integer');
    }
  }

  async prepare(
    request: EventActorPrepareRequest<TEvent>
  ): Promise<EventActorPreparation<TEvent>> {
    this.validatePrepareRequest(request);
    const adapterRequest: EventActorAdapterPrepareRequest<TEvent> = {
      ...request,
      checkpointNs: createInvocationCheckpointNs(request),
    };
    const preparation = await this.adapter.prepare(adapterRequest);
    if (preparation.status === 'ready') {
      try {
        validateInvocation(
          request,
          preparation.invocation,
          'warm',
          adapterRequest.checkpointNs
        );
      } catch (error) {
        await this.adapter.discard({
          invocation: preparation.invocation,
          reason: 'failed',
        });
        throw error;
      }
    } else {
      validateHead(preparation.head, request.actorThreadId);
    }
    return preparation;
  }

  async coldContinue(
    request: EventActorPrepareRequest<TEvent>,
    head: EventActorHead
  ): Promise<EventActorInvocation<TEvent>> {
    this.validatePrepareRequest(request);
    const adapterRequest: EventActorAdapterPrepareRequest<TEvent> = {
      ...request,
      checkpointNs: createInvocationCheckpointNs(request),
    };
    const invocation = await this.adapter.coldContinue(adapterRequest, head);
    try {
      validateInvocation(
        request,
        invocation,
        'cold',
        adapterRequest.checkpointNs,
        head
      );
    } catch (error) {
      await this.adapter.discard({ invocation, reason: 'failed' });
      throw error;
    }
    return invocation;
  }

  async invoke(
    invocation: EventActorInvocation<TEvent>,
    signal?: AbortSignal
  ): Promise<EventActorTerminalResult<TResult>> {
    validateInvocation(
      {
        actorThreadId: invocation.actorThreadId,
        invocationId: invocation.invocationId,
        depth: invocation.depth,
        event: invocation.event,
      },
      invocation,
      invocation.continuation,
      createInvocationCheckpointNs(invocation)
    );
    const controller = new AbortController();
    const abort = (): void => controller.abort(signal?.reason);
    if (isAborted(signal)) {
      abort();
    } else {
      signal?.addEventListener('abort', abort, { once: true });
    }
    const config = createRunnableConfig(invocation, controller.signal);
    try {
      if (controller.signal.aborted) {
        throw asError(controller.signal.reason ?? 'Event actor cancelled');
      }
      return await AsyncLocalStorageProviderSingleton.runWithConfig(
        config,
        () =>
          this.adapter.invoke(invocation, {
            signal: controller.signal,
            config,
          })
      );
    } finally {
      signal?.removeEventListener('abort', abort);
    }
  }

  async commit(
    invocation: EventActorInvocationReference,
    terminal: EventActorAppliedResult<TResult>
  ): Promise<EventActorCommitResult> {
    validateTerminalCheckpoint(invocation, terminal.checkpoint);
    const committed = await this.adapter.commit({
      invocation,
      expectedHead: invocation.base,
      checkpoint: terminal.checkpoint,
      result: terminal.result,
      retention: {
        committedCheckpoints: 2,
        dormantCheckpointTtlMs: this.dormantCheckpointTtlMs,
      },
    });
    if (committed.status === 'committed') {
      validateCommittedHead(invocation, committed.head);
    }
    return committed;
  }

  discard(
    invocation: EventActorInvocationReference,
    reason: EventActorDiscardReason
  ): Promise<void> {
    return this.adapter.discard({ invocation, reason });
  }

  private validatePrepareRequest(
    request: EventActorPrepareRequest<TEvent>
  ): void {
    requireNonEmpty(request.actorThreadId, 'actorThreadId');
    requireNonEmpty(request.invocationId, 'invocationId');
    if (
      !Number.isSafeInteger(request.depth) ||
      request.depth < 1 ||
      request.depth > this.maxDepth
    ) {
      throw new Error(
        `Event actor depth ${request.depth} exceeds maximum ${this.maxDepth}`
      );
    }
  }

  async execute(
    request: EventActorExecutionRequest<TEvent>
  ): Promise<EventActorExecutionResult<TResult>> {
    const depth = request.depth ?? 1;
    const prepareRequest: EventActorPrepareRequest<TEvent> = {
      actorThreadId: request.actorThreadId,
      invocationId: request.invocationId,
      depth,
      event: request.event,
    };
    const preparation = await this.prepare(prepareRequest);
    const invocation =
      preparation.status === 'ready'
        ? preparation.invocation
        : await this.coldContinue(prepareRequest, preparation.head);
    const continuation = preparation.status === 'ready' ? 'warm' : 'cold';
    const invocationReference: EventActorInvocationReference = invocation;
    if (isAborted(request.signal)) {
      await this.discard(invocationReference, 'cancelled');
      return { status: 'cancelled', continuation };
    }
    let terminal;
    try {
      terminal = await this.invoke(invocation, request.signal);
    } catch (error) {
      const reason = isAborted(request.signal) ? 'cancelled' : 'failed';
      await this.discard(invocationReference, reason);
      if (reason === 'cancelled') {
        return { status: 'cancelled', continuation };
      }
      return { status: 'failed', error: asError(error), continuation };
    }
    if (isAborted(request.signal)) {
      await this.discard(invocationReference, 'cancelled');
      return { status: 'cancelled', continuation };
    }
    if (terminal.status === 'completed_no_action') {
      await this.discard(invocationReference, 'completed_no_action');
      return { status: 'completed_no_action', continuation };
    }
    try {
      validateTerminalCheckpoint(invocationReference, terminal.checkpoint);
    } catch (error) {
      await this.discard(invocationReference, 'failed');
      return { status: 'failed', error: asError(error), continuation };
    }
    let committed;
    try {
      committed = await this.commit(invocationReference, terminal);
    } catch (error) {
      return {
        status: 'commit_indeterminate',
        error: asError(error),
        continuation,
      };
    }
    if (committed.status === 'stale') {
      await this.discard(invocationReference, 'stale');
      return { status: 'stale', continuation };
    }
    return {
      status: 'applied',
      result: terminal.result,
      head: committed.head,
      continuation,
    };
  }
}

export function createEventActorExecutor<TEvent, TResult>(
  adapter: EventActorHostAdapter<TEvent, TResult>,
  options: EventActorExecutorOptions = {}
): EventActorExecutor<TEvent, TResult> {
  return new EventActorExecutor(adapter, options);
}
