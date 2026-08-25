import { isGraphInterrupt, isParentCommand } from '@langchain/langgraph';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import {
  createHash,
  createHmac,
  randomBytes,
  randomUUID,
  timingSafeEqual,
} from 'node:crypto';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  EventActorAdapterPrepareRequest,
  EventActorAppliedResult,
  EventActorCheckpointFork,
  EventActorCheckpointReference,
  EventActorCommitResult,
  EventActorDiscardReason,
  EventActorEvent,
  EventActorExecutionRequest,
  EventActorExecutionResult,
  EventActorExecutorOptions,
  EventActorHead,
  EventActorHostAdapter,
  EventActorIndeterminateResult,
  EventActorInvocation,
  EventActorInvocationResult,
  EventActorInvocationReference,
  EventActorPreparedInvocation,
  EventActorPrepareRequest,
  EventActorPreparation,
  EventActorTerminalResult,
} from './types';

const DEFAULT_MAX_DEPTH = 1;
const DEFAULT_DORMANT_CHECKPOINT_TTL_MS = 24 * 60 * 60 * 1_000;

function createInvocationCheckpointNs(
  request: EventActorPrepareRequest<EventActorEvent>,
  attemptId = randomUUID()
): string {
  return `event-actor/${createHash('sha256')
    .update(request.actorThreadId)
    .update('\0')
    .update(request.invocationId)
    .update('\0')
    .update(attemptId)
    .digest('hex')
    .slice(0, 32)}`;
}

function snapshotEvent<TEvent extends EventActorEvent>(event: TEvent): TEvent {
  const ancestors = new WeakSet<object>();
  const clone = (value: unknown): EventActorEvent => {
    if (
      value === null ||
      typeof value === 'string' ||
      typeof value === 'boolean'
    ) {
      return value;
    }
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        throw new Error('Event actor event numbers must be finite');
      }
      return Object.is(value, -0) ? 0 : value;
    }
    if (typeof value !== 'object') {
      throw new Error('Event actor events must contain only JSON values');
    }
    if (ancestors.has(value)) {
      throw new Error('Event actor events must not contain cycles');
    }
    ancestors.add(value);
    try {
      if (Array.isArray(value)) {
        if (Object.getOwnPropertySymbols(value).length > 0) {
          throw new Error('Event actor event arrays must not contain symbols');
        }
        const snapshot: EventActorEvent[] = [];
        for (let index = 0; index < value.length; index += 1) {
          if (!Object.hasOwn(value, index)) {
            throw new Error('Event actor event arrays must not contain holes');
          }
          snapshot.push(clone(value[index]));
        }
        if (Object.keys(value).length !== value.length) {
          throw new Error(
            'Event actor event arrays must not contain named properties'
          );
        }
        return Object.freeze(snapshot);
      }
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== Object.prototype && prototype !== null) {
        throw new Error('Event actor events must contain only JSON objects');
      }
      if (Object.getOwnPropertySymbols(value).length > 0) {
        throw new Error('Event actor events must not contain symbol keys');
      }
      const snapshot: Record<string, EventActorEvent> = {};
      for (const key of Object.keys(value).sort()) {
        const item = value[key as keyof typeof value];
        Object.defineProperty(snapshot, key, {
          configurable: false,
          enumerable: true,
          writable: false,
          value: clone(item),
        });
      }
      return Object.freeze(snapshot);
    } finally {
      ancestors.delete(value);
    }
  };
  return clone(event) as TEvent;
}

function snapshotCheckpointReference(
  checkpoint: EventActorCheckpointReference
): EventActorCheckpointReference {
  return {
    threadId: checkpoint.threadId,
    ...(checkpoint.checkpointId == null
      ? {}
      : { checkpointId: checkpoint.checkpointId }),
    checkpointNs: checkpoint.checkpointNs,
  };
}

function snapshotCheckpointFork(
  checkpoint: EventActorCheckpointFork
): EventActorCheckpointFork {
  return {
    ...snapshotCheckpointReference(checkpoint),
    invocationId: checkpoint.invocationId,
  };
}

function snapshotHead(head: EventActorHead): EventActorHead {
  return {
    actorThreadId: head.actorThreadId,
    generation: head.generation,
    ...(head.checkpoint == null
      ? {}
      : { checkpoint: snapshotCheckpointReference(head.checkpoint) }),
  };
}

function snapshotInvocationReference(
  invocation: EventActorInvocationReference
): EventActorInvocationReference {
  return {
    actorThreadId: invocation.actorThreadId,
    invocationId: invocation.invocationId,
    depth: invocation.depth,
    continuation: invocation.continuation,
    base: snapshotHead(invocation.base),
    fork: snapshotCheckpointFork(invocation.fork),
  };
}

function snapshotInvocation<TEvent extends EventActorEvent>(
  invocation: EventActorInvocation<TEvent>
): EventActorInvocation<TEvent> {
  return {
    ...snapshotInvocationReference(invocation),
    event: snapshotEvent(invocation.event),
  };
}

function snapshotPrepareRequest<TEvent extends EventActorEvent>(
  request: EventActorPrepareRequest<TEvent>
): EventActorPrepareRequest<TEvent> {
  return {
    actorThreadId: request.actorThreadId,
    invocationId: request.invocationId,
    depth: request.depth,
    event: snapshotEvent(request.event),
  };
}

function freezeInvocationReference(
  invocation: EventActorInvocationReference
): EventActorInvocationReference {
  const snapshot = snapshotInvocationReference(invocation);
  if (snapshot.base.checkpoint != null) {
    Object.freeze(snapshot.base.checkpoint);
  }
  Object.freeze(snapshot.base);
  Object.freeze(snapshot.fork);
  return Object.freeze(snapshot);
}

function freezeInvocation<TEvent extends EventActorEvent>(
  invocation: EventActorInvocation<TEvent>
): EventActorInvocation<TEvent> {
  return Object.freeze({
    ...freezeInvocationReference(invocation),
    event: snapshotEvent(invocation.event),
  });
}

function snapshotPreparedInvocation<TEvent extends EventActorEvent>(
  invocation: EventActorPreparedInvocation<TEvent>
): EventActorPreparedInvocation<TEvent> {
  return Object.freeze({
    ...freezeInvocation(invocation),
    preparationDigest: invocation.preparationDigest,
  });
}

function canonicalHead(head: EventActorHead): object {
  if (head.checkpoint == null) {
    return {
      actorThreadId: head.actorThreadId,
      generation: head.generation,
      checkpoint: null,
    };
  }
  return {
    actorThreadId: head.actorThreadId,
    generation: head.generation,
    checkpoint: {
      threadId: head.checkpoint.threadId,
      checkpointId: head.checkpoint.checkpointId ?? null,
      checkpointNs: head.checkpoint.checkpointNs,
    },
  };
}

function serializeInvocationPreparation<TEvent extends EventActorEvent>(
  invocation: EventActorInvocation<TEvent>
): string {
  return JSON.stringify({
    kind: 'invocation',
    actorThreadId: invocation.actorThreadId,
    invocationId: invocation.invocationId,
    depth: invocation.depth,
    continuation: invocation.continuation,
    base: canonicalHead(invocation.base),
    fork: {
      invocationId: invocation.fork.invocationId,
      threadId: invocation.fork.threadId,
      checkpointId: invocation.fork.checkpointId ?? null,
      checkpointNs: invocation.fork.checkpointNs,
    },
    event: snapshotEvent(invocation.event),
  });
}

function serializeUnavailablePreparation<TEvent extends EventActorEvent>(
  request: EventActorPrepareRequest<TEvent>,
  head: EventActorHead
): string {
  return JSON.stringify({
    kind: 'checkpoint_unavailable',
    request: {
      actorThreadId: request.actorThreadId,
      invocationId: request.invocationId,
      depth: request.depth,
      event: snapshotEvent(request.event),
    },
    head: canonicalHead(head),
  });
}

function freezePrepareRequest<TEvent extends EventActorEvent>(
  request: EventActorPrepareRequest<TEvent>
): EventActorPrepareRequest<TEvent> {
  return Object.freeze(snapshotPrepareRequest(request));
}

function freezeHead(head: EventActorHead): EventActorHead {
  const snapshot = snapshotHead(head);
  if (snapshot.checkpoint != null) {
    Object.freeze(snapshot.checkpoint);
  }
  return Object.freeze(snapshot);
}

function snapshotAmbientConfig(
  config: RunnableConfig | undefined
): RunnableConfig | undefined {
  if (config == null) {
    return undefined;
  }
  return {
    ...config,
    ...(config.tags == null ? {} : { tags: [...config.tags] }),
    ...(config.metadata == null ? {} : { metadata: { ...config.metadata } }),
    ...(config.configurable == null
      ? {}
      : { configurable: { ...config.configurable } }),
  };
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
    if (head.generation > 0) {
      throw new Error('Advanced event actor head has no checkpoint');
    }
    return;
  }
  requireNonEmpty(head.checkpoint.threadId, 'head.checkpoint.threadId');
  if (typeof head.checkpoint.checkpointNs !== 'string') {
    throw new Error('head.checkpoint.checkpointNs must be a string');
  }
  requireNonEmpty(
    head.checkpoint.checkpointId ?? '',
    'head.checkpoint.checkpointId'
  );
}

function validateInvocation<TEvent extends EventActorEvent>(
  request: EventActorPrepareRequest<TEvent>,
  invocation: EventActorInvocation<TEvent>,
  continuation: 'warm' | 'cold',
  checkpointNs: string,
  maxDepth: number,
  expectedHead?: EventActorInvocation<TEvent>['base']
): void {
  validateInvocationReference(invocation, maxDepth);
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

function validateInvocationReference(
  invocation: EventActorInvocationReference,
  maxDepth?: number
): void {
  requireNonEmpty(invocation.actorThreadId, 'actorThreadId');
  requireNonEmpty(invocation.invocationId, 'invocationId');
  if (!Number.isSafeInteger(invocation.depth) || invocation.depth < 1) {
    throw new Error('Event actor invocation depth is invalid');
  }
  if (maxDepth != null && invocation.depth > maxDepth) {
    throw new Error(
      `Event actor depth ${invocation.depth} exceeds maximum ${maxDepth}`
    );
  }
  const continuation: unknown = invocation.continuation;
  if (continuation !== 'warm' && continuation !== 'cold') {
    throw new Error('Event actor invocation continuation is invalid');
  }
  validateHead(invocation.base, invocation.actorThreadId);
  if (invocation.fork.invocationId !== invocation.invocationId) {
    throw new Error(
      'Event actor invocation has mismatched checkpoint ownership'
    );
  }
  requireNonEmpty(invocation.fork.threadId, 'fork.threadId');
  requireNonEmpty(invocation.fork.checkpointNs, 'fork.checkpointNs');
  if (invocation.base.checkpoint != null) {
    if (invocation.fork.threadId !== invocation.base.checkpoint.threadId) {
      throw new Error('Event actor fork changed its logical checkpoint thread');
    }
    requireNonEmpty(
      invocation.fork.checkpointId ?? '',
      'fork.checkpointId for resumed actor'
    );
    if (
      invocation.continuation === 'warm' &&
      invocation.fork.checkpointId !== invocation.base.checkpoint.checkpointId
    ) {
      throw new Error(
        'Warm event actor fork did not start from the committed checkpoint'
      );
    }
  }
}

function checkpointIdsMatch(
  left: EventActorCheckpointFork | EventActorHead['checkpoint'],
  right: EventActorCheckpointFork | EventActorHead['checkpoint']
): boolean {
  return (
    left != null && right != null && left.checkpointId === right.checkpointId
  );
}

function checkpointsMatch(
  left: EventActorCheckpointFork | EventActorHead['checkpoint'],
  right: EventActorCheckpointFork | EventActorHead['checkpoint']
): boolean {
  return (
    checkpointIdsMatch(left, right) &&
    left?.threadId === right?.threadId &&
    left?.checkpointNs === right?.checkpointNs
  );
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
    checkpoint.checkpointId.trim() === '' ||
    checkpoint.checkpointId === invocation.fork.checkpointId ||
    checkpointIdsMatch(checkpoint, invocation.base.checkpoint)
  ) {
    throw new Error(
      'Event actor result escaped its invocation checkpoint fork'
    );
  }
}

function createRunnableConfig(
  invocation: EventActorInvocationReference,
  signal: AbortSignal,
  ambient?: RunnableConfig
): RunnableConfig {
  const {
    signal: _ambientSignal,
    runId: _ambientRunId,
    runName: _ambientRunName,
    callbacks: ambientCallbacks,
    tags: ambientTags,
    metadata: ambientMetadata,
    configurable: ambientConfigurable,
    ...ambientRuntime
  } = ambient ?? {};
  const configurable = Object.fromEntries(
    Object.entries(ambientConfigurable ?? {}).filter(
      ([key]) =>
        !key.startsWith('__pregel_') &&
        !key.startsWith('__librechat_') &&
        key !== 'lc_run_breaker_scope'
    )
  );
  delete configurable.run_id;
  delete configurable.thread_id;
  delete configurable.checkpoint_ns;
  delete configurable.checkpoint_id;
  delete configurable.checkpoint_map;
  delete configurable.event_actor_thread_id;
  delete configurable.event_actor_invocation_id;
  delete configurable.event_actor_generation;
  delete configurable.event_actor_depth;
  delete configurable.event_actor_continuation;
  const metadata = Object.fromEntries(
    Object.entries(ambientMetadata ?? {}).filter(
      ([key]) =>
        !key.startsWith('langgraph_') &&
        !key.startsWith('__pregel_') &&
        key !== 'run_id' &&
        key !== 'thread_id' &&
        key !== 'checkpoint_ns' &&
        key !== 'checkpoint_id' &&
        key !== 'checkpoint_map'
    )
  );
  return {
    ...ambientRuntime,
    signal,
    ...(ambientCallbacks == null ? {} : { callbacks: ambientCallbacks }),
    ...(ambientTags == null ? {} : { tags: ambientTags }),
    metadata: {
      ...metadata,
      thread_id: invocation.fork.threadId,
      checkpoint_ns: invocation.fork.checkpointNs,
      eventActorThreadId: invocation.actorThreadId,
      eventActorInvocationId: invocation.invocationId,
      eventActorGeneration: invocation.base.generation,
      eventActorDepth: invocation.depth,
      eventActorContinuation: invocation.continuation,
    },
    configurable: {
      ...configurable,
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

type EventActorAppliedSnapshot<TResult extends EventActorEvent> = {
  status: 'snapshot_ready';
  result: TResult;
  checkpoint: EventActorCheckpointFork;
  invocation: EventActorInvocationReference;
};

function createIndeterminateResult<TResult extends EventActorEvent>(
  invocation: EventActorInvocationReference,
  error: unknown,
  result?: TResult
): EventActorIndeterminateResult<TResult> {
  return Object.freeze({
    status: 'commit_indeterminate',
    ...(result === undefined ? {} : { result }),
    checkpoint: Object.freeze({
      invocationId: invocation.fork.invocationId,
      threadId: invocation.fork.threadId,
      checkpointNs: invocation.fork.checkpointNs,
    }),
    error: asError(error),
  });
}

function snapshotAppliedTerminal<TResult extends EventActorEvent>(
  invocation: EventActorInvocationReference,
  terminal: Extract<EventActorTerminalResult<TResult>, { status: 'applied' }>
): EventActorAppliedSnapshot<TResult> | EventActorIndeterminateResult<TResult> {
  let result: TResult | undefined;
  try {
    result = snapshotEvent(terminal.result);
    return {
      status: 'snapshot_ready',
      result,
      checkpoint: snapshotCheckpointFork(terminal.checkpoint),
      invocation: freezeInvocationReference(invocation),
    };
  } catch (error) {
    return createIndeterminateResult(invocation, error, result);
  }
}

function isAborted(signal?: AbortSignal): boolean {
  return signal?.aborted === true;
}

class EventActorPreparationCancelledError extends Error {
  constructor(
    readonly continuation: 'warm' | 'cold',
    reason: unknown
  ) {
    super(`Event actor ${continuation} preparation was cancelled`, {
      cause: reason,
    });
    this.name = 'EventActorPreparationCancelledError';
  }
}

function resolveExecutionDepth(
  requestedDepth: number | undefined,
  ambientConfig: RunnableConfig | undefined
): number {
  const ambientDepth = ambientConfig?.configurable?.event_actor_depth;
  if (ambientDepth == null) {
    return requestedDepth ?? 1;
  }
  if (!Number.isSafeInteger(ambientDepth) || Number(ambientDepth) < 1) {
    throw new Error('Ambient event actor depth is invalid');
  }
  const nestedDepth = Number(ambientDepth) + 1;
  if (requestedDepth != null && requestedDepth !== nestedDepth) {
    throw new Error(
      `Nested event actor depth ${requestedDepth} must advance parent depth ${ambientDepth}`
    );
  }
  return nestedDepth;
}

function validateCommittedHead(
  invocation: EventActorInvocationReference,
  checkpoint: EventActorCheckpointFork,
  head: EventActorInvocationReference['base']
): void {
  validateHead(head, invocation.actorThreadId, true);
  if (
    head.generation !== invocation.base.generation + 1 ||
    head.checkpoint?.threadId !== checkpoint.threadId ||
    head.checkpoint.checkpointNs !== checkpoint.checkpointNs ||
    head.checkpoint.checkpointId !== checkpoint.checkpointId
  ) {
    throw new Error('Event actor commit returned an invalid logical head');
  }
}

/**
 * Runs one event against an isolated checkpoint fork and advances the stable
 * actor head only through the host's atomic commit interface.
 */
export class EventActorExecutor<
  TEvent extends EventActorEvent,
  TResult extends EventActorEvent,
> {
  readonly #adapter: EventActorHostAdapter<TEvent, TResult>;
  readonly #maxDepth: number;
  readonly #dormantCheckpointTtlMs: number;
  readonly #preparationSigningKey: Uint8Array;
  readonly #issuedSettlements = new WeakSet<object>();

  constructor(
    adapter: EventActorHostAdapter<TEvent, TResult>,
    options: EventActorExecutorOptions = {}
  ) {
    this.#adapter = adapter;
    this.#maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH;
    this.#dormantCheckpointTtlMs =
      options.dormantCheckpointTtlMs ?? DEFAULT_DORMANT_CHECKPOINT_TTL_MS;
    const signingKey = Buffer.from(
      options.preparationSigningKey ?? randomBytes(32)
    );
    if (signingKey.byteLength < 32) {
      throw new Error('preparationSigningKey must contain at least 32 bytes');
    }
    this.#preparationSigningKey = signingKey;
    if (!Number.isSafeInteger(this.#maxDepth) || this.#maxDepth < 1) {
      throw new Error('maxDepth must be a positive safe integer');
    }
    if (
      !Number.isSafeInteger(this.#dormantCheckpointTtlMs) ||
      this.#dormantCheckpointTtlMs < 1
    ) {
      throw new Error('dormantCheckpointTtlMs must be a positive safe integer');
    }
  }

  #signPreparation(payload: string): string {
    return createHmac('sha256', this.#preparationSigningKey)
      .update(payload)
      .digest('hex');
  }

  #preparationSignatureMatches(signature: string, payload: string): boolean {
    if (!/^[a-f0-9]{64}$/.test(signature)) {
      return false;
    }
    return timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(this.#signPreparation(payload), 'hex')
    );
  }

  #createPreparedInvocation(
    invocation: EventActorInvocation<TEvent>
  ): EventActorPreparedInvocation<TEvent> {
    const trustedInvocation = freezeInvocation(invocation);
    return Object.freeze({
      ...trustedInvocation,
      preparationDigest: this.#signPreparation(
        serializeInvocationPreparation(trustedInvocation)
      ),
    });
  }

  #validatePreparedInvocation(
    invocation: EventActorPreparedInvocation<TEvent>
  ): void {
    requireNonEmpty(invocation.preparationDigest, 'preparationDigest');
    if (
      !this.#preparationSignatureMatches(
        invocation.preparationDigest,
        serializeInvocationPreparation(invocation)
      )
    ) {
      throw new Error('Event actor prepared invocation binding is invalid');
    }
  }

  async prepare(
    request: EventActorPrepareRequest<TEvent>,
    signal?: AbortSignal
  ): Promise<EventActorPreparation<TEvent>> {
    const trustedRequest = snapshotPrepareRequest(request);
    resolveExecutionDepth(
      trustedRequest.depth,
      AsyncLocalStorageProviderSingleton.getRunnableConfig()
    );
    this.#validatePrepareRequest(trustedRequest);
    const checkpointNs = createInvocationCheckpointNs(trustedRequest);
    const adapterRequest: EventActorAdapterPrepareRequest<TEvent> = {
      ...snapshotPrepareRequest(trustedRequest),
      checkpointNs,
    };
    const controller = new AbortController();
    const abort = (): void => controller.abort(signal?.reason);
    if (isAborted(signal)) {
      abort();
    } else {
      signal?.addEventListener('abort', abort, { once: true });
    }
    if (isAborted(controller.signal)) {
      signal?.removeEventListener('abort', abort);
      throw new EventActorPreparationCancelledError(
        'warm',
        controller.signal.reason
      );
    }
    let preparation;
    try {
      preparation = await this.#adapter.prepare(
        { ...adapterRequest },
        { signal: controller.signal }
      );
    } catch (error) {
      if (isAborted(controller.signal) && error === controller.signal.reason) {
        throw new EventActorPreparationCancelledError('warm', error);
      }
      throw error;
    } finally {
      signal?.removeEventListener('abort', abort);
    }
    if (preparation.status === 'ready') {
      const adapterInvocation = snapshotInvocation(preparation.invocation);
      validateInvocation(
        trustedRequest,
        adapterInvocation,
        'warm',
        checkpointNs,
        this.#maxDepth
      );
      const preparedInvocation = this.#createPreparedInvocation({
        ...adapterInvocation,
        event: snapshotEvent(trustedRequest.event),
      });
      if (isAborted(controller.signal)) {
        await this.#discardInvocationReference(
          snapshotInvocationReference(preparedInvocation),
          'cancelled'
        );
        throw new EventActorPreparationCancelledError(
          'warm',
          controller.signal.reason
        );
      }
      return Object.freeze({
        status: 'ready',
        invocation: preparedInvocation,
      });
    } else {
      const preparedHead = freezeHead(preparation.head);
      validateHead(preparedHead, trustedRequest.actorThreadId);
      if (isAborted(controller.signal)) {
        throw new EventActorPreparationCancelledError(
          'warm',
          controller.signal.reason
        );
      }
      const preparedRequest = freezePrepareRequest(trustedRequest);
      return Object.freeze({
        status: 'checkpoint_unavailable',
        request: preparedRequest,
        head: preparedHead,
        preparationDigest: this.#signPreparation(
          serializeUnavailablePreparation(preparedRequest, preparedHead)
        ),
      });
    }
  }

  async coldContinue(
    preparation: Extract<
      EventActorPreparation<TEvent>,
      { status: 'checkpoint_unavailable' }
    >,
    signal?: AbortSignal
  ): Promise<EventActorPreparedInvocation<TEvent>> {
    const request = snapshotPrepareRequest(preparation.request);
    const trustedHead = snapshotHead(preparation.head);
    const preparationDigest = preparation.preparationDigest;
    requireNonEmpty(preparationDigest, 'preparationDigest');
    if (
      !this.#preparationSignatureMatches(
        preparationDigest,
        serializeUnavailablePreparation(request, trustedHead)
      )
    ) {
      throw new Error('Event actor unavailable preparation binding is invalid');
    }
    resolveExecutionDepth(
      request.depth,
      AsyncLocalStorageProviderSingleton.getRunnableConfig()
    );
    this.#validatePrepareRequest(request);
    validateHead(trustedHead, request.actorThreadId);
    const checkpointNs = createInvocationCheckpointNs(request);
    const adapterRequest: EventActorAdapterPrepareRequest<TEvent> = {
      ...snapshotPrepareRequest(request),
      checkpointNs,
    };
    const controller = new AbortController();
    const abort = (): void => controller.abort(signal?.reason);
    if (isAborted(signal)) {
      abort();
    } else {
      signal?.addEventListener('abort', abort, { once: true });
    }
    if (isAborted(controller.signal)) {
      signal?.removeEventListener('abort', abort);
      throw new EventActorPreparationCancelledError(
        'cold',
        controller.signal.reason
      );
    }
    let invocation;
    try {
      invocation = await this.#adapter.coldContinue(
        { ...adapterRequest },
        snapshotHead(trustedHead),
        { signal: controller.signal }
      );
    } catch (error) {
      if (isAborted(controller.signal) && error === controller.signal.reason) {
        throw new EventActorPreparationCancelledError('cold', error);
      }
      throw error;
    } finally {
      signal?.removeEventListener('abort', abort);
    }
    const adapterInvocation = snapshotInvocation(invocation);
    validateInvocation(
      request,
      adapterInvocation,
      'cold',
      checkpointNs,
      this.#maxDepth,
      trustedHead
    );
    const trustedInvocation: EventActorInvocation<TEvent> = {
      ...adapterInvocation,
      event: snapshotEvent(request.event),
    };
    if (isAborted(controller.signal)) {
      await this.#discardInvocationReference(
        snapshotInvocationReference(trustedInvocation),
        'cancelled'
      );
      throw new EventActorPreparationCancelledError(
        'cold',
        controller.signal.reason
      );
    }
    return this.#createPreparedInvocation(trustedInvocation);
  }

  async invoke(
    invocation: EventActorPreparedInvocation<TEvent>,
    signal?: AbortSignal
  ): Promise<EventActorInvocationResult<TResult>> {
    const trustedInvocation = snapshotPreparedInvocation(invocation);
    this.#validatePreparedInvocation(trustedInvocation);
    const settlementInvocation = snapshotInvocationReference(trustedInvocation);
    const terminal = await this.#invokeWithConfig(
      snapshotInvocation(trustedInvocation),
      signal,
      AsyncLocalStorageProviderSingleton.getRunnableConfig()
    );
    if (terminal.status === 'applied') {
      const snapshot = snapshotAppliedTerminal(settlementInvocation, terminal);
      return snapshot.status === 'snapshot_ready'
        ? this.#issueSettlement(snapshot)
        : snapshot;
    }
    return Object.freeze({
      status: 'completed_no_action' as const,
      ...(terminal.result === undefined
        ? {}
        : { result: snapshotEvent(terminal.result) }),
    });
  }

  #issueSettlement(
    snapshot: EventActorAppliedSnapshot<TResult>
  ): EventActorAppliedResult<TResult> {
    const settlement = Object.freeze({
      status: 'applied' as const,
      result: snapshot.result,
      checkpoint: Object.freeze(snapshotCheckpointFork(snapshot.checkpoint)),
      invocation: freezeInvocationReference(snapshot.invocation),
    });
    this.#issuedSettlements.add(settlement);
    return settlement;
  }

  async #invokeWithConfig(
    invocation: EventActorInvocation<TEvent>,
    signal: AbortSignal | undefined,
    ambientConfig: RunnableConfig | undefined
  ): Promise<EventActorTerminalResult<TResult>> {
    resolveExecutionDepth(invocation.depth, ambientConfig);
    validateInvocation(
      {
        actorThreadId: invocation.actorThreadId,
        invocationId: invocation.invocationId,
        depth: invocation.depth,
        event: invocation.event,
      },
      invocation,
      invocation.continuation,
      invocation.fork.checkpointNs,
      this.#maxDepth
    );
    const controller = new AbortController();
    const abort = (): void => controller.abort(signal?.reason);
    if (isAborted(signal)) {
      abort();
    } else {
      signal?.addEventListener('abort', abort, { once: true });
    }
    const config = createRunnableConfig(
      invocation,
      controller.signal,
      ambientConfig
    );
    try {
      if (controller.signal.aborted) {
        throw asError(controller.signal.reason ?? 'Event actor cancelled');
      }
      return await AsyncLocalStorageProviderSingleton.runWithConfig(
        config,
        () =>
          this.#adapter.invoke(invocation, {
            signal: controller.signal,
            config,
          })
      );
    } finally {
      signal?.removeEventListener('abort', abort);
    }
  }

  async commit(
    settlement: EventActorAppliedResult<TResult>
  ): Promise<EventActorCommitResult> {
    if (!this.#issuedSettlements.has(settlement)) {
      throw new Error('Event actor settlement was not issued by this executor');
    }
    const trustedInvocation = snapshotInvocationReference(
      settlement.invocation
    );
    validateInvocationReference(trustedInvocation, this.#maxDepth);
    const trustedCheckpoint = snapshotCheckpointFork(settlement.checkpoint);
    validateTerminalCheckpoint(trustedInvocation, trustedCheckpoint);
    const committed = await this.#adapter.commit({
      invocation: snapshotInvocationReference(trustedInvocation),
      expectedHead: snapshotHead(trustedInvocation.base),
      checkpoint: { ...trustedCheckpoint },
      result: settlement.result,
      retention: {
        committedCheckpoints: 2,
        dormantCheckpointTtlMs: this.#dormantCheckpointTtlMs,
      },
    });
    if (committed.status === 'committed') {
      const committedHead = snapshotHead(committed.head);
      validateCommittedHead(
        trustedInvocation,
        trustedCheckpoint,
        committedHead
      );
      return { status: 'committed', head: committedHead };
    }
    if (committed.head != null) {
      const committedHead = snapshotHead(committed.head);
      validateHead(committedHead, trustedInvocation.actorThreadId);
      if (committedHead.generation <= trustedInvocation.base.generation) {
        throw new Error('Stale event actor head did not advance past its base');
      }
      if (
        trustedInvocation.base.checkpoint != null &&
        committedHead.checkpoint?.threadId !==
          trustedInvocation.base.checkpoint.threadId
      ) {
        throw new Error('Stale event actor head changed its checkpoint thread');
      }
      if (
        checkpointIdsMatch(
          committedHead.checkpoint,
          trustedInvocation.base.checkpoint
        ) ||
        checkpointIdsMatch(committedHead.checkpoint, trustedInvocation.fork) ||
        checkpointsMatch(committedHead.checkpoint, trustedCheckpoint)
      ) {
        throw new Error(
          'Stale event actor head does not identify a competing checkpoint'
        );
      }
      return { status: 'stale', head: committedHead };
    }
    return committed;
  }

  discard(
    invocation: EventActorPreparedInvocation<TEvent>,
    reason: EventActorDiscardReason
  ): Promise<void> {
    const trustedInvocation = snapshotPreparedInvocation(invocation);
    this.#validatePreparedInvocation(trustedInvocation);
    return this.#discardInvocationReference(trustedInvocation, reason);
  }

  #discardInvocationReference(
    invocation: EventActorInvocationReference,
    reason: EventActorDiscardReason
  ): Promise<void> {
    const trustedInvocation = snapshotInvocationReference(invocation);
    validateInvocationReference(trustedInvocation, this.#maxDepth);
    return this.#adapter.discard({
      invocation: trustedInvocation,
      reason,
    });
  }

  #validatePrepareRequest(request: EventActorPrepareRequest<TEvent>): void {
    requireNonEmpty(request.actorThreadId, 'actorThreadId');
    requireNonEmpty(request.invocationId, 'invocationId');
    if (
      !Number.isSafeInteger(request.depth) ||
      request.depth < 1 ||
      request.depth > this.#maxDepth
    ) {
      throw new Error(
        `Event actor depth ${request.depth} exceeds maximum ${this.#maxDepth}`
      );
    }
  }

  async execute(
    request: EventActorExecutionRequest<TEvent>
  ): Promise<EventActorExecutionResult<TResult>> {
    const trustedRequest: EventActorExecutionRequest<TEvent> = {
      actorThreadId: request.actorThreadId,
      invocationId: request.invocationId,
      event: snapshotEvent(request.event),
      ...(request.depth == null ? {} : { depth: request.depth }),
      ...(request.signal == null ? {} : { signal: request.signal }),
    };
    const ambientConfig = snapshotAmbientConfig(
      AsyncLocalStorageProviderSingleton.getRunnableConfig()
    );
    const depth = resolveExecutionDepth(trustedRequest.depth, ambientConfig);
    const prepareRequest: EventActorPrepareRequest<TEvent> = {
      actorThreadId: trustedRequest.actorThreadId,
      invocationId: trustedRequest.invocationId,
      depth,
      event: trustedRequest.event,
    };
    let preparation;
    try {
      preparation = await this.prepare(prepareRequest, trustedRequest.signal);
    } catch (error) {
      if (error instanceof EventActorPreparationCancelledError) {
        return { status: 'cancelled', continuation: error.continuation };
      }
      throw error;
    }
    if (
      preparation.status === 'checkpoint_unavailable' &&
      isAborted(trustedRequest.signal)
    ) {
      return { status: 'cancelled', continuation: 'cold' };
    }
    let invocation;
    try {
      invocation =
        preparation.status === 'ready'
          ? preparation.invocation
          : await this.coldContinue(preparation, trustedRequest.signal);
    } catch (error) {
      if (error instanceof EventActorPreparationCancelledError) {
        return { status: 'cancelled', continuation: 'cold' };
      }
      throw error;
    }
    const continuation = preparation.status === 'ready' ? 'warm' : 'cold';
    const invocationReference = snapshotInvocationReference(invocation);
    const invocationForAdapter = snapshotInvocation(invocation);
    if (isAborted(trustedRequest.signal)) {
      await this.#discardInvocationReference(invocationReference, 'cancelled');
      return { status: 'cancelled', continuation };
    }
    let terminal;
    try {
      terminal = await this.#invokeWithConfig(
        invocationForAdapter,
        trustedRequest.signal,
        ambientConfig
      );
    } catch (error) {
      if (isGraphInterrupt(error) || isParentCommand(error)) {
        throw error;
      }
      const reason = isAborted(trustedRequest.signal) ? 'cancelled' : 'failed';
      await this.#discardInvocationReference(invocationReference, reason);
      if (reason === 'cancelled') {
        return { status: 'cancelled', continuation };
      }
      return { status: 'failed', error: asError(error), continuation };
    }
    if (terminal.status === 'completed_no_action') {
      const result =
        terminal.result === undefined
          ? undefined
          : snapshotEvent(terminal.result);
      await this.#discardInvocationReference(
        invocationReference,
        'completed_no_action'
      );
      return {
        status: 'completed_no_action',
        ...(result === undefined ? {} : { result }),
        continuation,
      };
    }
    const appliedSnapshot = snapshotAppliedTerminal(
      invocationReference,
      terminal
    );
    if (appliedSnapshot.status === 'commit_indeterminate') {
      return { ...appliedSnapshot, continuation };
    }
    try {
      validateTerminalCheckpoint(
        invocationReference,
        appliedSnapshot.checkpoint
      );
    } catch (error) {
      return {
        ...createIndeterminateResult(
          invocationReference,
          error,
          appliedSnapshot.result
        ),
        continuation,
      };
    }
    const trustedTerminal = this.#issueSettlement(appliedSnapshot);
    let committed;
    try {
      committed = await this.commit(trustedTerminal);
    } catch (error) {
      return {
        status: 'commit_indeterminate',
        result: trustedTerminal.result,
        checkpoint: { ...trustedTerminal.checkpoint },
        error: asError(error),
        continuation,
      };
    }
    if (committed.status === 'stale') {
      return {
        status: 'commit_conflict',
        result: trustedTerminal.result,
        checkpoint: { ...trustedTerminal.checkpoint },
        ...(committed.head == null
          ? {}
          : { head: snapshotHead(committed.head) }),
        continuation,
      };
    }
    return {
      status: 'applied',
      result: trustedTerminal.result,
      head: committed.head,
      continuation,
    };
  }
}

export function createEventActorExecutor<
  TEvent extends EventActorEvent,
  TResult extends EventActorEvent,
>(
  adapter: EventActorHostAdapter<TEvent, TResult>,
  options: EventActorExecutorOptions = {}
): EventActorExecutor<TEvent, TResult> {
  return new EventActorExecutor(adapter, options);
}
