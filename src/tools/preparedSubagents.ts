import type { ToolCall } from '@langchain/core/messages/tool';
import { stableStringify, normalizeError } from './eagerEventExecution';

/** A model attempt cannot safely retry after delegated work has started. */
export class PreparedSubagentError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'PreparedSubagentError';
  }
}

type Outcome =
  | { output: unknown; error?: never }
  | { error: Error; output?: never };
type Attempt = { keys: Set<string>; cancelled: boolean };

const PREPARED_INVOCATION = Symbol('prepared-subagent-invocation');
type PreparedCall = ToolCall & { [PREPARED_INVOCATION]?: symbol };

type Reservation = {
  token: symbol;
  callId: string;
  attempt: string;
  fingerprint: string;
  controller: AbortController;
  outcome: Promise<Outcome>;
  committed: boolean;
};

/**
 * Owns speculative invocation work, never tool results or graph state. Only
 * explicitly open model attempts admit work; closing an attempt fences late
 * callbacks without retaining tombstones. Normal ToolNode execution adopts the
 * raw output and performs its existing lifecycle/output processing once.
 */
export class PreparedSubagents {
  private readonly attempts = new Map<string, Attempt>();
  private readonly reservations = new Map<string, Reservation>();
  private readonly running = new Set<AbortController>();
  private epoch = 0;

  begin(attempt: string): void {
    this.attempts.set(attempt, { keys: new Set(), cancelled: false });
  }

  isOpen(attempt: string): boolean {
    return this.attempts.get(attempt)?.cancelled === false;
  }

  start(
    attempt: string,
    owner: string,
    call: ToolCall,
    capacity: number,
    invoke: (signal: AbortSignal) => Promise<unknown>
  ): boolean {
    const entries = this.attempts.get(attempt);
    if (
      entries == null ||
      entries.cancelled ||
      call.id == null ||
      call.id === ''
    ) {
      return false;
    }
    const key = JSON.stringify([owner, call.id]);
    const canonical = fingerprint(call);
    const previous = this.reservations.get(key);
    if (previous != null) {
      if (previous.attempt !== attempt || previous.fingerprint !== canonical) {
        throw new PreparedSubagentError(
          'Conflicting eager subagent call identity.'
        );
      }
      return false;
    }
    if (
      !Number.isSafeInteger(capacity) ||
      capacity <= 0 ||
      this.reservations.size >= capacity ||
      this.running.size >= capacity
    ) {
      return false;
    }
    const controller = new AbortController();
    this.running.add(controller);
    const reservation: Reservation = {
      token: Symbol(),
      attempt,
      callId: call.id,
      fingerprint: canonical,
      controller,
      committed: false,
      outcome: Promise.resolve()
        .then(() => {
          controller.signal.throwIfAborted();
          return invoke(controller.signal);
        })
        .then(
          (output): Outcome => ({ output }),
          (error): Outcome => ({ error: normalizeError(error) })
        )
        .finally(() => {
          this.running.delete(controller);
        }),
    };
    entries.keys.add(key);
    this.reservations.set(key, reservation);
    return true;
  }

  finish(attempt: string, calls?: ToolCall[], cause?: unknown): void {
    const entries = this.attempts.get(attempt);
    this.attempts.delete(attempt);
    if (entries == null || entries.keys.size === 0) {
      return;
    }
    const finalCalls = new Map(calls?.map((call) => [call.id, call]));
    for (const key of entries.keys) {
      const record = this.reservations.get(key);
      const finalCall =
        record == null ? undefined : finalCalls.get(record.callId);
      if (
        entries.cancelled ||
        record == null ||
        record.attempt !== attempt ||
        finalCall == null ||
        fingerprint(finalCall) !== record.fingerprint
      ) {
        const error = new PreparedSubagentError(
          'The model attempt ended or changed after a subagent started; refusing automatic retry.',
          { cause }
        );
        for (const pendingKey of entries.keys) {
          const pending = this.reservations.get(pendingKey);
          if (pending?.attempt === attempt) {
            pending.controller.abort(error);
            this.reservations.delete(pendingKey);
          }
        }
        throw error;
      }
      record.committed = true;
      (finalCall as PreparedCall)[PREPARED_INVOCATION] = record.token;
    }
  }

  owns(owner: string, call: ToolCall): boolean {
    const record = this.reservations.get(JSON.stringify([owner, call.id]));
    return record != null && (call as PreparedCall)[PREPARED_INVOCATION] === record.token;
  }

  take(owner: string, call: ToolCall): Promise<unknown> | undefined {
    const key = JSON.stringify([owner, call.id]);
    const record = this.reservations.get(key);
    const token = (call as PreparedCall)[PREPARED_INVOCATION];
    if (record == null && token == null) {
      return undefined;
    }
    if (record == null || token !== record.token) {
      return Promise.reject(new PreparedSubagentError('Eager subagent invocation is no longer owned by this call.'));
    }
    this.reservations.delete(key);
    if (!record.committed || record.fingerprint !== fingerprint(call)) {
      const error = new PreparedSubagentError(
        'Subagent arguments changed after eager invocation.'
      );
      record.controller.abort(error);
      return Promise.reject(error);
    }
    const epoch = this.epoch;
    return record.outcome.then((result) => {
      if (this.epoch !== epoch) {
        throw new PreparedSubagentError('Eager subagent result belongs to a retired run.');
      }
      record.controller.signal.throwIfAborted();
      if ('error' in result) {
        throw result.error;
      }
      return result.output;
    });
  }

  clear(): void {
    this.epoch++;
    const reason = new PreparedSubagentError(
      'Eager subagent execution was cancelled.'
    );
    for (const controller of this.running) {
      controller.abort(reason);
    }
    this.reservations.clear();
    for (const attempt of this.attempts.values()) {
      attempt.cancelled = true;
    }
  }
}

function fingerprint(call: ToolCall): string {
  return stableStringify({ name: call.name, args: call.args });
}
