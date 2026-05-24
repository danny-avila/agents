import { ensureHandler } from '@langchain/core/callbacks/manager';
import type {
  BaseCallbackHandler,
  CallbackHandlerMethods,
} from '@langchain/core/callbacks/base';
import type { Callbacks } from '@langchain/core/callbacks/manager';

export type CallbackEntry = BaseCallbackHandler | CallbackHandlerMethods;

export function appendCallbacks(
  callbacks: Callbacks | undefined,
  additions: readonly CallbackEntry[]
): Callbacks {
  if (additions.length === 0) {
    return callbacks ?? [];
  }

  if (callbacks == null) {
    return [...additions];
  }

  if (Array.isArray(callbacks)) {
    return callbacks.concat(additions);
  }

  return callbacks.copy(additions.map(ensureHandler));
}

export function findCallback(
  callbacks: Callbacks | undefined,
  predicate: (callback: CallbackEntry) => boolean
): CallbackEntry | undefined {
  if (callbacks == null) {
    return undefined;
  }

  const handlers = Array.isArray(callbacks) ? callbacks : callbacks.handlers;
  return handlers.find(predicate);
}
