// src/utils/events.ts
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import type { RunnableConfig } from '@langchain/core/runnables';

/**
 * Safely dispatches a custom event and swallows async rejections to avoid
 * unhandled promise rejections from tracer run-map mismatches.
 */
export function safeDispatchCustomEvent(
  event: string,
  payload: unknown,
  config?: RunnableConfig
): void {
  try {
    const maybePromise = dispatchCustomEvent(
      event,
      payload,
      config
    ) as unknown as Promise<unknown> | void;
    if (
      maybePromise != null &&
      typeof (maybePromise as Promise<unknown>).then === 'function'
    ) {
      void Promise.resolve(maybePromise as Promise<unknown>).catch(() => {});
    }
  } catch {
    // Swallow synchronous errors as well; eventing should not break execution
  }
}
