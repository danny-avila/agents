// src/utils/events.ts
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import type { RunnableConfig } from '@langchain/core/runnables';

/**
 * Safely dispatches a custom event and properly awaits it to avoid
 * race conditions where events are dispatched after run cleanup.
 */
export async function safeDispatchCustomEvent(
  event: string,
  payload: unknown,
  config?: RunnableConfig
): Promise<void> {
  try {
    await dispatchCustomEvent(event, payload, config);
  } catch (e) {
    // eslint-disable-next-line no-console
    console.error('Error dispatching custom event:', e);
  }
}
