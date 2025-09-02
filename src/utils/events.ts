/* eslint-disable no-console */
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
    // Check if this is the known EventStreamCallbackHandler error
    if (
      e instanceof Error &&
      e.message.includes('handleCustomEvent: Run ID') &&
      e.message.includes('not found in run map')
    ) {
      // Suppress this specific error - it's expected during parallel execution
      // when EventStreamCallbackHandler loses track of run IDs
      // console.debug('Suppressed error dispatching custom event:', e);
      return;
    }
    // Log other errors
    console.error('Error dispatching custom event:', e);
  }
}
