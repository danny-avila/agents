/* eslint-disable no-console */
// src/utils/events.ts
import { dispatchCustomEvent } from '@langchain/core/callbacks/dispatch';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { AgentLogEvent } from '@/types/graph';
import { GraphEvents } from '@/common';

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

/**
 * Fire-and-forget diagnostic log event.
 * Debug-level logs are gated behind AGENT_DEBUG_LOGGING=true to avoid
 * overhead in production. Info/warn/error always flow through.
 * Pass `force: true` to bypass the env-var gate (e.g. invoke timing).
 */
export function emitAgentLog(
  config: RunnableConfig | undefined,
  level: AgentLogEvent['level'],
  scope: AgentLogEvent['scope'],
  message: string,
  data?: Record<string, unknown>,
  meta?: { runId?: string; agentId?: string },
  options?: { force?: boolean }
): void {
  if (!config) return;
  if (
    level === 'debug' &&
    !(options?.force ?? false) &&
    process.env.AGENT_DEBUG_LOGGING !== 'true'
  )
    return;
  void safeDispatchCustomEvent(
    GraphEvents.ON_AGENT_LOG,
    { level, scope, message, data, ...meta } satisfies AgentLogEvent,
    config
  );
}
