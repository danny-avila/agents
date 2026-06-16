import type * as t from '@/types';

/**
 * Reconciles a context-usage breakdown's instruction/available/message fields
 * from the pruner's budget metrics. `messageTokens` and `availableForMessages`
 * are DERIVED from `contextBudget` / `effectiveInstructionTokens` /
 * `remainingContextTokens` rather than summed from the index map — that map is
 * keyed by pre-prune indices, so summing it over the kept context would missum.
 * Shared by the live snapshot path (`Graph.createCallModel`) and the pre-send
 * projection (`AgentContext.projectContextUsage`) so both yield identical numbers.
 */
export function syncBudgetDerivedFields(usage: t.ContextUsageEvent): void {
  const { breakdown, contextBudget, effectiveInstructionTokens } = usage;
  if (effectiveInstructionTokens == null) {
    return;
  }
  breakdown.instructionTokens = effectiveInstructionTokens;
  if (contextBudget == null) {
    return;
  }
  breakdown.availableForMessages = Math.max(
    0,
    contextBudget - effectiveInstructionTokens
  );
  if (usage.remainingContextTokens == null) {
    return;
  }
  breakdown.messageTokens = Math.max(
    0,
    contextBudget - effectiveInstructionTokens - usage.remainingContextTokens
  );
}
