import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { AgentContext } from './AgentContext';

export interface ProjectAgentContextUsageParams {
  /** Same `AgentInputs` a run is built from (instructions, tools, model, window). */
  agent: t.AgentInputs;
  /** Branch messages to project, in send order (no leading system message). */
  messages: BaseMessage[];
  tokenCounter: t.TokenCounter;
  /** Per-message counts aligned to `messages` (e.g. from `formatAgentMessages`).
   *  When omitted, counts are recounted via `tokenCounter`. */
  indexTokenCountMap?: Record<string, number>;
  /** Provider-calibrated ratio from a prior snapshot, applied as a static seed. */
  calibrationRatio?: number;
  runId?: string;
  agentId?: string;
}

/**
 * Projects a pre-send context-usage snapshot for a branch under an agent config
 * WITHOUT invoking the model — the host-side (page-load / branch-switch /
 * window-switch) counterpart to the live `ON_CONTEXT_USAGE` event. Builds a
 * throwaway `AgentContext` from the same `AgentInputs` a run uses, awaits its
 * instruction/tool token accounting, then runs the shared pruner + budget math
 * via `AgentContext.projectContextUsage` (which never mutates the supplied
 * messages). Returns null when the config has no tokenizer or context window.
 */
export async function projectAgentContextUsage({
  agent,
  messages,
  tokenCounter,
  indexTokenCountMap,
  calibrationRatio,
  runId,
  agentId,
}: ProjectAgentContextUsageParams): Promise<t.ContextUsageEvent | null> {
  const context = AgentContext.fromConfig(agent, tokenCounter, indexTokenCountMap);
  await context.tokenCalculationPromise;
  return context.projectContextUsage(messages, {
    runId,
    agentId: agentId ?? agent.agentId,
    calibrationRatio,
    indexTokenCountMap,
  });
}
