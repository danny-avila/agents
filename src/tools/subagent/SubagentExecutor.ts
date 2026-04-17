import { nanoid } from 'nanoid';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type {
  AgentInputs,
  StandardGraphInput,
  ResolvedSubagentConfig,
  SubagentConfig,
  TokenCounter,
} from '@/types';
import type { AggregatedHookResult, HookRegistry } from '@/hooks';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs/Graph';
import { executeHooks } from '@/hooks';

const DEFAULT_MAX_TURNS = 25;
const RECURSION_MULTIPLIER = 3;
const ERROR_MESSAGE_MAX_CHARS = 200;

const HOOK_FALLBACK: AggregatedHookResult = Object.freeze({
  additionalContexts: [] as string[],
  errors: [] as string[],
});

export type SubagentExecuteParams = {
  description: string;
  subagentType: string;
  threadId?: string;
};

export type SubagentExecuteResult = {
  content: string;
  messages: BaseMessage[];
};

/**
 * Factory that constructs a child graph for subagent execution. Injected
 * rather than imported so that `SubagentExecutor` does not have a runtime
 * dependency on `StandardGraph` — this avoids a circular dependency between
 * `src/graphs/Graph.ts` and `src/tools/subagent/` that would otherwise break
 * Rollup's chunking under `preserveModules`.
 */
export type ChildGraphFactory = (input: StandardGraphInput) => StandardGraph;

export type SubagentExecutorOptions = {
  configs: Map<string, ResolvedSubagentConfig>;
  parentSignal?: AbortSignal;
  hookRegistry?: HookRegistry;
  parentRunId: string;
  parentAgentId?: string;
  tokenCounter?: TokenCounter;
  /** Remaining nesting budget. 0 or negative blocks execution. */
  maxDepth?: number;
  /**
   * Factory for constructing the isolated child graph. Callers pass
   * `(input) => new StandardGraph(input)` — injected to break a circular
   * module dependency.
   */
  createChildGraph: ChildGraphFactory;
};

export class SubagentExecutor {
  private readonly configs: Map<string, ResolvedSubagentConfig>;
  private readonly parentSignal?: AbortSignal;
  private readonly hookRegistry?: HookRegistry;
  private readonly parentRunId: string;
  private readonly parentAgentId?: string;
  private readonly tokenCounter?: TokenCounter;
  private readonly maxDepth: number;
  private readonly createChildGraph: ChildGraphFactory;

  constructor(options: SubagentExecutorOptions) {
    this.configs = options.configs;
    this.parentSignal = options.parentSignal;
    this.hookRegistry = options.hookRegistry;
    this.parentRunId = options.parentRunId;
    this.parentAgentId = options.parentAgentId;
    this.tokenCounter = options.tokenCounter;
    this.maxDepth = options.maxDepth ?? 1;
    this.createChildGraph = options.createChildGraph;
  }

  async execute(params: SubagentExecuteParams): Promise<SubagentExecuteResult> {
    const { description, subagentType, threadId } = params;
    const config = this.configs.get(subagentType);

    if (!config) {
      const available = [...this.configs.keys()].join(', ');
      return {
        content: `Error: Unknown subagent type "${subagentType}". Available types: ${available}`,
        messages: [],
      };
    }

    if (this.maxDepth <= 0) {
      return {
        content: 'Error: Maximum subagent nesting depth exceeded.',
        messages: [],
      };
    }

    const childAgentId =
      config.agentInputs.agentId ||
      `${this.parentAgentId ?? 'agent'}_sub_${nanoid(8)}`;

    if (
      this.hookRegistry?.hasHookFor('SubagentStart', this.parentRunId) === true
    ) {
      const hookResult = await executeHooks({
        registry: this.hookRegistry,
        input: {
          hook_event_name: 'SubagentStart',
          runId: this.parentRunId,
          threadId,
          parentAgentId: this.parentAgentId,
          agentId: childAgentId,
          agentType: subagentType,
          inputs: [new HumanMessage(description)],
        },
        sessionId: this.parentRunId,
        matchQuery: subagentType,
      }).catch((): AggregatedHookResult => HOOK_FALLBACK);

      /**
       * `ask` is treated identically to `deny` in the subagent context:
       * subagents are non-interactive, so there is no prompt path for `ask`.
       * Both decisions block execution and return a "Blocked" tool result.
       */
      if (hookResult.decision === 'deny' || hookResult.decision === 'ask') {
        return {
          content: `Blocked: ${hookResult.reason ?? 'Blocked by hook'}`,
          messages: [],
        };
      }
    }

    const childInputs = buildChildInputs(config, childAgentId, this.maxDepth);
    const childRunId = `${this.parentRunId}_sub_${nanoid(8)}`;
    const maxTurns = config.maxTurns ?? DEFAULT_MAX_TURNS;

    const childGraph = this.createChildGraph({
      runId: childRunId,
      signal: this.parentSignal,
      agents: [childInputs],
      tokenCounter: this.tokenCounter,
    });

    let result: { messages: BaseMessage[] };
    try {
      const workflow = childGraph.createWorkflow();
      /**
       * Detach the child invocation from the parent's callback chain.
       * Without this, `streamEvents` in the parent's `Run.processStream`
       * captures events from the child graph's LLM calls (e.g.
       * `on_chat_model_stream` for the "researcher" agent) and delivers
       * them to the parent's handlers. The parent then tries to resolve
       * the child's agent ID in its own `agentContexts` map and throws
       * "No agent context found for agent ID …". Setting `callbacks: []`
       * overrides the inherited callbacks for this invoke; combined with
       * the child's own empty `handlerRegistry`/`hookRegistry`, the child
       * runs fully isolated.
       *
       * `runName` gives the child a distinct LangSmith trace root (avoids
       * nested trace pollution).
       */
      result = await workflow.invoke(
        { messages: [new HumanMessage(description)] },
        {
          recursionLimit: maxTurns * RECURSION_MULTIPLIER,
          signal: this.parentSignal,
          callbacks: [],
          runName: `subagent:${subagentType}`,
          configurable: {
            thread_id: childRunId,
          },
        }
      );
    } catch (error) {
      childGraph.clearHeavyState();
      return {
        content: `Subagent error: ${truncateErrorMessage(error)}`,
        messages: [],
      };
    }

    const filteredContent = filterSubagentResult(result.messages);

    if (
      this.hookRegistry?.hasHookFor('SubagentStop', this.parentRunId) === true
    ) {
      /**
       * Awaited (not fire-and-forget) for deterministic test synchronization
       * and consistency with PostCompact. The parent is already waiting on the
       * tool result, so the small extra latency is acceptable. Errors are
       * swallowed — SubagentStop is observational.
       */
      await executeHooks({
        registry: this.hookRegistry,
        input: {
          hook_event_name: 'SubagentStop',
          runId: this.parentRunId,
          threadId,
          agentId: childAgentId,
          agentType: subagentType,
          messages: result.messages,
        },
        sessionId: this.parentRunId,
        matchQuery: subagentType,
      }).catch(() => {
        /* SubagentStop is observational — swallow errors */
      });
    }

    childGraph.clearHeavyState();

    return { content: filteredContent, messages: result.messages };
  }
}

/**
 * Walk messages from last to first, returning the text content of the most
 * recent AIMessage that has any. Non-text blocks (tool_use, thinking,
 * redacted_thinking, tool_result) are stripped. If the last AIMessage is
 * pure tool_use (e.g. the subagent hit `maxTurns` mid-tool-call), the walk
 * continues to earlier AIMessages so partial progress is salvaged — this
 * matches Claude Code's behavior in `agentToolUtils.finalizeAgentTool`.
 * Returns "Task completed" only when no AIMessage in the history contains
 * any text.
 */
export function filterSubagentResult(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]._getType() !== 'ai') {
      continue;
    }

    const content = messages[i].content;

    if (typeof content === 'string') {
      if (content) return content;
      continue;
    }

    if (!Array.isArray(content)) {
      continue;
    }

    const textParts: string[] = [];
    for (const block of content) {
      if (typeof block === 'string') {
        textParts.push(block);
      } else if ('type' in block && block.type === 'text' && 'text' in block) {
        textParts.push(block.text as string);
      }
    }

    if (textParts.length > 0) {
      return textParts.join('\n');
    }
  }

  return 'Task completed';
}

/**
 * Resolve self-spawn configs by filling in agentInputs from the parent context.
 * Returns configs with agentInputs guaranteed present. Throws on duplicate
 * `type` values to prevent silent config shadowing.
 */
export function resolveSubagentConfigs(
  configs: SubagentConfig[],
  parentContext: AgentContext
): ResolvedSubagentConfig[] {
  const resolved = configs
    .map((config) => {
      if (config.agentInputs != null) {
        return config as ResolvedSubagentConfig;
      }
      if (config.self !== true || parentContext._sourceInputs == null) {
        return null;
      }
      return {
        ...config,
        agentInputs: { ...parentContext._sourceInputs },
      } as ResolvedSubagentConfig;
    })
    .filter((c): c is ResolvedSubagentConfig => c != null);

  const seenTypes = new Set<string>();
  for (const config of resolved) {
    if (seenTypes.has(config.type)) {
      throw new Error(
        `Duplicate subagent type "${config.type}". Each SubagentConfig must have a unique "type" field.`
      );
    }
    seenTypes.add(config.type);
  }

  return resolved;
}

/**
 * Build child AgentInputs from a resolved config, stripping nesting and
 * event-driven fields. When `allowNested: true`, the child's
 * `maxSubagentDepth` is decremented so that depth is consumed as the call
 * chain deepens across graph boundaries — the parent's executor-level check
 * alone cannot see into the child graph's separate executor.
 *
 * @remarks Advanced utility: exported primarily for testing and by
 * {@link SubagentExecutor}. Host applications configuring subagents should
 * not need to call this directly — it is invoked internally when a subagent
 * tool is dispatched. The depth-countdown contract (parent's `maxDepth` in,
 * child's decremented `maxSubagentDepth` on the returned inputs) is the
 * mechanism that bounds nesting across graph boundaries; callers must
 * respect it.
 */
export function buildChildInputs(
  config: ResolvedSubagentConfig,
  childAgentId: string,
  parentMaxDepth: number
): AgentInputs {
  const { agentInputs } = config;
  const childInputs: AgentInputs = {
    ...agentInputs,
    agentId: childAgentId,
    toolDefinitions: undefined,
  };

  if (config.allowNested === true) {
    childInputs.maxSubagentDepth = Math.max(0, parentMaxDepth - 1);
  } else {
    childInputs.subagentConfigs = undefined;
    childInputs.maxSubagentDepth = undefined;
  }

  return childInputs;
}

function truncateErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.length <= ERROR_MESSAGE_MAX_CHARS) {
    return message;
  }
  return `${message.slice(0, ERROR_MESSAGE_MAX_CHARS)}...`;
}
