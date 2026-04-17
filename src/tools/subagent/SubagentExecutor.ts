import { nanoid } from 'nanoid';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { Callbacks } from '@langchain/core/callbacks/manager';
import type {
  AgentInputs,
  StandardGraphInput,
  ResolvedSubagentConfig,
  SubagentConfig,
  SubagentUpdateEvent,
  SubagentUpdatePhase,
  TokenCounter,
} from '@/types';
import type { AggregatedHookResult, HookRegistry } from '@/hooks';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs/Graph';
import { GraphEvents, Callback } from '@/common';
import type { HandlerRegistry } from '@/events';
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
  /**
   * Parent-side `tool_call_id` of the `subagent` tool invocation that
   * triggered this execution. Surfaced on {@link SubagentUpdateEvent} so
   * hosts can correlate child updates back to the originating tool call
   * without relying on event ordering heuristics.
   */
  parentToolCallId?: string;
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
  /**
   * Parent's event handler registry. When provided, child-graph events are
   * forwarded through this registry so hosts can:
   *   (a) execute event-driven tools (`ON_TOOL_EXECUTE` routed to parent's handler),
   *   (b) surface child activity to a UI via wrapped {@link GraphEvents.ON_SUBAGENT_UPDATE}.
   * When omitted, the child runs fully isolated (legacy behavior).
   *
   * Can be a direct `HandlerRegistry` or a zero-arg getter — use the getter
   * form when the registry is assigned to the graph AFTER the executor is
   * constructed (the current `Run.create` flow sets `handlerRegistry`
   * post-`createWorkflow`, so `createAgentNode` must capture lazily).
   */
  parentHandlerRegistry?: HandlerRegistry | (() => HandlerRegistry | undefined);
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
  private readonly resolveParentHandlerRegistry?: () =>
    | HandlerRegistry
    | undefined;

  constructor(options: SubagentExecutorOptions) {
    this.configs = options.configs;
    this.parentSignal = options.parentSignal;
    this.hookRegistry = options.hookRegistry;
    this.parentRunId = options.parentRunId;
    this.parentAgentId = options.parentAgentId;
    this.tokenCounter = options.tokenCounter;
    this.maxDepth = options.maxDepth ?? 1;
    this.createChildGraph = options.createChildGraph;
    const rawRegistry = options.parentHandlerRegistry;
    if (typeof rawRegistry === 'function') {
      this.resolveParentHandlerRegistry = rawRegistry;
    } else if (rawRegistry != null) {
      this.resolveParentHandlerRegistry = (): HandlerRegistry => rawRegistry;
    }
  }

  /** Snapshot of the parent's registry at the moment a subagent is dispatched. */
  private getParentHandlerRegistry(): HandlerRegistry | undefined {
    return this.resolveParentHandlerRegistry?.();
  }

  async execute(params: SubagentExecuteParams): Promise<SubagentExecuteResult> {
    const { description, subagentType, threadId, parentToolCallId } = params;
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

    const parentRegistry = this.getParentHandlerRegistry();
    const forwardingEnabled = parentRegistry != null;
    /**
     * Keep `toolDefinitions` only when the host has actually wired an
     * `ON_TOOL_EXECUTE` handler. `Run` always constructs a `HandlerRegistry`,
     * so treating any registry as "forwarding enabled" would leak
     * `toolDefinitions` into children whose hosts cannot execute them — the
     * child's `ToolNode` batch promise would hang forever with no handler to
     * resolve/reject. Gating on the tool-execute handler preserves the
     * recoverable "no tools" path for registry-but-no-handler configs.
     */
    const hasToolExecuteHandler =
      parentRegistry?.getHandler(GraphEvents.ON_TOOL_EXECUTE) != null;
    const childInputs = buildChildInputs(
      config,
      childAgentId,
      this.maxDepth,
      /* keepToolDefinitions */ hasToolExecuteHandler
    );
    const childRunId = `${this.parentRunId}_sub_${nanoid(8)}`;
    const maxTurns = config.maxTurns ?? DEFAULT_MAX_TURNS;

    const childGraph = this.createChildGraph({
      runId: childRunId,
      signal: this.parentSignal,
      agents: [childInputs],
      tokenCounter: this.tokenCounter,
    });

    const forwarder = forwardingEnabled
      ? this.createForwarderCallback({
        parentRegistry: parentRegistry!,
        childGraph,
        subagentType,
        subagentAgentId: childAgentId,
        childRunId,
        parentToolCallId,
      })
      : undefined;

    if (forwarder) {
      await this.emitSubagentUpdate(parentRegistry!, {
        childRunId,
        subagentType,
        subagentAgentId: childAgentId,
        parentToolCallId,
        phase: 'start',
        label: `Subagent "${subagentType}" started`,
      });
    }

    let result: { messages: BaseMessage[] };
    try {
      const workflow = childGraph.createWorkflow();
      /**
       * When `parentHandlerRegistry` is provided (forwarding mode), attach a
       * lightweight callback that intercepts the child's `on_custom_event`
       * dispatches and routes them to the parent's registry — either as
       * operational events (ON_TOOL_EXECUTE) or wrapped ON_SUBAGENT_UPDATE
       * envelopes. Native LangChain streaming events (on_chat_model_stream,
       * etc.) still do NOT propagate to the parent's outer streamEvents
       * iterator — the `callbacks` array REPLACES the inherited chain, so
       * parent handlers won't receive child stream chunks and raise "No
       * agent context found" lookups on the parent's agentContexts map.
       *
       * When no registry is provided (legacy isolation), `callbacks: []`
       * fully detaches the child.
       *
       * `runName` gives the child a distinct LangSmith trace root (avoids
       * nested trace pollution).
       */
      const callbacks: Callbacks = forwarder ? [forwarder] : [];
      result = await workflow.invoke(
        { messages: [new HumanMessage(description)] },
        {
          recursionLimit: maxTurns * RECURSION_MULTIPLIER,
          signal: this.parentSignal,
          callbacks,
          runName: `subagent:${subagentType}`,
          configurable: {
            thread_id: childRunId,
          },
        }
      );
    } catch (error) {
      const errorMessage = truncateErrorMessage(error);
      if (forwarder) {
        await this.emitSubagentUpdate(parentRegistry!, {
          childRunId,
          subagentType,
          subagentAgentId: childAgentId,
          parentToolCallId,
          phase: 'error',
          label: `Subagent "${subagentType}" errored: ${errorMessage}`,
          data: { message: errorMessage },
        });
      }
      childGraph.clearHeavyState();
      return {
        content: `Subagent error: ${errorMessage}`,
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

    if (forwarder) {
      await this.emitSubagentUpdate(parentRegistry!, {
        childRunId,
        subagentType,
        subagentAgentId: childAgentId,
        parentToolCallId,
        phase: 'stop',
        label: `Subagent "${subagentType}" finished`,
      });
    }

    childGraph.clearHeavyState();

    return { content: filteredContent, messages: result.messages };
  }

  /**
   * Emits a single {@link GraphEvents.ON_SUBAGENT_UPDATE} envelope through the
   * parent's handler registry. Silent no-op when no parent registry is set.
   * Errors are swallowed — update events are observational.
   */
  private async emitSubagentUpdate(
    parentRegistry: HandlerRegistry,
    args: {
      childRunId: string;
      subagentType: string;
      subagentAgentId: string;
      parentToolCallId?: string;
      phase: SubagentUpdatePhase;
      data?: unknown;
      label?: string;
    }
  ): Promise<void> {
    const handler = parentRegistry.getHandler(GraphEvents.ON_SUBAGENT_UPDATE);
    if (!handler) {
      return;
    }
    const event: SubagentUpdateEvent = {
      runId: this.parentRunId,
      subagentRunId: args.childRunId,
      subagentType: args.subagentType,
      subagentAgentId: args.subagentAgentId,
      parentAgentId: this.parentAgentId,
      parentToolCallId: args.parentToolCallId,
      phase: args.phase,
      data: args.data,
      label: args.label,
      timestamp: new Date().toISOString(),
    };
    try {
      await handler.handle(GraphEvents.ON_SUBAGENT_UPDATE, event);
    } catch {
      /* observational — swallow */
    }
  }

  /**
   * Builds a BaseCallbackHandler that intercepts the child graph's custom
   * events. Routing rules:
   *   - `ON_TOOL_EXECUTE` → forwarded as-is to the parent's ON_TOOL_EXECUTE
   *     handler (so event-driven tools work identically for child and parent).
   *   - `ON_RUN_STEP` / `ON_RUN_STEP_DELTA` / `ON_RUN_STEP_COMPLETED` /
   *     `ON_MESSAGE_DELTA` / `ON_REASONING_DELTA` → wrapped in a
   *     {@link GraphEvents.ON_SUBAGENT_UPDATE} envelope with a human-readable
   *     label, delivered to the parent's subagent-update handler.
   *   - Everything else → ignored (keeps parent's UI scoped to the events it
   *     cares about; host apps can extend by registering more phases).
   */
  private createForwarderCallback(args: {
    parentRegistry: HandlerRegistry;
    childGraph: StandardGraph;
    subagentType: string;
    subagentAgentId: string;
    childRunId: string;
    parentToolCallId?: string;
  }): BaseCallbackHandler {
    const {
      parentRegistry,
      childGraph: _childGraph,
      subagentType,
      subagentAgentId,
      childRunId,
      parentToolCallId,
    } = args;
    const parentRunId = this.parentRunId;
    const parentAgentId = this.parentAgentId;

    const wrap = async (
      eventName: string,
      phase: SubagentUpdatePhase,
      data: unknown
    ): Promise<void> => {
      const handler = parentRegistry.getHandler(GraphEvents.ON_SUBAGENT_UPDATE);
      if (!handler) {
        return;
      }
      const event: SubagentUpdateEvent = {
        runId: parentRunId,
        subagentRunId: childRunId,
        subagentType,
        subagentAgentId,
        parentAgentId,
        parentToolCallId,
        phase,
        data,
        label: summarizeEvent(eventName, data),
        timestamp: new Date().toISOString(),
      };
      try {
        await handler.handle(GraphEvents.ON_SUBAGENT_UPDATE, event);
      } catch {
        /* observational — swallow */
      }
    };

    const handler = BaseCallbackHandler.fromMethods({
      [Callback.CUSTOM_EVENT]: async (
        eventName: string,
        data: unknown
      ): Promise<void> => {
        if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
          const toolHandler = parentRegistry.getHandler(
            GraphEvents.ON_TOOL_EXECUTE
          );
          if (toolHandler) {
            await toolHandler.handle(
              GraphEvents.ON_TOOL_EXECUTE,
              data as never
            );
          }
          /**
           * We also surface a short notice in the subagent-update stream so
           * the UI can show "calling <tool>" for each tool the child spawns.
           */
          await wrap(eventName, 'run_step', data);
          return;
        }

        if (eventName === GraphEvents.ON_RUN_STEP) {
          await wrap(eventName, 'run_step', data);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_DELTA) {
          await wrap(eventName, 'run_step_delta', data);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
          await wrap(eventName, 'run_step_completed', data);
          return;
        }
        if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
          await wrap(eventName, 'message_delta', data);
          return;
        }
        if (eventName === GraphEvents.ON_REASONING_DELTA) {
          await wrap(eventName, 'reasoning_delta', data);
          return;
        }
      },
    });
    handler.awaitHandlers = true;
    return handler;
  }
}

/**
 * Produces a short single-line label for an arbitrary forwarded child event.
 * Used to populate {@link SubagentUpdateEvent.label} so the host UI can show
 * a compact status ticker without parsing the raw payload.
 */
function summarizeEvent(eventName: string, data: unknown): string {
  if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
    const req = data as { toolCalls?: Array<{ name?: string }> };
    const names = (req.toolCalls ?? [])
      .map((c) => c.name)
      .filter((n): n is string => typeof n === 'string');
    return names.length > 0 ? `Calling ${names.join(', ')}` : 'Calling tool';
  }
  if (eventName === GraphEvents.ON_RUN_STEP) {
    const step = data as {
      type?: string;
      stepDetails?: { type?: string; tool_calls?: Array<{ name?: string }> };
    };
    const detailType = step.stepDetails?.type ?? step.type ?? 'step';
    if (detailType === 'tool_calls') {
      const names = (step.stepDetails?.tool_calls ?? [])
        .map((c) => c.name)
        .filter((n): n is string => typeof n === 'string');
      return names.length > 0
        ? `Using tool: ${names.join(', ')}`
        : 'Planning tool call';
    }
    if (detailType === 'message_creation') {
      return 'Thinking…';
    }
    return `Step: ${detailType}`;
  }
  if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
    const step = data as {
      result?: {
        type?: string;
        tool_call?: { name?: string; output?: string };
      };
    };
    const tool = step.result?.tool_call;
    if (tool?.name != null && tool.name !== '') {
      return `Tool ${tool.name} complete`;
    }
    return 'Step complete';
  }
  if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
    return 'Streaming…';
  }
  return eventName;
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
 * (optionally) event-driven fields. When `allowNested: true`, the child's
 * `maxSubagentDepth` is decremented so that depth is consumed as the call
 * chain deepens across graph boundaries — the parent's executor-level check
 * alone cannot see into the child graph's separate executor.
 *
 * When `keepToolDefinitions` is `true`, the child retains the parent's
 * `toolDefinitions` so event-driven tools remain usable. This is only safe
 * when the caller has wired a forwarder for `ON_TOOL_EXECUTE` to a
 * registered handler — otherwise the child will hang on tool dispatch.
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
  parentMaxDepth: number,
  keepToolDefinitions: boolean = false
): AgentInputs {
  const { agentInputs } = config;
  const childInputs: AgentInputs = {
    ...agentInputs,
    agentId: childAgentId,
    toolDefinitions: keepToolDefinitions
      ? agentInputs.toolDefinitions
      : undefined,
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
