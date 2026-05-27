import { nanoid } from 'nanoid';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { Callbacks } from '@langchain/core/callbacks/manager';
import type {
  AgentInputs,
  MessageDeltaEvent,
  ProcessedToolCall,
  ReasoningDeltaEvent,
  RunStep,
  RunStepDeltaEvent,
  StandardGraphInput,
  ResolvedSubagentConfig,
  StepCompleted,
  SubagentConfig,
  SubagentUpdateEvent,
  SubagentUpdatePhase,
  ToolExecuteBatchRequest,
  ToolCallDelta,
  TokenCounter,
} from '@/types';
import type { AggregatedHookResult, HookRegistry } from '@/hooks';
import type { AgentContext } from '@/agents/AgentContext';
import type { StandardGraph } from '@/graphs/Graph';
import { GraphEvents, Callback, StepTypes } from '@/common';
import type { HandlerRegistry } from '@/events';
import { executeHooks } from '@/hooks';

const DEFAULT_MAX_TURNS = 25;
const RECURSION_MULTIPLIER = 3;
const ERROR_MESSAGE_MAX_CHARS = 200;
const MAX_PENDING_SUBAGENT_UPDATES = 64;

const HOOK_FALLBACK: AggregatedHookResult = Object.freeze({
  additionalContexts: [] as string[],
  errors: [] as string[],
});

type SanitizedSubagentToolCall = {
  id: string;
  name: string;
  args?: ToolExecuteBatchRequest['toolCalls'][number]['args'];
};

type SanitizedSubagentToolExecuteData = {
  toolCalls: SanitizedSubagentToolCall[];
  agentId?: string;
};

type SanitizedRunStep = Partial<
  Pick<
    RunStep,
    | 'agentId'
    | 'groupId'
    | 'id'
    | 'index'
    | 'runId'
    | 'stepIndex'
    | 'summary'
    | 'type'
    | 'usage'
  >
> & {
  stepDetails?: SanitizedStepDetails;
};

type SanitizedStepDetails =
  | {
      type: StepTypes.MESSAGE_CREATION;
      message_creation?: {
        message_id?: string;
      };
    }
  | {
      type: StepTypes.TOOL_CALLS;
      tool_calls?: SanitizedAgentToolCall[];
    };

type SanitizedAgentToolCall = {
  id?: string;
  name?: string;
  args?: string | object;
  type?: string;
  function?: {
    name?: string;
    arguments?: string | object;
  };
};

type SanitizedRunStepDelta = Partial<Pick<RunStepDeltaEvent, 'id'>> & {
  delta?: SanitizedToolCallDelta;
};

type SanitizedToolCallDelta = Partial<
  Pick<ToolCallDelta, 'auth' | 'expires_at' | 'summary' | 'type'>
> & {
  tool_calls?: SanitizedAgentToolCall[];
};

type SanitizedStepCompleted =
  | {
      id?: string;
      index?: number;
      type: 'tool_call';
      tool_call?: SanitizedProcessedToolCall;
    }
  | {
      type: 'summary';
      summary?: Extract<StepCompleted, { type: 'summary' }>['summary'];
    };

type SanitizedProcessedToolCall = Partial<
  Pick<ProcessedToolCall, 'args' | 'id' | 'name' | 'output' | 'progress'>
>;

type SanitizedRunStepCompleted = {
  result?: SanitizedStepCompleted;
};

type SanitizedMessageDelta = Partial<Pick<MessageDeltaEvent, 'id'>> & {
  delta?: {
    content?: MessageDeltaEvent['delta']['content'];
    tool_call_ids?: MessageDeltaEvent['delta']['tool_call_ids'];
  };
};

type SanitizedReasoningDelta = Partial<Pick<ReasoningDeltaEvent, 'id'>> & {
  delta?: {
    content?: ReasoningDeltaEvent['delta']['content'];
  };
};

type QueuedSubagentUpdate = {
  eventName: string;
  phase: SubagentUpdatePhase;
  data: unknown;
};

type ForwarderCallback = {
  handler: BaseCallbackHandler;
  drain: () => Promise<void>;
};

const LANGGRAPH_RUNTIME_CONFIG_PREFIX = '__pregel_';
const LANGGRAPH_CHECKPOINT_CONFIG_KEYS = new Set([
  'checkpoint_id',
  'checkpoint_map',
  'checkpoint_ns',
]);

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
  /**
   * Snapshot of the parent invocation's host `config.configurable` at
   * the spawn-tool call site. Host-set fields (`requestBody`, `user`,
   * `userMCPAuthMap`, etc.) propagate into the child workflow's
   * `configurable` — fixing MCP body-placeholder substitution and
   * per-user lookups for subagent tool calls. LangGraph runtime keys
   * (`__pregel_*`, checkpoint bookkeeping) are intentionally not
   * inherited; the child graph recreates its own runtime config.
   *
   * Inheritance details (verified empirically against LangGraph):
   *   - host-set keys propagate as-is into the child's tool dispatches;
   *   - `thread_id` propagates (with `childRunId` as a fallback when
   *     parent did not supply one) — matches the "subagent is part of
   *     the same conversation" mental model and aligns with the
   *     `sessionId: this.parentRunId` convention this executor already
   *     uses for `SubagentStart` / `SubagentStop` hooks;
   *   - `parent_run_id` propagates when the host put it on parent's
   *     configurable;
   *   - `run_id` is *overwritten by the LangGraph runtime* at child
   *     invoke time regardless of what we forward — child's tool
   *     dispatches see the child graph's runtime runId in
   *     `configurable.run_id`, not the parent's. Hosts that need
   *     parent-scoped run identity for downstream consumers should
   *     plumb it via a host-defined key (e.g. `requestBody.messageId`),
   *     not `run_id`.
   *
   * A future revision will likely make this inheritance configurable
   * per spawn type — background / async subagents may want isolation
   * rather than sharing parent's host context.
   */
  parentConfigurable?: Record<string, unknown>;
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
  langfuse?: StandardGraphInput['langfuse'];
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
  private readonly langfuse?: StandardGraphInput['langfuse'];
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
    this.langfuse = options.langfuse;
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
      langfuse: this.langfuse,
      tokenCounter: this.tokenCounter,
    });

    let forwarding: ForwarderCallback | undefined;
    if (forwardingEnabled) {
      forwarding = this.createForwarderCallback({
        parentRegistry: parentRegistry!,
        subagentType,
        subagentAgentId: childAgentId,
        childRunId,
        parentToolCallId,
      });
    }
    const forwarder = forwarding?.handler;

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
      /**
       * Inherit the parent's host `configurable` — host-set fields
       * (`requestBody`, `user`, `userMCPAuthMap`, etc.) AND the run-
       * identity fields (`run_id`, `parent_run_id`, `thread_id`) all
       * propagate. LangGraph's own runtime keys are excluded because the
       * child graph creates its own scratchpad/checkpoint/abort plumbing.
       *
       * Run-identity propagation is intentional and matches the
       * convention this executor itself already uses for `SubagentStart`
       * / `SubagentStop` hooks (`sessionId: this.parentRunId`): the
       * subagent runs under the parent's session scope, not its own.
       * Forwarding `run_id` / `parent_run_id` / `thread_id` makes
       * `ToolNode`'s hook lookups (`hasHookFor(eventName, runId)`),
       * `ToolOutputReferenceRegistry` keying, and trace lineage all
       * resolve to the parent's session for tools dispatched from the
       * subagent — so `PreToolUse` / `PostToolUse` hooks the host
       * registered against the parent's run fire for subagent tool
       * calls too. "Same run" matches the user-perceptual mental model.
       *
       * `thread_id` falls back to `childRunId` only when the parent
       * didn't supply one (legacy behavior preserved for hosts that
       * never set thread_id).
       *
       * NOTE: a future revision will likely make this configurable per
       * spawn type — e.g. a background / async subagent that runs after
       * the parent's run completes wants isolation, not inheritance.
       * For now the inheritance path matches LibreChat's primary use
       * case (synchronous subagents within a single user turn).
       */
      const inheritedConfigurable: Record<string, unknown> =
        sanitizeChildConfigurable(params.parentConfigurable);
      result = await workflow.invoke(
        { messages: [new HumanMessage(description)] },
        {
          recursionLimit: maxTurns * RECURSION_MULTIPLIER,
          signal: this.parentSignal,
          callbacks,
          runName: `subagent:${subagentType}`,
          configurable: {
            thread_id: childRunId,
            ...inheritedConfigurable,
          },
        }
      );
    } catch (error) {
      const errorMessage = truncateErrorMessage(error);
      if (forwarding) {
        await forwarding.drain();
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

    if (forwarding) {
      await forwarding.drain();
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
    subagentType: string;
    subagentAgentId: string;
    childRunId: string;
    parentToolCallId?: string;
  }): ForwarderCallback {
    const {
      parentRegistry,
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
      try {
        const event: SubagentUpdateEvent = {
          runId: parentRunId,
          subagentRunId: childRunId,
          subagentType,
          subagentAgentId,
          parentAgentId,
          parentToolCallId,
          phase,
          data: sanitizeForwardedSubagentUpdateData(eventName, data),
          label: summarizeEvent(eventName, data),
          timestamp: new Date().toISOString(),
        };
        await handler.handle(GraphEvents.ON_SUBAGENT_UPDATE, event);
      } catch {
        /* observational — swallow */
      }
    };

    const queuedUpdates: QueuedSubagentUpdate[] = [];
    let drainPromise: Promise<void> | undefined;

    const enqueue = (update: QueuedSubagentUpdate): void => {
      if (queuedUpdates.length >= MAX_PENDING_SUBAGENT_UPDATES) {
        const dropIndex = queuedUpdates.findIndex((queued) =>
          isDroppableSubagentUpdatePhase(queued.phase)
        );
        if (dropIndex >= 0) {
          queuedUpdates.splice(dropIndex, 1);
        } else if (isDroppableSubagentUpdatePhase(update.phase)) {
          return;
        }
      }
      queuedUpdates.push(update);
    };

    const drain = async (): Promise<void> => {
      if (drainPromise != null) {
        await drainPromise;
        return;
      }
      drainPromise = (async (): Promise<void> => {
        while (queuedUpdates.length > 0) {
          const update = queuedUpdates.shift();
          if (update == null) {
            continue;
          }
          await wrap(update.eventName, update.phase, update.data);
        }
      })();
      try {
        await drainPromise;
      } finally {
        drainPromise = undefined;
        if (queuedUpdates.length > 0) {
          await drain();
        }
      }
    };

    const scheduleWrap = (
      eventName: string,
      phase: SubagentUpdatePhase,
      data: unknown
    ): void => {
      enqueue({ eventName, phase, data });
      void drain();
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
              data as ToolExecuteBatchRequest
            );
          }
          /**
           * We also surface a short notice in the subagent-update stream so
           * the UI can show "calling <tool>" for each tool the child spawns.
           */
          scheduleWrap(eventName, 'run_step', data);
          return;
        }

        if (eventName === GraphEvents.ON_RUN_STEP) {
          scheduleWrap(eventName, 'run_step', data);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_DELTA) {
          scheduleWrap(eventName, 'run_step_delta', data);
          return;
        }
        if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
          scheduleWrap(eventName, 'run_step_completed', data);
          return;
        }
        if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
          scheduleWrap(eventName, 'message_delta', data);
          return;
        }
        if (eventName === GraphEvents.ON_REASONING_DELTA) {
          scheduleWrap(eventName, 'reasoning_delta', data);
          return;
        }
      },
    });
    /**
     * `awaitHandlers = true` is required so the child's `ToolNode` actually
     * blocks on the parent's `ON_TOOL_EXECUTE` handler until it resolves
     * the batch request. Observational `ON_SUBAGENT_UPDATE` calls are queued
     * behind a bounded sequential dispatcher so host UI publication cannot
     * backpressure each child emission or run unbounded concurrent publishes.
     * The executor drains this queue before terminal stop/error envelopes to
     * preserve phase ordering.
     */
    handler.awaitHandlers = true;
    return { handler, drain };
  }
}

function sanitizeChildConfigurable(
  parentConfigurable: Record<string, unknown> | undefined
): Record<string, unknown> {
  if (parentConfigurable == null) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(parentConfigurable).filter(
      ([key]) => !isLangGraphRuntimeConfigKey(key)
    )
  );
}

function isLangGraphRuntimeConfigKey(key: string): boolean {
  return (
    key.startsWith(LANGGRAPH_RUNTIME_CONFIG_PREFIX) ||
    LANGGRAPH_CHECKPOINT_CONFIG_KEYS.has(key)
  );
}

export function sanitizeForwardedSubagentUpdateData(
  eventName: string,
  data: unknown
): unknown {
  if (eventName === GraphEvents.ON_TOOL_EXECUTE) {
    return sanitizeToolExecuteUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP) {
    return sanitizeRunStepUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP_DELTA) {
    return sanitizeRunStepDeltaUpdateData(data);
  }
  if (eventName === GraphEvents.ON_RUN_STEP_COMPLETED) {
    return sanitizeRunStepCompletedUpdateData(data);
  }
  if (eventName === GraphEvents.ON_MESSAGE_DELTA) {
    return sanitizeMessageDeltaUpdateData(data);
  }
  if (eventName === GraphEvents.ON_REASONING_DELTA) {
    return sanitizeReasoningDeltaUpdateData(data);
  }
  return undefined;
}

function isDroppableSubagentUpdatePhase(phase: SubagentUpdatePhase): boolean {
  return (
    phase === 'message_delta' ||
    phase === 'reasoning_delta' ||
    phase === 'run_step_delta'
  );
}

function sanitizeToolExecuteUpdateData(
  data: unknown
): SanitizedSubagentToolExecuteData {
  const request = data as Partial<ToolExecuteBatchRequest>;
  const toolCalls = Array.isArray(request.toolCalls)
    ? request.toolCalls.map(sanitizeToolCallForUpdate)
    : [];
  const sanitized: SanitizedSubagentToolExecuteData = { toolCalls };
  if (typeof request.agentId === 'string') {
    sanitized.agentId = request.agentId;
  }
  return sanitized;
}

function sanitizeToolCallForUpdate(
  call: ToolExecuteBatchRequest['toolCalls'][number]
): SanitizedSubagentToolCall {
  const sanitized: SanitizedSubagentToolCall = {
    id: call.id,
    name: call.name,
    args: call.args,
  };
  return sanitized;
}

function sanitizeRunStepUpdateData(data: unknown): SanitizedRunStep | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const step = data as Partial<RunStep>;
  const sanitized: SanitizedRunStep = {};
  assignString(sanitized, 'agentId', step.agentId);
  assignNumber(sanitized, 'groupId', step.groupId);
  assignString(sanitized, 'id', step.id);
  assignNumber(sanitized, 'index', step.index);
  assignString(sanitized, 'runId', step.runId);
  assignNumber(sanitized, 'stepIndex', step.stepIndex);
  assignString(sanitized, 'type', step.type);
  if (step.summary !== undefined) {
    sanitized.summary = step.summary;
  }
  if (step.usage !== undefined) {
    sanitized.usage = step.usage;
  }
  sanitized.stepDetails = sanitizeStepDetails(step.stepDetails);
  return sanitized;
}

function sanitizeRunStepDeltaUpdateData(
  data: unknown
): SanitizedRunStepDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<RunStepDeltaEvent>;
  const sanitized: SanitizedRunStepDelta = {};
  assignString(sanitized, 'id', event.id);
  sanitized.delta = sanitizeToolCallDelta(event.delta);
  return sanitized;
}

function sanitizeRunStepCompletedUpdateData(
  data: unknown
): SanitizedRunStepCompleted | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as { result?: unknown };
  return { result: sanitizeStepCompleted(event.result) };
}

function sanitizeMessageDeltaUpdateData(
  data: unknown
): SanitizedMessageDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<MessageDeltaEvent>;
  const sanitized: SanitizedMessageDelta = {};
  assignString(sanitized, 'id', event.id);
  if (event.delta != null) {
    sanitized.delta = {};
    if (event.delta.content !== undefined) {
      sanitized.delta.content = event.delta.content;
    }
    if (event.delta.tool_call_ids !== undefined) {
      sanitized.delta.tool_call_ids = event.delta.tool_call_ids;
    }
  }
  return sanitized;
}

function sanitizeReasoningDeltaUpdateData(
  data: unknown
): SanitizedReasoningDelta | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const event = data as Partial<ReasoningDeltaEvent>;
  const sanitized: SanitizedReasoningDelta = {};
  assignString(sanitized, 'id', event.id);
  if (event.delta?.content !== undefined) {
    sanitized.delta = { content: event.delta.content };
  }
  return sanitized;
}

function sanitizeStepDetails(stepDetails: unknown): SanitizedStepDetails | undefined {
  if (!isObjectLike(stepDetails)) {
    return undefined;
  }
  const rawDetails = stepDetails as {
    message_creation?: { message_id?: unknown };
    tool_calls?: unknown[];
    type?: unknown;
  };
  if (rawDetails.type === StepTypes.MESSAGE_CREATION) {
    const sanitized: SanitizedStepDetails = {
      type: StepTypes.MESSAGE_CREATION,
    };
    const messageId = rawDetails.message_creation?.message_id;
    if (typeof messageId === 'string') {
      sanitized.message_creation = { message_id: messageId };
    }
    return sanitized;
  }
  if (rawDetails.type === StepTypes.TOOL_CALLS) {
    const sanitized: SanitizedStepDetails = {
      type: StepTypes.TOOL_CALLS,
    };
    if (Array.isArray(rawDetails.tool_calls)) {
      sanitized.tool_calls = rawDetails.tool_calls.map(sanitizeAgentToolCall);
    }
    return sanitized;
  }
  return undefined;
}

function sanitizeToolCallDelta(
  delta: ToolCallDelta | undefined
): SanitizedToolCallDelta | undefined {
  if (!isObjectLike(delta)) {
    return undefined;
  }
  const sanitized: SanitizedToolCallDelta = {};
  assignString(sanitized, 'auth', delta.auth);
  assignNumber(sanitized, 'expires_at', delta.expires_at);
  assignString(sanitized, 'type', delta.type);
  if (delta.summary !== undefined) {
    sanitized.summary = delta.summary;
  }
  if (Array.isArray(delta.tool_calls)) {
    sanitized.tool_calls = delta.tool_calls.map(sanitizeAgentToolCall);
  }
  return sanitized;
}

function sanitizeStepCompleted(data: unknown): SanitizedStepCompleted | undefined {
  if (!isObjectLike(data)) {
    return undefined;
  }
  const completed = data as Partial<StepCompleted> & {
    id?: unknown;
    index?: unknown;
    tool_call?: unknown;
  };
  if (completed.type === 'summary') {
    return {
      type: 'summary',
      summary: completed.summary,
    };
  }
  if (completed.type !== 'tool_call') {
    return undefined;
  }
  const sanitized: SanitizedStepCompleted = { type: 'tool_call' };
  assignString(sanitized, 'id', completed.id);
  assignNumber(sanitized, 'index', completed.index);
  sanitized.tool_call = sanitizeProcessedToolCall(completed.tool_call);
  return sanitized;
}

function sanitizeProcessedToolCall(
  toolCall: unknown
): SanitizedProcessedToolCall | undefined {
  if (!isObjectLike(toolCall)) {
    return undefined;
  }
  const call = toolCall as Partial<ProcessedToolCall>;
  const sanitized: SanitizedProcessedToolCall = {};
  assignString(sanitized, 'id', call.id);
  assignString(sanitized, 'name', call.name);
  if (call.args !== undefined) {
    sanitized.args = call.args;
  }
  assignString(sanitized, 'output', call.output);
  assignNumber(sanitized, 'progress', call.progress);
  return sanitized;
}

function sanitizeAgentToolCall(toolCall: unknown): SanitizedAgentToolCall {
  if (!isObjectLike(toolCall)) {
    return {};
  }
  const call = toolCall as SanitizedAgentToolCall;
  const sanitized: SanitizedAgentToolCall = {};
  assignString(sanitized, 'id', call.id);
  assignString(sanitized, 'name', call.name);
  assignString(sanitized, 'type', call.type);
  if (call.args !== undefined) {
    sanitized.args = call.args;
  }
  if (isObjectLike(call.function)) {
    const fn: SanitizedAgentToolCall['function'] = {};
    assignString(fn, 'name', call.function.name);
    if (
      typeof call.function.arguments === 'string' ||
      isObjectLike(call.function.arguments)
    ) {
      fn.arguments = call.function.arguments;
    }
    sanitized.function = fn;
  }
  return sanitized;
}

function isObjectLike(value: unknown): value is object {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assignString<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: unknown
): void {
  if (typeof value === 'string') {
    target[key] = value as T[K];
  }
}

function assignNumber<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: unknown
): void {
  if (typeof value === 'number') {
    target[key] = value as T[K];
  }
}

/**
 * Produces a short single-line label for an arbitrary forwarded child event.
 * Used to populate {@link SubagentUpdateEvent.label} so the host UI can show
 * a compact status ticker without parsing the raw payload.
 */
export function summarizeEvent(eventName: string, data: unknown): string {
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
    /**
     * Subagents run in an isolated context by contract. Parent-run-scoped
     * fields that would otherwise survive the shallow-spread clone — the
     * cross-run conversation summary and the prior-turn tool-discovery
     * set — are cleared here so the child starts fresh. Host applications
     * that want a subagent to see parent context must thread it in
     * explicitly (e.g. via the `description` argument to the subagent
     * tool), not via inherited state.
     */
    initialSummary: undefined,
    discoveredTools: undefined,
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
