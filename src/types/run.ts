// src/types/run.ts
import type { Callbacks } from '@langchain/core/callbacks/manager';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import type { StructuredTool } from '@langchain/core/tools';
import type * as z from 'zod';
import type {
  ToolSessionMap,
  ToolExecutionConfig,
  ToolOutputReferencesConfig,
  EagerEventToolExecutionConfig,
} from '@/types/tools';
import type { HumanInTheLoopConfig } from '@/types/hitl';
import type { HookRegistry } from '@/hooks';
import type * as s from '@/types/stream';
import type * as e from '@/common/enum';
import type * as g from '@/types/graph';
import type * as l from '@/types/llm';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ZodObjectAny = z.ZodObject<any, any, any, any>;
export type BaseGraphConfig = {
  llmConfig: l.LLMConfig;
  provider?: e.Providers;
  clientOptions?: l.ClientOptions;
  /** Optional compile options for workflow.compile() */
  compileOptions?: g.CompileOptions;
};
export type LegacyGraphConfig = BaseGraphConfig & {
  type?: 'standard';
} & Omit<g.StandardGraphInput, 'provider' | 'clientOptions' | 'agents'> &
  Omit<g.AgentInputs, 'provider' | 'clientOptions' | 'agentId'>;

/* Supervised graph (opt-in) */
export type SupervisedGraphConfig = BaseGraphConfig & {
  type: 'supervised';
  /** Enable supervised router; when false, fall back to standard loop */
  routerEnabled?: boolean;
  /** Table-driven routing policy per stage */
  routingPolicies?: Array<{
    stage: string;
    agents?: string[];
    model?: e.Providers;
    parallel?: boolean;
    /** Optional simple condition on content/tools */
    when?:
      | 'always'
      | 'has_tools'
      | 'no_tools'
      | { includes?: string[]; excludes?: string[] };
  }>;
  /** Opt-in feature flags */
  featureFlags?: {
    multi_model_routing?: boolean;
    fan_out?: boolean;
    fan_out_retries?: number;
    fan_out_backoff_ms?: number;
    fan_out_concurrency?: number;
  };
  /** Optional per-stage model configs */
  models?: Record<string, l.LLMConfig>;
} & Omit<g.StandardGraphInput, 'provider' | 'clientOptions'>;

export type RunTitleOptions = {
  inputText: string;
  provider: e.Providers;
  contentParts: (s.MessageContentComplex | undefined)[];
  titlePrompt?: string;
  skipLanguage?: boolean;
  clientOptions?: l.ClientOptions;
  chainOptions?: Partial<RunnableConfig> | undefined;
  omitOptions?: Set<string>;
  titleMethod?: e.TitleMethod;
  titlePromptTemplate?: string;
};

export interface AgentStateChannels {
  messages: BaseMessage[];
  next: string;
  [key: string]: unknown;
  /** Stable/cacheable system instructions. */
  instructions?: string;
  /** Dynamic system tail appended after stable instructions. */
  additional_instructions?: string;
}

export interface Member {
  name: string;
  systemPrompt: string;
  tools: StructuredTool[];
  llmConfig: l.LLMConfig;
}

export type CollaborativeGraphConfig = {
  type: 'collaborative';
  members: Member[];
  supervisorConfig: { systemPrompt?: string; llmConfig: l.LLMConfig };
};

export type TaskManagerGraphConfig = {
  type: 'taskmanager';
  members: Member[];
  supervisorConfig: { systemPrompt?: string; llmConfig: l.LLMConfig };
};

export type MultiAgentGraphConfig = {
  type: 'multi-agent';
  compileOptions?: g.CompileOptions;
  agents: g.AgentInputs[];
  edges: g.GraphEdge[];
};

export type StandardGraphConfig = Omit<
  MultiAgentGraphConfig,
  'edges' | 'type'
> & { type?: 'standard'; signal?: AbortSignal };

export type RunConfig = {
  runId: string;
  graphConfig: LegacyGraphConfig | StandardGraphConfig | MultiAgentGraphConfig;
  /**
   * Run-scoped Langfuse configuration. Per-agent `AgentInputs.langfuse`
   * takes precedence for agent-specific spans; this object supplies defaults
   * for run-wide tracing controls such as tool-output redaction.
   */
  langfuse?: g.LangfuseConfig;
  customHandlers?: Record<string, g.EventHandler>;
  /**
   * Receives token usage for every model call made inside subagent child
   * runs (including nested subagents). Child graphs execute via `invoke()`
   * outside this run's `streamEvents` loop, so their model-end events never
   * reach `customHandlers` — without this sink, child usage is invisible to
   * the host. Parent-graph calls are not reported here; they flow through
   * the registered `CHAT_MODEL_END` handler as usual.
   */
  subagentUsageSink?: g.SubagentUsageSink;
  /**
   * Pre-constructed hook registry for this run. Hooks fire at lifecycle
   * points in `processStream` (RunStart, UserPromptSubmit, Stop,
   * StopFailure) and around tool calls (PreToolUse, PostToolUse,
   * PostToolUseFailure, PermissionDenied).
   *
   * Pass `undefined` (the default) to skip all hook dispatch. When a
   * registry is provided, the run attaches it to the `Graph` so internal
   * nodes can fire hooks too, and clears the session in the `finally`
   * block to prevent leaks.
   */
  hooks?: HookRegistry;
  returnContent?: boolean;
  tokenCounter?: TokenCounter;
  indexTokenCountMap?: Record<string, number>;
  /**
   * Calibration ratio from a previous run's contextMeta.
   * Seeds the pruner's EMA so new messages are scaled immediately.
   *
   * Hosts should persist the value returned by `Run.getCalibrationRatio()`
   * after each run and pass it back here on subsequent runs for the same
   * conversation. Without this, the EMA resets to 1 on every new Run instance.
   */
  calibrationRatio?: number;
  /** Skip post-stream cleanup (clearHeavyState) — useful for tests that inspect graph state after processStream */
  skipCleanup?: boolean;
  /**
   * Initial session state to seed the Graph's ToolSessionMap.
   * Used to carry over code environment sessions from skill file priming
   * at run start, so ToolNode can inject session_id + files into tool calls.
   */
  initialSessions?: ToolSessionMap;
  /**
   * Run-scoped tool output reference configuration. When `enabled` is
   * `true`, tool outputs are registered under stable keys
   * (`tool<idx>turn<turn>`) and subsequent tool calls can pipe previous
   * outputs into their arguments via `{{tool<idx>turn<turn>}}`
   * placeholders. Disabled by default so existing runs are unaffected.
   */
  toolOutputReferences?: ToolOutputReferencesConfig;
  /**
   * Opt-in latency optimization for event-driven tools. When enabled,
   * the streaming layer may start a tool call as soon as it sees a
   * complete, parseable tool call; ToolNode still waits for the final
   * assistant message before appending ToolMessages, preserving provider
   * ordering. The SDK automatically falls back to normal batch dispatch
   * when hooks, HITL, output references, or ambiguous stream shapes make
   * eager dispatch unsafe.
   */
  eagerEventToolExecution?: EagerEventToolExecutionConfig;
  /**
   * Names of host tools that write to the code-execution sandbox but are not
   * built-in `CODE_EXECUTION_TOOLS` (e.g. LibreChat's create_file/edit_file).
   * Their successful results fold the returned exec `session_id` into the
   * shared code session, so a file such a tool writes is visible to later
   * bash_tool/execute_code calls running in the same sandbox. Host-declared so
   * the SDK stays name-agnostic.
   */
  codeSessionToolNames?: string[];
  /**
   * Names of host tools whose in-process body may raise a LangGraph
   * `interrupt()` mid-execution — the canonical example is an
   * `ask_user_question` tool that suspends the run to collect a human
   * answer. Within a single tool-call batch, a named tool that is a real
   * in-process graphTool (the only kind whose body can reach `interrupt()`;
   * graphTools are auto-marked direct) is scheduled **ahead of** its
   * non-interrupting direct siblings. That ordering guarantees a mid-body
   * interrupt unwinds the tool batch before a non-idempotent sibling
   * (send_email, billing) executes, so the sibling cannot run once on the
   * first pass and AGAIN when LangGraph re-runs the interrupted batch on
   * resume.
   *
   * This only reorders the direct group — it does NOT force a name onto
   * the direct path. A name that is only an inherited event `toolDefinition`
   * (schema-only stub, e.g. in a self-spawned child) stays event-dispatched;
   * the guard applies only to tools that are independently direct.
   * Host-declared so the SDK stays name-agnostic; omit to keep the prior
   * (unguarded) behavior.
   */
  interruptingToolNames?: string[];
  /**
   * Selects the execution backend for built-in code tools. Omit this to keep
   * the remote LibreChat Code API sandbox. Set `{ engine: 'local' }` to run
   * code execution locally and auto-bind the local coding tool suite unless
   * `local.includeCodingTools` is set to `false`.
   */
  toolExecution?: ToolExecutionConfig;
  /**
   * First-class human-in-the-loop (HITL) flow for this run.
   *
   * **HITL is OFF by default.** Omitting this field — or passing
   * `{ enabled: false }` — keeps the pre-HITL fail-closed semantics
   * where `ask` decisions collapse into a synchronous deny. Hosts opt
   * in explicitly with `{ enabled: true }` once their UI can render
   * and resolve `tool_approval` interrupts (otherwise the run just
   * pauses with no resolver, which surfaces to end users as a hung
   * tool-call card).
   *
   * Plan of record: the default flips back to ON in a future minor
   * once the consumer ecosystem (notably LibreChat) ships HITL UI
   * end-to-end. See `HumanInTheLoopConfig` JSDoc.
   *
   * When enabled (`{ enabled: true }`):
   *   - `PreToolUse` hooks returning `decision: 'ask'` raise a real
   *     LangGraph `interrupt()` instead of being treated as a synchronous
   *     deny. The graph pauses and the run exits cleanly.
   *   - If `graphConfig.compileOptions.checkpointer` is missing, the SDK
   *     installs an in-memory `MemorySaver` as a fallback so scripts and
   *     tests can resume without external infrastructure. Production
   *     hosts should always provide a durable checkpointer.
   *   - Hosts inspect the pending interrupt via `run.getInterrupt()` and
   *     continue with `Run.resume(decisions)` against a Run rebuilt with
   *     the same `thread_id` and checkpointer.
   *
   * When disabled (the default): `ask` decisions remain fail-closed
   * (blocked with an error `ToolMessage`) and no checkpointer is
   * implicitly attached.
   */
  humanInTheLoop?: HumanInTheLoopConfig;
};

export type ProvidedCallbacks = Callbacks | undefined;

export type TokenCounter = (message: BaseMessage) => number;

/** Structured breakdown of how context token budget is consumed. */
export type TokenBudgetBreakdown = {
  /** Total context window budget (maxContextTokens). */
  maxContextTokens: number;
  /** Total instruction tokens (system + tools + summary). */
  instructionTokens: number;
  /** Tokens from the system message text alone. */
  systemMessageTokens: number;
  /** Tokens from instruction text emitted outside the system message. */
  dynamicInstructionTokens: number;
  /** Tokens from tool schema definitions. */
  toolSchemaTokens: number;
  /** Tokens from the conversation summary. */
  summaryTokens: number;
  /** Number of registered tools. */
  toolCount: number;
  /** Number of messages in the conversation. */
  messageCount: number;
  /** Total tokens consumed by messages (excluding system). */
  messageTokens: number;
  /** Tokens available for messages after instructions. */
  availableForMessages: number;
  /** Per-tool schema token counts (post-multiplier), keyed by tool name. */
  toolTokenCounts?: Record<string, number>;
  /** Names of counted tools that are deferred (`defer_loading`) and discovered. */
  deferredToolNames?: string[];
};

export type EventStreamOptions = {
  callbacks?: g.ClientCallbacks;
  keepContent?: boolean;
};

/**
 * When to persist checkpoints during a run. Mirrors langgraph's option:
 * `'async'`/`'sync'` checkpoint every superstep; `'exit'` checkpoints only
 * at the graph's exit/interrupt boundary. Kept as a local union to avoid
 * coupling to langgraph internals (it is not exported from the root).
 */
export type Durability = 'async' | 'sync' | 'exit';

/**
 * Config accepted by `processStream`/`resume`. Extends `RunnableConfig` with
 * the stream `version`, an optional `run_id`, and an optional `durability`
 * override forwarded to langgraph's run options.
 */
export type RunStreamConfig = Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  run_id?: string;
  durability?: Durability;
};
