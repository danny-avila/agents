// src/types/tools.ts
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableToolLike } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { HookRegistry } from '@/hooks';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type { MessageContentComplex, ToolErrorData } from './stream';
import type { HumanInTheLoopConfig } from './hitl';

/** Replacement type for `import type { ToolCall } from '@langchain/core/messages/tool'` in order to have stringified args typed */
export type CustomToolCall = {
  name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  args: string | Record<string, any>;
  id?: string;
  type?: 'tool_call';
  output?: string;
};

export type GenericTool = (StructuredToolInterface | RunnableToolLike) & {
  mcp?: boolean;
};

export type ToolMap = Map<string, GenericTool>;
export type ToolRefs = {
  tools: GenericTool[];
  toolMap?: ToolMap;
};

export type ToolRefGenerator = (tool_calls: ToolCall[]) => ToolRefs;

export type ToolNodeOptions = {
  name?: string;
  tags?: string[];
  handleToolErrors?: boolean;
  loadRuntimeTools?: ToolRefGenerator;
  toolCallStepIds?: Map<string, string>;
  errorHandler?: (
    data: ToolErrorData,
    metadata?: Record<string, unknown>
  ) => Promise<void>;
  /** Tool registry for lazy computation of programmatic tools and tool search */
  toolRegistry?: LCToolRegistry;
  /** Reference to Graph's sessions map for automatic session injection */
  sessions?: ToolSessionMap;
  /** When true, dispatches ON_TOOL_EXECUTE events instead of invoking tools directly */
  eventDrivenMode?: boolean;
  /** Tool definitions for event-driven mode (used for context, not invocation) */
  toolDefinitions?: Map<string, LCTool>;
  /** Agent ID for event-driven mode (used to identify which agent's context to use) */
  agentId?: string;
  /** Tool names that must be executed directly (via runTool) even in event-driven mode (e.g., graph-managed handoff tools) */
  directToolNames?: Set<string>;
  /**
   * Hook registry for PreToolUse/PostToolUse/PostToolUseFailure/
   * PermissionDenied lifecycle hooks. Fires for **every** tool the
   * ToolNode invokes — both event-dispatched tools (via
   * `dispatchToolEvents`) and direct-path tools (via
   * `runDirectToolWithLifecycleHooks`). The pre-existing limitation
   * that direct-path tools "bypass hook dispatch entirely" was
   * lifted in a follow-up to the HITL surface; both paths now route
   * through the same hook lifecycle.
   */
  hookRegistry?: HookRegistry;
  /**
   * Run-scoped HITL config. **HITL is OFF by default** — omitting this
   * field (or passing `{ enabled: false }`) keeps the pre-HITL
   * fail-closed behavior where a `PreToolUse` `ask` decision collapses
   * into a blocked `ToolMessage`. Hosts opt in with
   * `{ enabled: true }` once their UI can render and resolve a
   * `tool_approval` interrupt; that engages the interrupt path where
   * `ask` raises a real LangGraph `interrupt()` carrying a
   * `HumanInterruptPayload` and the host resumes with
   * `Run.resume(decisions)`.
   *
   * Mirrors `RunConfig.humanInTheLoop` (which is the canonical place
   * to set this); the Graph threads it down to every ToolNode it
   * compiles. The interrupt path is wired into both the event
   * dispatch (`dispatchToolEvents`) and the direct path
   * (`runDirectToolWithLifecycleHooks`) — direct tools no longer
   * bypass HITL. See `HumanInTheLoopConfig` for the resume re-execution
   * contract that applies equally to both paths.
   */
  humanInTheLoop?: HumanInTheLoopConfig;
  /** Max context tokens for the agent — used to compute tool result truncation limits. */
  maxContextTokens?: number;
  /**
   * Maximum characters allowed in a single tool result before truncation.
   * When provided, takes precedence over the value computed from maxContextTokens.
   */
  maxToolResultChars?: number;
  /**
   * Run-scoped tool output reference configuration. When `enabled` is
   * `true`, ToolNode registers successful outputs and substitutes
   * `{{tool<idx>turn<turn>}}` placeholders found in string args.
   *
   * Ignored when `toolOutputRegistry` is also provided (host-supplied
   * registry wins).
   */
  toolOutputReferences?: ToolOutputReferencesConfig;
  /**
   * Pre-constructed registry instance shared across ToolNodes for the
   * run. Graphs pass the same registry to every ToolNode they compile
   * so cross-agent `{{tool<i>turn<n>}}` substitutions resolve. Takes
   * precedence over `toolOutputReferences` when both are set.
   */
  toolOutputRegistry?: ToolOutputReferenceRegistry;
  /**
   * Selects where built-in code execution tools run. Defaults to the
   * remote LibreChat Code API sandbox; `local` swaps those same tool names
   * to process-based local executors at ToolNode construction time.
   */
  toolExecution?: ToolExecutionConfig;
  /**
   * Pre-constructed file checkpointer shared across every ToolNode in
   * a Run so `Run.rewindFiles()` (and direct
   * `Run.getFileCheckpointer()` callers) sees a single unified
   * snapshot store. The Graph layer creates one per Run when
   * `toolExecution.local.fileCheckpointing === true` and threads it
   * to every ToolNode it compiles; without this shared instance,
   * multi-agent graphs would each get their own private checkpointer
   * and the host couldn't reach any of them. Wins over the
   * auto-created bundle checkpointer in
   * `resolveLocalExecutionTools`.
   */
  fileCheckpointer?: LocalFileCheckpointer;
};

export type ToolNodeConstructorParams = ToolRefs & ToolNodeOptions;

export type ToolEndEvent = {
  /** The Step Id of the Tool Call */
  id: string;
  /** The Completed Tool Call */
  tool_call: ToolCall;
  /** The content index of the tool call */
  index: number;
  type?: 'tool_call';
};

export type CodeEnvFile = {
  id: string;
  name: string;
  session_id: string;
};

export type CodeExecutionToolParams =
  | undefined
  | {
      session_id?: string;
      user_id?: string;
      files?: CodeEnvFile[];
    };

export type FileRef = {
  id: string;
  name: string;
  path?: string;
  /** Session ID this file belongs to (for multi-session file tracking) */
  session_id?: string;
  /**
   * `true` when the codeapi sandbox echoed this entry as an unchanged
   * passthrough of an input the caller already owns (skill files,
   * downloaded inputs whose hash matched the baseline, inherited
   * `.dirkeep` markers). The tool-result formatter renders these as
   * "Available files" rather than "Generated files" so the LLM doesn't
   * conflate infrastructure inputs with newly-produced outputs.
   */
  inherited?: true;
};

export type FileRefs = FileRef[];

export type ExecuteResult = {
  session_id: string;
  stdout: string;
  stderr: string;
  files?: FileRefs;
};

/** JSON Schema type definition for tool parameters */
export type JsonSchemaType = {
  type:
    | 'string'
    | 'number'
    | 'integer'
    | 'float'
    | 'boolean'
    | 'array'
    | 'object';
  enum?: string[];
  items?: JsonSchemaType;
  properties?: Record<string, JsonSchemaType>;
  required?: string[];
  description?: string;
  additionalProperties?: boolean | JsonSchemaType;
};

/**
 * Specifies which contexts can invoke a tool (inspired by Anthropic's allowed_callers)
 * - 'direct': Only callable directly by the LLM (default if omitted)
 * - 'code_execution': Only callable from within programmatic code execution
 */
export type AllowedCaller = 'direct' | 'code_execution';

/** Tool definition with optional deferred loading and caller restrictions */
export type LCTool = {
  name: string;
  description?: string;
  parameters?: JsonSchemaType;
  /** When true, tool is not loaded into context initially (for tool search) */
  defer_loading?: boolean;
  /**
   * Which contexts can invoke this tool.
   * Default: ['direct'] (only callable directly by LLM)
   * Options: 'direct', 'code_execution'
   */
  allowed_callers?: AllowedCaller[];
  /** Response format for the tool output */
  responseFormat?: 'content' | 'content_and_artifact';
  /** Server name for MCP tools */
  serverName?: string;
  /** Tool type classification */
  toolType?: 'builtin' | 'mcp' | 'action';
};

/** Single tool call within a batch request for event-driven execution */
export type ToolCallRequest = {
  /** Tool call ID from the LLM */
  id: string;
  /** Tool name */
  name: string;
  /** Tool arguments */
  args: Record<string, unknown>;
  /** Step ID for tracking */
  stepId?: string;
  /** Usage turn count for this tool */
  turn?: number;
  /** Code execution session context for session continuity in event-driven mode */
  codeSessionContext?: {
    session_id: string;
    files?: CodeEnvFile[];
  };
};

/** Batch request containing ALL tool calls for a graph step */
export type ToolExecuteBatchRequest = {
  /** All tool calls from the AIMessage */
  toolCalls: ToolCallRequest[];
  /** User ID for context */
  userId?: string;
  /** Agent ID for context */
  agentId?: string;
  /** Runtime configurable from RunnableConfig (includes user, userMCPAuthMap, etc.) */
  configurable?: Record<string, unknown>;
  /** Runtime metadata from RunnableConfig (includes thread_id, run_id, provider, etc.) */
  metadata?: Record<string, unknown>;
  /** Promise resolver - handler calls this with ALL results */
  resolve: (results: ToolExecuteResult[]) => void;
  /** Promise rejector - handler calls this on fatal error */
  reject: (error: Error) => void;
};

/**
 * A message injected into graph state by any tool execution handler.
 * Generic mechanism: any tool returning `injectedMessages` in its `ToolExecuteResult`
 * will have these appended to state after the ToolMessage for this call.
 */
export type InjectedMessage = {
  /** 'user' for skill body injection, 'system' for context hints.
   *  Both are converted to HumanMessage at runtime; the original role
   *  is preserved in additional_kwargs.role. */
  role: 'user' | 'system';
  /** Message content: string for simple text, array for complex multi-part content */
  content: string | MessageContentComplex[];
  /** When true, the message is framework-internal: not shown in UI, not counted as a user turn */
  isMeta?: boolean;
  /** Origin tag for downstream consumers (UI, pruner, compaction) */
  source?: 'skill' | 'hook' | 'system';
  /** Only set when source is 'skill', for compaction preservation */
  skillName?: string;
};

/** Result for a single tool call in event-driven execution */
export type ToolExecuteResult = {
  /** Matches ToolCallRequest.id */
  toolCallId: string;
  /** Tool output content */
  content: string | unknown[];
  /** Optional artifact (for content_and_artifact format) */
  artifact?: unknown;
  /** Execution status */
  status: 'success' | 'error';
  /** Error message if status is 'error' */
  errorMessage?: string;
  /**
   * Messages to inject into graph state after the ToolMessage for this call.
   * Placed after tool results to respect provider message ordering (tool_call -> tool_result adjacency).
   * The host's message formatter may merge injected user messages with the preceding tool_result turn.
   * Generic mechanism: any tool execution handler can use this.
   */
  injectedMessages?: InjectedMessage[];
};

/** Map of tool names to tool definitions */
export type LCToolRegistry = Map<string, LCTool>;

/**
 * Run-scoped configuration for tool output references.
 *
 * When enabled, each successful tool result is registered under a stable
 * key (`tool<idx>turn<turn>`). Later tool calls can pipe a previous
 * output into their arguments by including the literal placeholder
 * `{{tool<idx>turn<turn>}}` anywhere in a string argument; ToolNode
 * substitutes it with the stored output immediately before invoking
 * the tool.
 *
 * The registry stores the *raw, untruncated* tool output (subject to
 * its own size caps) so a later substitution can pipe the full payload
 * into the next tool even when the LLM only saw a head+tail-truncated
 * preview in `ToolMessage.content`. Size limits are decoupled from the
 * LLM-visible truncation budget and default to 5 MB total.
 *
 * Known limitations:
 *  - Tools that return a `ToolMessage` with array-type content
 *    (multi-part content blocks such as text + image) are not
 *    registered and cannot be cited via `{{tool<i>turn<n>}}`. A
 *    warning is logged so the missing reference is visible.
 *  - When a `PostToolUse` hook replaces `ToolMessage.content`, the
 *    *post-hook* content is what gets stored in the registry (and
 *    what the model sees), so `{{…}}` substitutions deliver the
 *    hooked output rather than the raw tool return. This matches the
 *    hook's "authoritative" role for output shaping.
 */
export type ToolOutputReferencesConfig = {
  /** Enable the registry and placeholder substitution. Defaults to `false`. */
  enabled?: boolean;
  /**
   * Maximum characters stored (and substituted) per registered output.
   * Applied to the *raw* output before storage. Defaults to
   * `HARD_MAX_TOOL_RESULT_CHARS` (~400 KB) — matching the
   * LLM-visible tool-result truncation budget, which is also a safe
   * payload size for shell `ARG_MAX` limits when a `{{…}}` expansion
   * gets piped into a bash `command`. Hosts that want to preserve
   * fuller fidelity (for example for non-bash API consumers) can
   * raise this up to `maxTotalSize` (defaults to 5 MB) — be aware
   * that large single-output substitutions may exceed shell
   * argument-size limits on typical Linux/macOS.
   */
  maxOutputSize?: number;
  /**
   * Hard cap on total characters retained across all registered outputs
   * for the run. When exceeded, the oldest entries are evicted FIFO
   * until the total fits. The effective per-output cap is
   * `min(maxOutputSize, maxTotalSize)` so a single stored output can
   * never exceed the aggregate bound. Defaults to
   * `calculateMaxTotalToolOutputSize(maxOutputSize)` (5 MB).
   */
  maxTotalSize?: number;
};

export type ToolExecutionEngine = 'sandbox' | 'local';

/**
 * Records pre-write file contents so callers can rewind edits/writes
 * made by the local engine. Implementations live in `src/tools/local`.
 */
export interface LocalFileCheckpointer {
  /**
   * Captures the current contents of `absolutePath` before a write or
   * edit. Idempotent: capturing the same path twice keeps the first
   * snapshot. Records "did not exist" so creates can be undone with
   * deletion.
   */
  captureBeforeWrite(absolutePath: string): Promise<void>;
  /** Restores all captured snapshots. Returns the number of files restored. */
  rewind(): Promise<number>;
  /** Returns paths that have been captured during this run. */
  capturedPaths(): string[];
}

/**
 * Pluggable process launcher used by the local execution engine. When
 * provided, the engine calls this in place of `child_process.spawn`,
 * letting callers route shell commands through SSH, containers, or
 * remote runners without forking the engine. The implementation must
 * return a `ChildProcess`-shaped value whose `stdout`/`stderr` streams
 * emit `data` events and that resolves a `close` event when finished.
 */
export type LocalSpawn = (
  command: string,
  args: string[],
  options: import('child_process').SpawnOptions
) => import('child_process').ChildProcessWithoutNullStreams;

/** Bash command-validation strictness for the local engine. */
export type LocalBashAstMode = 'auto' | 'off' | 'strict';

export type LocalSandboxConfig = {
  /**
   * Enable Anthropic Sandbox Runtime wrapping for local process tools.
   * Defaults to false; requires @anthropic-ai/sandbox-runtime to be installed.
   */
  enabled?: boolean;
  /** Throw when native sandbox dependencies are unavailable. Defaults to false. */
  failIfUnavailable?: boolean;
  filesystem?: {
    denyRead?: string[];
    allowRead?: string[];
    allowWrite?: string[];
    denyWrite?: string[];
    allowGitConfig?: boolean;
  };
  network?: {
    allowedDomains?: string[];
    deniedDomains?: string[];
    allowUnixSockets?: string[];
    allowAllUnixSockets?: boolean;
    allowLocalBinding?: boolean;
    allowMachLookup?: string[];
  };
};

/**
 * Workspace boundary the local-coding tools clamp file operations to.
 *
 * `root` is the canonical "project directory" — every file tool that
 * accepts a path resolves it relative to `root` and refuses to touch
 * paths outside it (after symlink resolution). `additionalRoots` lets
 * monorepos extend the boundary to sibling directories without
 * disabling clamping entirely.
 *
 * `allowReadOutside` and `allowWriteOutside` are escape hatches that
 * disable clamping for read- and write-shaped tools respectively.
 * Most hosts shouldn't need them — prefer wiring
 * `createWorkspacePolicyHook` if you want "ask the user" semantics
 * via the existing PreToolUse / HITL machinery instead of an
 * unconditional bypass.
 */
export type LocalWorkspaceConfig = {
  /** Required. The canonical workspace root. */
  root: string;
  /**
   * Sibling roots that also count as inside the workspace. Useful in
   * monorepos where a project legitimately needs to reach a paired
   * directory without disabling clamping.
   */
  additionalRoots?: readonly string[];
  /** When true, disable the read-outside-workspace clamp. */
  allowReadOutside?: boolean;
  /** When true, disable the write-outside-workspace clamp. */
  allowWriteOutside?: boolean;
};

/**
 * Engine-agnostic execution seam. Default uses Node's
 * `child_process.spawn` and `fs/promises`. A future engine (e.g.
 * stateful remote sandbox) supplies its own `spawn` and `fs` and
 * inherits every tool factory unchanged.
 */
export type LocalExecConfig = {
  /** Pluggable spawn (for SSH, container, remote workers, etc.). */
  spawn?: LocalSpawn;
  /** Pluggable filesystem (for remote-workspace engines). */
  fs?: import('@/tools/local/workspaceFS').WorkspaceFS;
};

export type LocalExecutionConfig = {
  /**
   * Working directory for local commands. Defaults to process.cwd().
   * Back-compat shorthand for `workspace.root`.
   *
   * @deprecated Prefer `workspace.root`. Kept for backward
   *   compatibility with pre-workspace configs.
   */
  cwd?: string;
  /**
   * Workspace boundary configuration. When omitted, the implementation
   * derives `{ root: cwd ?? process.cwd() }` so existing call sites
   * keep working.
   */
  workspace?: LocalWorkspaceConfig;
  /**
   * Execution seam (spawn + fs). When omitted, defaults to Node-host
   * child_process and fs/promises. Same back-compat: a top-level
   * `spawn` field is honoured if `exec.spawn` isn't set.
   */
  exec?: LocalExecConfig;
  /** Shell executable for bash-style tools. Defaults to `bash`. */
  shell?: string;
  /** Default timeout for local processes, in milliseconds. */
  timeoutMs?: number;
  /** Maximum stdout/stderr characters surfaced to the model. */
  maxOutputChars?: number;
  /**
   * Hard cap on total bytes a single child process can stream across
   * stdout+stderr before its process tree is killed. Independent from
   * `maxOutputChars` (which controls what the model sees); this is the
   * OOM backstop for noisy / runaway commands (`yes`, `cat /dev/urandom`,
   * a verbose build that loops). Defaults to 50 MiB. Set higher for
   * legitimately large logs; setting to 0 disables the cap and risks
   * an OOM crash on the host.
   */
  maxSpawnedBytes?: number;
  /** Extra environment variables merged over process.env. */
  env?: NodeJS.ProcessEnv;
  /** Optional process sandboxing via @anthropic-ai/sandbox-runtime. */
  sandbox?: LocalSandboxConfig;
  /**
   * When true, block obviously mutating shell commands before execution.
   * Useful for read-only agent modes and dry-run workflows.
   */
  readOnly?: boolean;
  /** Permit dangerous commands that the validator otherwise blocks. */
  allowDangerousCommands?: boolean;
  /**
   * Permit file tools to resolve paths outside `cwd`. Defaults to false.
   *
   * @deprecated Prefer the granular
   *   `workspace.allowReadOutside` / `workspace.allowWriteOutside`.
   *   Kept for backward compatibility; setting either of the
   *   workspace fields takes precedence over this flag.
   */
  allowOutsideWorkspace?: boolean;
  /**
   * Add the built-in local coding suite (`read_file`, `write_file`,
   * `edit_file`, `grep_search`, `glob_search`, `list_directory`, plus local
   * code/bash execution tools) when `engine` is `local`. Defaults to true.
   */
  includeCodingTools?: boolean;
  /**
   * Override the process launcher. When set, replaces
   * `child_process.spawn` for every local tool invocation, allowing
   * SSH/container delegation. Default: native spawn.
   */
  spawn?: LocalSpawn;
  /**
   * Tree-sitter-bash AST validation pass on bash commands.
   * - `'off'` skips AST validation (regex + `bash -n` only — current behavior).
   * - `'auto'` runs the AST validator when `tree-sitter` modules are
   *   available; falls back silently otherwise.
   * - `'strict'` requires the AST validator and fails closed when
   *   parsing is unavailable or the command is too complex to verify.
   * Default: `'off'` to preserve historical behavior.
   */
  bashAst?: LocalBashAstMode;
  /**
   * Enable per-Run file checkpointing for `edit_file` / `write_file`
   * so callers can rewind file changes after a failed batch. When
   * true and the run is constructed via `Run.create(...)`, the host
   * can call `Run.rewindFiles()` (or `Run.getFileCheckpointer()` for
   * the raw checkpointer). Defaults to false.
   */
  fileCheckpointing?: boolean;
  /**
   * Maximum bytes to read in `read_file` before returning a stub.
   * Defaults to 10 MiB.
   */
  maxReadBytes?: number;
  /**
   * Controls whether `read_file` returns binary files as inline
   * `MessageContentComplex[]` attachments (so vision-capable models
   * see them) or as a textual stub.
   *
   *   - `'off'`        : never embed; current binary-stub behavior.
   *   - `'images-only'`: embed images (png/jpeg/gif/webp) as
   *     `image_url` blocks; other binaries get the stub.
   *   - `'images-and-pdf'` : also embed PDFs as `image_url` data URLs
   *     (Anthropic accepts these in tool_result; other providers may
   *     degrade to JSON).
   *
   * Defaults to `'off'` to preserve current behavior.
   */
  attachReadAttachments?: 'off' | 'images-only' | 'images-and-pdf';
  /**
   * Maximum pre-encoding byte size to embed inline. Anything larger
   * degrades to an `<oversize>` stub. Defaults to 5 MiB to bound the
   * post-base64 token cost.
   */
  maxAttachmentBytes?: number;
  /**
   * Run a fast per-file syntax check after every successful
   * `edit_file` / `write_file`. When the checker finds an error,
   * the diagnostics are appended to the tool result so the model
   * can self-correct without a separate read round-trip.
   *
   *   - `'off'` (default) : skip; current behavior.
   *   - `'auto'`          : run the checker for known file types
   *     when the corresponding tool is on PATH. Silently skip
   *     otherwise.
   *   - `'strict'`        : same as `'auto'`, plus fail the tool
   *     call with the error so the model is forced to react. Use
   *     when you don't trust the model to read a non-blocking
   *     advisory.
   *
   * Built-in checkers: Node `node --check` for `.js/.mjs/.cjs`,
   * Python `py_compile` for `.py`, `JSON.parse` for `.json`,
   * `bash -n` for `.sh/.bash`. TypeScript falls back to `compile_check`
   * (project-level) since per-file `.ts` syntax check requires the
   * `typescript` package; the host can wire a per-file checker via
   * `local.spawn` if desired.
   */
  postEditSyntaxCheck?: 'off' | 'auto' | 'strict';
  /**
   * Configuration for the `compile_check` tool. When `engine` is
   * `local` and `includeCodingTools` is on, the SDK exposes a
   * `compile_check` tool that runs the project's standard
   * type/lint command (`tsc --noEmit`, `cargo check`, etc.).
   */
  compileCheck?: {
    /**
     * Override the auto-detected command. Runs verbatim from `cwd`
     * via the local engine's standard spawn pipeline (sandbox / AST
     * validation / output overflow all apply).
     */
    command?: string;
    /** Default timeout for `compile_check`, in milliseconds. Defaults to 120s. */
    timeoutMs?: number;
  };
};

export type ToolExecutionConfig = {
  /** `sandbox` preserves the remote Code API behavior and is the default. */
  engine?: ToolExecutionEngine;
  /** Local process execution settings used when `engine` is `local`. */
  local?: LocalExecutionConfig;
};

export type ProgrammaticCache = {
  toolMap: ToolMap;
  toolDefs: LCTool[];
  /**
   * Hook context plumbed through by ToolNode for the local
   * programmatic-tool path so the in-process bridge can run
   * `PreToolUse` hooks (deny / updatedInput) for inner tool calls.
   * Not present for non-local (remote-engine) programmatic calling
   * which dispatches inner tools through the host's own pipeline.
   */
  hookContext?: ProgrammaticHookContext;
};

export type ProgrammaticHookContext = {
  registry: import('@/hooks').HookRegistry | undefined;
  runId: string;
  threadId?: string;
  agentId?: string;
};

/** Search mode: code_interpreter uses external sandbox, local uses safe substring matching */
export type ToolSearchMode = 'code_interpreter' | 'local';

/** Format for MCP tool names in search results */
export type McpNameFormat = 'full' | 'base';

/** Parameters for creating a Tool Search tool */
export type ToolSearchParams = {
  toolRegistry?: LCToolRegistry;
  onlyDeferred?: boolean;
  baseUrl?: string;
  /** Search mode: 'code_interpreter' (default) uses sandbox for regex, 'local' uses safe substring matching */
  mode?: ToolSearchMode;
  /** Filter tools to only those from specific MCP server(s). Can be a single name or array of names. */
  mcpServer?: string | string[];
  /** Format for MCP tool names: 'full' (tool_mcp_server) or 'base' (tool only). Default: 'full' */
  mcpNameFormat?: McpNameFormat;
};

/** Simplified tool metadata for search purposes */
export type ToolMetadata = {
  name: string;
  description: string;
  parameters?: JsonSchemaType;
};

/** Individual search result for a matching tool */
export type ToolSearchResult = {
  tool_name: string;
  match_score: number;
  matched_field: string;
  snippet: string;
};

/** Response from the tool search operation */
export type ToolSearchResponse = {
  tool_references: ToolSearchResult[];
  total_tools_searched: number;
  pattern_used: string;
};

/** Artifact returned alongside the formatted search results */
export type ToolSearchArtifact = {
  tool_references: ToolSearchResult[];
  metadata: {
    total_searched: number;
    pattern: string;
    error?: string;
  };
};

// ============================================================================
// Programmatic Tool Calling Types
// ============================================================================

/**
 * Tool call requested by the Code API during programmatic execution
 */
export type PTCToolCall = {
  /** Unique ID like "call_001" */
  id: string;
  /** Tool name */
  name: string;
  /** Input parameters */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  input: Record<string, any>;
};

/**
 * Tool result sent back to the Code API
 */
export type PTCToolResult = {
  /** Matches PTCToolCall.id */
  call_id: string;
  /** Tool execution result (any JSON-serializable value) */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result: any;
  /** Whether tool execution failed */
  is_error: boolean;
  /** Error details if is_error=true */
  error_message?: string;
};

/**
 * Response from the Code API for programmatic execution
 */
export type ProgrammaticExecutionResponse = {
  status: 'tool_call_required' | 'completed' | 'error' | unknown;
  session_id?: string;

  /** Present when status='tool_call_required' */
  continuation_token?: string;
  tool_calls?: PTCToolCall[];

  /** Present when status='completed' */
  stdout?: string;
  stderr?: string;
  files?: FileRefs;

  /** Present when status='error' */
  error?: string;
};

/**
 * Artifact returned by the PTC tool
 */
export type ProgrammaticExecutionArtifact = {
  session_id?: string;
  files?: FileRefs;
};

/** Parameters for creating a bash execution tool (same API as CodeExecutor, bash-only) */
export type BashExecutionToolParams = CodeExecutionToolParams;

/** Parameters for creating a bash programmatic tool calling tool (same API as PTC, bash-only) */
export type BashProgrammaticToolCallingParams = ProgrammaticToolCallingParams;

/**
 * Initialization parameters for the PTC tool
 */
export type ProgrammaticToolCallingParams = {
  /** Code API base URL (or use CODE_BASEURL env var) */
  baseUrl?: string;
  /** Safety limit for round-trips (default: 20) */
  maxRoundTrips?: number;
  /** HTTP proxy URL */
  proxy?: string;
  /** Enable debug logging (or set PTC_DEBUG=true env var) */
  debug?: boolean;
};

// ============================================================================
// Tool Session Context Types
// ============================================================================

/**
 * Tracks code execution session state for automatic file persistence.
 * Stored in Graph.sessions and injected into subsequent tool invocations.
 */
export type CodeSessionContext = {
  /** Session ID from the code execution environment */
  session_id: string;
  /** Files generated in this session (for context/tracking) */
  files?: FileRefs;
  /** Timestamp of last update */
  lastUpdated: number;
};

/**
 * Artifact structure returned by code execution tools (CodeExecutor, PTC).
 * Used to extract session context after tool completion.
 */
export type CodeExecutionArtifact = {
  session_id?: string;
  files?: FileRefs;
};

/**
 * Generic session context union type for different tool types.
 * Extend this as new tool session types are added.
 */
export type ToolSessionContext = CodeSessionContext;

/**
 * Map of tool names to their session contexts.
 * Keys are tool constants (e.g., Constants.EXECUTE_CODE, Constants.PROGRAMMATIC_TOOL_CALLING).
 */
export type ToolSessionMap = Map<string, ToolSessionContext>;
