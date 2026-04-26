// src/types/tools.ts
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableToolLike } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { HookRegistry } from '@/hooks';
import type { ToolOutputReferenceRegistry } from '@/tools/toolOutputReferences';
import type { MessageContentComplex, ToolErrorData } from './stream';

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
   * Hook registry for PreToolUse/PostToolUse lifecycle hooks.
   * Only fires for event-driven tool calls (`dispatchToolEvents`). Tools
   * routed through `directToolNames` bypass hook dispatch entirely.
   */
  hookRegistry?: HookRegistry;
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

export type ProgrammaticCache = { toolMap: ToolMap; toolDefs: LCTool[] };

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
