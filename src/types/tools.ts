// src/types/tools.ts
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableToolLike } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ToolErrorData } from './stream';
import { EnvVar } from '@/common';

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
};

export type ToolNodeConstructorParams = ToolRefs & ToolNodeOptions;

export type ToolEndEvent = {
  /** The Step Id of the Tool Call */
  id: string;
  /** The Completed Tool Call */
  tool_call: ToolCall;
  /** The content index of the tool call */
  index: number;
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
      apiKey?: string;
      files?: CodeEnvFile[];
      [EnvVar.CODE_API_KEY]?: string;
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
};

/** Map of tool names to tool definitions */
export type LCToolRegistry = Map<string, LCTool>;

export type ProgrammaticCache = { toolMap: ToolMap; toolDefs: LCTool[] };

/** Search mode: code_interpreter uses external sandbox, local uses safe substring matching */
export type ToolSearchMode = 'code_interpreter' | 'local';

/** Parameters for creating a Tool Search tool */
export type ToolSearchParams = {
  apiKey?: string;
  toolRegistry?: LCToolRegistry;
  onlyDeferred?: boolean;
  baseUrl?: string;
  /** Search mode: 'code_interpreter' (default) uses sandbox for regex, 'local' uses safe substring matching */
  mode?: ToolSearchMode;
  /** Filter tools to only those from specific MCP server(s). Can be a single name or array of names. */
  mcpServer?: string | string[];
  [key: string]: unknown;
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

/**
 * Initialization parameters for the PTC tool
 */
export type ProgrammaticToolCallingParams = {
  /** Code API key (or use CODE_API_KEY env var) */
  apiKey?: string;
  /** Code API base URL (or use CODE_BASEURL env var) */
  baseUrl?: string;
  /** Safety limit for round-trips (default: 20) */
  maxRoundTrips?: number;
  /** HTTP proxy URL */
  proxy?: string;
  /** Enable debug logging (or set PTC_DEBUG=true env var) */
  debug?: boolean;
  /** Environment variable key for API key */
  [key: string]: unknown;
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
  files: FileRefs;
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
