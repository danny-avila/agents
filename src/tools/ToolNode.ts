import { ToolCall } from '@langchain/core/messages/tool';
import {
  ToolMessage,
  HumanMessage,
  isAIMessage,
  isBaseMessage,
} from '@langchain/core/messages';
import {
  END,
  Send,
  Command,
  isCommand,
  interrupt,
  isGraphInterrupt,
  MessagesAnnotation,
} from '@langchain/langgraph';
import { AsyncLocalStorageProviderSingleton } from '@langchain/core/singletons';
import type {
  RunnableConfig,
  RunnableToolLike,
} from '@langchain/core/runnables';
import type { BaseMessage, AIMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type {
  ToolOutputResolveView,
  PreResolvedArgsMap,
  ResolvedArgsByCallId,
} from '@/tools/toolOutputReferences';
import type {
  HookRegistry,
  AggregatedHookResult,
  PostToolBatchEntry,
} from '@/hooks';
import type * as t from '@/types';
import { RunnableCallable } from '@/utils';
import {
  calculateMaxToolResultChars,
  truncateToolResultContent,
} from '@/utils/truncation';
import { safeDispatchCustomEvent } from '@/utils/events';
import { executeHooks } from '@/hooks';
import { toLangChainContent } from '@/messages/langchain';
import { Constants, GraphEvents, CODE_EXECUTION_TOOLS } from '@/common';
import {
  buildReferenceKey,
  ToolOutputReferenceRegistry,
} from '@/tools/toolOutputReferences';

/**
 * Per-call batch context for `runTool`. Bundles every optional
 * batch-scoped value the method needs so the signature stays at
 * three positional parameters even as new context fields are added.
 */
type RunToolBatchContext = {
  /** Position of this call within the parent ToolNode batch. */
  batchIndex?: number;
  /** Batch turn shared across every call in the batch. */
  turn?: number;
  /** Registry partition scope (run id or anonymous batch id). */
  batchScopeId?: string;
  /** Batch-local sink for post-substitution args. */
  resolvedArgsByCallId?: ResolvedArgsByCallId;
};

/**
 * Per-batch context for `dispatchToolEvents` / `executeViaEvent`.
 * Mirrors {@link RunToolBatchContext} for the event-driven path,
 * with bulk indices and the snapshot/pre-resolved-args carriers
 * used in the mixed direct+event flow.
 */
type DispatchBatchContext = {
  /** Per-call batch indices, parallel to the `toolCalls` array. */
  batchIndices?: number[];
  /** Batch turn shared across every call in the batch. */
  turn?: number;
  /** Registry partition scope (run id or anonymous batch id). */
  batchScopeId?: string;
  /**
   * Pre-resolved args keyed by `toolCallId`. Populated by the mixed
   * path so event calls don't re-resolve against a registry that
   * already contains same-turn direct outputs.
   */
  preResolvedArgs?: PreResolvedArgsMap;
  /**
   * Frozen pre-batch registry view used to re-resolve placeholders
   * a `PreToolUse` hook injects via `updatedInput` — preserves the
   * same-turn isolation guarantee for hook-rewritten args.
   */
  preBatchSnapshot?: ToolOutputResolveView;
};

/**
 * Helper to check if a value is a Send object
 */
function isSend(value: unknown): value is Send {
  return value instanceof Send;
}

/**
 * Build a `tool_call_id → ToolApprovalDecision` map from the host's
 * resume value. Hosts may return decisions either as an array (one per
 * action_request, in order) or as a record keyed by `tool_call_id`. Any
 * unrecognized shape (or a decision missing for a given call id) is
 * treated as "no decision" by callers — typically rejected so the run
 * doesn't silently invoke a tool the human never approved.
 */
function normalizeApprovalDecisions(
  callIds: string[],
  resumeValue: t.ToolApprovalDecision[] | t.ToolApprovalDecisionMap | undefined
): Map<string, t.ToolApprovalDecision> {
  const map = new Map<string, t.ToolApprovalDecision>();
  if (resumeValue == null) {
    return map;
  }
  if (Array.isArray(resumeValue)) {
    const limit = Math.min(callIds.length, resumeValue.length);
    for (let i = 0; i < limit; i++) {
      map.set(callIds[i], resumeValue[i]);
    }
    return map;
  }
  if (typeof resumeValue === 'object') {
    for (const callId of callIds) {
      const decision = (resumeValue as Partial<t.ToolApprovalDecisionMap>)[
        callId
      ];
      if (decision !== undefined) {
        map.set(callId, decision);
      }
    }
  }
  return map;
}

/**
 * Merges code execution session context into the sessions map.
 *
 * The codeapi worker reports two distinct ids on a code-execution result:
 *  - `artifact.session_id` (the `sessionId` arg here) is the EXEC session
 *    — the sandbox VM that ran the code. It's transient and torn down
 *    post-execution; subsequent calls cannot reuse it as a sandbox.
 *  - `file.session_id` on each `artifact.files[i]` is the STORAGE
 *    session — the file-server bucket prefix where the artifact actually
 *    lives and is served from.
 *
 * Per-file `session_id` is preserved (not overwritten with the exec id)
 * because `_injected_files` are looked up against the file-server's
 * storage path on subsequent tool calls. Stomping the storage id with
 * the exec id silently 404s every follow-up tool call within the same
 * run — `cat /mnt/data/foo.txt` reports "No such file or directory"
 * because the worker can't mount a file at a path the storage doesn't
 * know about. Fall back to `sessionId` only when the per-file id is
 * absent (older worker payloads).
 */
function updateCodeSession(
  sessions: t.ToolSessionMap,
  sessionId: string,
  files: t.FileRefs | undefined
): void {
  const newFiles = files ?? [];
  const existingSession = sessions.get(Constants.EXECUTE_CODE) as
    | t.CodeSessionContext
    | undefined;
  const existingFiles = existingSession?.files ?? [];

  if (newFiles.length > 0) {
    const filesWithSession: t.FileRefs = newFiles.map((file) => ({
      ...file,
      session_id: file.session_id ?? sessionId,
    }));
    const newFileNames = new Set(filesWithSession.map((f) => f.name));
    const filteredExisting = existingFiles.filter(
      (f) => !newFileNames.has(f.name)
    );
    sessions.set(Constants.EXECUTE_CODE, {
      session_id: sessionId,
      files: [...filteredExisting, ...filesWithSession],
      lastUpdated: Date.now(),
    });
  } else {
    sessions.set(Constants.EXECUTE_CODE, {
      session_id: sessionId,
      files: existingFiles,
      lastUpdated: Date.now(),
    });
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export class ToolNode<T = any> extends RunnableCallable<T, T> {
  private toolMap: Map<string, StructuredToolInterface | RunnableToolLike>;
  private loadRuntimeTools?: t.ToolRefGenerator;
  handleToolErrors = true;
  trace = false;
  toolCallStepIds?: Map<string, string>;
  errorHandler?: t.ToolNodeConstructorParams['errorHandler'];
  private toolUsageCount: Map<string, number>;
  /** Maps toolCallId → turn captured in runTool, used by handleRunToolCompletions */
  private toolCallTurns: Map<string, number> = new Map();
  /** Tool registry for filtering (lazy computation of programmatic maps) */
  private toolRegistry?: t.LCToolRegistry;
  /** Cached programmatic tools (computed once on first PTC call) */
  private programmaticCache?: t.ProgrammaticCache;
  /** Reference to Graph's sessions map for automatic session injection */
  private sessions?: t.ToolSessionMap;
  /** When true, dispatches ON_TOOL_EXECUTE events instead of invoking tools directly */
  private eventDrivenMode: boolean = false;
  /** Agent ID for event-driven mode */
  private agentId?: string;
  /** Tool names that bypass event dispatch and execute directly (e.g., graph-managed handoff tools) */
  private directToolNames?: Set<string>;
  /** Maximum characters allowed in a single tool result before truncation. */
  private maxToolResultChars: number;
  /** Hook registry for PreToolUse/PostToolUse lifecycle hooks */
  private hookRegistry?: HookRegistry;
  /**
   * Run-scoped HITL config. When `enabled`, `ask` decisions from
   * PreToolUse hooks raise a LangGraph `interrupt()` instead of being
   * treated as fail-closed denies.
   */
  private humanInTheLoop?: t.HumanInTheLoopConfig;
  /**
   * Registry of tool outputs keyed by `tool<idx>turn<turn>`.
   *
   * Populated only when `toolOutputReferences.enabled` is true. The
   * registry owns the run-scoped state (turn counter, last-seen runId,
   * warn-once memo, stored outputs), so sharing a single instance
   * across multiple ToolNodes in a run lets cross-agent `{{…}}`
   * references resolve — which is why multi-agent graphs pass the
   * *same* instance to every ToolNode they compile rather than each
   * ToolNode building its own.
   */
  private toolOutputRegistry?: ToolOutputReferenceRegistry;
  /**
   * Monotonic counter used to mint a unique scope id for anonymous
   * batches (ones invoked without a `run_id` in
   * `config.configurable`). Each such batch gets its own registry
   * partition so concurrent anonymous invocations can't delete each
   * other's in-flight state.
   */
  private anonBatchCounter: number = 0;

  constructor({
    tools,
    toolMap,
    name,
    tags,
    errorHandler,
    toolCallStepIds,
    handleToolErrors,
    loadRuntimeTools,
    toolRegistry,
    sessions,
    eventDrivenMode,
    agentId,
    directToolNames,
    maxContextTokens,
    maxToolResultChars,
    hookRegistry,
    humanInTheLoop,
    toolOutputReferences,
    toolOutputRegistry,
  }: t.ToolNodeConstructorParams) {
    super({ name, tags, func: (input, config) => this.run(input, config) });
    this.toolMap = toolMap ?? new Map(tools.map((tool) => [tool.name, tool]));
    this.toolCallStepIds = toolCallStepIds;
    this.handleToolErrors = handleToolErrors ?? this.handleToolErrors;
    this.loadRuntimeTools = loadRuntimeTools;
    this.errorHandler = errorHandler;
    this.toolUsageCount = new Map<string, number>();
    this.toolRegistry = toolRegistry;
    this.sessions = sessions;
    this.eventDrivenMode = eventDrivenMode ?? false;
    this.agentId = agentId;
    this.directToolNames = directToolNames;
    this.maxToolResultChars =
      maxToolResultChars ?? calculateMaxToolResultChars(maxContextTokens);
    this.hookRegistry = hookRegistry;
    this.humanInTheLoop = humanInTheLoop;
    /**
     * Precedence: an explicitly passed `toolOutputRegistry` instance
     * wins over a config object so a host (`Graph`) can share one
     * registry across many ToolNodes. When only the config is
     * provided (direct ToolNode usage), build a local registry so
     * the feature still works without graph-level plumbing. Registry
     * caps are intentionally decoupled from `maxToolResultChars`:
     * the registry stores the raw untruncated output so a later
     * `{{…}}` substitution pipes the full payload into the next
     * tool, even when the LLM saw a truncated preview.
     */
    if (toolOutputRegistry != null) {
      this.toolOutputRegistry = toolOutputRegistry;
    } else if (toolOutputReferences?.enabled === true) {
      this.toolOutputRegistry = new ToolOutputReferenceRegistry({
        maxOutputSize: toolOutputReferences.maxOutputSize,
        maxTotalSize: toolOutputReferences.maxTotalSize,
      });
    }
  }

  /**
   * Returns the run-scoped tool output registry, or `undefined` when
   * the feature is disabled.
   *
   * @internal Exposed for test observation only. Host code should rely
   * on `{{tool<i>turn<n>}}` substitution at tool-invocation time and
   * not mutate the registry directly.
   */
  public _unsafeGetToolOutputRegistry():
    | ToolOutputReferenceRegistry
    | undefined {
    return this.toolOutputRegistry;
  }

  /**
   * Returns cached programmatic tools, computing once on first access.
   * Single iteration builds both toolMap and toolDefs simultaneously.
   */
  private getProgrammaticTools(): { toolMap: t.ToolMap; toolDefs: t.LCTool[] } {
    if (this.programmaticCache) return this.programmaticCache;

    const toolMap: t.ToolMap = new Map();
    const toolDefs: t.LCTool[] = [];

    if (this.toolRegistry) {
      for (const [name, toolDef] of this.toolRegistry) {
        if (
          (toolDef.allowed_callers ?? ['direct']).includes('code_execution')
        ) {
          toolDefs.push(toolDef);
          const tool = this.toolMap.get(name);
          if (tool) toolMap.set(name, tool);
        }
      }
    }

    this.programmaticCache = { toolMap, toolDefs };
    return this.programmaticCache;
  }

  /**
   * Returns a snapshot of the current tool usage counts.
   * @returns A ReadonlyMap where keys are tool names and values are their usage counts.
   */
  public getToolUsageCounts(): ReadonlyMap<string, number> {
    return new Map(this.toolUsageCount); // Return a copy
  }

  /**
   * Runs a single tool call with error handling.
   *
   * `batchIndex` is the tool's position within the current ToolNode
   * batch and, together with `this.currentTurn`, forms the key used to
   * register the output for future `{{tool<idx>turn<turn>}}`
   * substitutions. Omit when no registration should occur.
   */
  protected async runTool(
    call: ToolCall,
    config: RunnableConfig,
    batchContext: RunToolBatchContext = {}
  ): Promise<BaseMessage | Command> {
    const { batchIndex, turn, batchScopeId, resolvedArgsByCallId } =
      batchContext;
    const tool = this.toolMap.get(call.name);
    const registry = this.toolOutputRegistry;
    /**
     * Precompute the reference key once per call — captured locally
     * so concurrent `invoke()` calls on the same ToolNode cannot race
     * on a shared turn field.
     */
    const refKey =
      registry != null && batchIndex != null && turn != null
        ? buildReferenceKey(batchIndex, turn)
        : undefined;
    /**
     * Hoisted outside the try so the catch branch can append
     * `[unresolved refs: …]` to error messages — otherwise the LLM
     * only sees a generic error when it references a bad key, losing
     * the self-correction signal this feature is meant to provide.
     */
    let unresolvedRefs: string[] = [];
    /**
     * Use the caller-provided `batchScopeId` when threaded from
     * `run()` (so anonymous batches get their own unique scope).
     * Fall back to the config's `run_id` when runTool is invoked
     * from a context that doesn't thread it — that still preserves
     * the runId-based partitioning for named runs.
     */
    const runId =
      batchScopeId ?? (config.configurable?.run_id as string | undefined);
    try {
      if (tool === undefined) {
        throw new Error(`Tool "${call.name}" not found.`);
      }
      /**
       * `usageCount` is the per-tool-name invocation index that
       * web-search and other tools observe via `invokeParams.turn`.
       * It is intentionally distinct from the outer `turn` parameter
       * (the batch turn used for ref keys); the latter is captured
       * before the try block when constructing `refKey`.
       */
      const usageCount = this.toolUsageCount.get(call.name) ?? 0;
      this.toolUsageCount.set(call.name, usageCount + 1);
      if (call.id != null && call.id !== '') {
        this.toolCallTurns.set(call.id, usageCount);
      }
      let args = call.args;
      if (registry != null) {
        const { resolved, unresolved } = registry.resolve(runId, args);
        args = resolved;
        unresolvedRefs = unresolved;
        /**
         * Expose the post-substitution args to downstream completion
         * events so audit logs / host-side `ON_RUN_STEP_COMPLETED`
         * handlers observe what actually ran, not the `{{…}}`
         * template. Only string/object args are worth recording.
         */
        if (
          resolvedArgsByCallId != null &&
          call.id != null &&
          call.id !== '' &&
          resolved !== call.args &&
          typeof resolved === 'object'
        ) {
          resolvedArgsByCallId.set(
            call.id,
            resolved as Record<string, unknown>
          );
        }
      }
      const stepId = this.toolCallStepIds?.get(call.id!);

      // Build invoke params - LangChain extracts non-schema fields to config.toolCall
      // `turn` here is the per-tool usage count (matches what tools have
      // observed historically via config.toolCall.turn — e.g. web search).
      let invokeParams: Record<string, unknown> = {
        ...call,
        args,
        type: 'tool_call',
        stepId,
        turn: usageCount,
      };

      // Inject runtime data for special tools (becomes available at config.toolCall)
      if (
        call.name === Constants.PROGRAMMATIC_TOOL_CALLING ||
        call.name === Constants.BASH_PROGRAMMATIC_TOOL_CALLING
      ) {
        const { toolMap, toolDefs } = this.getProgrammaticTools();
        invokeParams = {
          ...invokeParams,
          toolMap,
          toolDefs,
        };
      } else if (call.name === Constants.TOOL_SEARCH) {
        invokeParams = {
          ...invokeParams,
          toolRegistry: this.toolRegistry,
        };
      }

      /**
       * Inject session context for code execution tools when available.
       * Each file uses its own session_id (supporting multi-session file tracking).
       * Both session_id and _injected_files are injected directly to invokeParams
       * (not inside args) so they bypass Zod schema validation and reach config.toolCall.
       *
       * session_id is always injected when available (even without tracked files)
       * so the CodeExecutor can fall back to the /files endpoint for session continuity.
       */
      if (CODE_EXECUTION_TOOLS.has(call.name)) {
        const codeSession = this.sessions?.get(Constants.EXECUTE_CODE) as
          | t.CodeSessionContext
          | undefined;
        if (codeSession?.session_id != null && codeSession.session_id !== '') {
          invokeParams = {
            ...invokeParams,
            session_id: codeSession.session_id,
          };

          if (codeSession.files != null && codeSession.files.length > 0) {
            const fileRefs: t.CodeEnvFile[] = codeSession.files.map((file) => ({
              session_id: file.session_id ?? codeSession.session_id,
              id: file.id,
              name: file.name,
            }));
            invokeParams._injected_files = fileRefs;
          }
        }
      }

      const output = await tool.invoke(invokeParams, config);
      if (isCommand(output)) {
        return output;
      }
      if (isBaseMessage(output) && output._getType() === 'tool') {
        const toolMsg = output as ToolMessage;
        const isError = toolMsg.status === 'error';
        if (isError) {
          /**
           * Error ToolMessages bypass registration but still stamp the
           * unresolved-refs hint into `additional_kwargs` so the lazy
           * annotation transform surfaces it to the LLM, letting the
           * model self-correct when its reference key caused the
           * failure. Persisted `content` stays clean.
           */
          if (unresolvedRefs.length > 0) {
            toolMsg.additional_kwargs = {
              ...toolMsg.additional_kwargs,
              _unresolvedRefs: unresolvedRefs,
            };
          }
          return toolMsg;
        }
        if (this.toolOutputRegistry != null || unresolvedRefs.length > 0) {
          if (typeof toolMsg.content === 'string') {
            const rawContent = toolMsg.content;
            const llmContent = truncateToolResultContent(
              rawContent,
              this.maxToolResultChars
            );
            toolMsg.content = llmContent;
            const refMeta = this.recordOutputReference(
              runId,
              rawContent,
              refKey,
              unresolvedRefs
            );
            if (refMeta != null) {
              toolMsg.additional_kwargs = {
                ...toolMsg.additional_kwargs,
                ...refMeta,
              };
            }
          } else {
            /**
             * Non-string content (multi-part content blocks — text +
             * image). Known limitation: we cannot register under a
             * reference key because there's no canonical serialized
             * form. Warn once per tool per run when the caller
             * intended to register. The unresolved-refs hint is still
             * stamped as metadata; the lazy transform prepends a text
             * block at request time so the LLM gets the self-correction
             * signal.
             */
            if (unresolvedRefs.length > 0) {
              toolMsg.additional_kwargs = {
                ...toolMsg.additional_kwargs,
                _unresolvedRefs: unresolvedRefs,
              };
            }
            if (
              refKey != null &&
              this.toolOutputRegistry!.claimWarnOnce(runId, call.name)
            ) {
              // eslint-disable-next-line no-console
              console.warn(
                `[ToolNode] Skipping tool output reference for "${call.name}": ` +
                  'ToolMessage content is not a string (further occurrences for this tool in the same run are suppressed).'
              );
            }
          }
        }
        return toolMsg;
      }
      const rawContent =
        typeof output === 'string' ? output : JSON.stringify(output);
      const truncated = truncateToolResultContent(
        rawContent,
        this.maxToolResultChars
      );
      const refMeta = this.recordOutputReference(
        runId,
        rawContent,
        refKey,
        unresolvedRefs
      );
      return new ToolMessage({
        status: 'success',
        name: tool.name,
        content: truncated,
        tool_call_id: call.id!,
        ...(refMeta != null && {
          additional_kwargs: refMeta as Record<string, unknown>,
        }),
      });
    } catch (_e: unknown) {
      const e = _e as Error;
      if (!this.handleToolErrors) {
        throw e;
      }
      if (isGraphInterrupt(e)) {
        throw e;
      }
      if (this.errorHandler) {
        try {
          await this.errorHandler(
            {
              error: e,
              id: call.id!,
              name: call.name,
              input: call.args,
            },
            config.metadata
          );
        } catch (handlerError) {
          // eslint-disable-next-line no-console
          console.error('Error in errorHandler:', {
            toolName: call.name,
            toolCallId: call.id,
            toolArgs: call.args,
            stepId: this.toolCallStepIds?.get(call.id!),
            turn: this.toolUsageCount.get(call.name),
            originalError: {
              message: e.message,
              stack: e.stack ?? undefined,
            },
            handlerError:
              handlerError instanceof Error
                ? {
                  message: handlerError.message,
                  stack: handlerError.stack ?? undefined,
                }
                : {
                  message: String(handlerError),
                  stack: undefined,
                },
          });
        }
      }
      const errorContent = `Error: ${e.message}\n Please fix your mistakes.`;
      const refMeta =
        unresolvedRefs.length > 0
          ? this.recordOutputReference(
            runId,
            errorContent,
            undefined,
            unresolvedRefs
          )
          : undefined;
      return new ToolMessage({
        status: 'error',
        content: errorContent,
        name: call.name,
        tool_call_id: call.id ?? '',
        ...(refMeta != null && {
          additional_kwargs: refMeta as Record<string, unknown>,
        }),
      });
    }
  }

  /**
   * Registers the full, raw output under `refKey` (when provided) and
   * builds the per-message ref metadata stamped onto the resulting
   * `ToolMessage.additional_kwargs`. The metadata is read at LLM-
   * request time by `annotateMessagesForLLM` to produce a transient
   * annotated copy of the message — the persisted `content` itself
   * stays clean.
   *
   * @param registryContent  The full, untruncated output to store in
   *   the registry so `{{tool<i>turn<n>}}` substitutions deliver the
   *   complete payload. Ignored when `refKey` is undefined.
   * @param refKey  Precomputed `tool<i>turn<n>` key, or undefined when
   *   the output is not to be registered (errors, disabled feature,
   *   unavailable batch/turn).
   * @param unresolved  Placeholder keys that did not resolve; surfaced
   *   to the LLM lazily so it can self-correct.
   * @returns A `ToolMessageRefMetadata` object when there is anything
   *   to stamp, otherwise `undefined`.
   */
  private recordOutputReference(
    runId: string | undefined,
    registryContent: string,
    refKey: string | undefined,
    unresolved: string[]
  ): t.ToolMessageRefMetadata | undefined {
    if (this.toolOutputRegistry != null && refKey != null) {
      this.toolOutputRegistry.set(runId, refKey, registryContent);
    }
    if (refKey == null && unresolved.length === 0) return undefined;
    const meta: t.ToolMessageRefMetadata = {};
    if (refKey != null) {
      meta._refKey = refKey;
      /**
       * Stamp the registry scope alongside the key so the lazy
       * annotation transform can look up the right bucket. Anonymous
       * invocations get a synthetic per-batch scope (`\0anon-<n>`)
       * that `attemptInvoke` cannot derive from
       * `config.configurable.run_id` — without this, anonymous-run
       * refs would silently fail registry lookup and the LLM would
       * never see `[ref: …]` markers for outputs that were registered.
       */
      if (runId != null) meta._refScope = runId;
    }
    if (unresolved.length > 0) meta._unresolvedRefs = unresolved;
    return meta;
  }

  /**
   * Builds code session context for injection into event-driven tool calls.
   * Mirrors the session injection logic in runTool() for direct execution.
   */
  private getCodeSessionContext(): t.ToolCallRequest['codeSessionContext'] {
    if (!this.sessions) {
      return undefined;
    }

    const codeSession = this.sessions.get(Constants.EXECUTE_CODE) as
      | t.CodeSessionContext
      | undefined;
    if (!codeSession) {
      return undefined;
    }

    const context: NonNullable<t.ToolCallRequest['codeSessionContext']> = {
      session_id: codeSession.session_id,
    };

    if (codeSession.files && codeSession.files.length > 0) {
      context.files = codeSession.files.map((file) => ({
        session_id: file.session_id ?? codeSession.session_id,
        id: file.id,
        name: file.name,
      }));
    }

    return context;
  }

  /**
   * Extracts code execution session context from tool results and stores in Graph.sessions.
   * Mirrors the session storage logic in handleRunToolCompletions for direct execution.
   */
  private storeCodeSessionFromResults(
    results: t.ToolExecuteResult[],
    requestMap: Map<string, t.ToolCallRequest>
  ): void {
    if (!this.sessions) {
      return;
    }

    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      if (result.status !== 'success' || result.artifact == null) {
        continue;
      }

      const request = requestMap.get(result.toolCallId);
      if (
        !request?.name ||
        (!CODE_EXECUTION_TOOLS.has(request.name) &&
          request.name !== Constants.SKILL_TOOL)
      ) {
        continue;
      }

      const artifact = result.artifact as t.CodeExecutionArtifact | undefined;
      if (artifact?.session_id == null || artifact.session_id === '') {
        continue;
      }

      updateCodeSession(this.sessions, artifact.session_id!, artifact.files);
    }
  }

  /**
   * Post-processes standard runTool outputs: dispatches ON_RUN_STEP_COMPLETED
   * and stores code session context. Mirrors the completion handling in
   * dispatchToolEvents for the event-driven path.
   *
   * By handling completions here in graph context (rather than in the
   * stream consumer via ToolEndHandler), the race between the stream
   * consumer and graph execution is eliminated.
   *
   * @param resolvedArgsByCallId  Per-batch resolved-args sink populated
   *   by `runTool`. Threaded as a local map (instead of instance state)
   *   so concurrent batches cannot read each other's entries.
   */
  private handleRunToolCompletions(
    calls: ToolCall[],
    outputs: (BaseMessage | Command)[],
    config: RunnableConfig,
    resolvedArgsByCallId?: ResolvedArgsByCallId
  ): void {
    for (let i = 0; i < calls.length; i++) {
      const call = calls[i];
      const output = outputs[i];
      const turn = this.toolCallTurns.get(call.id!) ?? 0;

      if (isCommand(output)) {
        continue;
      }

      const toolMessage = output as ToolMessage;
      const toolCallId = call.id ?? '';

      // Skip error ToolMessages when errorHandler already dispatched ON_RUN_STEP_COMPLETED
      // via handleToolCallErrorStatic. Without this check, errors would be double-dispatched.
      if (toolMessage.status === 'error' && this.errorHandler != null) {
        continue;
      }

      if (this.sessions && CODE_EXECUTION_TOOLS.has(call.name)) {
        const artifact = toolMessage.artifact as
          | t.CodeExecutionArtifact
          | undefined;
        if (artifact?.session_id != null && artifact.session_id !== '') {
          updateCodeSession(this.sessions, artifact.session_id, artifact.files);
        }
      }

      // Dispatch ON_RUN_STEP_COMPLETED via custom event (same path as dispatchToolEvents)
      const stepId = this.toolCallStepIds?.get(toolCallId) ?? '';
      if (!stepId) {
        continue;
      }

      const contentString =
        typeof toolMessage.content === 'string'
          ? toolMessage.content
          : JSON.stringify(toolMessage.content);

      /**
       * Prefer the post-substitution args when a `{{…}}` placeholder
       * was resolved in `runTool`. This keeps
       * `ON_RUN_STEP_COMPLETED.tool_call.args` consistent with what
       * the tool actually received rather than leaking the template.
       */
      const effectiveArgs = resolvedArgsByCallId?.get(toolCallId) ?? call.args;
      const tool_call: t.ProcessedToolCall = {
        args:
          typeof effectiveArgs === 'string'
            ? (effectiveArgs as string)
            : JSON.stringify((effectiveArgs as unknown) ?? {}),
        name: call.name,
        id: toolCallId,
        output: contentString,
        progress: 1,
      };

      safeDispatchCustomEvent(
        GraphEvents.ON_RUN_STEP_COMPLETED,
        {
          result: {
            id: stepId,
            index: turn,
            type: 'tool_call' as const,
            tool_call,
          },
        },
        config
      );
    }
  }

  /**
   * Dispatches tool calls to the host via ON_TOOL_EXECUTE event and returns raw ToolMessages.
   * Core logic for event-driven execution, separated from output shaping.
   *
   * Hook lifecycle (when `hookRegistry` is set):
   * 1. **PreToolUse** fires per call in parallel before dispatch. Denied
   *    calls produce error ToolMessages and fire **PermissionDenied**;
   *    surviving calls proceed with optional `updatedInput`.
   * 2. Surviving calls are dispatched to the host via `ON_TOOL_EXECUTE`.
   * 3. **PostToolUse** / **PostToolUseFailure** fire per result. Post hooks
   *    can replace tool output via `updatedOutput`.
   * 4. Injected messages from results are collected and returned alongside
   *    ToolMessages (appended AFTER to respect provider ordering).
   */
  private async dispatchToolEvents(
    toolCalls: ToolCall[],
    config: RunnableConfig,
    batchContext: DispatchBatchContext = {}
  ): Promise<{ toolMessages: ToolMessage[]; injected: BaseMessage[] }> {
    const {
      batchIndices,
      turn,
      batchScopeId,
      preResolvedArgs,
      preBatchSnapshot,
    } = batchContext;
    const runId = (config.configurable?.run_id as string | undefined) ?? '';
    /**
     * Registry-facing scope id — prefers the caller-threaded
     * `batchScopeId` so anonymous batches target their own unique
     * bucket and don't step on concurrent anonymous invocations.
     * Hooks and event payloads keep using the empty-string coerced
     * `runId` for backward compat.
     */
    const registryRunId =
      batchScopeId ?? (config.configurable?.run_id as string | undefined);
    const threadId = config.configurable?.thread_id as string | undefined;
    const registry = this.toolOutputRegistry;
    const unresolvedByCallId = new Map<string, string[]>();

    const preToolCalls = toolCalls.map((call, i) => {
      const originalArgs = call.args as Record<string, unknown>;
      let resolvedArgs = originalArgs;
      /**
       * When the caller provided a pre-resolved map (the mixed
       * direct+event path snapshots event args synchronously before
       * awaiting directs so they can't accidentally resolve
       * same-turn direct outputs), use those entries verbatim instead
       * of re-resolving against a registry that may have changed
       * since the batch started.
       */
      const pre = call.id != null ? preResolvedArgs?.get(call.id) : undefined;
      if (pre != null) {
        resolvedArgs = pre.resolved;
        if (pre.unresolved.length > 0 && call.id != null) {
          unresolvedByCallId.set(call.id, pre.unresolved);
        }
      } else if (registry != null) {
        const { resolved, unresolved } = registry.resolve(
          registryRunId,
          originalArgs
        );
        resolvedArgs = resolved as Record<string, unknown>;
        if (unresolved.length > 0 && call.id != null) {
          unresolvedByCallId.set(call.id, unresolved);
        }
      }
      return {
        call,
        stepId: this.toolCallStepIds?.get(call.id!) ?? '',
        args: resolvedArgs,
        batchIndex: batchIndices?.[i],
      };
    });

    const messageByCallId = new Map<string, ToolMessage>();
    const approvedEntries: typeof preToolCalls = [];
    /**
     * Batch-level accumulator for `additionalContext` strings returned
     * by any PreToolUse / PostToolUse / PostToolUseFailure hook in this
     * dispatch. We emit one consolidated `HumanMessage` after all tool
     * results land so the next model turn sees the injected context
     * exactly once, ordered after the ToolMessages.
     */
    const batchAdditionalContexts: string[] = [];
    /**
     * Batch-level outcome record. Captures every tool call's final
     * result (success / error from the host, blocked from HITL reject,
     * substituted from HITL respond) so a single `PostToolBatch` hook
     * sees the whole set in batch order.
     */
    const postToolBatchEntries: PostToolBatchEntry[] = [];
    const HOOK_FALLBACK: AggregatedHookResult = Object.freeze({
      additionalContexts: [] as string[],
      errors: [] as string[],
    });

    if (this.hookRegistry?.hasHookFor('PreToolUse', runId) === true) {
      /**
       * Capture as a non-null local so the inner `blockEntry` closure
       * doesn't lose narrowing on `this.hookRegistry` and we don't have
       * to defensively `?.` it across every reference inside.
       */
      const hookRegistry = this.hookRegistry;
      const preResults = await Promise.all(
        preToolCalls.map((entry) =>
          executeHooks({
            registry: hookRegistry,
            input: {
              hook_event_name: 'PreToolUse',
              runId,
              threadId,
              agentId: this.agentId,
              toolName: entry.call.name,
              toolInput: entry.args,
              toolUseId: entry.call.id!,
              stepId: entry.stepId,
              turn: this.toolUsageCount.get(entry.call.name) ?? 0,
            },
            sessionId: runId,
            matchQuery: entry.call.name,
          }).catch((): AggregatedHookResult => HOOK_FALLBACK)
        )
      );

      type PendingEntry = (typeof preToolCalls)[number];

      const blockEntry = (entry: PendingEntry, reason: string): void => {
        const contentString = `Blocked: ${reason}`;
        messageByCallId.set(
          entry.call.id!,
          new ToolMessage({
            status: 'error',
            content: contentString,
            name: entry.call.name,
            tool_call_id: entry.call.id!,
          })
        );
        postToolBatchEntries.push({
          toolName: entry.call.name,
          toolInput: entry.args,
          toolUseId: entry.call.id!,
          stepId: entry.stepId,
          /**
           * Mirrors the `turn` value the executed-path push records, so
           * batch hooks see uniform entry shapes regardless of whether
           * the tool was blocked or executed. Read pre-increment, just
           * like the executed path captures it before the usage counter
           * advances.
           */
          turn: this.toolUsageCount.get(entry.call.name) ?? 0,
          status: 'error',
          error: contentString,
        });
        this.dispatchStepCompleted(
          entry.call.id!,
          entry.call.name,
          entry.args,
          contentString,
          config
        );
        if (hookRegistry.hasHookFor('PermissionDenied', runId)) {
          executeHooks({
            registry: hookRegistry,
            input: {
              hook_event_name: 'PermissionDenied',
              runId,
              threadId,
              agentId: this.agentId,
              toolName: entry.call.name,
              toolInput: entry.args,
              toolUseId: entry.call.id!,
              reason,
            },
            sessionId: runId,
            matchQuery: entry.call.name,
          }).catch(() => {
            /* PermissionDenied is observational — swallow errors */
          });
        }
      };

      /**
       * Apply a hook-supplied or host-supplied input override to a pending
       * entry, re-running the `{{tool<i>turn<n>}}` resolver so any new
       * placeholders introduced by the override are substituted (and any
       * formerly-unresolved refs cleared from the unresolved set).
       *
       * Mixed direct+event batches must use the pre-batch snapshot so a
       * hook-introduced placeholder cannot accidentally resolve to a
       * same-turn direct output that has just registered. Pure event
       * batches don't have a snapshot and resolve against the live
       * registry — safe because no event-side registrations have happened
       * yet.
       */
      const applyInputOverride = (
        entry: PendingEntry,
        nextArgs: Record<string, unknown>
      ): void => {
        if (registry != null) {
          const view: ToolOutputResolveView = preBatchSnapshot ?? {
            resolve: <T>(args: T) => registry.resolve(registryRunId, args),
          };
          const { resolved, unresolved } = view.resolve(nextArgs);
          entry.args = resolved as Record<string, unknown>;
          if (entry.call.id != null) {
            if (unresolved.length > 0) {
              unresolvedByCallId.set(entry.call.id, unresolved);
            } else {
              unresolvedByCallId.delete(entry.call.id);
            }
          }
          return;
        }
        entry.args = nextArgs;
      };

      const askEntries: Array<{
        entry: PendingEntry;
        reason?: string;
        allowedDecisions?: ReadonlyArray<
          'approve' | 'reject' | 'edit' | 'respond'
        >;
      }> = [];

      for (let i = 0; i < preToolCalls.length; i++) {
        const hookResult = preResults[i];
        const entry = preToolCalls[i];

        for (const ctx of hookResult.additionalContexts) {
          batchAdditionalContexts.push(ctx);
        }

        if (hookResult.decision === 'deny') {
          blockEntry(entry, hookResult.reason ?? 'Blocked by hook');
          continue;
        }

        if (hookResult.decision === 'ask') {
          /**
           * HITL is on by default — only the explicit opt-out
           * (`humanInTheLoop: { enabled: false }`) falls back to the
           * pre-HITL fail-closed path where `ask` collapses into a
           * blocked tool with an error `ToolMessage`. Otherwise the
           * entry queues for a single batched `interrupt()` call below.
           */
          if (this.humanInTheLoop?.enabled === false) {
            blockEntry(entry, hookResult.reason ?? 'Blocked by hook');
            continue;
          }
          askEntries.push({
            entry,
            reason: hookResult.reason,
            allowedDecisions: hookResult.allowedDecisions,
          });
          continue;
        }

        if (hookResult.updatedInput != null) {
          applyInputOverride(entry, hookResult.updatedInput);
        }
        approvedEntries.push(entry);
      }

      /**
       * If any entries asked for approval, raise a single LangGraph
       * `interrupt()` carrying every pending request together. The host
       * pauses, gathers human input, and resumes the run with one
       * decision per request. On resume LangGraph re-executes this node
       * from the start; `interrupt()` then returns the resume value
       * instead of throwing, so the loop above re-runs and the same
       * `askEntries` list is rebuilt deterministically (assuming hooks
       * are pure — see `humanInTheLoop` docs).
       */
      if (askEntries.length > 0) {
        const payload: t.ToolApprovalInterruptPayload = {
          type: 'tool_approval',
          action_requests: askEntries.map(({ entry, reason }) => {
            const request: t.ToolApprovalRequest = {
              tool_call_id: entry.call.id!,
              name: entry.call.name,
              arguments: entry.args,
            };
            if (reason != null) {
              request.description = reason;
            }
            return request;
          }),
          review_configs: askEntries.map(({ entry, allowedDecisions }) => ({
            action_name: entry.call.name,
            allowed_decisions: (allowedDecisions ?? [
              'approve',
              'reject',
              'edit',
              'respond',
            ]) as t.ToolApprovalDecisionType[],
          })),
        };

        /**
         * `interrupt()` reads the current `RunnableConfig` from
         * AsyncLocalStorage, but our `RunnableCallable` sets
         * `trace = false` for ToolNode (intentional — avoids LangSmith
         * tracing per tool call). Without the trace path, the upstream
         * `runWithConfig` frame is never established, so we re-anchor
         * here using the node's own `config` — Pregel hands us a
         * config that already carries every checkpoint/scratchpad key
         * `interrupt()` needs to suspend and resume.
         */
        const resumeValue = AsyncLocalStorageProviderSingleton.runWithConfig(
          config,
          () =>
            interrupt<
              t.ToolApprovalInterruptPayload,
              t.ToolApprovalDecision[] | t.ToolApprovalDecisionMap
            >(payload)
        );

        const decisionByCallId = normalizeApprovalDecisions(
          askEntries.map(({ entry }) => entry.call.id!),
          resumeValue
        );

        for (const { entry, reason: askReason } of askEntries) {
          const decision = decisionByCallId.get(entry.call.id!) ?? {
            type: 'reject' as const,
            reason: 'No decision provided for tool approval',
          };

          if (decision.type === 'reject') {
            blockEntry(
              entry,
              decision.reason ?? askReason ?? 'Rejected by user'
            );
            continue;
          }

          /**
           * `respond` short-circuits tool execution: the human supplies
           * the result the model should see in place of running the
           * tool. We emit a successful `ToolMessage` directly and skip
           * dispatch — no host event fires, no real tool side effect
           * occurs. Mirrors LangChain HITL middleware semantics.
           */
          if (decision.type === 'respond') {
            messageByCallId.set(
              entry.call.id!,
              new ToolMessage({
                status: 'success',
                content: decision.responseText,
                name: entry.call.name,
                tool_call_id: entry.call.id!,
              })
            );
            postToolBatchEntries.push({
              toolName: entry.call.name,
              toolInput: entry.args,
              toolUseId: entry.call.id!,
              stepId: entry.stepId,
              turn: this.toolUsageCount.get(entry.call.name) ?? 0,
              status: 'success',
              toolOutput: decision.responseText,
            });
            this.dispatchStepCompleted(
              entry.call.id!,
              entry.call.name,
              entry.args,
              decision.responseText,
              config
            );
            continue;
          }

          if (decision.type === 'edit') {
            applyInputOverride(entry, decision.updatedInput);
            approvedEntries.push(entry);
            continue;
          }

          /**
           * Defensive type widening: hosts deserialize resume payloads
           * from untyped JSON, so the `decision.type` value at runtime
           * is whatever string the wire sent — not necessarily one of
           * the four union variants TS knows about. We compare against
           * the literal `'approve'` through a widened view so a typo
           * or schema drift (`'aproved'`, `null`, `undefined`) hits the
           * fail-closed branch below instead of silently approving the
           * tool. Without this widening, TS narrows the union after the
           * three earlier branches and treats `=== 'approve'` as
           * trivially true.
           */
          const declaredType = (decision as { type?: unknown }).type;
          if (declaredType === 'approve') {
            approvedEntries.push(entry);
            continue;
          }

          /**
           * Unknown / missing decision type — fail closed. The whole
           * point of an approval gate is that "no decision" or
           * "garbled decision" deny by default.
           */
          const unknownType =
            typeof declaredType === 'string' ? declaredType : '<missing>';
          blockEntry(
            entry,
            `Unknown approval decision type "${unknownType}" — failing closed`
          );
        }
      }
    } else {
      approvedEntries.push(...preToolCalls);
    }

    const injected: BaseMessage[] = [];

    const batchIndexByCallId = new Map<string, number>();

    if (approvedEntries.length > 0) {
      const requests: t.ToolCallRequest[] = approvedEntries.map((entry) => {
        const turn = this.toolUsageCount.get(entry.call.name) ?? 0;
        this.toolUsageCount.set(entry.call.name, turn + 1);

        if (entry.batchIndex != null && entry.call.id != null) {
          batchIndexByCallId.set(entry.call.id, entry.batchIndex);
        }

        const request: t.ToolCallRequest = {
          id: entry.call.id!,
          name: entry.call.name,
          args: entry.args,
          stepId: entry.stepId,
          turn,
        };

        /**
         * Emit `codeSessionContext` for any tool whose host handler may need
         * to reach into the code-execution sandbox:
         *   - `CODE_EXECUTION_TOOLS` — direct executors that POST to /exec.
         *   - `SKILL_TOOL` — skill files live alongside code-env state.
         *   - `READ_FILE` — when the requested path is a code-env artifact
         *     (e.g. `/mnt/data/...`) the host falls back to reading via the
         *     same sandbox session; without the seeded `session_id` /
         *     `_injected_files` here, that fallback can't see prior-turn
         *     artifacts on the very first call of a turn.
         */
        if (
          CODE_EXECUTION_TOOLS.has(entry.call.name) ||
          entry.call.name === Constants.SKILL_TOOL ||
          entry.call.name === Constants.READ_FILE
        ) {
          request.codeSessionContext = this.getCodeSessionContext();
        }

        return request;
      });

      const requestMap = new Map(requests.map((r) => [r.id, r]));

      const results = await new Promise<t.ToolExecuteResult[]>(
        (resolve, reject) => {
          const batchRequest: t.ToolExecuteBatchRequest = {
            toolCalls: requests,
            userId: config.configurable?.user_id as string | undefined,
            agentId: this.agentId,
            configurable: config.configurable as
              | Record<string, unknown>
              | undefined,
            metadata: config.metadata as Record<string, unknown> | undefined,
            resolve,
            reject,
          };

          safeDispatchCustomEvent(
            GraphEvents.ON_TOOL_EXECUTE,
            batchRequest,
            config
          );
        }
      );

      this.storeCodeSessionFromResults(results, requestMap);

      const hasPostHook =
        this.hookRegistry?.hasHookFor('PostToolUse', runId) === true;
      const hasFailureHook =
        this.hookRegistry?.hasHookFor('PostToolUseFailure', runId) === true;

      for (const result of results) {
        if (result.injectedMessages && result.injectedMessages.length > 0) {
          try {
            injected.push(
              ...this.convertInjectedMessages(result.injectedMessages)
            );
          } catch (e) {
            // eslint-disable-next-line no-console
            console.warn(
              `[ToolNode] Failed to convert injectedMessages for toolCallId=${result.toolCallId}:`,
              e instanceof Error ? e.message : e
            );
          }
        }
        const request = requestMap.get(result.toolCallId);
        const toolName = request?.name ?? 'unknown';

        let contentString: string;
        let toolMessage: ToolMessage;
        /**
         * Tracks the post-PostToolUse-hook output so the
         * `PostToolBatch` entry below sees the final transformed value
         * even when a hook replaced the original via `updatedOutput`.
         * Lives at the loop-iteration scope so the success branch can
         * mutate it; the error branch leaves it unset (and the batch
         * entry uses `error` instead of `toolOutput` in that case).
         */
        let finalToolOutput: unknown = result.content;

        if (result.status === 'error') {
          contentString = `Error: ${result.errorMessage ?? 'Unknown error'}\n Please fix your mistakes.`;
          /**
           * Error results bypass registration but stamp the
           * unresolved-refs hint into `additional_kwargs` so the lazy
           * annotation transform surfaces it to the LLM at request
           * time, letting the model self-correct when its reference
           * key caused the failure. Persisted `content` stays clean.
           */
          const unresolved = unresolvedByCallId.get(result.toolCallId) ?? [];
          const errorRefMeta =
            unresolved.length > 0
              ? this.recordOutputReference(
                registryRunId,
                contentString,
                undefined,
                unresolved
              )
              : undefined;
          toolMessage = new ToolMessage({
            status: 'error',
            content: contentString,
            name: toolName,
            tool_call_id: result.toolCallId,
            ...(errorRefMeta != null && {
              additional_kwargs: errorRefMeta as Record<string, unknown>,
            }),
          });

          if (hasFailureHook) {
            const failureHookResult = await executeHooks({
              registry: this.hookRegistry!,
              input: {
                hook_event_name: 'PostToolUseFailure',
                runId,
                threadId,
                agentId: this.agentId,
                toolName,
                toolInput: request?.args ?? {},
                toolUseId: result.toolCallId,
                error: result.errorMessage ?? 'Unknown error',
                stepId: request?.stepId,
                turn: request?.turn,
              },
              sessionId: runId,
              matchQuery: toolName,
            }).catch((): undefined => undefined);
            /**
             * Collect `additionalContext` from failure hooks too. Without
             * this, recovery guidance returned on tool errors (e.g.
             * "if this tool errors with X, suggest Y to the user") is
             * silently dropped even though the API surface advertises
             * `additionalContext` for this event. PostToolUseFailure
             * remains observational for errors thrown by the hook
             * itself, but a successfully-returned result is honored.
             */
            if (failureHookResult != null) {
              for (const ctx of failureHookResult.additionalContexts) {
                batchAdditionalContexts.push(ctx);
              }
            }
          }
        } else {
          let registryRaw =
            typeof result.content === 'string'
              ? result.content
              : JSON.stringify(result.content);
          contentString = truncateToolResultContent(
            registryRaw,
            this.maxToolResultChars
          );

          if (hasPostHook) {
            const hookResult = await executeHooks({
              registry: this.hookRegistry!,
              input: {
                hook_event_name: 'PostToolUse',
                runId,
                threadId,
                agentId: this.agentId,
                toolName,
                toolInput: request?.args ?? {},
                toolOutput: result.content,
                toolUseId: result.toolCallId,
                stepId: request?.stepId,
                turn: request?.turn,
              },
              sessionId: runId,
              matchQuery: toolName,
            }).catch((): undefined => undefined);
            if (hookResult != null) {
              for (const ctx of hookResult.additionalContexts) {
                batchAdditionalContexts.push(ctx);
              }
            }
            if (hookResult?.updatedOutput != null) {
              const replaced =
                typeof hookResult.updatedOutput === 'string'
                  ? hookResult.updatedOutput
                  : JSON.stringify(hookResult.updatedOutput);
              registryRaw = replaced;
              contentString = truncateToolResultContent(
                replaced,
                this.maxToolResultChars
              );
              finalToolOutput = hookResult.updatedOutput;
            }
          }

          const batchIndex = batchIndexByCallId.get(result.toolCallId);
          const unresolved = unresolvedByCallId.get(result.toolCallId) ?? [];
          const refKey =
            this.toolOutputRegistry != null &&
            batchIndex != null &&
            turn != null
              ? buildReferenceKey(batchIndex, turn)
              : undefined;
          const successRefMeta = this.recordOutputReference(
            registryRunId,
            registryRaw,
            refKey,
            unresolved
          );

          toolMessage = new ToolMessage({
            status: 'success',
            name: toolName,
            content: contentString,
            artifact: result.artifact,
            tool_call_id: result.toolCallId,
            ...(successRefMeta != null && {
              additional_kwargs: successRefMeta as Record<string, unknown>,
            }),
          });
        }

        this.dispatchStepCompleted(
          result.toolCallId,
          toolName,
          request?.args ?? {},
          contentString,
          config,
          request?.turn
        );

        postToolBatchEntries.push({
          toolName,
          toolInput: request?.args ?? {},
          toolUseId: result.toolCallId,
          stepId: request?.stepId,
          turn: request?.turn,
          status: result.status === 'error' ? 'error' : 'success',
          ...(result.status === 'error'
            ? { error: result.errorMessage ?? 'Unknown error' }
            : { toolOutput: finalToolOutput }),
        });

        messageByCallId.set(result.toolCallId, toolMessage);
      }
    }

    const toolMessages = toolCalls
      .map((call) => messageByCallId.get(call.id!))
      .filter((m): m is ToolMessage => m != null);

    /**
     * Fire `PostToolBatch` after every per-tool hook has settled. We
     * collect its `additionalContexts` into the same batch accumulator
     * so a host that wants to inject a single batch-level convention
     * can do so via the same mechanism per-tool hooks already use.
     */
    if (
      this.hookRegistry?.hasHookFor('PostToolBatch', runId) === true &&
      postToolBatchEntries.length > 0
    ) {
      const batchHookResult = await executeHooks({
        registry: this.hookRegistry,
        input: {
          hook_event_name: 'PostToolBatch',
          runId,
          threadId,
          agentId: this.agentId,
          entries: postToolBatchEntries,
        },
        sessionId: runId,
      }).catch((): undefined => undefined);
      if (batchHookResult != null) {
        for (const ctx of batchHookResult.additionalContexts) {
          batchAdditionalContexts.push(ctx);
        }
      }
    }

    if (batchAdditionalContexts.length > 0) {
      /**
       * `HumanMessage` carrying a metadata `role: 'system'` marker —
       * see `convertInjectedMessages` for the wider rationale. Anthropic
       * and Google reject mid-conversation `SystemMessage`s, so we use
       * a user-role message and surface the system intent through
       * `additional_kwargs` for hosts inspecting state. The model sees
       * a user message; `role` is metadata only.
       */
      injected.push(
        new HumanMessage({
          content: batchAdditionalContexts.join('\n\n'),
          additional_kwargs: { role: 'system', source: 'hook' },
        })
      );
    }

    return { toolMessages, injected };
  }

  private dispatchStepCompleted(
    toolCallId: string,
    toolName: string,
    args: Record<string, unknown>,
    output: string,
    config: RunnableConfig,
    turn?: number
  ): void {
    const stepId = this.toolCallStepIds?.get(toolCallId) ?? '';
    if (!stepId) {
      // eslint-disable-next-line no-console
      console.warn(
        `[ToolNode] toolCallStepIds missing entry for toolCallId=${toolCallId} (tool=${toolName}). ` +
          'This indicates a race between the stream consumer and graph execution. ' +
          `Map size: ${this.toolCallStepIds?.size ?? 0}`
      );
    }

    safeDispatchCustomEvent(
      GraphEvents.ON_RUN_STEP_COMPLETED,
      {
        result: {
          id: stepId,
          index: turn ?? this.toolUsageCount.get(toolName) ?? 0,
          type: 'tool_call' as const,
          tool_call: {
            args: JSON.stringify(args),
            name: toolName,
            id: toolCallId,
            output,
            progress: 1,
          } as t.ProcessedToolCall,
        },
      },
      config
    );
  }

  /**
   * Converts InjectedMessage instances to LangChain HumanMessage objects.
   * Both 'user' and 'system' roles become HumanMessage to avoid provider
   * rejections (Anthropic/Google reject non-leading SystemMessages).
   * The original role is preserved in additional_kwargs for downstream consumers.
   */
  private convertInjectedMessages(
    messages: t.InjectedMessage[]
  ): BaseMessage[] {
    const converted: BaseMessage[] = [];
    for (const msg of messages) {
      const additional_kwargs: Record<string, unknown> = {
        role: msg.role,
      };
      if (msg.isMeta != null) additional_kwargs.isMeta = msg.isMeta;
      if (msg.source != null) additional_kwargs.source = msg.source;
      if (msg.skillName != null) additional_kwargs.skillName = msg.skillName;

      converted.push(
        new HumanMessage({
          content: toLangChainContent(msg.content),
          additional_kwargs,
        })
      );
    }
    return converted;
  }

  /**
   * Execute all tool calls via ON_TOOL_EXECUTE event dispatch.
   * Injected messages are placed AFTER ToolMessages to respect provider
   * message ordering (AIMessage tool_calls must be immediately followed
   * by their ToolMessage results).
   *
   * `batchIndices` mirrors `toolCalls` and carries each call's position
   * within the parent batch. `turn` is the per-`run()` batch index
   * captured locally by the caller. Both are threaded so concurrent
   * invocations cannot race on shared mutable state.
   */
  private async executeViaEvent(
    toolCalls: ToolCall[],
    config: RunnableConfig,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: any,
    batchContext: DispatchBatchContext = {}
  ): Promise<T> {
    const { toolMessages, injected } = await this.dispatchToolEvents(
      toolCalls,
      config,
      batchContext
    );
    const outputs: BaseMessage[] = [...toolMessages, ...injected];
    return (Array.isArray(input) ? outputs : { messages: outputs }) as T;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  protected async run(input: any, config: RunnableConfig): Promise<T> {
    this.toolCallTurns.clear();
    /**
     * Per-batch local map for resolved (post-substitution) args.
     * Lives on the stack so concurrent `run()` calls on the same
     * ToolNode cannot read or wipe each other's entries.
     */
    const resolvedArgsByCallId = new Map<string, Record<string, unknown>>();
    /**
     * Claim this batch's turn synchronously from the registry (or
     * fall back to 0 when the feature is disabled). The registry is
     * partitioned by scope id so overlapping batches cannot
     * overwrite each other's state even under a shared registry.
     *
     * For anonymous callers (no `run_id` in config), mint a unique
     * per-batch scope id so two concurrent anonymous invocations
     * don't target the same bucket. The scope is threaded down to
     * every subsequent registry call on this batch.
     */
    const incomingRunId = config.configurable?.run_id as string | undefined;
    const batchScopeId = incomingRunId ?? `\0anon-${this.anonBatchCounter++}`;
    const turn = this.toolOutputRegistry?.nextTurn(batchScopeId) ?? 0;
    let outputs: (BaseMessage | Command)[];

    if (this.isSendInput(input)) {
      const isDirectTool = this.directToolNames?.has(input.lg_tool_call.name);
      if (this.eventDrivenMode && isDirectTool !== true) {
        return this.executeViaEvent([input.lg_tool_call], config, input, {
          batchIndices: [0],
          turn,
          batchScopeId,
        });
      }
      outputs = [
        await this.runTool(input.lg_tool_call, config, {
          batchIndex: 0,
          turn,
          batchScopeId,
          resolvedArgsByCallId,
        }),
      ];
      this.handleRunToolCompletions(
        [input.lg_tool_call],
        outputs,
        config,
        resolvedArgsByCallId
      );
    } else {
      let messages: BaseMessage[];
      if (Array.isArray(input)) {
        messages = input;
      } else if (this.isMessagesState(input)) {
        messages = input.messages;
      } else {
        throw new Error(
          'ToolNode only accepts BaseMessage[] or { messages: BaseMessage[] } as input.'
        );
      }

      const toolMessageIds: Set<string> = new Set(
        messages
          .filter((msg) => msg._getType() === 'tool')
          .map((msg) => (msg as ToolMessage).tool_call_id)
      );

      let aiMessage: AIMessage | undefined;
      for (let i = messages.length - 1; i >= 0; i--) {
        const message = messages[i];
        if (isAIMessage(message)) {
          aiMessage = message;
          break;
        }
      }

      if (aiMessage == null || !isAIMessage(aiMessage)) {
        throw new Error('ToolNode only accepts AIMessages as input.');
      }

      if (this.loadRuntimeTools) {
        const { tools, toolMap } = this.loadRuntimeTools(
          aiMessage.tool_calls ?? []
        );
        this.toolMap =
          toolMap ?? new Map(tools.map((tool) => [tool.name, tool]));
        this.programmaticCache = undefined; // Invalidate cache on toolMap change
      }

      const filteredCalls =
        aiMessage.tool_calls?.filter((call) => {
          /**
           * Filter out:
           * 1. Already processed tool calls (present in toolMessageIds)
           * 2. Server tool calls (e.g., web_search with IDs starting with 'srvtoolu_')
           *    which are executed by the provider's API and don't require invocation
           */
          return (
            (call.id == null || !toolMessageIds.has(call.id)) &&
            !(
              call.id?.startsWith(Constants.ANTHROPIC_SERVER_TOOL_PREFIX) ??
              false
            )
          );
        }) ?? [];

      if (this.eventDrivenMode && filteredCalls.length > 0) {
        const filteredIndices = filteredCalls.map((_, idx) => idx);

        if (!this.directToolNames || this.directToolNames.size === 0) {
          return this.executeViaEvent(filteredCalls, config, input, {
            batchIndices: filteredIndices,
            turn,
            batchScopeId,
          });
        }

        const directEntries: Array<{ call: ToolCall; batchIndex: number }> = [];
        const eventEntries: Array<{ call: ToolCall; batchIndex: number }> = [];
        for (let i = 0; i < filteredCalls.length; i++) {
          const call = filteredCalls[i];
          const entry = { call, batchIndex: i };
          if (this.directToolNames!.has(call.name)) {
            directEntries.push(entry);
          } else {
            eventEntries.push(entry);
          }
        }

        const directCalls = directEntries.map((e) => e.call);
        const directIndices = directEntries.map((e) => e.batchIndex);
        const eventCalls = eventEntries.map((e) => e.call);
        const eventIndices = eventEntries.map((e) => e.batchIndex);

        /**
         * Snapshot the event calls' args against the *pre-batch*
         * registry state synchronously, before any await runs. The
         * directs are then awaited first (preserving fail-fast
         * semantics — a thrown error in a direct tool, e.g. with
         * `handleToolErrors=false` or a `GraphInterrupt`, aborts
         * before we dispatch any event-driven tools to the host).
         * Because the event args were captured pre-await, they stay
         * isolated from same-turn direct outputs that register
         * during the await.
         */
        const preResolvedEventArgs = new Map<
          string,
          { resolved: Record<string, unknown>; unresolved: string[] }
        >();
        /**
         * Take a frozen snapshot of the registry state before any
         * direct registrations land. The snapshot resolves
         * placeholders against this point-in-time view, so a
         * `PreToolUse` hook later rewriting event args via
         * `updatedInput` can introduce placeholders that resolve
         * cross-batch (against prior runs) without ever picking up
         * same-turn direct outputs.
         */
        const preBatchSnapshot =
          this.toolOutputRegistry?.snapshot(batchScopeId);
        if (preBatchSnapshot != null) {
          for (const entry of eventEntries) {
            if (entry.call.id != null) {
              const { resolved, unresolved } = preBatchSnapshot.resolve(
                entry.call.args as Record<string, unknown>
              );
              preResolvedEventArgs.set(entry.call.id, {
                resolved: resolved as Record<string, unknown>,
                unresolved,
              });
            }
          }
        }

        const directOutputs: (BaseMessage | Command)[] =
          directCalls.length > 0
            ? await Promise.all(
              directCalls.map((call, i) =>
                this.runTool(call, config, {
                  batchIndex: directIndices[i],
                  turn,
                  batchScopeId,
                  resolvedArgsByCallId,
                })
              )
            )
            : [];

        if (directCalls.length > 0 && directOutputs.length > 0) {
          this.handleRunToolCompletions(
            directCalls,
            directOutputs,
            config,
            resolvedArgsByCallId
          );
        }

        const eventResult =
          eventCalls.length > 0
            ? await this.dispatchToolEvents(eventCalls, config, {
              batchIndices: eventIndices,
              turn,
              batchScopeId,
              preResolvedArgs: preResolvedEventArgs,
              preBatchSnapshot,
            })
            : {
              toolMessages: [] as ToolMessage[],
              injected: [] as BaseMessage[],
            };

        outputs = [
          ...directOutputs,
          ...eventResult.toolMessages,
          ...eventResult.injected,
        ];
      } else {
        outputs = await Promise.all(
          filteredCalls.map((call, i) =>
            this.runTool(call, config, {
              batchIndex: i,
              turn,
              batchScopeId,
              resolvedArgsByCallId,
            })
          )
        );
        this.handleRunToolCompletions(
          filteredCalls,
          outputs,
          config,
          resolvedArgsByCallId
        );
      }
    }

    if (!outputs.some(isCommand)) {
      return (Array.isArray(input) ? outputs : { messages: outputs }) as T;
    }

    const combinedOutputs: (
      | { messages: BaseMessage[] }
      | BaseMessage[]
      | Command
    )[] = [];
    let parentCommand: Command | null = null;

    /**
     * Collect handoff commands (Commands with string goto and Command.PARENT)
     * for potential parallel handoff aggregation
     */
    const handoffCommands: Command[] = [];
    const nonCommandOutputs: BaseMessage[] = [];

    for (const output of outputs) {
      if (isCommand(output)) {
        if (
          output.graph === Command.PARENT &&
          Array.isArray(output.goto) &&
          output.goto.every((send): send is Send => isSend(send))
        ) {
          /** Aggregate Send-based commands */
          if (parentCommand) {
            (parentCommand.goto as Send[]).push(...(output.goto as Send[]));
          } else {
            parentCommand = new Command({
              graph: Command.PARENT,
              goto: output.goto,
            });
          }
        } else if (output.graph === Command.PARENT) {
          /**
           * Handoff Command with destination.
           * Handle both string ('agent') and array (['agent']) formats.
           * Collect for potential parallel aggregation.
           */
          const goto = output.goto;
          const isSingleStringDest = typeof goto === 'string';
          const isSingleArrayDest =
            Array.isArray(goto) &&
            goto.length === 1 &&
            typeof goto[0] === 'string';

          if (isSingleStringDest || isSingleArrayDest) {
            handoffCommands.push(output);
          } else {
            /** Multi-destination or other command - pass through */
            combinedOutputs.push(output);
          }
        } else {
          /** Other commands - pass through */
          combinedOutputs.push(output);
        }
      } else {
        nonCommandOutputs.push(output);
        combinedOutputs.push(
          Array.isArray(input) ? [output] : { messages: [output] }
        );
      }
    }

    /**
     * Handle handoff commands - convert to Send objects for parallel execution
     * when multiple handoffs are requested
     */
    if (handoffCommands.length > 1) {
      /**
       * Multiple parallel handoffs - convert to Send objects.
       * Each Send carries its own state with the appropriate messages.
       * This enables LLM-initiated parallel execution when calling multiple
       * transfer tools simultaneously.
       */

      /** Collect all destinations for sibling tracking */
      const allDestinations = handoffCommands.map((cmd) => {
        const goto = cmd.goto;
        return typeof goto === 'string' ? goto : (goto as string[])[0];
      });

      const sends = handoffCommands.map((cmd, idx) => {
        const destination = allDestinations[idx];
        /** Get siblings (other destinations, not this one) */
        const siblings = allDestinations.filter((d) => d !== destination);

        /** Add siblings to ToolMessage additional_kwargs */
        const update = cmd.update as { messages?: BaseMessage[] } | undefined;
        if (update && update.messages) {
          for (const msg of update.messages) {
            if (msg.getType() === 'tool') {
              (msg as ToolMessage).additional_kwargs.handoff_parallel_siblings =
                siblings;
            }
          }
        }

        return new Send(destination, cmd.update);
      });

      const parallelCommand = new Command({
        graph: Command.PARENT,
        goto: sends,
      });
      combinedOutputs.push(parallelCommand);
    } else if (handoffCommands.length === 1) {
      /** Single handoff - pass through as-is */
      combinedOutputs.push(handoffCommands[0]);
    }

    if (parentCommand) {
      combinedOutputs.push(parentCommand);
    }

    return combinedOutputs as T;
  }

  private isSendInput(input: unknown): input is { lg_tool_call: ToolCall } {
    return (
      typeof input === 'object' && input != null && 'lg_tool_call' in input
    );
  }

  private isMessagesState(
    input: unknown
  ): input is { messages: BaseMessage[] } {
    return (
      typeof input === 'object' &&
      input != null &&
      'messages' in input &&
      Array.isArray((input as { messages: unknown }).messages) &&
      (input as { messages: unknown[] }).messages.every(isBaseMessage)
    );
  }
}

function areToolCallsInvoked(
  message: AIMessage,
  invokedToolIds?: Set<string>
): boolean {
  if (!invokedToolIds || invokedToolIds.size === 0) return false;
  return (
    message.tool_calls?.every(
      (toolCall) => toolCall.id != null && invokedToolIds.has(toolCall.id)
    ) ?? false
  );
}

export function toolsCondition<T extends string>(
  state: BaseMessage[] | typeof MessagesAnnotation.State,
  toolNode: T,
  invokedToolIds?: Set<string>
): T | typeof END {
  const messages = Array.isArray(state) ? state : state.messages;
  const message = messages[messages.length - 1] as AIMessage | undefined;

  if (
    message &&
    'tool_calls' in message &&
    (message.tool_calls?.length ?? 0) > 0 &&
    !areToolCallsInvoked(message, invokedToolIds)
  ) {
    return toolNode;
  }
  return END;
}
