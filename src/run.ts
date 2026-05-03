// src/run.ts
import './instrumentation';
import { CallbackHandler } from '@langfuse/langchain';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda } from '@langchain/core/runnables';
import { AzureChatOpenAI, ChatOpenAI } from '@langchain/openai';
import {
  Command,
  INTERRUPT,
  MemorySaver,
  isInterrupted,
} from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type {
  MessageContentComplex,
  BaseMessage,
} from '@langchain/core/messages';
import type { StringPromptValue } from '@langchain/core/prompt_values';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import {
  createCompletionTitleRunnable,
  createTitleRunnable,
} from '@/utils/title';
import { createTokenCounter, encodingForModel } from '@/utils/tokens';
import { GraphEvents, Callback, TitleMethod } from '@/common';
import { MultiAgentGraph } from '@/graphs/MultiAgentGraph';
import { StandardGraph } from '@/graphs/Graph';
import { initializeModel } from '@/llm/init';
import { HandlerRegistry } from '@/events';
import { executeHooks } from '@/hooks';
import { isOpenAILike } from '@/utils/llm';
import { isPresent } from '@/utils/misc';
import type { HookRegistry } from '@/hooks';

export const defaultOmitOptions = new Set([
  'stream',
  'thinking',
  'streaming',
  'maxTokens',
  'clientOptions',
  'thinkingConfig',
  'thinkingBudget',
  'includeThoughts',
  'maxOutputTokens',
  'additionalModelRequestFields',
]);

export class Run<_T extends t.BaseGraphState> {
  id: string;
  private tokenCounter?: t.TokenCounter;
  private handlerRegistry?: HandlerRegistry;
  private hookRegistry?: HookRegistry;
  private humanInTheLoop?: t.HumanInTheLoopConfig;
  private toolOutputReferences?: t.ToolOutputReferencesConfig;
  private indexTokenCountMap?: Record<string, number>;
  calibrationRatio: number = 1;
  graphRunnable?: t.CompiledStateWorkflow;
  Graph: StandardGraph | MultiAgentGraph | undefined;
  returnContent: boolean = false;
  private skipCleanup: boolean = false;
  private _streamResult: t.MessageContentComplex[] | undefined;
  /**
   * Captured interrupt payload typed as `unknown` because the SDK
   * does not validate the runtime shape — custom graph nodes can
   * raise interrupts with arbitrary payloads (not just the SDK's
   * `HumanInterruptPayload` union). The public `getInterrupt<T>()`
   * lets callers assert the type they expect.
   */
  private _interrupt: t.RunInterruptResult<unknown> | undefined;
  private _haltedReason: string | undefined;

  private constructor(config: Partial<t.RunConfig>) {
    const runId = config.runId ?? '';
    if (!runId) {
      throw new Error('Run ID not provided');
    }

    this.id = runId;
    this.tokenCounter = config.tokenCounter;
    this.indexTokenCountMap = config.indexTokenCountMap;
    if (config.calibrationRatio != null && config.calibrationRatio > 0) {
      this.calibrationRatio = config.calibrationRatio;
    }

    const handlerRegistry = new HandlerRegistry();

    if (config.customHandlers) {
      for (const [eventType, handler] of Object.entries(
        config.customHandlers
      )) {
        handlerRegistry.register(eventType, handler);
      }
    }

    this.handlerRegistry = handlerRegistry;
    this.hookRegistry = config.hooks;
    this.humanInTheLoop = config.humanInTheLoop;
    this.toolOutputReferences = config.toolOutputReferences;

    if (!config.graphConfig) {
      throw new Error('Graph config not provided');
    }

    /** Handle different graph types */
    if (config.graphConfig.type === 'multi-agent') {
      this.graphRunnable = this.createMultiAgentGraph(config.graphConfig);
      if (this.Graph) {
        this.Graph.handlerRegistry = handlerRegistry;
      }
    } else {
      /** Default to legacy graph for 'standard' or undefined type */
      this.graphRunnable = this.createLegacyGraph(config.graphConfig);
      if (this.Graph) {
        this.Graph.compileOptions =
          config.graphConfig.compileOptions ?? this.Graph.compileOptions;
        this.Graph.handlerRegistry = handlerRegistry;
      }
    }

    if (config.initialSessions && this.Graph) {
      for (const [key, value] of config.initialSessions) {
        this.Graph.sessions.set(key, value);
      }
    }

    this.returnContent = config.returnContent ?? false;
    this.skipCleanup = config.skipCleanup ?? false;
  }

  private createLegacyGraph(
    config: t.LegacyGraphConfig | t.StandardGraphConfig
  ): t.CompiledStateWorkflow {
    let agentConfig: t.AgentInputs;
    let signal: AbortSignal | undefined;

    /** Check if this is a multi-agent style config (has agents array) */
    if ('agents' in config && Array.isArray(config.agents)) {
      if (config.agents.length === 0) {
        throw new Error('At least one agent must be provided');
      }
      agentConfig = config.agents[0];
      signal = config.signal;
    } else {
      /** Legacy path: build agent config from llmConfig */
      const {
        type: _type,
        llmConfig,
        signal: legacySignal,
        tools = [],
        ...agentInputs
      } = config as t.LegacyGraphConfig;
      const { provider, ...clientOptions } = llmConfig;

      agentConfig = {
        ...agentInputs,
        tools,
        provider,
        clientOptions,
        agentId: 'default',
      };
      signal = legacySignal;
    }

    const standardGraph = new StandardGraph({
      signal,
      runId: this.id,
      agents: [agentConfig],
      tokenCounter: this.tokenCounter,
      indexTokenCountMap: this.indexTokenCountMap,
      calibrationRatio: this.calibrationRatio,
    });
    /** Propagate compile options from graph config */
    standardGraph.compileOptions = this.applyHITLCheckpointerFallback(
      config.compileOptions
    );
    standardGraph.hookRegistry = this.hookRegistry;
    standardGraph.humanInTheLoop = this.humanInTheLoop;
    standardGraph.toolOutputReferences = this.toolOutputReferences;
    this.Graph = standardGraph;
    return standardGraph.createWorkflow();
  }

  private createMultiAgentGraph(
    config: t.MultiAgentGraphConfig
  ): t.CompiledStateWorkflow {
    const { agents, edges, compileOptions } = config;

    const multiAgentGraph = new MultiAgentGraph({
      runId: this.id,
      agents,
      edges,
      tokenCounter: this.tokenCounter,
      indexTokenCountMap: this.indexTokenCountMap,
      calibrationRatio: this.calibrationRatio,
    });

    multiAgentGraph.compileOptions =
      this.applyHITLCheckpointerFallback(compileOptions);

    multiAgentGraph.hookRegistry = this.hookRegistry;
    multiAgentGraph.humanInTheLoop = this.humanInTheLoop;
    multiAgentGraph.toolOutputReferences = this.toolOutputReferences;
    this.Graph = multiAgentGraph;
    return multiAgentGraph.createWorkflow();
  }

  /**
   * When HITL is enabled (the default) and the host did not supply a
   * checkpointer, install an in-memory `MemorySaver` so `interrupt()`
   * can persist checkpoints and `Command({ resume })` can rebuild state.
   * The fallback is intentionally process-local: production hosts that
   * need durable resumption across processes / restarts must provide
   * their own checkpointer (Redis, Postgres, etc.) on
   * `compileOptions.checkpointer`.
   *
   * No-op when the host opted out via `humanInTheLoop: { enabled: false }`
   * or already supplied a checkpointer of their own.
   */
  private applyHITLCheckpointerFallback(
    compileOptions: t.CompileOptions | undefined
  ): t.CompileOptions | undefined {
    if (this.humanInTheLoop?.enabled === false) {
      return compileOptions;
    }
    if (compileOptions?.checkpointer != null) {
      return compileOptions;
    }
    return {
      ...(compileOptions ?? {}),
      checkpointer: new MemorySaver(),
    };
  }

  /**
   * Run RunStart + UserPromptSubmit hooks before the graph stream
   * begins, accumulate any `additionalContext` strings into the input
   * messages, and short-circuit when a hook signals the run should not
   * proceed (deny / ask decision on the prompt, or `preventContinuation`
   * on either hook).
   *
   * Returns `true` when the caller should bail with `undefined` (run
   * was halted before any model call); returns `false` to proceed
   * into the stream loop.
   *
   * ## Side effects
   *
   * On the success path:
   *   - Mutates `stateInputs.messages` in place to append a
   *     consolidated `HumanMessage` carrying any hook
   *     `additionalContext` strings. Safe because the host owns the
   *     array and `processStream` is the only consumer until LangGraph
   *     reads it.
   *
   * On the halt path (returning `true`):
   *   - Sets `this._haltedReason` so callers (and the eventual host)
   *     can distinguish a hook-driven halt from a natural completion.
   *   - Calls `registry.clearSession(this.id)` and
   *     `registry.clearHaltSignal(this.id)` because no resume is
   *     expected from a pre-stream halt — the run never entered the
   *     graph, so the session/halt state for this run would otherwise
   *     leak to the next `processStream` invocation on the same
   *     registry. Other concurrent runs on the same registry are
   *     untouched (halt signals are scoped per session id).
   *   - Sets `config.callbacks = undefined` to drop the callback
   *     references the caller built (langfuse handler, custom event
   *     handler, etc.) since they won't be exercised. Mirrors the
   *     equivalent cleanup the `processStream` `finally` block does
   *     on the natural-completion path.
   */
  private async runPreStreamHooks(
    stateInputs: t.IState,
    threadId: string | undefined,
    config: Partial<RunnableConfig>
  ): Promise<boolean> {
    const registry = this.hookRegistry;
    /**
     * Defensive guard: `processStream` already validated `this.Graph`
     * before calling this helper, but TypeScript can't propagate that
     * narrowing across method boundaries. The check keeps the body
     * free of `this.Graph!` non-null assertions.
     */
    if (registry == null || this.Graph == null) {
      return false;
    }

    const preStreamContexts: string[] = [];

    const runStartResult = await executeHooks({
      registry,
      input: {
        hook_event_name: 'RunStart',
        runId: this.id,
        threadId,
        agentId: this.Graph.defaultAgentId,
        messages: stateInputs.messages,
      },
      sessionId: this.id,
    });
    for (const ctx of runStartResult.additionalContexts) {
      preStreamContexts.push(ctx);
    }
    /**
     * Honor `preventContinuation` from RunStart before the stream
     * starts. Mid-flight halts (from tool/compact/subagent hooks)
     * route through `HookRegistry.haltRun` and are polled by the
     * stream loop in `processStream` — different mechanism, same
     * intent.
     */
    if (runStartResult.preventContinuation === true) {
      this._haltedReason = runStartResult.stopReason ?? 'preventContinuation';
      registry.clearSession(this.id);
      registry.clearHaltSignal(this.id);
      config.callbacks = undefined;
      return true;
    }

    const lastHuman = findLastMessageOfType(stateInputs.messages, 'human');
    if (lastHuman != null) {
      const promptResult = await executeHooks({
        registry,
        input: {
          hook_event_name: 'UserPromptSubmit',
          runId: this.id,
          threadId,
          agentId: this.Graph.defaultAgentId,
          prompt: extractPromptText(lastHuman),
          // attachments: not yet wired — Phase 2 will extract
          // non-text content blocks (images, files) from messages
        },
        sessionId: this.id,
      });
      if (
        promptResult.decision === 'deny' ||
        promptResult.decision === 'ask' ||
        promptResult.preventContinuation === true
      ) {
        if (promptResult.preventContinuation === true) {
          this._haltedReason = promptResult.stopReason ?? 'preventContinuation';
        }
        registry.clearSession(this.id);
        registry.clearHaltSignal(this.id);
        config.callbacks = undefined;
        return true;
      }
      for (const ctx of promptResult.additionalContexts) {
        preStreamContexts.push(ctx);
      }
    }

    if (preStreamContexts.length > 0) {
      /**
       * Wraps the joined hook contexts as a `HumanMessage` even though
       * the intent is system-level guidance. Using a `SystemMessage`
       * mid-conversation is rejected by Anthropic and Google providers
       * (system messages must be the leading entry), so the LangChain
       * convention — also used by `ToolNode.convertInjectedMessages`
       * — is `HumanMessage` carrying `additional_kwargs.role` as a
       * marker for hosts inspecting state. The model still sees a
       * user-role message; the `role: 'system'` field is metadata
       * only. Hosts that want a true system message should compose
       * it into the agent's `instructions` config instead.
       */
      stateInputs.messages.push(
        new HumanMessage({
          content: preStreamContexts.join('\n\n'),
          additional_kwargs: { role: 'system', source: 'hook' },
        })
      );
    }

    return false;
  }

  static async create<T extends t.BaseGraphState>(
    config: t.RunConfig
  ): Promise<Run<T>> {
    /** Create tokenCounter if indexTokenCountMap is provided but tokenCounter is not */
    if (config.indexTokenCountMap && !config.tokenCounter) {
      const gc = config.graphConfig;
      const clientOpts =
        'agents' in gc ? gc.agents[0]?.clientOptions : gc.clientOptions;
      const model = (clientOpts as { model?: string } | undefined)?.model ?? '';
      config.tokenCounter = await createTokenCounter(encodingForModel(model));
    }
    return new Run<T>(config);
  }

  getRunMessages(): BaseMessage[] | undefined {
    if (!this.Graph) {
      throw new Error(
        'Graph not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    return this.Graph.getRunMessages();
  }

  /**
   * Returns the current calibration ratio (EMA of provider-vs-estimate token ratios).
   * Hosts should persist this value and pass it back as `RunConfig.calibrationRatio`
   * on the next run for the same conversation so the pruner starts with an accurate
   * scaling factor instead of the default (1).
   */
  getCalibrationRatio(): number {
    return this.calibrationRatio;
  }

  getResolvedInstructionOverhead(): number | undefined {
    return this.Graph?.getResolvedInstructionOverhead();
  }

  getToolCount(): number {
    return this.Graph?.getToolCount() ?? 0;
  }

  /**
   * Creates a custom event callback handler that intercepts custom events
   * and processes them through our handler registry instead of EventStreamCallbackHandler
   */
  private createCustomEventCallback() {
    return async (
      eventName: string,
      data: unknown,
      runId: string,
      tags?: string[],
      metadata?: Record<string, unknown>
    ): Promise<void> => {
      // ON_RUN_STEP is dispatched directly via handler registry in
      // Graph.dispatchRunStep (primary, reliable path).  Skip the
      // callback-based dispatch to prevent double handling.
      if (
        eventName === GraphEvents.ON_RUN_STEP &&
        this.Graph != null &&
        this.Graph.handlerDispatchedStepIds.has((data as t.RunStep).id)
      ) {
        return;
      }
      const handler = this.handlerRegistry?.getHandler(eventName);
      if (handler && this.Graph) {
        return await handler.handle(
          eventName,
          data as
            | t.StreamEventData
            | t.ModelEndData
            | t.RunStep
            | t.RunStepDeltaEvent
            | t.MessageDeltaEvent
            | t.ReasoningDeltaEvent
            | { result: t.ToolEndEvent },
          metadata,
          this.Graph
        );
      }
    };
  }

  async processStream(
    inputs: t.IState | Command,
    callerConfig: Partial<RunnableConfig> & {
      version: 'v1' | 'v2';
      run_id?: string;
    },
    streamOptions?: t.EventStreamOptions
  ): Promise<MessageContentComplex[] | undefined> {
    if (this.graphRunnable == null) {
      throw new Error(
        'Run not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }
    if (!this.Graph) {
      throw new Error(
        'Graph not initialized. Make sure to use Run.create() to instantiate the Run.'
      );
    }

    /**
     * `Command` inputs (currently only `Command({ resume })`) are
     * resume-mode invocations: LangGraph rebuilds graph state from the
     * checkpointer, so we skip RunStart / UserPromptSubmit hooks (no
     * new prompt to evaluate) and read run-state from the Graph wrapper
     * instead of `inputs.messages`.
     */
    const isResume = inputs instanceof Command;
    const stateInputs = isResume ? undefined : (inputs as t.IState);

    const config: Partial<RunnableConfig> & {
      version: 'v1' | 'v2';
      run_id?: string;
    } = {
      recursionLimit: 50,
      ...callerConfig,
      configurable: { ...callerConfig.configurable },
    };

    this.Graph.resetValues(streamOptions?.keepContent);
    this._interrupt = undefined;
    this._haltedReason = undefined;
    this.hookRegistry?.clearHaltSignal(this.id);

    /** Custom event callback to intercept and handle custom events */
    const customEventCallback = this.createCustomEventCallback();

    const baseCallbacks = (config.callbacks as t.ProvidedCallbacks) ?? [];
    const streamCallbacks = streamOptions?.callbacks
      ? this.getCallbacks(streamOptions.callbacks)
      : [];

    const customHandler = BaseCallbackHandler.fromMethods({
      [Callback.CUSTOM_EVENT]: customEventCallback,
    });
    customHandler.awaitHandlers = true;

    config.callbacks = baseCallbacks
      .concat(streamCallbacks)
      .concat(customHandler);

    if (
      isPresent(process.env.LANGFUSE_SECRET_KEY) &&
      isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
      isPresent(process.env.LANGFUSE_BASE_URL)
    ) {
      const userId = config.configurable?.user_id;
      const sessionId = config.configurable?.thread_id;
      const primaryContext = this.Graph.agentContexts.get(
        this.Graph.defaultAgentId
      );
      const traceMetadata = {
        messageId: this.id,
        parentMessageId: config.configurable?.requestBody?.parentMessageId,
        agentName: primaryContext?.name,
      };
      const handler = new CallbackHandler({
        userId,
        sessionId,
        traceMetadata,
      });
      config.callbacks = (
        (config.callbacks as t.ProvidedCallbacks) ?? []
      ).concat([handler]);
    }

    if (!this.id) {
      throw new Error('Run ID not provided');
    }

    config.run_id = this.id;
    config.configurable = Object.assign(config.configurable ?? {}, {
      run_id: this.id,
    });

    const threadId = config.configurable.thread_id as string | undefined;

    if (this.hookRegistry != null && stateInputs != null) {
      const shouldHalt = await this.runPreStreamHooks(
        stateInputs,
        threadId,
        config
      );
      if (shouldHalt) {
        return undefined;
      }
    }

    /**
     * `streamEvents` accepts both state inputs and `Command` (resume) at
     * runtime, but our `CompiledStateWorkflow` type narrows the first
     * arg to `BaseGraphState`. Cast on the call so the resume path
     * type-checks without widening the wrapper for every caller.
     */
    const stream = this.graphRunnable.streamEvents(inputs as t.IState, config, {
      raiseError: true,
      /**
       * Prevent EventStreamCallbackHandler from processing custom events.
       * Custom events are already handled via our createCustomEventCallback()
       * which routes them through the handlerRegistry.
       * Without this flag, EventStreamCallbackHandler throws errors when
       * custom events are dispatched for run IDs not in its internal map
       * (due to timing issues in parallel execution or after run cleanup).
       */
      ignoreCustomEvent: true,
    });

    /**
     * Tracks whether the stream loop threw. Used by the `finally`
     * block to decide whether to honor the interrupt-preservation
     * guard for session hooks: a captured `_interrupt` is only
     * meaningful if the stream completed cleanly. If the loop errored
     * after stashing an interrupt (e.g. a downstream handler throws
     * after the interrupt event landed), the interrupt is stale —
     * preserving session hooks would leak them into the next run.
     */
    let streamThrew = false;

    try {
      for await (const event of stream) {
        const { data, metadata, ...info } = event;

        const eventName: t.EventName = info.event;

        /** Skip custom events as they're handled by our callback */
        if (eventName === GraphEvents.ON_CUSTOM_EVENT) {
          continue;
        }

        /**
         * Detect interrupts surfaced by LangGraph as a synthetic
         * `__interrupt__` field on the streamed chunk and stash the
         * first one for the host to read via `run.getInterrupt()`
         * once the stream drains. Captured as `unknown` because the
         * SDK does not validate the runtime payload shape — the
         * built-in ToolNode raises a `HumanInterruptPayload`
         * (`tool_approval` / `ask_user_question`), but custom nodes
         * can pass any payload to `interrupt()`. Callers narrow with
         * the `isToolApprovalInterrupt` / `isAskUserQuestionInterrupt`
         * guards or assert via `getInterrupt<T>()`.
         */
        if (
          this._interrupt == null &&
          data.chunk != null &&
          isInterrupted<unknown>(data.chunk)
        ) {
          const interrupts = data.chunk[INTERRUPT];
          if (interrupts.length > 0) {
            const first = interrupts[0];
            /**
             * Capture the interrupt unconditionally — `interrupt(null)`
             * and `interrupt(undefined)` are valid pauses (a custom
             * node may want to pause without metadata) and the host
             * still needs to know the run is awaiting resume. Gating
             * on `payload != null` would silently downgrade a paused
             * run to "completed" and let the `Stop` hook fire,
             * breaking host resume handling.
             */
            this._interrupt = {
              interruptId: first.id ?? '',
              threadId,
              payload: first.value,
            };
          }
        }

        const handler = this.handlerRegistry?.getHandler(eventName);
        if (handler) {
          await handler.handle(eventName, data, metadata, this.Graph);
        }

        /**
         * Mid-flight halt: any hook (PreToolUse, PostToolUse,
         * PostToolBatch, SubagentStart/Stop, PreCompact, PostCompact)
         * that returned `preventContinuation: true` raises a halt
         * signal on the registry via `executeHooks`. We poll between
         * stream events and break out as soon as one is set so the
         * graph doesn't take another model turn after the halting
         * operation completes.
         *
         * Limitation: the current step (in-flight model call, ongoing
         * tool batch) is not aborted — only the next step is skipped.
         * This matches Claude Code's `continue: false` semantic where
         * the active operation finishes before halting takes effect.
         */
        const haltSignal = this.hookRegistry?.getHaltSignal(this.id);
        if (haltSignal != null) {
          this._haltedReason = haltSignal.reason;
          break;
        }
      }

      /**
       * Skip the Stop hook when the run paused on a HITL interrupt
       * (still pending human input) or was halted by a hook (the host
       * already chose to stop, so a Stop hook firing now would be
       * misleading). The host fires Stop on the resumed-and-completed
       * run instead.
       */
      if (
        this._interrupt == null &&
        this._haltedReason == null &&
        this.hookRegistry?.hasHookFor('Stop', this.id) === true
      ) {
        await executeHooks({
          registry: this.hookRegistry,
          input: {
            hook_event_name: 'Stop',
            runId: this.id,
            threadId,
            agentId: this.Graph.defaultAgentId,
            messages:
              this.Graph.getRunMessages() ?? stateInputs?.messages ?? [],
            stopHookActive: false, // will be true when stop is triggered by a hook (Phase 2)
          },
          sessionId: this.id,
        }).catch(() => {
          /* Stop hook errors must not masquerade as stream failures */
        });
      }
    } catch (err) {
      streamThrew = true;
      if (this.hookRegistry?.hasHookFor('StopFailure', this.id) === true) {
        const runMessages = this.Graph.getRunMessages() ?? [];
        await executeHooks({
          registry: this.hookRegistry,
          input: {
            hook_event_name: 'StopFailure',
            runId: this.id,
            threadId,
            agentId: this.Graph.defaultAgentId,
            error: err instanceof Error ? err.message : String(err),
            lastAssistantMessage: findLastMessageOfType(runMessages, 'ai'),
          },
          sessionId: this.id,
        }).catch(() => {
          /* swallow hook errors — the original error must propagate */
        });
      }
      throw err;
    } finally {
      /**
       * Preserve session-scoped hooks when the run paused on a HITL
       * interrupt — the very next call will be `Run.resume()`, which
       * needs the same policy hooks (e.g., the `PreToolUse` matcher
       * that triggered the interrupt) to fire on the re-executed node
       * and uphold the approval flow. Clearing here would leak the
       * approval gate on resume. The session is cleared instead at
       * natural completion, error (including errors that happen AFTER
       * an interrupt was captured — those interrupts are stale), or
       * hook-driven halt — every state where no resume is expected.
       */
      if (this._interrupt == null || streamThrew) {
        this.hookRegistry?.clearSession(this.id);
      }
      /**
       * Drop any halt signal raised mid-stream for this run so a
       * subsequent `processStream` / `resume` starts with clean state.
       * The Run captured `_haltedReason` already; the registry entry
       * for this `sessionId` would otherwise spuriously trip the next
       * loop. Other concurrent runs sharing this registry are
       * unaffected — their entries live under their own session ids.
       */
      this.hookRegistry?.clearHaltSignal(this.id);

      /**
       * Break the reference chain that keeps heavy data alive via
       * LangGraph's internal `__pregel_scratchpad.currentTaskInput` →
       * `@langchain/core` `RunTree.extra[lc:child_config]` →
       * Node.js `AsyncLocalStorage` context captured by timers/promises.
       *
       * Without this, base64-encoded images/PDFs in message content remain
       * reachable from lingering `Timeout` handles until GC runs.
       */
      if (!this.skipCleanup) {
        if (
          (config.configurable as Record<string, unknown> | undefined) != null
        ) {
          for (const key of Object.getOwnPropertySymbols(config.configurable)) {
            const val = config.configurable[key as unknown as string] as
              | Record<string, unknown>
              | undefined;
            if (
              val != null &&
              typeof val === 'object' &&
              'currentTaskInput' in val
            ) {
              (val as Record<string, unknown>).currentTaskInput = undefined;
            }
            delete config.configurable[key as unknown as string];
          }
          config.configurable = undefined;
        }
        config.callbacks = undefined;
      }

      const result = this.returnContent
        ? this.Graph.getContentParts()
        : undefined;

      this.calibrationRatio = this.Graph.getCalibrationRatio();

      if (!this.skipCleanup) {
        this.Graph.clearHeavyState();
      }

      this._streamResult = result;
    }

    return this._streamResult;
  }

  /**
   * Returns the pending interrupt captured during the most recent
   * `processStream` (or `resume`) invocation. `undefined` when the run
   * either has not been streamed yet or completed without pausing.
   *
   * Hosts call this immediately after `processStream` returns to decide
   * whether the run is awaiting human input. Persist the returned
   * descriptor (alongside `thread_id` and the agent run config) so a
   * later `resume(decisions)` can rebuild the run.
   *
   * The default `TPayload` is the SDK's `HumanInterruptPayload` union
   * (`tool_approval` / `ask_user_question`), suitable for the common
   * case where interrupts come from the built-in ToolNode or
   * `askUserQuestion()` helper. Hosts that raise custom interrupts
   * from custom graph nodes pass their own type — the SDK does not
   * validate the runtime shape, it just transports whatever the
   * `interrupt()` call carried. When in doubt, narrow with the
   * `isToolApprovalInterrupt` / `isAskUserQuestionInterrupt` type
   * guards (which accept `unknown`) before reading variant-specific
   * fields.
   */
  getInterrupt<TPayload = t.HumanInterruptPayload>():
    | t.RunInterruptResult<TPayload>
    | undefined {
    return this._interrupt as t.RunInterruptResult<TPayload> | undefined;
  }

  /**
   * Returns the reason a hook halted the run via
   * `preventContinuation: true`, or `undefined` if no hook halted.
   *
   * Hosts inspect this after `processStream` returns to distinguish a
   * natural completion (`undefined`) from a hook-driven halt (a
   * truthy string). Independent from `getInterrupt()` — a halted run
   * has no interrupt; an interrupted run has no halt reason.
   */
  getHaltReason(): string | undefined {
    return this._haltedReason;
  }

  /**
   * Resume a paused HITL run with the value the user (or whatever
   * decided the interrupt) supplied. The default `TResume` covers the
   * `tool_approval` interrupt (the common case): an array of decisions
   * in `action_requests` order, or a record keyed by `tool_call_id`.
   *
   * For other interrupt types (e.g., `ask_user_question` →
   * `AskUserQuestionResolution`, or any custom interrupt a host raises
   * from a custom node), pass the type parameter and the SDK forwards
   * the value through unchanged. LangGraph delivers it as the return
   * value of the original `interrupt()` call inside the paused node.
   *
   * The host MUST construct this Run with the same `thread_id` and the
   * same checkpointer as the original paused run; LangGraph rebuilds
   * graph state from the checkpoint and re-enters the interrupted node
   * from the start.
   */
  async resume<TResume = t.ToolApprovalDecision[] | t.ToolApprovalDecisionMap>(
    resumeValue: TResume,
    callerConfig: Partial<RunnableConfig> & {
      version: 'v1' | 'v2';
      run_id?: string;
    },
    streamOptions?: t.EventStreamOptions
  ): Promise<MessageContentComplex[] | undefined> {
    return this.processStream(
      new Command({ resume: resumeValue }),
      callerConfig,
      streamOptions
    );
  }

  private createSystemCallback<K extends keyof t.ClientCallbacks>(
    clientCallbacks: t.ClientCallbacks,
    key: K
  ): t.SystemCallbacks[K] {
    return ((...args: unknown[]) => {
      const clientCallback = clientCallbacks[key];
      if (clientCallback && this.Graph) {
        (clientCallback as (...args: unknown[]) => void)(this.Graph, ...args);
      }
    }) as t.SystemCallbacks[K];
  }

  getCallbacks(clientCallbacks: t.ClientCallbacks): t.SystemCallbacks {
    return {
      [Callback.TOOL_ERROR]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_ERROR
      ),
      [Callback.TOOL_START]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_START
      ),
      [Callback.TOOL_END]: this.createSystemCallback(
        clientCallbacks,
        Callback.TOOL_END
      ),
    };
  }

  async generateTitle({
    provider,
    inputText,
    contentParts,
    titlePrompt,
    clientOptions,
    chainOptions,
    skipLanguage,
    titleMethod = TitleMethod.COMPLETION,
    titlePromptTemplate,
  }: t.RunTitleOptions): Promise<{ language?: string; title?: string }> {
    if (
      chainOptions != null &&
      isPresent(process.env.LANGFUSE_SECRET_KEY) &&
      isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
      isPresent(process.env.LANGFUSE_BASE_URL)
    ) {
      const userId = chainOptions.configurable?.user_id;
      const sessionId = chainOptions.configurable?.thread_id;
      const titleContext = this.Graph?.agentContexts.get(
        this.Graph.defaultAgentId
      );
      const traceMetadata = {
        messageId: 'title-' + this.id,
        agentName: titleContext?.name,
      };
      const handler = new CallbackHandler({
        userId,
        sessionId,
        traceMetadata,
      });
      chainOptions.callbacks = (
        (chainOptions.callbacks as t.ProvidedCallbacks) ?? []
      ).concat([handler]);
    }

    const convoTemplate = PromptTemplate.fromTemplate(
      titlePromptTemplate ?? 'User: {input}\nAI: {output}'
    );

    const response = contentParts
      .map((part) => {
        if (part?.type === 'text') return part.text;
        return '';
      })
      .join('\n');

    const model = initializeModel({
      provider,
      clientOptions,
    }) as t.ChatModelInstance;

    if (
      isOpenAILike(provider) &&
      (model instanceof ChatOpenAI || model instanceof AzureChatOpenAI)
    ) {
      model.temperature = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.temperature as number;
      model.topP = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.topP as number;
      model.frequencyPenalty = (
        clientOptions as t.OpenAIClientOptions | undefined
      )?.frequencyPenalty as number;
      model.presencePenalty = (
        clientOptions as t.OpenAIClientOptions | undefined
      )?.presencePenalty as number;
      model.n = (clientOptions as t.OpenAIClientOptions | undefined)
        ?.n as number;
    }

    const convoToTitleInput = new RunnableLambda({
      func: (
        promptValue: StringPromptValue
      ): { convo: string; inputText: string; skipLanguage?: boolean } => ({
        convo: promptValue.value,
        inputText,
        skipLanguage,
      }),
    }).withConfig({ runName: 'ConvoTransform' });

    const titleChain =
      titleMethod === TitleMethod.COMPLETION
        ? await createCompletionTitleRunnable(model, titlePrompt)
        : await createTitleRunnable(model, titlePrompt);

    /** Pipes `convoTemplate` -> `transformer` -> `titleChain` */
    const fullChain = convoTemplate
      .withConfig({ runName: 'ConvoTemplate' })
      .pipe(convoToTitleInput)
      .pipe(titleChain)
      .withConfig({ runName: 'TitleChain' });

    const invokeConfig = Object.assign({}, chainOptions, {
      run_id: this.id,
      runId: this.id,
    });

    try {
      return await fullChain.invoke(
        { input: inputText, output: response },
        invokeConfig
      );
    } catch (_e) {
      // Fallback: strip callbacks to avoid EventStream tracer errors in certain environments
      // But preserve langfuse handler if it exists
      const langfuseHandler = (
        invokeConfig.callbacks as t.ProvidedCallbacks
      )?.find((cb) => cb instanceof CallbackHandler);
      const { callbacks: _cb, ...rest } = invokeConfig;
      const safeConfig = Object.assign({}, rest, {
        callbacks: langfuseHandler ? [langfuseHandler] : [],
      });
      return await fullChain.invoke(
        { input: inputText, output: response },
        safeConfig as Partial<RunnableConfig>
      );
    }
  }
}

function findLastMessageOfType(
  messages: BaseMessage[],
  type: string
): BaseMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].getType() === type) {
      return messages[i];
    }
  }
  return undefined;
}

function extractPromptText(message: BaseMessage): string {
  const content = message.content;
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    return String(content);
  }
  const parts: string[] = [];
  for (const block of content) {
    if (
      typeof block === 'object' &&
      'type' in block &&
      block.type === 'text' &&
      'text' in block &&
      typeof block.text === 'string'
    ) {
      parts.push(block.text);
    }
  }
  return parts.join('\n');
}
