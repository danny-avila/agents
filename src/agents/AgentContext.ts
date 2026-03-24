/* eslint-disable no-console */
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { RunnableLambda } from '@langchain/core/runnables';
import type {
  UsageMetadata,
  BaseMessage,
  BaseMessageFields,
} from '@langchain/core/messages';
import type { RunnableConfig, Runnable } from '@langchain/core/runnables';
import type { createPruneMessages } from '@/messages';
import type * as t from '@/types';
import {
  ANTHROPIC_TOOL_TOKEN_MULTIPLIER,
  DEFAULT_TOOL_TOKEN_MULTIPLIER,
  ContentTypes,
  Providers,
} from '@/common';
import { createSchemaOnlyTools } from '@/tools/schema';
import { addCacheControl } from '@/messages/cache';
import { DEFAULT_RESERVE_RATIO } from '@/messages';
import { toJsonSchema } from '@/utils/schema';

/**
 * Encapsulates agent-specific state that can vary between agents in a multi-agent system
 */
export class AgentContext {
  /**
   * Create an AgentContext from configuration with token accounting initialization
   */
  static fromConfig(
    agentConfig: t.AgentInputs,
    tokenCounter?: t.TokenCounter,
    indexTokenCountMap?: Record<string, number>
  ): AgentContext {
    const {
      agentId,
      name,
      provider,
      clientOptions,
      tools,
      toolMap,
      toolEnd,
      toolRegistry,
      toolDefinitions,
      instructions,
      additional_instructions,
      streamBuffer,
      maxContextTokens,
      reasoningKey,
      useLegacyContent,
      discoveredTools,
      summarizationEnabled,
      summarizationConfig,
      initialSummary,
      contextPruningConfig,
      maxToolResultChars,
      toolSchemaTokens,
    } = agentConfig;

    const agentContext = new AgentContext({
      agentId,
      name: name ?? agentId,
      provider,
      clientOptions,
      maxContextTokens,
      streamBuffer,
      tools,
      toolMap,
      toolRegistry,
      toolDefinitions,
      instructions,
      additionalInstructions: additional_instructions,
      reasoningKey,
      toolEnd,
      instructionTokens: 0,
      tokenCounter,
      useLegacyContent,
      discoveredTools,
      summarizationEnabled,
      summarizationConfig,
      contextPruningConfig,
      maxToolResultChars,
    });

    if (initialSummary?.text != null && initialSummary.text !== '') {
      agentContext.setInitialSummary(
        initialSummary.text,
        initialSummary.tokenCount
      );
    }

    if (tokenCounter) {
      agentContext.initializeSystemRunnable();

      const tokenMap = indexTokenCountMap || {};
      agentContext.baseIndexTokenCountMap = { ...tokenMap };
      agentContext.indexTokenCountMap = tokenMap;

      if (toolSchemaTokens != null && toolSchemaTokens > 0) {
        /** Use pre-computed (cached) tool schema tokens — skip calculateInstructionTokens */
        agentContext.toolSchemaTokens = toolSchemaTokens;
        agentContext.tokenCalculationPromise = Promise.resolve();
        agentContext.updateTokenMapWithInstructions(tokenMap);
      } else {
        agentContext.tokenCalculationPromise = agentContext
          .calculateInstructionTokens(tokenCounter)
          .then(() => {
            agentContext.updateTokenMapWithInstructions(tokenMap);
          })
          .catch((err) => {
            console.error('Error calculating instruction tokens:', err);
          });
      }
    } else if (indexTokenCountMap) {
      agentContext.baseIndexTokenCountMap = { ...indexTokenCountMap };
      agentContext.indexTokenCountMap = indexTokenCountMap;
    }

    return agentContext;
  }

  /** Agent identifier */
  agentId: string;
  /** Human-readable name for this agent (used in handoff context). Falls back to agentId if not provided. */
  name?: string;
  /** Provider for this specific agent */
  provider: Providers;
  /** Client options for this agent */
  clientOptions?: t.ClientOptions;
  /** Token count map indexed by message position */
  indexTokenCountMap: Record<string, number | undefined> = {};
  /** Canonical pre-run token map used to restore token accounting on reset */
  baseIndexTokenCountMap: Record<string, number> = {};
  /** Maximum context tokens for this agent */
  maxContextTokens?: number;
  /** Current usage metadata for this agent */
  currentUsage?: Partial<UsageMetadata>;
  /**
   * Usage from the most recent LLM call only (not accumulated).
   * Used for accurate provider calibration in pruning.
   */
  lastCallUsage?: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    cacheRead?: number;
    cacheCreation?: number;
  };
  /**
   * Whether totalTokens data is fresh (set true when provider usage arrives,
   * false at the start of each turn before the LLM responds).
   * Prevents stale token data from driving pruning/trigger decisions.
   */
  totalTokensFresh: boolean = false;
  /** Context pruning configuration. */
  contextPruningConfig?: t.ContextPruningConfig;
  maxToolResultChars?: number;
  /** Prune messages function configured for this agent */
  pruneMessages?: ReturnType<typeof createPruneMessages>;
  /** Token counter function for this agent */
  tokenCounter?: t.TokenCounter;
  /** Token count for the system message (instructions text). */
  systemMessageTokens: number = 0;
  /** Token count for tool schemas only. */
  toolSchemaTokens: number = 0;
  /** Running calibration ratio from the pruner — persisted across runs via contextMeta. */
  calibrationRatio: number = 1;
  /** Provider-observed instruction overhead from the pruner's best-variance turn. */
  resolvedInstructionOverhead?: number;
  /** Pre-masking tool content keyed by message index, consumed by the summarize node. */
  pendingOriginalToolContent?: Map<number, string>;

  /** Total instruction overhead: system message + tool schemas + pending summary. */
  get instructionTokens(): number {
    const summaryOverhead =
      this._summaryLocation === 'user_message' ? this.summaryTokenCount : 0;
    return this.systemMessageTokens + this.toolSchemaTokens + summaryOverhead;
  }
  /** The amount of time that should pass before another consecutive API call */
  streamBuffer?: number;
  /** Last stream call timestamp for rate limiting */
  lastStreamCall?: number;
  /** Tools available to this agent */
  tools?: t.GraphTools;
  /** Graph-managed tools (e.g., handoff tools created by MultiAgentGraph) that bypass event-driven dispatch */
  graphTools?: t.GraphTools;
  /** Tool map for this agent */
  toolMap?: t.ToolMap;
  /**
   * Tool definitions registry (includes deferred and programmatic tool metadata).
   * Used for tool search and programmatic tool calling.
   */
  toolRegistry?: t.LCToolRegistry;
  /**
   * Serializable tool definitions for event-driven execution.
   * When provided, ToolNode operates in event-driven mode.
   */
  toolDefinitions?: t.LCTool[];
  /** Set of tool names discovered via tool search (to be loaded) */
  discoveredToolNames: Set<string> = new Set();
  /** Instructions for this agent */
  instructions?: string;
  /** Additional instructions for this agent */
  additionalInstructions?: string;
  /** Reasoning key for this agent */
  reasoningKey: 'reasoning_content' | 'reasoning' = 'reasoning_content';
  /** Last token for reasoning detection */
  lastToken?: string;
  /** Token type switch state */
  tokenTypeSwitch?: 'reasoning' | 'content';
  /** Tracks how many reasoning→text transitions have occurred (ensures unique post-reasoning step keys) */
  reasoningTransitionCount = 0;
  /** Current token type being processed */
  currentTokenType: ContentTypes.TEXT | ContentTypes.THINK | 'think_and_text' =
    ContentTypes.TEXT;
  /** Whether tools should end the workflow */
  toolEnd: boolean = false;
  /** Cached system runnable (created lazily) */
  private cachedSystemRunnable?: Runnable<
    BaseMessage[],
    (BaseMessage | SystemMessage)[],
    RunnableConfig<Record<string, unknown>>
  >;
  /** Whether system runnable needs rebuild (set when discovered tools change) */
  private systemRunnableStale: boolean = true;
  /** Promise for token calculation initialization */
  tokenCalculationPromise?: Promise<void>;
  /** Format content blocks as strings (for legacy compatibility) */
  useLegacyContent: boolean = false;
  /** Enables graph-level summarization for this agent */
  summarizationEnabled?: boolean;
  /** Summarization runtime settings used by graph pruning hooks */
  summarizationConfig?: t.SummarizationConfig;
  /** Current summary text produced by the summarize node, integrated into system message */
  private summaryText?: string;
  /** Token count of the current summary (tracked for token accounting) */
  private summaryTokenCount: number = 0;
  /**
   * Where the summary should be injected:
   * - `'system_prompt'`: cross-run summary, included in `buildInstructionsString`
   * - `'user_message'`: mid-run compaction, injected as HumanMessage on clean slate
   * - `'none'`: no summary present
   */
  private _summaryLocation: 'system_prompt' | 'user_message' | 'none' = 'none';
  /**
   * Durable summary that survives reset() calls. Set from initialSummary
   * during fromConfig() and updated by setSummary() so that the latest
   * summary (whether cross-run or intra-run) is always restored after
   * processStream's resetValues() cycle.
   */
  private _durableSummaryText?: string;
  private _durableSummaryTokenCount: number = 0;
  /** Number of summarization cycles that have occurred for this agent context */
  private _summaryVersion: number = 0;
  /**
   * Message count at the time summarization was last triggered.
   * Used to prevent re-summarizing the same unchanged message set.
   * Summarization is allowed to fire again only when new messages appear.
   */
  private _lastSummarizationMsgCount: number = 0;
  /**
   * Handoff context when this agent receives control via handoff.
   * Contains source and parallel execution info for system message context.
   */
  handoffContext?: {
    /** Source agent that transferred control */
    sourceAgentName: string;
    /** Names of sibling agents executing in parallel (empty if sequential) */
    parallelSiblings: string[];
  };

  constructor({
    agentId,
    name,
    provider,
    clientOptions,
    maxContextTokens,
    streamBuffer,
    tokenCounter,
    tools,
    toolMap,
    toolRegistry,
    toolDefinitions,
    instructions,
    additionalInstructions,
    reasoningKey,
    toolEnd,
    instructionTokens,
    useLegacyContent,
    discoveredTools,
    summarizationEnabled,
    summarizationConfig,
    contextPruningConfig,
    maxToolResultChars,
  }: {
    agentId: string;
    name?: string;
    provider: Providers;
    clientOptions?: t.ClientOptions;
    maxContextTokens?: number;
    streamBuffer?: number;
    tokenCounter?: t.TokenCounter;
    tools?: t.GraphTools;
    toolMap?: t.ToolMap;
    toolRegistry?: t.LCToolRegistry;
    toolDefinitions?: t.LCTool[];
    instructions?: string;
    additionalInstructions?: string;
    reasoningKey?: 'reasoning_content' | 'reasoning';
    toolEnd?: boolean;
    instructionTokens?: number;
    useLegacyContent?: boolean;
    discoveredTools?: string[];
    summarizationEnabled?: boolean;
    summarizationConfig?: t.SummarizationConfig;
    contextPruningConfig?: t.ContextPruningConfig;
    maxToolResultChars?: number;
  }) {
    this.agentId = agentId;
    this.name = name;
    this.provider = provider;
    this.clientOptions = clientOptions;
    this.maxContextTokens = maxContextTokens;
    this.streamBuffer = streamBuffer;
    this.tokenCounter = tokenCounter;
    this.tools = tools;
    this.toolMap = toolMap;
    this.toolRegistry = toolRegistry;
    this.toolDefinitions = toolDefinitions;
    this.instructions = instructions;
    this.additionalInstructions = additionalInstructions;
    if (reasoningKey) {
      this.reasoningKey = reasoningKey;
    }
    if (toolEnd !== undefined) {
      this.toolEnd = toolEnd;
    }
    if (instructionTokens !== undefined) {
      this.systemMessageTokens = instructionTokens;
    }

    this.useLegacyContent = useLegacyContent ?? false;
    this.summarizationEnabled = summarizationEnabled;
    this.summarizationConfig = summarizationConfig;
    this.contextPruningConfig = contextPruningConfig;
    this.maxToolResultChars = maxToolResultChars;

    if (discoveredTools && discoveredTools.length > 0) {
      for (const toolName of discoveredTools) {
        this.discoveredToolNames.add(toolName);
      }
    }
  }

  /**
   * Builds instructions text for tools that are ONLY callable via programmatic code execution.
   * These tools cannot be called directly by the LLM but are available through the
   * run_tools_with_code tool.
   *
   * Includes:
   * - Code_execution-only tools that are NOT deferred
   * - Code_execution-only tools that ARE deferred but have been discovered via tool search
   */
  private buildProgrammaticOnlyToolsInstructions(): string {
    if (!this.toolRegistry) return '';

    const programmaticOnlyTools: t.LCTool[] = [];
    for (const [name, toolDef] of this.toolRegistry) {
      const allowedCallers = toolDef.allowed_callers ?? ['direct'];
      const isCodeExecutionOnly =
        allowedCallers.includes('code_execution') &&
        !allowedCallers.includes('direct');

      if (!isCodeExecutionOnly) continue;

      const isDeferred = toolDef.defer_loading === true;
      const isDiscovered = this.discoveredToolNames.has(name);
      if (!isDeferred || isDiscovered) {
        programmaticOnlyTools.push(toolDef);
      }
    }

    if (programmaticOnlyTools.length === 0) return '';

    const toolDescriptions = programmaticOnlyTools
      .map((tool) => {
        let desc = `- **${tool.name}**`;
        if (tool.description != null && tool.description !== '') {
          desc += `: ${tool.description}`;
        }
        if (tool.parameters) {
          desc += `\n  Parameters: ${JSON.stringify(tool.parameters, null, 2).replace(/\n/g, '\n  ')}`;
        }
        return desc;
      })
      .join('\n\n');

    return (
      '\n\n## Programmatic-Only Tools\n\n' +
      'The following tools are available exclusively through the `run_tools_with_code` tool. ' +
      'You cannot call these tools directly; instead, use `run_tools_with_code` with Python code that invokes them.\n\n' +
      toolDescriptions
    );
  }

  /**
   * Gets the system runnable, creating it lazily if needed.
   * Includes instructions, additional instructions, and programmatic-only tools documentation.
   * Only rebuilds when marked stale (via markToolsAsDiscovered).
   */
  get systemRunnable():
    | Runnable<
        BaseMessage[],
        (BaseMessage | SystemMessage)[],
        RunnableConfig<Record<string, unknown>>
      >
    | undefined {
    if (!this.systemRunnableStale && this.cachedSystemRunnable !== undefined) {
      return this.cachedSystemRunnable;
    }

    const instructionsString = this.buildInstructionsString();
    this.cachedSystemRunnable = this.buildSystemRunnable(instructionsString);
    this.systemRunnableStale = false;
    return this.cachedSystemRunnable;
  }

  /**
   * Explicitly initializes the system runnable.
   * Call this before async token calculation to ensure system message tokens are counted first.
   */
  initializeSystemRunnable(): void {
    if (this.systemRunnableStale || this.cachedSystemRunnable === undefined) {
      const instructionsString = this.buildInstructionsString();
      this.cachedSystemRunnable = this.buildSystemRunnable(instructionsString);
      this.systemRunnableStale = false;
    }
  }

  /**
   * Builds the raw instructions string (without creating SystemMessage).
   * Includes agent identity preamble and handoff context when available.
   */
  private buildInstructionsString(): string {
    const parts: string[] = [];

    const identityPreamble = this.buildIdentityPreamble();
    if (identityPreamble) {
      parts.push(identityPreamble);
    }

    if (this.instructions != null && this.instructions !== '') {
      parts.push(this.instructions);
    }

    if (
      this.additionalInstructions != null &&
      this.additionalInstructions !== ''
    ) {
      parts.push(this.additionalInstructions);
    }

    const programmaticToolsDoc = this.buildProgrammaticOnlyToolsInstructions();
    if (programmaticToolsDoc) {
      parts.push(programmaticToolsDoc);
    }

    // Cross-run summary: include in system prompt so the model has context
    // from the prior run.  Mid-run summaries are injected as a HumanMessage
    // on the post-compaction clean slate instead (see buildSystemRunnable).
    if (
      this._summaryLocation === 'system_prompt' &&
      this.summaryText != null &&
      this.summaryText !== ''
    ) {
      parts.push('## Conversation Summary\n\n' + this.summaryText);
    }

    return parts.join('\n\n');
  }

  /**
   * Builds the agent identity preamble including handoff context if present.
   * This helps the agent understand its role in the multi-agent workflow.
   */
  private buildIdentityPreamble(): string {
    if (!this.handoffContext) return '';

    const displayName = this.name ?? this.agentId;
    const { sourceAgentName, parallelSiblings } = this.handoffContext;
    const isParallel = parallelSiblings.length > 0;

    const lines: string[] = [];
    lines.push('## Multi-Agent Workflow');
    lines.push(
      `You are "${displayName}", transferred from "${sourceAgentName}".`
    );

    if (isParallel) {
      lines.push(`Running in parallel with: ${parallelSiblings.join(', ')}.`);
    }

    lines.push(
      'Execute only tasks relevant to your role. Routing is already handled if requested, unless you can route further.'
    );

    return lines.join('\n');
  }

  /**
   * Build system runnable from pre-built instructions string.
   * Only called when content has actually changed.
   */
  private buildSystemRunnable(
    instructionsString: string
  ):
    | Runnable<
        BaseMessage[],
        (BaseMessage | SystemMessage)[],
        RunnableConfig<Record<string, unknown>>
      >
    | undefined {
    const hasMidRunSummary =
      this._summaryLocation === 'user_message' &&
      this.summaryText != null &&
      this.summaryText !== '';

    if (!instructionsString && !hasMidRunSummary) {
      this.systemMessageTokens = 0;
      return undefined;
    }

    let finalInstructions: string | BaseMessageFields = instructionsString;

    let usePromptCache = false;
    if (this.provider === Providers.ANTHROPIC) {
      const anthropicOptions = this.clientOptions as
        | t.AnthropicClientOptions
        | undefined;
      if (anthropicOptions?.promptCache === true) {
        usePromptCache = true;
        finalInstructions = {
          content: [
            {
              type: 'text',
              text: instructionsString,
              cache_control: { type: 'ephemeral' },
            },
          ],
        };
      }
    }

    const systemMessage = instructionsString
      ? new SystemMessage(finalInstructions)
      : undefined;

    if (this.tokenCounter) {
      this.systemMessageTokens = systemMessage
        ? this.tokenCounter(systemMessage)
        : 0;
    }

    return RunnableLambda.from((messages: BaseMessage[]) => {
      const prefix: BaseMessage[] = systemMessage ? [systemMessage] : [];

      // Build the non-system portion (summary + conversation), then apply
      // cache markers separately so addCacheControl doesn't strip the
      // SystemMessage's own cache_control breakpoint set above.
      const hasSummaryBody =
        this._summaryLocation === 'user_message' &&
        this.summaryText != null &&
        this.summaryText !== '';

      let body: BaseMessage[];
      if (hasSummaryBody) {
        const wrappedSummary =
          '<summary>\n' +
          (this.summaryText as string) +
          '\n</summary>\n\n' +
          'This is your own checkpoint: you wrote it to preserve context after compaction. Pick up where you left off based on the summary above. Do not repeat prior tasks, information or acknowledge this checkpoint message directly.';

        const summaryMsg = usePromptCache
          ? new HumanMessage({
            content: [
              {
                type: 'text',
                text: wrappedSummary,
                cache_control: { type: 'ephemeral' },
              },
            ],
          })
          : new HumanMessage(wrappedSummary);
        body = [summaryMsg, ...messages];
      } else {
        body = messages;
      }

      if (usePromptCache && body.length >= 2) {
        body = addCacheControl(body);
      }
      return [...prefix, ...body];
    }).withConfig({ runName: 'prompt' });
  }

  /**
   * Reset context for a new run
   */
  reset(): void {
    this.systemMessageTokens = 0;
    this.toolSchemaTokens = 0;
    this.cachedSystemRunnable = undefined;
    this.systemRunnableStale = true;
    this.lastToken = undefined;
    this.indexTokenCountMap = { ...this.baseIndexTokenCountMap };
    this.currentUsage = undefined;
    this.pruneMessages = undefined;
    this.lastStreamCall = undefined;
    this.tokenTypeSwitch = undefined;
    this.reasoningTransitionCount = 0;
    this.currentTokenType = ContentTypes.TEXT;
    this.discoveredToolNames.clear();
    this.handoffContext = undefined;

    this.summaryText = this._durableSummaryText;
    this.summaryTokenCount = this._durableSummaryTokenCount;
    this._lastSummarizationMsgCount = 0;
    this.lastCallUsage = undefined;
    this.totalTokensFresh = false;

    if (this.tokenCounter) {
      this.initializeSystemRunnable();
      const baseTokenMap = { ...this.baseIndexTokenCountMap };
      this.indexTokenCountMap = baseTokenMap;
      this.tokenCalculationPromise = this.calculateInstructionTokens(
        this.tokenCounter
      )
        .then(() => {
          this.updateTokenMapWithInstructions(baseTokenMap);
        })
        .catch((err) => {
          console.error('Error calculating instruction tokens:', err);
        });
    } else {
      this.tokenCalculationPromise = undefined;
    }
  }

  /**
   * Update the token count map from a base map.
   *
   * Previously this inflated index 0 with instructionTokens to indirectly
   * reserve budget for the system prompt.  That approach was imprecise: with
   * large tool-schema overhead (e.g. 26 MCP tools ~5 000 tokens) the first
   * conversation message appeared enormous and was always pruned, while the
   * real available budget was never explicitly computed.
   *
   * Now instruction tokens are passed to getMessagesWithinTokenLimit via
   * the `getInstructionTokens` factory param so the pruner subtracts them
   * from the budget directly.  The token map contains only real per-message
   * token counts.
   */
  updateTokenMapWithInstructions(baseTokenMap: Record<string, number>): void {
    this.indexTokenCountMap = { ...baseTokenMap };
  }

  /**
   * Calculate tool tokens and add to instruction tokens
   * Note: System message tokens are calculated during systemRunnable creation
   */
  async calculateInstructionTokens(
    tokenCounter: t.TokenCounter
  ): Promise<void> {
    let toolTokens = 0;
    const countedToolNames = new Set<string>();

    if (this.tools && this.tools.length > 0) {
      for (const tool of this.tools) {
        const genericTool = tool as Record<string, unknown>;
        if (
          genericTool.schema != null &&
          typeof genericTool.schema === 'object'
        ) {
          const toolName = (genericTool.name as string | undefined) ?? '';
          const jsonSchema = toJsonSchema(
            genericTool.schema,
            toolName,
            (genericTool.description as string | undefined) ?? ''
          );
          toolTokens += tokenCounter(
            new SystemMessage(JSON.stringify(jsonSchema))
          );
          if (toolName) {
            countedToolNames.add(toolName);
          }
        }
      }
    }

    if (this.toolDefinitions && this.toolDefinitions.length > 0) {
      for (const def of this.toolDefinitions) {
        if (countedToolNames.has(def.name)) {
          continue;
        }
        const schema = {
          type: 'function',
          function: {
            name: def.name,
            description: def.description ?? '',
            parameters: def.parameters ?? {},
          },
        };
        toolTokens += tokenCounter(new SystemMessage(JSON.stringify(schema)));
      }
    }

    const isAnthropic =
      this.provider !== Providers.BEDROCK &&
      (this.provider === Providers.ANTHROPIC ||
        /anthropic|claude/i.test(
          String(
            (this.clientOptions as { model?: string } | undefined)?.model ?? ''
          )
        ));
    const toolTokenMultiplier = isAnthropic
      ? ANTHROPIC_TOOL_TOKEN_MULTIPLIER
      : DEFAULT_TOOL_TOKEN_MULTIPLIER;
    this.toolSchemaTokens = Math.ceil(toolTokens * toolTokenMultiplier);
  }

  /**
   * Gets the tool registry for deferred tools (for tool search).
   * @param onlyDeferred If true, only returns tools with defer_loading=true
   * @returns LCToolRegistry with tool definitions
   */
  getDeferredToolRegistry(onlyDeferred: boolean = true): t.LCToolRegistry {
    const registry: t.LCToolRegistry = new Map();

    if (!this.toolRegistry) {
      return registry;
    }

    for (const [name, toolDef] of this.toolRegistry) {
      if (!onlyDeferred || toolDef.defer_loading === true) {
        registry.set(name, toolDef);
      }
    }

    return registry;
  }

  /**
   * Sets the handoff context for this agent.
   * Call this when the agent receives control via handoff from another agent.
   * Marks system runnable as stale to include handoff context in system message.
   * @param sourceAgentName - Name of the agent that transferred control
   * @param parallelSiblings - Names of other agents executing in parallel with this one
   */
  setHandoffContext(sourceAgentName: string, parallelSiblings: string[]): void {
    this.handoffContext = { sourceAgentName, parallelSiblings };
    this.systemRunnableStale = true;
  }

  /**
   * Clears any handoff context.
   * Call this when resetting the agent or when handoff context is no longer relevant.
   */
  clearHandoffContext(): void {
    if (this.handoffContext) {
      this.handoffContext = undefined;
      this.systemRunnableStale = true;
    }
  }

  setSummary(text: string, tokenCount: number): void {
    this.summaryText = text;
    this.summaryTokenCount = tokenCount;
    this._summaryLocation = 'user_message';
    this._durableSummaryText = text;
    this._durableSummaryTokenCount = tokenCount;
    this._summaryVersion += 1;
    this.systemRunnableStale = true;
    this.pruneMessages = undefined;
  }

  /** Sets a cross-run summary that is injected into the system prompt. */
  setInitialSummary(text: string, tokenCount: number): void {
    this.summaryText = text;
    this.summaryTokenCount = tokenCount;
    this._summaryLocation = 'system_prompt';
    this._durableSummaryText = text;
    this._durableSummaryTokenCount = tokenCount;
    this._summaryVersion += 1;
    this.systemRunnableStale = true;
  }

  /**
   * Replaces the indexTokenCountMap with a fresh map keyed to the surviving
   * context messages after summarization.  Called by the summarize node after
   * it emits RemoveMessage operations that shift message indices.
   */
  rebuildTokenMapAfterSummarization(newTokenMap: Record<string, number>): void {
    this.indexTokenCountMap = newTokenMap;
    this.baseIndexTokenCountMap = { ...newTokenMap };
    this._lastSummarizationMsgCount = Object.keys(newTokenMap).length;
    this.currentUsage = undefined;
    this.lastCallUsage = undefined;
    this.totalTokensFresh = false;
  }

  hasSummary(): boolean {
    return this.summaryText != null && this.summaryText !== '';
  }

  /** True when a mid-run compaction summary is ready to be injected as a HumanMessage. */
  hasPendingCompactionSummary(): boolean {
    return this._summaryLocation === 'user_message' && this.hasSummary();
  }

  getSummaryText(): string | undefined {
    return this.summaryText;
  }

  get summaryVersion(): number {
    return this._summaryVersion;
  }

  /**
   * Returns true when the message count hasn't changed since the last
   * summarization — re-summarizing would produce an identical result.
   * Oversized individual messages are handled by fit-to-budget truncation
   * in the pruner, which keeps them in context without triggering overflow.
   */
  shouldSkipSummarization(currentMsgCount: number): boolean {
    return (
      this._lastSummarizationMsgCount > 0 &&
      currentMsgCount <= this._lastSummarizationMsgCount
    );
  }

  /**
   * Records the message count at which summarization was triggered,
   * so subsequent calls with the same count are suppressed.
   */
  markSummarizationTriggered(msgCount: number): void {
    this._lastSummarizationMsgCount = msgCount;
  }

  clearSummary(): void {
    if (this.summaryText != null) {
      this.summaryText = undefined;
      this.summaryTokenCount = 0;
      this._durableSummaryText = undefined;
      this._durableSummaryTokenCount = 0;
      this._summaryLocation = 'none';
      this.systemRunnableStale = true;
    }
  }

  /**
   * Returns a structured breakdown of how the context token budget is consumed.
   * Useful for diagnostics when context overflow or pruning issues occur.
   */
  getTokenBudgetBreakdown(messages?: BaseMessage[]): t.TokenBudgetBreakdown {
    const maxContextTokens = this.maxContextTokens ?? 0;
    const toolCount =
      (this.tools?.length ?? 0) + (this.toolDefinitions?.length ?? 0);
    const messageCount = messages?.length ?? 0;

    let messageTokens = 0;
    if (messages != null) {
      for (let i = 0; i < messages.length; i++) {
        messageTokens +=
          (this.indexTokenCountMap[i] as number | undefined) ?? 0;
      }
    }

    const reserveTokens = Math.round(maxContextTokens * DEFAULT_RESERVE_RATIO);
    const availableForMessages = Math.max(
      0,
      maxContextTokens - reserveTokens - this.instructionTokens
    );

    return {
      maxContextTokens,
      instructionTokens: this.instructionTokens,
      systemMessageTokens: this.systemMessageTokens,
      toolSchemaTokens: this.toolSchemaTokens,
      summaryTokens: this.summaryTokenCount,
      toolCount,
      messageCount,
      messageTokens,
      availableForMessages,
    };
  }

  /**
   * Returns a human-readable string of the token budget breakdown
   * for inclusion in error messages and diagnostics.
   */
  formatTokenBudgetBreakdown(messages?: BaseMessage[]): string {
    const b = this.getTokenBudgetBreakdown(messages);
    const lines = [
      'Token budget breakdown:',
      `  maxContextTokens:    ${b.maxContextTokens}`,
      `  instructionTokens:   ${b.instructionTokens} (system: ${b.systemMessageTokens}, tools: ${b.toolSchemaTokens} [${b.toolCount} tools])`,
      `  summaryTokens:       ${b.summaryTokens}`,
      `  messageTokens:       ${b.messageTokens} (${b.messageCount} messages)`,
      `  availableForMessages: ${b.availableForMessages}`,
    ];
    return lines.join('\n');
  }

  /**
   * Updates the last-call usage with data from the most recent LLM response.
   * Unlike `currentUsage` which accumulates, this captures only the single call.
   */
  updateLastCallUsage(usage: Partial<UsageMetadata>): void {
    const baseInputTokens = Number(usage.input_tokens) || 0;
    const cacheCreation =
      Number(usage.input_token_details?.cache_creation) || 0;
    const cacheRead = Number(usage.input_token_details?.cache_read) || 0;

    const outputTokens = Number(usage.output_tokens) || 0;
    const cacheSum = cacheCreation + cacheRead;
    const cacheIsAdditive = cacheSum > 0 && cacheSum > baseInputTokens;
    const totalInputTokens = cacheIsAdditive
      ? baseInputTokens + cacheSum
      : baseInputTokens;

    this.lastCallUsage = {
      inputTokens: totalInputTokens,
      outputTokens,
      totalTokens: totalInputTokens + outputTokens,
      cacheRead: cacheRead || undefined,
      cacheCreation: cacheCreation || undefined,
    };
    this.totalTokensFresh = true;
  }

  /** Marks token data as stale before a new LLM call. */
  markTokensStale(): void {
    this.totalTokensFresh = false;
  }

  /**
   * Marks tools as discovered via tool search.
   * Discovered tools will be included in the next model binding.
   * Only marks system runnable stale if NEW tools were actually added.
   * @param toolNames - Array of discovered tool names
   * @returns true if any new tools were discovered
   */
  markToolsAsDiscovered(toolNames: string[]): boolean {
    let hasNewDiscoveries = false;
    for (const name of toolNames) {
      if (!this.discoveredToolNames.has(name)) {
        this.discoveredToolNames.add(name);
        hasNewDiscoveries = true;
      }
    }
    if (hasNewDiscoveries) {
      this.systemRunnableStale = true;
    }
    return hasNewDiscoveries;
  }

  /**
   * Gets tools that should be bound to the LLM.
   * In event-driven mode (toolDefinitions present, tools empty), creates schema-only tools.
   * Otherwise filters tool instances based on:
   * 1. Non-deferred tools with allowed_callers: ['direct']
   * 2. Discovered tools (from tool search)
   * @returns Array of tools to bind to model
   */
  getToolsForBinding(): t.GraphTools | undefined {
    if (this.toolDefinitions && this.toolDefinitions.length > 0) {
      return this.getEventDrivenToolsForBinding();
    }

    const filtered =
      !this.tools || !this.toolRegistry
        ? this.tools
        : this.filterToolsForBinding(this.tools);

    if (this.graphTools && this.graphTools.length > 0) {
      return [...(filtered ?? []), ...this.graphTools];
    }

    return filtered;
  }

  /** Creates schema-only tools from toolDefinitions for event-driven mode, merged with native tools */
  private getEventDrivenToolsForBinding(): t.GraphTools {
    if (!this.toolDefinitions) {
      return this.graphTools ?? [];
    }

    const defsToInclude = this.toolDefinitions.filter((def) => {
      const allowedCallers = def.allowed_callers ?? ['direct'];
      if (!allowedCallers.includes('direct')) {
        return false;
      }
      if (
        def.defer_loading === true &&
        !this.discoveredToolNames.has(def.name)
      ) {
        return false;
      }
      return true;
    });

    const schemaTools = createSchemaOnlyTools(defsToInclude) as t.GraphTools;

    const allTools = [...schemaTools];

    if (this.graphTools && this.graphTools.length > 0) {
      allTools.push(...this.graphTools);
    }

    if (this.tools && this.tools.length > 0) {
      allTools.push(...this.tools);
    }

    return allTools;
  }

  /** Filters tool instances for binding based on registry config */
  private filterToolsForBinding(tools: t.GraphTools): t.GraphTools {
    return tools.filter((tool) => {
      if (!('name' in tool)) {
        return true;
      }

      const toolDef = this.toolRegistry?.get(tool.name);
      if (!toolDef) {
        return true;
      }

      if (this.discoveredToolNames.has(tool.name)) {
        const allowedCallers = toolDef.allowed_callers ?? ['direct'];
        return allowedCallers.includes('direct');
      }

      const allowedCallers = toolDef.allowed_callers ?? ['direct'];
      return (
        allowedCallers.includes('direct') && toolDef.defer_loading !== true
      );
    });
  }
}
