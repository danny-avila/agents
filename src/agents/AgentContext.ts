/* eslint-disable no-console */
// src/agents/AgentContext.ts
import { zodToJsonSchema } from 'zod-to-json-schema';
import { SystemMessage } from '@langchain/core/messages';
import { RunnableLambda } from '@langchain/core/runnables';
import type {
  UsageMetadata,
  BaseMessage,
  BaseMessageFields,
} from '@langchain/core/messages';
import type { RunnableConfig, Runnable } from '@langchain/core/runnables';
import type * as t from '@/types';
import type { createPruneMessages } from '@/messages';
import { ContentTypes, Providers } from '@/common';

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
      instructions,
      additional_instructions,
      streamBuffer,
      maxContextTokens,
      reasoningKey,
      useLegacyContent,
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
      instructions,
      additionalInstructions: additional_instructions,
      reasoningKey,
      toolEnd,
      instructionTokens: 0,
      tokenCounter,
      useLegacyContent,
    });

    if (tokenCounter) {
      // Initialize system runnable BEFORE async tool token calculation
      // This ensures system message tokens are in instructionTokens before
      // updateTokenMapWithInstructions is called
      agentContext.initializeSystemRunnable();

      const tokenMap = indexTokenCountMap || {};
      agentContext.indexTokenCountMap = tokenMap;
      agentContext.tokenCalculationPromise = agentContext
        .calculateInstructionTokens(tokenCounter)
        .then(() => {
          // Update token map with instruction tokens (includes system + tool tokens)
          agentContext.updateTokenMapWithInstructions(tokenMap);
        })
        .catch((err) => {
          console.error('Error calculating instruction tokens:', err);
        });
    } else if (indexTokenCountMap) {
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
  /** Maximum context tokens for this agent */
  maxContextTokens?: number;
  /** Current usage metadata for this agent */
  currentUsage?: Partial<UsageMetadata>;
  /** Prune messages function configured for this agent */
  pruneMessages?: ReturnType<typeof createPruneMessages>;
  /** Token counter function for this agent */
  tokenCounter?: t.TokenCounter;
  /** Instructions/system message token count */
  instructionTokens: number = 0;
  /** The amount of time that should pass before another consecutive API call */
  streamBuffer?: number;
  /** Last stream call timestamp for rate limiting */
  lastStreamCall?: number;
  /** Tools available to this agent */
  tools?: t.GraphTools;
  /** Tool map for this agent */
  toolMap?: t.ToolMap;
  /**
   * Tool definitions registry (includes deferred and programmatic tool metadata).
   * Used for tool search and programmatic tool calling.
   */
  toolRegistry?: t.LCToolRegistry;
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
  /** Cached system message token count (separate from tool tokens) */
  private systemMessageTokens: number = 0;
  /** Promise for token calculation initialization */
  tokenCalculationPromise?: Promise<void>;
  /** Format content blocks as strings (for legacy compatibility) */
  useLegacyContent: boolean = false;
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
    instructions,
    additionalInstructions,
    reasoningKey,
    toolEnd,
    instructionTokens,
    useLegacyContent,
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
    instructions?: string;
    additionalInstructions?: string;
    reasoningKey?: 'reasoning_content' | 'reasoning';
    toolEnd?: boolean;
    instructionTokens?: number;
    useLegacyContent?: boolean;
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
    this.instructions = instructions;
    this.additionalInstructions = additionalInstructions;
    if (reasoningKey) {
      this.reasoningKey = reasoningKey;
    }
    if (toolEnd !== undefined) {
      this.toolEnd = toolEnd;
    }
    if (instructionTokens !== undefined) {
      this.instructionTokens = instructionTokens;
    }

    this.useLegacyContent = useLegacyContent ?? false;
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

      // Include if: not deferred OR deferred but discovered
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
    // Return cached if not stale
    if (!this.systemRunnableStale && this.cachedSystemRunnable !== undefined) {
      return this.cachedSystemRunnable;
    }

    // Stale or first access - rebuild
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

    /** Build agent identity and handoff context preamble */
    const identityPreamble = this.buildIdentityPreamble();
    if (identityPreamble) {
      parts.push(identityPreamble);
    }

    /** Add main instructions */
    if (this.instructions != null && this.instructions !== '') {
      parts.push(this.instructions);
    }

    /** Add additional instructions */
    if (
      this.additionalInstructions != null &&
      this.additionalInstructions !== ''
    ) {
      parts.push(this.additionalInstructions);
    }

    /** Add programmatic tools documentation */
    const programmaticToolsDoc = this.buildProgrammaticOnlyToolsInstructions();
    if (programmaticToolsDoc) {
      parts.push(programmaticToolsDoc);
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
    if (!instructionsString) {
      // Remove previous tokens if we had a system message before
      this.instructionTokens -= this.systemMessageTokens;
      this.systemMessageTokens = 0;
      return undefined;
    }

    let finalInstructions: string | BaseMessageFields = instructionsString;

    // Handle Anthropic prompt caching
    if (this.provider === Providers.ANTHROPIC) {
      const anthropicOptions = this.clientOptions as
        | t.AnthropicClientOptions
        | undefined;
      if (anthropicOptions?.promptCache === true) {
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

    const systemMessage = new SystemMessage(finalInstructions);

    // Update token counts (subtract old, add new)
    if (this.tokenCounter) {
      this.instructionTokens -= this.systemMessageTokens;
      this.systemMessageTokens = this.tokenCounter(systemMessage);
      this.instructionTokens += this.systemMessageTokens;
    }

    return RunnableLambda.from((messages: BaseMessage[]) => {
      return [systemMessage, ...messages];
    }).withConfig({ runName: 'prompt' });
  }

  /**
   * Reset context for a new run
   */
  reset(): void {
    this.instructionTokens = 0;
    this.systemMessageTokens = 0;
    this.cachedSystemRunnable = undefined;
    this.systemRunnableStale = true;
    this.lastToken = undefined;
    this.indexTokenCountMap = {};
    this.currentUsage = undefined;
    this.pruneMessages = undefined;
    this.lastStreamCall = undefined;
    this.tokenTypeSwitch = undefined;
    this.currentTokenType = ContentTypes.TEXT;
    this.discoveredToolNames.clear();
    this.handoffContext = undefined;
  }

  /**
   * Update the token count map with instruction tokens
   */
  updateTokenMapWithInstructions(baseTokenMap: Record<string, number>): void {
    if (this.instructionTokens > 0) {
      // Shift all indices by the instruction token count
      const shiftedMap: Record<string, number> = {};
      for (const [key, value] of Object.entries(baseTokenMap)) {
        const index = parseInt(key, 10);
        if (!isNaN(index)) {
          shiftedMap[String(index)] =
            value + (index === 0 ? this.instructionTokens : 0);
        }
      }
      this.indexTokenCountMap = shiftedMap;
    } else {
      this.indexTokenCountMap = { ...baseTokenMap };
    }
  }

  /**
   * Calculate tool tokens and add to instruction tokens
   * Note: System message tokens are calculated during systemRunnable creation
   */
  async calculateInstructionTokens(
    tokenCounter: t.TokenCounter
  ): Promise<void> {
    let toolTokens = 0;
    if (this.tools && this.tools.length > 0) {
      for (const tool of this.tools) {
        const genericTool = tool as Record<string, unknown>;
        if (
          genericTool.schema != null &&
          typeof genericTool.schema === 'object'
        ) {
          const schema = genericTool.schema as {
            describe: (desc: string) => unknown;
          };
          const describedSchema = schema.describe(
            (genericTool.description as string) || ''
          );
          const jsonSchema = zodToJsonSchema(
            describedSchema as Parameters<typeof zodToJsonSchema>[0],
            (genericTool.name as string) || ''
          );
          toolTokens += tokenCounter(
            new SystemMessage(JSON.stringify(jsonSchema))
          );
        }
      }
    }

    // Add tool tokens to existing instruction tokens (which may already include system message tokens)
    this.instructionTokens += toolTokens;
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
   * Includes:
   * 1. Non-deferred tools with allowed_callers: ['direct']
   * 2. Discovered tools (from tool search)
   * @returns Array of tools to bind to model
   */
  getToolsForBinding(): t.GraphTools | undefined {
    if (!this.tools || !this.toolRegistry) {
      return this.tools;
    }

    const toolsToInclude = this.tools.filter((tool) => {
      if (!('name' in tool)) {
        return true; // No name, include by default
      }

      const toolDef = this.toolRegistry?.get(tool.name);
      if (!toolDef) {
        return true; // Not in registry, include by default
      }

      // Check if discovered (overrides defer_loading)
      if (this.discoveredToolNames.has(tool.name)) {
        // Discovered tools must still have allowed_callers: ['direct']
        const allowedCallers = toolDef.allowed_callers ?? ['direct'];
        return allowedCallers.includes('direct');
      }

      // Not discovered: must be direct-callable AND not deferred
      const allowedCallers = toolDef.allowed_callers ?? ['direct'];
      return (
        allowedCallers.includes('direct') && toolDef.defer_loading !== true
      );
    });

    return toolsToInclude;
  }
}
