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
      provider,
      clientOptions,
      tools,
      toolMap,
      toolEnd,
      instructions,
      additional_instructions,
      streamBuffer,
      maxContextTokens,
      reasoningKey,
    } = agentConfig;

    const agentContext = new AgentContext({
      agentId,
      provider,
      clientOptions,
      maxContextTokens,
      streamBuffer,
      tools,
      toolMap,
      instructions,
      additionalInstructions: additional_instructions,
      reasoningKey,
      toolEnd,
      instructionTokens: 0,
      tokenCounter,
    });

    if (tokenCounter) {
      const tokenMap = indexTokenCountMap || {};
      agentContext.indexTokenCountMap = tokenMap;
      agentContext.tokenCalculationPromise = agentContext
        .calculateInstructionTokens(tokenCounter)
        .then(() => {
          // Update token map with instruction tokens
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
  /** System runnable for this agent */
  systemRunnable?: Runnable<
    BaseMessage[],
    (BaseMessage | SystemMessage)[],
    RunnableConfig<Record<string, unknown>>
  >;
  /** Promise for token calculation initialization */
  tokenCalculationPromise?: Promise<void>;

  constructor({
    agentId,
    provider,
    clientOptions,
    maxContextTokens,
    streamBuffer,
    tokenCounter,
    tools,
    toolMap,
    instructions,
    additionalInstructions,
    reasoningKey,
    toolEnd,
    instructionTokens,
  }: {
    agentId: string;
    provider: Providers;
    clientOptions?: t.ClientOptions;
    maxContextTokens?: number;
    streamBuffer?: number;
    tokenCounter?: t.TokenCounter;
    tools?: t.GraphTools;
    toolMap?: t.ToolMap;
    instructions?: string;
    additionalInstructions?: string;
    reasoningKey?: 'reasoning_content' | 'reasoning';
    toolEnd?: boolean;
    instructionTokens?: number;
  }) {
    this.agentId = agentId;
    this.provider = provider;
    this.clientOptions = clientOptions;
    this.maxContextTokens = maxContextTokens;
    this.streamBuffer = streamBuffer;
    this.tokenCounter = tokenCounter;
    this.tools = tools;
    this.toolMap = toolMap;
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

    this.systemRunnable = this.createSystemRunnable();
  }

  /**
   * Create system runnable from instructions and calculate tokens if tokenCounter is available
   */
  private createSystemRunnable():
    | Runnable<
        BaseMessage[],
        (BaseMessage | SystemMessage)[],
        RunnableConfig<Record<string, unknown>>
      >
    | undefined {
    let finalInstructions: string | BaseMessageFields | undefined =
      this.instructions;

    if (
      this.additionalInstructions != null &&
      this.additionalInstructions !== ''
    ) {
      finalInstructions =
        finalInstructions != null && finalInstructions
          ? `${finalInstructions}\n\n${this.additionalInstructions}`
          : this.additionalInstructions;
    }

    // Handle Anthropic prompt caching
    if (
      finalInstructions != null &&
      finalInstructions !== '' &&
      this.provider === Providers.ANTHROPIC
    ) {
      const anthropicOptions = this.clientOptions as
        | t.AnthropicClientOptions
        | undefined;
      const defaultHeaders = anthropicOptions?.clientOptions?.defaultHeaders as
        | Record<string, string>
        | undefined;
      const anthropicBeta = defaultHeaders?.['anthropic-beta'];
      if (
        typeof anthropicBeta === 'string' &&
        anthropicBeta.includes('prompt-caching')
      ) {
        finalInstructions = {
          content: [
            {
              type: 'text',
              text: this.instructions,
              cache_control: { type: 'ephemeral' },
            },
          ],
        };
      }
    }

    if (finalInstructions != null && finalInstructions !== '') {
      const systemMessage = new SystemMessage(finalInstructions);

      if (this.tokenCounter) {
        this.instructionTokens += this.tokenCounter(systemMessage);
      }

      return RunnableLambda.from((messages: BaseMessage[]) => {
        return [systemMessage, ...messages];
      }).withConfig({ runName: 'prompt' });
    }

    return undefined;
  }

  /**
   * Reset context for a new run
   */
  reset(): void {
    this.instructionTokens = 0;
    this.lastToken = undefined;
    this.indexTokenCountMap = {};
    this.currentUsage = undefined;
    this.pruneMessages = undefined;
    this.lastStreamCall = undefined;
    this.tokenTypeSwitch = undefined;
    this.currentTokenType = ContentTypes.TEXT;
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
}
