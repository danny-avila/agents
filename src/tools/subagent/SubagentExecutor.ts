import { nanoid } from 'nanoid';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { AggregatedHookResult, HookRegistry } from '@/hooks';
import type {
  AgentInputs,
  ResolvedSubagentConfig,
  SubagentConfig,
  TokenCounter,
} from '@/types';
import type { AgentContext } from '@/agents/AgentContext';
import { StandardGraph } from '@/graphs/Graph';
import { executeHooks } from '@/hooks';

const DEFAULT_MAX_TURNS = 25;
const RECURSION_MULTIPLIER = 3;

const HOOK_FALLBACK: AggregatedHookResult = Object.freeze({
  additionalContexts: [] as string[],
  errors: [] as string[],
});

export type SubagentExecuteParams = {
  description: string;
  subagentType: string;
  threadId?: string;
};

export type SubagentExecuteResult = {
  content: string;
  messages: BaseMessage[];
};

export type SubagentExecutorOptions = {
  configs: Map<string, ResolvedSubagentConfig>;
  parentSignal?: AbortSignal;
  hookRegistry?: HookRegistry;
  parentRunId: string;
  parentAgentId?: string;
  tokenCounter?: TokenCounter;
  depth?: number;
  maxDepth?: number;
};

export class SubagentExecutor {
  private readonly configs: Map<string, ResolvedSubagentConfig>;
  private readonly parentSignal?: AbortSignal;
  private readonly hookRegistry?: HookRegistry;
  private readonly parentRunId: string;
  private readonly parentAgentId?: string;
  private readonly tokenCounter?: TokenCounter;
  private readonly depth: number;
  private readonly maxDepth: number;

  constructor(options: SubagentExecutorOptions) {
    this.configs = options.configs;
    this.parentSignal = options.parentSignal;
    this.hookRegistry = options.hookRegistry;
    this.parentRunId = options.parentRunId;
    this.parentAgentId = options.parentAgentId;
    this.tokenCounter = options.tokenCounter;
    this.depth = options.depth ?? 0;
    this.maxDepth = options.maxDepth ?? 1;
  }

  async execute(params: SubagentExecuteParams): Promise<SubagentExecuteResult> {
    const { description, subagentType, threadId } = params;
    const config = this.configs.get(subagentType);

    if (!config) {
      const available = [...this.configs.keys()].join(', ');
      return {
        content: `Error: Unknown subagent type "${subagentType}". Available types: ${available}`,
        messages: [],
      };
    }

    if (this.depth >= this.maxDepth) {
      return {
        content: `Error: Maximum subagent nesting depth (${this.maxDepth}) exceeded.`,
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

      if (hookResult.decision === 'deny' || hookResult.decision === 'ask') {
        return {
          content: `Blocked: ${hookResult.reason ?? 'Blocked by hook'}`,
          messages: [],
        };
      }
    }

    const childInputs = buildChildInputs(config, childAgentId);
    const childRunId = `${this.parentRunId}_sub_${nanoid(8)}`;
    const maxTurns = config.maxTurns ?? DEFAULT_MAX_TURNS;

    const childGraph = new StandardGraph({
      runId: childRunId,
      signal: this.parentSignal,
      agents: [childInputs],
      tokenCounter: this.tokenCounter,
    });

    let result: { messages: BaseMessage[] };
    try {
      const workflow = childGraph.createWorkflow();
      result = await workflow.invoke(
        { messages: [new HumanMessage(description)] },
        {
          recursionLimit: maxTurns * RECURSION_MULTIPLIER,
          signal: this.parentSignal,
        }
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      childGraph.clearHeavyState();
      return {
        content: `Subagent error: ${message}`,
        messages: [],
      };
    }

    const filteredContent = filterSubagentResult(result.messages);

    if (
      this.hookRegistry?.hasHookFor('SubagentStop', this.parentRunId) === true
    ) {
      executeHooks({
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

    childGraph.clearHeavyState();

    return { content: filteredContent, messages: result.messages };
  }
}

/**
 * Extract the last AI message's text content, stripping non-text blocks
 * (tool_use, thinking, redacted_thinking, tool_result).
 * Returns "Task completed" when no valid text content is found.
 */
export function filterSubagentResult(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]._getType() !== 'ai') {
      continue;
    }

    const content = messages[i].content;

    if (typeof content === 'string') {
      return content || 'Task completed';
    }

    if (!Array.isArray(content)) {
      return 'Task completed';
    }

    const textParts: string[] = [];
    for (const block of content) {
      if (typeof block === 'string') {
        textParts.push(block);
      } else if ('type' in block && block.type === 'text' && 'text' in block) {
        textParts.push(block.text as string);
      }
    }

    return textParts.join('\n') || 'Task completed';
  }

  return 'Task completed';
}

/**
 * Resolve self-spawn configs by filling in agentInputs from the parent context.
 * Returns configs with agentInputs guaranteed present.
 */
export function resolveSubagentConfigs(
  configs: SubagentConfig[],
  parentContext: AgentContext
): ResolvedSubagentConfig[] {
  return configs
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
}

/** Build child AgentInputs from a resolved config, stripping nesting and event-driven fields. */
function buildChildInputs(
  config: ResolvedSubagentConfig,
  childAgentId: string
): AgentInputs {
  const { agentInputs } = config;
  const childInputs: AgentInputs = {
    ...agentInputs,
    agentId: childAgentId,
    toolDefinitions: undefined,
  };

  if (config.allowNested !== true) {
    childInputs.subagentConfigs = undefined;
    childInputs.maxSubagentDepth = undefined;
  }

  return childInputs;
}
