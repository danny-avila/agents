import type { ToolCall, ToolMessage } from '@langchain/core/messages/tool';
import type { RunnableConfig } from '@langchain/core/runnables';

export const SUBAGENT_REPLAY_CONTROLLER = Symbol.for(
  '@librechat/agents/subagent-replay-controller'
);

export type SettledSubagentToolOutput = {
  output: ToolMessage;
  additionalContexts: string[];
  resolvedArgs?: Record<string, unknown>;
  referenceContent?: string;
};

export interface SubagentReplayController {
  getSettledOutput(
    call: ToolCall,
    config: RunnableConfig
  ): Promise<SettledSubagentToolOutput | undefined>;
  persistSettledOutput(
    call: ToolCall,
    config: RunnableConfig,
    settled: SettledSubagentToolOutput
  ): Promise<void>;
}

export type ReplayableSubagentTool = {
  [SUBAGENT_REPLAY_CONTROLLER]?: SubagentReplayController;
};
