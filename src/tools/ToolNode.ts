import {
  END,
  isCommand,
  isGraphInterrupt,
  MessagesAnnotation,
} from '@langchain/langgraph';
import { ToolCall } from '@langchain/core/messages/tool';
import { ToolMessage, isBaseMessage } from '@langchain/core/messages';
import type {
  RunnableConfig,
  RunnableToolLike,
} from '@langchain/core/runnables';
import type { BaseMessage, AIMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { Command } from '@langchain/langgraph';
import type * as t from '@/types';
import { RunnableCallable } from '@/utils';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export class ToolNode<T = any> extends RunnableCallable<T, T> {
  tools: t.GenericTool[];
  private toolMap: Map<string, StructuredToolInterface | RunnableToolLike>;
  private loadRuntimeTools?: t.ToolRefGenerator;
  handleToolErrors = true;
  toolCallStepIds?: Map<string, string>;
  errorHandler?: t.ToolNodeConstructorParams['errorHandler'];
  private toolUsageCount: Map<string, number>;

  constructor({
    tools,
    toolMap,
    name,
    tags,
    errorHandler,
    toolCallStepIds,
    handleToolErrors,
    loadRuntimeTools,
  }: t.ToolNodeConstructorParams) {
    super({ name, tags, func: (input, config) => this.run(input, config) });
    this.tools = tools;
    this.toolMap = toolMap ?? new Map(tools.map((tool) => [tool.name, tool]));
    this.toolCallStepIds = toolCallStepIds;
    this.handleToolErrors = handleToolErrors ?? this.handleToolErrors;
    this.loadRuntimeTools = loadRuntimeTools;
    this.errorHandler = errorHandler;
    this.toolUsageCount = new Map<string, number>();
  }

  /**
   * Returns a snapshot of the current tool usage counts.
   * @returns A ReadonlyMap where keys are tool names and values are their usage counts.
   */
  public getToolUsageCounts(): ReadonlyMap<string, number> {
    return new Map(this.toolUsageCount); // Return a copy
  }

  /**
   * Runs a single tool call with error handling
   */
  protected async runTool(
    call: ToolCall,
    config: RunnableConfig
  ): Promise<BaseMessage | Command> {
    const tool = this.toolMap.get(call.name);
    try {
      if (tool === undefined) {
        throw new Error(`Tool "${call.name}" not found.`);
      }
      const turn = this.toolUsageCount.get(call.name) ?? 0;
      this.toolUsageCount.set(call.name, turn + 1);
      const args = call.args;
      const stepId = this.toolCallStepIds?.get(call.id!);
      const output = await tool.invoke(
        { ...call, args, type: 'tool_call', stepId, turn },
        config
      );
      if (
        (isBaseMessage(output) && output._getType() === 'tool') ||
        isCommand(output)
      ) {
        return output;
      } else {
        return new ToolMessage({
          name: tool.name,
          content: typeof output === 'string' ? output : JSON.stringify(output),
          tool_call_id: call.id!,
        });
      }
    } catch (_e: unknown) {
      const e = _e as Error;
      if (!this.handleToolErrors) {
        throw e;
      }
      if (isGraphInterrupt(e)) {
        throw e;
      }
      if (this.errorHandler) {
        await this.errorHandler(
          {
            error: e,
            id: call.id!,
            name: call.name,
            input: call.args,
          },
          config.metadata
        );
      }
      return new ToolMessage({
        content: `Error: ${e.message}\n Please fix your mistakes.`,
        name: call.name,
        tool_call_id: call.id ?? '',
      });
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  protected async run(input: any, config: RunnableConfig): Promise<T> {
    const message = Array.isArray(input)
      ? input[input.length - 1]
      : input.messages[input.messages.length - 1];

    if (message._getType() !== 'ai') {
      throw new Error('ToolNode only accepts AIMessages as input.');
    }

    if (this.loadRuntimeTools) {
      const { tools, toolMap } = this.loadRuntimeTools(
        (message as AIMessage).tool_calls ?? []
      );
      this.tools = tools;
      this.toolMap = toolMap ?? new Map(tools.map((tool) => [tool.name, tool]));
    }
    const outputs = await Promise.all(
      (message as AIMessage).tool_calls?.map((call) =>
        this.runTool(call, config)
      ) ?? []
    );

    if (!outputs.some(isCommand)) {
      return (Array.isArray(input) ? outputs : { messages: outputs }) as T;
    }

    const combinedOutputs = outputs.map((output) => {
      if (isCommand(output)) {
        return output;
      }
      return Array.isArray(input) ? [output] : { messages: [output] };
    });
    return combinedOutputs as T;
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
  const message: AIMessage = Array.isArray(state)
    ? state[state.length - 1]
    : state.messages[state.messages.length - 1];

  if (
    'tool_calls' in message &&
    (message.tool_calls?.length ?? 0) > 0 &&
    !areToolCallsInvoked(message, invokedToolIds)
  ) {
    return toolNode;
  } else {
    return END;
  }
}
