import {
  END,
  MessagesAnnotation,
  isCommand,
  isGraphInterrupt,
} from '@langchain/langgraph';
import { ToolMessage, isBaseMessage } from '@langchain/core/messages';
import type {
  RunnableConfig,
  RunnableToolLike,
} from '@langchain/core/runnables';
import type { BaseMessage, AIMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { RunnableCallable } from '@/utils';
import { GraphNodeKeys } from '@/common';

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
      (message as AIMessage).tool_calls?.map(async (call) => {
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

          // Filter out content items with metadata property.
          // This property is used to pass UI Resources from the tool response to the frontend.
          // But it should not be sent to the LLM since it's not part of the tool response schema.
          if (
            isBaseMessage(output) &&
            output.content &&
            Array.isArray(output.content)
          ) {
            output.content = output.content.filter(
              (item: t.MessageContentComplex) =>
                typeof item === 'string' || !('metadata' in item)
            );
          }

          if (
            (isBaseMessage(output) && output._getType() === 'tool') ||
            isCommand(output)
          ) {
            return output;
          } else {
            return new ToolMessage({
              name: tool.name,
              content:
                typeof output === 'string' ? output : JSON.stringify(output),
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
          this.errorHandler?.(
            {
              error: e,
              id: call.id!,
              name: call.name,
              input: call.args,
            },
            config.metadata
          );
          return new ToolMessage({
            content: `Error: ${e.message}\n Please fix your mistakes.`,
            name: call.name,
            tool_call_id: call.id ?? '',
          });
        }
      }) ?? []
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

export function toolsCondition(
  state: BaseMessage[] | typeof MessagesAnnotation.State,
  invokedToolIds?: Set<string>
): 'tools' | typeof END {
  const message: AIMessage = Array.isArray(state)
    ? state[state.length - 1]
    : state.messages[state.messages.length - 1];

  if (
    'tool_calls' in message &&
    (message.tool_calls?.length ?? 0) > 0 &&
    !areToolCallsInvoked(message, invokedToolIds)
  ) {
    return GraphNodeKeys.TOOLS;
  } else {
    return END;
  }
}
