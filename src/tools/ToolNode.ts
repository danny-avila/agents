import { ToolCall } from '@langchain/core/messages/tool';
import {
  ToolMessage,
  isAIMessage,
  isBaseMessage,
} from '@langchain/core/messages';
import {
  END,
  Send,
  Command,
  isCommand,
  isGraphInterrupt,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type {
  RunnableConfig,
  RunnableToolLike,
} from '@langchain/core/runnables';
import type { BaseMessage, AIMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { RunnableCallable } from '@/utils';
import { Constants } from '@/common';

/**
 * Helper to check if a value is a Send object
 */
function isSend(value: unknown): value is Send {
  return value instanceof Send;
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
  /** Tool registry for filtering (lazy computation of programmatic maps) */
  private toolRegistry?: t.LCToolRegistry;
  /** Cached programmatic tools (computed once on first PTC call) */
  private programmaticCache?: t.ProgrammaticCache;
  /** Reference to Graph's sessions map for automatic session injection */
  private sessions?: t.ToolSessionMap;
  /** Whether the agent supports vision capabilities (for artifact filtering) */
  private visionCapable: boolean;

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
    visionCapable,
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
    this.visionCapable = visionCapable ?? false;
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
   * Checks if an image_url content item contains base64 data (not an HTTP URL).
   * Base64 data URLs start with "data:" and can cause context overflow when sent to non-vision models.
   */
  private isBase64ImageUrl(item: unknown): boolean {
    if (
      typeof item !== 'object' ||
      item === null ||
      !('type' in item) ||
      item.type !== 'image_url'
    ) {
      return false;
    }

    const imageUrlItem = item as t.ImageUrlContent;
    const imageUrl = imageUrlItem.image_url;

    if (typeof imageUrl === 'string') {
      return imageUrl.startsWith('data:');
    }

    if (typeof imageUrl === 'object' && imageUrl && 'url' in imageUrl) {
      return (
        typeof imageUrl.url === 'string' && imageUrl.url.startsWith('data:')
      );
    }

    return false;
  }

  /**
   * Processes MCP artifact content by:
   * 1. Converting MCP image format to standard image_url format
   * 2. Filtering images if vision is disabled
   */
  private processArtifact(artifact: unknown): t.MCPArtifact | undefined {
    if (
      artifact === null ||
      artifact === undefined ||
      typeof artifact !== 'object' ||
      !('content' in artifact) ||
      !Array.isArray(artifact.content)
    ) {
      return undefined;
    }

    if (typeof this.visionCapable !== 'boolean') {
      console.warn(
        '[ToolNode] Invalid visionCapable value, defaulting to false',
        {
          visionCapable: this.visionCapable,
        }
      );
      this.visionCapable = false;
    }

    const artifactObj = artifact as t.MCPArtifact;

    // Convert MCP format (type: 'image' with data:) to image_url format
    artifactObj.content = artifactObj.content.map((item) => {
      if (
        typeof item === 'object' &&
        item !== null &&
        'type' in item &&
        item.type === 'image' &&
        'data' in item
      ) {
        const mimeType = item.mimeType ?? 'image/png';
        const dataUrl =
          typeof item.data === 'string' && item.data.startsWith('http')
            ? item.data
            : `data:${mimeType};base64,${item.data}`;
        return {
          type: 'image_url' as const,
          image_url: { url: dataUrl },
        };
      }
      return item;
    });

    if (!this.visionCapable) {
      artifactObj.content = artifactObj.content.filter((item) => {
        if (
          item &&
          typeof item === 'object' &&
          'type' in item &&
          item.type === 'image_url'
        ) {
          return false;
        }
        return true;
      });
    }

    return artifactObj;
  }

  /**
   * Runs a single tool call with error handling
   */
  protected async runTool(
    call: ToolCall,
    config: RunnableConfig
  ): Promise<BaseMessage | Command> {
    let tool = this.toolMap.get(call.name);

    // If tool not found and loadRuntimeTools is available, try loading it
    if (tool === undefined && this.loadRuntimeTools) {
      const { tools, toolMap } = this.loadRuntimeTools([call]);
      this.toolMap = toolMap ?? new Map(tools.map((t) => [t.name, t]));
      this.programmaticCache = undefined; // Invalidate cache on toolMap change
      tool = this.toolMap.get(call.name);
    }

    try {
      if (tool === undefined) {
        const availableTools = Array.from(this.toolMap.keys()).join(', ');
        throw new Error(
          `Tool "${call.name}" not found. Available tools: ${availableTools || 'none'}. ` +
            'If this is a runtime tool, ensure loadRuntimeTools is properly configured.'
        );
      }
      const turn = this.toolUsageCount.get(call.name) ?? 0;
      this.toolUsageCount.set(call.name, turn + 1);
      const args = call.args;
      const stepId = this.toolCallStepIds?.get(call.id!);

      // Build invoke params - LangChain extracts non-schema fields to config.toolCall
      let invokeParams: Record<string, unknown> = {
        ...call,
        args,
        type: 'tool_call',
        stepId,
        turn,
      };

      // Inject runtime data for special tools (becomes available at config.toolCall)
      if (call.name === Constants.PROGRAMMATIC_TOOL_CALLING) {
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
       */
      if (
        call.name === Constants.EXECUTE_CODE ||
        call.name === Constants.PROGRAMMATIC_TOOL_CALLING
      ) {
        const codeSession = this.sessions?.get(Constants.EXECUTE_CODE) as
          | t.CodeSessionContext
          | undefined;
        if (codeSession?.files != null && codeSession.files.length > 0) {
          /**
           * Convert tracked files to CodeEnvFile format for the API.
           * Each file uses its own session_id (set when file was created).
           * This supports files from multiple parallel/sequential executions.
           */
          const fileRefs: t.CodeEnvFile[] = codeSession.files.map((file) => ({
            session_id: file.session_id ?? codeSession.session_id,
            id: file.id,
            name: file.name,
          }));
          /** Inject latest session_id and files - bypasses Zod, reaches config.toolCall */
          invokeParams = {
            ...invokeParams,
            session_id: codeSession.session_id,
            _injected_files: fileRefs,
          };
        }
      }

      // Invoke tool (standard path for all tools)
      const output = await tool.invoke(invokeParams, config);

      // Handle MCP tuple [content, artifact]
      if (
        Array.isArray(output) &&
        output.length === 2 &&
        output[1] !== null &&
        output[1] !== undefined
      ) {
        const [content, artifact] = output;
        const processedArtifact = this.processArtifact(artifact);

        const toolMessage = new ToolMessage({
          status: 'success',
          name: tool.name,
          content:
            typeof content === 'string' ? content : JSON.stringify(content),
          tool_call_id: call.id!,
          additional_kwargs: processedArtifact
            ? { artifact: processedArtifact }
            : {},
        });
        return toolMessage;
      }

      // Handle ToolMessage output (from tools with responseFormat: 'content_and_artifact')
      if (isBaseMessage(output) && output._getType() === 'tool') {
        const toolMessage = output as ToolMessage;

        // If ToolMessage already has artifact, ensure it's in additional_kwargs for formatArtifactPayload
        if (toolMessage.artifact && !toolMessage.additional_kwargs.artifact) {
          toolMessage.additional_kwargs.artifact = toolMessage.artifact;
        }

        return toolMessage;
      }

      // Handle Command output
      if (isCommand(output)) {
        return output;
      }

      // Default: create ToolMessage without artifact
      return new ToolMessage({
        status: 'success',
        name: tool.name,
        content: typeof output === 'string' ? output : JSON.stringify(output),
        tool_call_id: call.id!,
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
      return new ToolMessage({
        status: 'error',
        content: `Error: ${e.message}\n Please fix your mistakes.`,
        name: call.name,
        tool_call_id: call.id ?? '',
      });
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  protected async run(input: any, config: RunnableConfig): Promise<T> {
    let outputs: (BaseMessage | Command)[];

    if (this.isSendInput(input)) {
      outputs = [await this.runTool(input.lg_tool_call, config)];
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

      outputs = await Promise.all(
        aiMessage.tool_calls
          ?.filter((call) => {
            /**
             * Filter out:
             * 1. Already processed tool calls (present in toolMessageIds)
             * 2. Server tool calls (e.g., web_search with IDs starting with 'srvtoolu_')
             *    which are executed by the provider's API and don't require invocation
             */
            return (
              (call.id == null || !toolMessageIds.has(call.id)) &&
              !(call.id?.startsWith('srvtoolu_') ?? false)
            );
          })
          .map((call) => this.runTool(call, config)) ?? []
      );
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
