import { tool } from '@langchain/core/tools';
import { PromptTemplate } from '@langchain/core/prompts';
import {
  AIMessage,
  ToolMessage,
  HumanMessage,
  getBufferString,
} from '@langchain/core/messages';
import {
  END,
  START,
  Command,
  StateGraph,
  Annotation,
  getCurrentTaskInput,
} from '@langchain/langgraph';
import { messagesStateReducer } from '@/messages/reducer';
import type { LangGraphRunnableConfig } from '@langchain/langgraph';
import type { BaseMessage, AIMessageChunk } from '@langchain/core/messages';
import type { ToolRunnableConfig } from '@langchain/core/tools';
import type * as t from '@/types';
import { StandardGraph } from './Graph';
import { Constants } from '@/common';

/** Pattern to extract instructions from transfer ToolMessage content */
const HANDOFF_INSTRUCTIONS_PATTERN = /(?:Instructions?|Context):\s*(.+)/is;

/**
 * MultiAgentGraph extends StandardGraph to support dynamic multi-agent workflows
 * with handoffs, fan-in/fan-out, and other composable patterns.
 *
 * Key behavior:
 * - Agents with ONLY handoff edges: Can dynamically route to any handoff destination
 * - Agents with ONLY direct edges: Always follow their direct edges
 * - Agents with BOTH: Use Command for exclusive routing (handoff OR direct, not both)
 *   - If handoff occurs: Only the handoff destination executes
 *   - If no handoff: Direct edges execute (potentially in parallel)
 *
 * This enables the common pattern where an agent either delegates (handoff)
 * OR continues its workflow (direct edges), but not both simultaneously.
 */
export class MultiAgentGraph extends StandardGraph {
  private edges: t.GraphEdge[];
  private startingNodes: Set<string> = new Set();
  private directEdges: t.GraphEdge[] = [];
  private handoffEdges: t.GraphEdge[] = [];
  /**
   * Map of agentId to parallel group info.
   * Contains groupId (incrementing number reflecting execution order) for agents in parallel groups.
   * Sequential agents (not in any parallel group) have undefined entry.
   *
   * Example for: researcher -> [analyst1, analyst2, analyst3] -> summarizer
   * - researcher: undefined (sequential, order 0)
   * - analyst1, analyst2, analyst3: { groupId: 1 } (parallel group, order 1)
   * - summarizer: undefined (sequential, order 2)
   */
  private agentParallelGroups: Map<string, number> = new Map();

  constructor(input: t.MultiAgentGraphInput) {
    super(input);
    this.edges = input.edges;
    this.categorizeEdges();
    this.analyzeGraph();
    this.createHandoffTools();
  }

  /**
   * Categorize edges into handoff and direct types
   */
  private categorizeEdges(): void {
    for (const edge of this.edges) {
      // Default behavior: edges with conditions or explicit 'handoff' type are handoff edges
      // Edges with explicit 'direct' type or multi-destination without conditions are direct edges
      if (edge.edgeType === 'direct') {
        this.directEdges.push(edge);
      } else if (edge.edgeType === 'handoff' || edge.condition != null) {
        this.handoffEdges.push(edge);
      } else {
        // Default: single-to-single edges are handoff, single-to-multiple are direct
        const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];

        if (sources.length === 1 && destinations.length > 1) {
          // Fan-out pattern defaults to direct
          this.directEdges.push(edge);
        } else {
          // Everything else defaults to handoff
          this.handoffEdges.push(edge);
        }
      }
    }
  }

  /**
   * Analyze graph structure to determine starting nodes and connections
   */
  private analyzeGraph(): void {
    const hasIncomingEdge = new Set<string>();

    // Track all nodes that have incoming edges
    for (const edge of this.edges) {
      const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
      destinations.forEach((dest) => hasIncomingEdge.add(dest));
    }

    // Starting nodes are those without incoming edges
    for (const agentId of this.agentContexts.keys()) {
      if (!hasIncomingEdge.has(agentId)) {
        this.startingNodes.add(agentId);
      }
    }

    // If no starting nodes found, use the first agent
    if (this.startingNodes.size === 0 && this.agentContexts.size > 0) {
      this.startingNodes.add(this.agentContexts.keys().next().value!);
    }

    // Determine if graph has parallel execution capability
    this.computeParallelCapability();
  }

  /**
   * Compute parallel groups by traversing the graph in execution order.
   * Assigns incrementing group IDs that reflect the sequential order of execution.
   *
   * For: researcher -> [analyst1, analyst2, analyst3] -> summarizer
   * - researcher: no group (first sequential node)
   * - analyst1, analyst2, analyst3: groupId 1 (first parallel group)
   * - summarizer: no group (next sequential node)
   *
   * This allows frontend to render in order:
   * Row 0: researcher
   * Row 1: [analyst1, analyst2, analyst3] (grouped)
   * Row 2: summarizer
   */
  private computeParallelCapability(): void {
    let groupCounter = 1; // Start at 1, 0 reserved for "no group"

    // Check 1: Multiple starting nodes means parallel from the start (group 1)
    if (this.startingNodes.size > 1) {
      for (const agentId of this.startingNodes) {
        this.agentParallelGroups.set(agentId, groupCounter);
      }
      groupCounter++;
    }

    // Check 2: Traverse direct edges in order to find fan-out patterns
    // Build a simple execution order by following edges from starting nodes
    const visited = new Set<string>();
    const queue: string[] = [...this.startingNodes];

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current)) continue;
      visited.add(current);

      // Find direct edges from this node
      for (const edge of this.directEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (!sources.includes(current)) continue;

        const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];

        // Fan-out: multiple destinations = parallel group
        if (destinations.length > 1) {
          for (const dest of destinations) {
            // Only set if not already in a group (first group wins)
            if (!this.agentParallelGroups.has(dest)) {
              this.agentParallelGroups.set(dest, groupCounter);
            }
            if (!visited.has(dest)) {
              queue.push(dest);
            }
          }
          groupCounter++;
        } else {
          // Single destination - add to queue for traversal
          for (const dest of destinations) {
            if (!visited.has(dest)) {
              queue.push(dest);
            }
          }
        }
      }

      // Also follow handoff edges for traversal (but they don't create parallel groups)
      for (const edge of this.handoffEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (!sources.includes(current)) continue;

        const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
        for (const dest of destinations) {
          if (!visited.has(dest)) {
            queue.push(dest);
          }
        }
      }
    }
  }

  /**
   * Get the parallel group ID for an agent, if any.
   * Returns undefined if the agent is not part of a parallel group.
   * Group IDs are incrementing numbers reflecting execution order.
   */
  getParallelGroupId(agentId: string): number | undefined {
    return this.agentParallelGroups.get(agentId);
  }

  /**
   * Override to indicate this is a multi-agent graph.
   * Enables agentId to be included in RunStep for frontend agent labeling.
   */
  protected override isMultiAgentGraph(): boolean {
    return true;
  }

  /**
   * Override base class method to provide parallel group IDs for run steps.
   */
  protected override getParallelGroupIdForAgent(
    agentId: string
  ): number | undefined {
    return this.agentParallelGroups.get(agentId);
  }

  /**
   * Create handoff tools for agents based on handoff edges only
   */
  private createHandoffTools(): void {
    // Group handoff edges by source agent(s)
    const handoffsByAgent = new Map<string, t.GraphEdge[]>();

    // Only process handoff edges for tool creation
    for (const edge of this.handoffEdges) {
      const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
      sources.forEach((source) => {
        if (!handoffsByAgent.has(source)) {
          handoffsByAgent.set(source, []);
        }
        handoffsByAgent.get(source)!.push(edge);
      });
    }

    // Create handoff tools for each agent
    for (const [agentId, edges] of handoffsByAgent) {
      const agentContext = this.agentContexts.get(agentId);
      if (!agentContext) continue;

      // Create handoff tools for this agent's outgoing edges
      const handoffTools: t.GenericTool[] = [];
      const sourceAgentName = agentContext.name ?? agentId;
      for (const edge of edges) {
        handoffTools.push(
          ...this.createHandoffToolsForEdge(edge, agentId, sourceAgentName)
        );
      }

      if (!agentContext.graphTools) {
        agentContext.graphTools = [];
      }
      agentContext.graphTools.push(...handoffTools);
    }
  }

  /**
   * Create handoff tools for an edge (handles multiple destinations)
   * @param edge - The graph edge defining the handoff
   * @param sourceAgentId - The ID of the agent that will perform the handoff
   * @param sourceAgentName - The human-readable name of the source agent
   */
  private createHandoffToolsForEdge(
    edge: t.GraphEdge,
    sourceAgentId: string,
    sourceAgentName: string
  ): t.GenericTool[] {
    const tools: t.GenericTool[] = [];
    const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];

    /** If there's a condition, create a single conditional handoff tool */
    if (edge.condition != null) {
      const toolName = 'conditional_transfer';
      const toolDescription =
        edge.description ?? 'Conditionally transfer control based on state';

      /** Check if we have a prompt for handoff input */
      const hasHandoffInput =
        edge.prompt != null && typeof edge.prompt === 'string';
      const handoffInputDescription = hasHandoffInput ? edge.prompt : undefined;
      const promptKey = edge.promptKey ?? 'instructions';

      tools.push(
        tool(
          async (rawInput, config) => {
            const input = rawInput as Record<string, unknown>;
            const state = getCurrentTaskInput() as t.BaseGraphState;
            const toolCallId =
              (config as ToolRunnableConfig | undefined)?.toolCall?.id ??
              'unknown';

            /** Evaluated condition */
            const result = edge.condition!(state);
            let destination: string;

            if (typeof result === 'boolean') {
              /** If true, use first destination; if false, don't transfer */
              if (!result) return null;
              destination = destinations[0];
            } else if (typeof result === 'string') {
              destination = result;
            } else {
              /** Array of destinations - for now, use the first */
              destination = Array.isArray(result) ? result[0] : destinations[0];
            }

            let content = `Conditionally transferred to ${destination}`;
            if (
              hasHandoffInput &&
              promptKey in input &&
              input[promptKey] != null
            ) {
              content += `\n\n${promptKey.charAt(0).toUpperCase() + promptKey.slice(1)}: ${input[promptKey]}`;
            }

            const toolMessage = new ToolMessage({
              content,
              name: toolName,
              tool_call_id: toolCallId,
              additional_kwargs: {
                /** Store destination for programmatic access in handoff detection */
                handoff_destination: destination,
                /** Store source agent name for receiving agent to know who handed off */
                handoff_source_name: sourceAgentName,
              },
            });

            return new Command({
              goto: destination,
              update: { messages: state.messages.concat(toolMessage) },
              graph: Command.PARENT,
            });
          },
          {
            name: toolName,
            schema: hasHandoffInput
              ? {
                type: 'object',
                properties: {
                  [promptKey]: {
                    type: 'string',
                    description: handoffInputDescription as string,
                  },
                },
                required: [],
              }
              : { type: 'object', properties: {}, required: [] },
            description: toolDescription,
          }
        )
      );
    } else {
      /** Create individual tools for each destination */
      for (const destination of destinations) {
        const toolName = `${Constants.LC_TRANSFER_TO_}${destination}`;
        const toolDescription =
          edge.description ?? `Transfer control to agent '${destination}'`;

        /** Check if we have a prompt for handoff input */
        const hasHandoffInput =
          edge.prompt != null && typeof edge.prompt === 'string';
        const handoffInputDescription = hasHandoffInput
          ? edge.prompt
          : undefined;
        const promptKey = edge.promptKey ?? 'instructions';

        tools.push(
          tool(
            async (rawInput, config) => {
              const input = rawInput as Record<string, unknown>;
              const toolCallId =
                (config as ToolRunnableConfig | undefined)?.toolCall?.id ??
                'unknown';

              let content = `Successfully transferred to ${destination}`;
              if (
                hasHandoffInput &&
                promptKey in input &&
                input[promptKey] != null
              ) {
                content += `\n\n${promptKey.charAt(0).toUpperCase() + promptKey.slice(1)}: ${input[promptKey]}`;
              }

              const toolMessage = new ToolMessage({
                content,
                name: toolName,
                tool_call_id: toolCallId,
                additional_kwargs: {
                  /** Store source agent name for receiving agent to know who handed off */
                  handoff_source_name: sourceAgentName,
                },
              });

              const state = getCurrentTaskInput() as t.BaseGraphState;

              /**
               * For parallel handoff support:
               * Build messages that include ONLY this tool call's context.
               * This prevents errors when LLM calls multiple transfers simultaneously -
               * each destination gets a valid AIMessage with matching tool_call and tool_result.
               *
               * Strategy:
               * 1. Find the AIMessage containing this tool call
               * 2. Create a filtered AIMessage with ONLY this tool_call
               * 3. Include all messages before the AIMessage plus the filtered pair
               */
              const messages = state.messages;
              let filteredMessages = messages;
              let aiMessageIndex = -1;

              /** Find the AIMessage containing this tool call */
              for (let i = messages.length - 1; i >= 0; i--) {
                const msg = messages[i];
                if (msg.getType() === 'ai') {
                  const aiMsg = msg as AIMessage;
                  const hasThisCall = aiMsg.tool_calls?.some(
                    (tc) => tc.id === toolCallId
                  );
                  if (hasThisCall === true) {
                    aiMessageIndex = i;
                    break;
                  }
                }
              }

              if (aiMessageIndex >= 0) {
                const originalAiMsg = messages[aiMessageIndex] as AIMessage;
                const thisToolCall = originalAiMsg.tool_calls?.find(
                  (tc) => tc.id === toolCallId
                );

                if (
                  thisToolCall != null &&
                  (originalAiMsg.tool_calls?.length ?? 0) > 1
                ) {
                  /**
                   * Multiple tool calls - create filtered AIMessage with ONLY this call.
                   * This ensures valid message structure for parallel handoffs.
                   */
                  const filteredAiMsg = new AIMessage({
                    content: originalAiMsg.content,
                    tool_calls: [thisToolCall],
                    id: originalAiMsg.id,
                  });

                  filteredMessages = [
                    ...messages.slice(0, aiMessageIndex),
                    filteredAiMsg,
                    toolMessage,
                  ];
                } else {
                  /** Single tool call - use messages as-is */
                  filteredMessages = messages.concat(toolMessage);
                }
              } else {
                /** Fallback - append tool message */
                filteredMessages = messages.concat(toolMessage);
              }

              return new Command({
                goto: destination,
                update: { messages: filteredMessages },
                graph: Command.PARENT,
              });
            },
            {
              name: toolName,
              schema: hasHandoffInput
                ? {
                  type: 'object',
                  properties: {
                    [promptKey]: {
                      type: 'string',
                      description: handoffInputDescription as string,
                    },
                  },
                  required: [],
                }
                : { type: 'object', properties: {}, required: [] },
              description: toolDescription,
            }
          )
        );
      }
    }

    return tools;
  }

  /**
   * Create a complete agent subgraph (similar to createReactAgent)
   */
  private createAgentSubgraph(agentId: string): t.CompiledAgentWorfklow {
    /** This is essentially the same as `createAgentNode` from `StandardGraph` */
    return this.createAgentNode(agentId);
  }

  /**
   * Detects if the current agent is receiving a handoff and processes the messages accordingly.
   * Returns filtered messages with the transfer tool call/message removed, plus any instructions,
   * source agent, and parallel sibling information extracted from the transfer.
   *
   * Supports both single handoffs (last message is the transfer) and parallel handoffs
   * (multiple transfer ToolMessages, need to find the one targeting this agent).
   *
   * @param messages - Current state messages
   * @param agentId - The agent ID to check for handoff reception
   * @returns Object with filtered messages, extracted instructions, source agent, and parallel siblings
   */
  private processHandoffReception(
    messages: BaseMessage[],
    agentId: string
  ): {
    filteredMessages: BaseMessage[];
    instructions: string | null;
    sourceAgentName: string | null;
    parallelSiblings: string[];
  } | null {
    if (messages.length === 0) return null;

    /**
     * Search for a transfer ToolMessage targeting this agent.
     * For parallel handoffs, multiple transfer messages may exist - find ours.
     * Search backwards from the end to find the most recent transfer to this agent.
     */
    let toolMessage: ToolMessage | null = null;
    let toolMessageIndex = -1;

    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.getType() !== 'tool') continue;

      const candidateMsg = msg as ToolMessage;
      const toolName = candidateMsg.name;

      if (typeof toolName !== 'string') continue;

      /** Check for standard transfer pattern */
      const isTransferMessage = toolName.startsWith(Constants.LC_TRANSFER_TO_);
      const isConditionalTransfer = toolName === 'conditional_transfer';

      if (!isTransferMessage && !isConditionalTransfer) continue;

      /** Extract destination from tool name or additional_kwargs */
      let destinationAgent: string | null = null;

      if (isTransferMessage) {
        destinationAgent = toolName.replace(Constants.LC_TRANSFER_TO_, '');
      } else if (isConditionalTransfer) {
        const handoffDest = candidateMsg.additional_kwargs.handoff_destination;
        destinationAgent = typeof handoffDest === 'string' ? handoffDest : null;
      }

      /** Check if this transfer targets our agent */
      if (destinationAgent === agentId) {
        toolMessage = candidateMsg;
        toolMessageIndex = i;
        break;
      }
    }

    /** No transfer targeting this agent found */
    if (toolMessage === null || toolMessageIndex < 0) return null;

    /** Extract instructions from the ToolMessage content */
    const contentStr =
      typeof toolMessage.content === 'string'
        ? toolMessage.content
        : JSON.stringify(toolMessage.content);

    const instructionsMatch = contentStr.match(HANDOFF_INSTRUCTIONS_PATTERN);
    const instructions = instructionsMatch?.[1]?.trim() ?? null;

    /** Extract source agent name from additional_kwargs */
    const handoffSourceName = toolMessage.additional_kwargs.handoff_source_name;
    const sourceAgentName =
      typeof handoffSourceName === 'string' ? handoffSourceName : null;

    /** Extract parallel siblings (set by ToolNode for parallel handoffs) */
    const rawSiblings = toolMessage.additional_kwargs.handoff_parallel_siblings;
    const siblingIds: string[] = Array.isArray(rawSiblings)
      ? rawSiblings.filter((s): s is string => typeof s === 'string')
      : [];
    /** Convert IDs to display names */
    const parallelSiblings = siblingIds.map((id) => {
      const ctx = this.agentContexts.get(id);
      return ctx?.name ?? id;
    });

    /** Get the tool_call_id to find and filter the AI message's tool call */
    const toolCallId = toolMessage.tool_call_id;

    /**
     * Collect all transfer tool_call_ids to filter out.
     * For parallel handoffs, we filter ALL transfer messages (not just ours)
     * to give the receiving agent a clean context without handoff noise.
     */
    const transferToolCallIds = new Set<string>([toolCallId]);
    for (const msg of messages) {
      if (msg.getType() !== 'tool') continue;
      const tm = msg as ToolMessage;
      const tName = tm.name;
      if (typeof tName !== 'string') continue;
      if (
        tName.startsWith(Constants.LC_TRANSFER_TO_) ||
        tName === 'conditional_transfer'
      ) {
        transferToolCallIds.add(tm.tool_call_id);
      }
    }

    /** Filter out all transfer messages */
    const filteredMessages: BaseMessage[] = [];

    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      const msgType = msg.getType();

      /** Skip transfer ToolMessages */
      if (msgType === 'tool') {
        const tm = msg as ToolMessage;
        if (transferToolCallIds.has(tm.tool_call_id)) {
          continue;
        }
      }

      if (msgType === 'ai') {
        /** Check if this AI message contains any transfer tool calls */
        const aiMsg = msg as AIMessage | AIMessageChunk;
        const toolCalls = aiMsg.tool_calls;

        if (toolCalls && toolCalls.length > 0) {
          /** Filter out all transfer tool calls */
          const remainingToolCalls = toolCalls.filter(
            (tc) => tc.id == null || !transferToolCallIds.has(tc.id)
          );

          const hasTransferCalls = remainingToolCalls.length < toolCalls.length;

          if (hasTransferCalls) {
            if (
              remainingToolCalls.length > 0 ||
              (typeof aiMsg.content === 'string' && aiMsg.content.trim())
            ) {
              /** Keep the message but without transfer tool calls */
              const filteredAiMsg = new AIMessage({
                content: aiMsg.content,
                tool_calls: remainingToolCalls,
                id: aiMsg.id,
              });
              filteredMessages.push(filteredAiMsg);
            }
            /** If no remaining content or tool calls, skip this message entirely */
            continue;
          }
        }
      }

      /** Keep all other messages */
      filteredMessages.push(msg);
    }

    return {
      filteredMessages,
      instructions,
      sourceAgentName,
      parallelSiblings,
    };
  }

  /**
   * Create the multi-agent workflow with dynamic handoffs
   */
  override createWorkflow(): t.CompiledMultiAgentWorkflow {
    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (a, b) => {
          if (!a.length) {
            this.startIndex = a.length + b.length;
          }
          const result = messagesStateReducer(a, b);
          this.messages = result;
          return result;
        },
        default: () => [],
      }),
      /** Channel for passing filtered messages to agents when excludeResults is true */
      agentMessages: Annotation<BaseMessage[]>({
        /** Replaces state entirely */
        reducer: (a, b) => b,
        default: () => [],
      }),
    });

    const builder = new StateGraph(StateAnnotation);

    // Add all agents as complete subgraphs
    for (const [agentId] of this.agentContexts) {
      // Get all possible destinations for this agent
      const handoffDestinations = new Set<string>();
      const directDestinations = new Set<string>();

      // Check handoff edges for destinations
      for (const edge of this.handoffEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (sources.includes(agentId) === true) {
          const dests = Array.isArray(edge.to) ? edge.to : [edge.to];
          dests.forEach((dest) => handoffDestinations.add(dest));
        }
      }

      // Check direct edges for destinations
      for (const edge of this.directEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (sources.includes(agentId) === true) {
          const dests = Array.isArray(edge.to) ? edge.to : [edge.to];
          dests.forEach((dest) => directDestinations.add(dest));
        }
      }

      /** Check if this agent has BOTH handoff and direct edges */
      const hasHandoffEdges = handoffDestinations.size > 0;
      const hasDirectEdges = directDestinations.size > 0;
      const needsCommandRouting = hasHandoffEdges && hasDirectEdges;

      /** Collect all possible destinations for this agent */
      const allDestinations = new Set([
        ...handoffDestinations,
        ...directDestinations,
      ]);
      if (handoffDestinations.size > 0 || directDestinations.size === 0) {
        allDestinations.add(END);
      }

      /** Agent subgraph (includes agent + tools) */
      const agentSubgraph = this.createAgentSubgraph(agentId);

      /** Wrapper function that handles agentMessages channel, handoff reception, and conditional routing */
      const agentWrapper = async (
        state: t.MultiAgentGraphState,
        config?: LangGraphRunnableConfig
      ): Promise<t.MultiAgentGraphState | Command> => {
        let result: t.MultiAgentGraphState;

        /**
         * Check if this agent is receiving a handoff.
         * If so, filter out the transfer messages and inject instructions as preamble.
         * This prevents the receiving agent from seeing the transfer as "completed work"
         * and prematurely producing an end token.
         */
        const handoffContext = this.processHandoffReception(
          state.messages,
          agentId
        );

        if (handoffContext !== null) {
          const {
            filteredMessages,
            instructions,
            sourceAgentName,
            parallelSiblings,
          } = handoffContext;

          /**
           * Set handoff context on the receiving agent.
           * Uses pre-computed graph position for depth and parallel info.
           */
          const agentContext = this.agentContexts.get(agentId);
          if (
            agentContext &&
            sourceAgentName != null &&
            sourceAgentName !== ''
          ) {
            agentContext.setHandoffContext(sourceAgentName, parallelSiblings);
          }

          /** Build messages for the receiving agent */
          let messagesForAgent = filteredMessages;

          /**
           * If there are instructions, inject them as a HumanMessage to
           * ground the receiving agent.
           *
           * When the last filtered message is a ToolMessage (e.g. from a
           * non-handoff tool the router called before handing off), a
           * synthetic AIMessage is inserted first to satisfy the
           * tool → assistant role ordering required by chat APIs.  Without
           * this bridge, appending a HumanMessage directly after a
           * ToolMessage causes "400 Unexpected role 'user' after role
           * 'tool'" errors (see issue #54).
           */
          const hasInstructions = instructions !== null && instructions !== '';
          if (hasInstructions) {
            const lastMsg =
              filteredMessages.length > 0
                ? filteredMessages[filteredMessages.length - 1]
                : null;

            if (lastMsg != null && lastMsg.getType() === 'tool') {
              messagesForAgent = [
                ...filteredMessages,
                new AIMessage(
                  `[Processed tool result and transferring to ${agentId}]`
                ),
                new HumanMessage(instructions),
              ];
            } else {
              messagesForAgent = [
                ...filteredMessages,
                new HumanMessage(instructions),
              ];
            }
          }

          /** Update token map if we have a token counter */
          if (agentContext?.tokenCounter && hasInstructions) {
            const freshTokenMap: Record<string, number> = {};
            for (
              let i = 0;
              i < Math.min(filteredMessages.length, this.startIndex);
              i++
            ) {
              const tokenCount = agentContext.indexTokenCountMap[i];
              if (tokenCount !== undefined) {
                freshTokenMap[i] = tokenCount;
              }
            }
            /** Add tokens for the bridge AIMessage + instructions HumanMessage */
            for (
              let i = filteredMessages.length;
              i < messagesForAgent.length;
              i++
            ) {
              freshTokenMap[i] = agentContext.tokenCounter(messagesForAgent[i]);
            }
            agentContext.updateTokenMapWithInstructions(freshTokenMap);
          }

          const transformedState: t.MultiAgentGraphState = {
            ...state,
            messages: messagesForAgent,
          };
          result = await agentSubgraph.invoke(transformedState, config);
          result = {
            ...result,
            agentMessages: [],
          };
        } else if (
          state.agentMessages != null &&
          state.agentMessages.length > 0
        ) {
          /**
           * When using agentMessages (excludeResults=true), we need to update
           * the token map to account for the new prompt message
           */
          const agentContext = this.agentContexts.get(agentId);
          if (agentContext && agentContext.tokenCounter) {
            /** The agentMessages contains:
             * 1. Filtered messages (0 to startIndex) - already have token counts
             * 2. New prompt message - needs token counting
             */
            const freshTokenMap: Record<string, number> = {};

            /** Copy existing token counts for filtered messages (0 to startIndex) */
            for (let i = 0; i < this.startIndex; i++) {
              const tokenCount = agentContext.indexTokenCountMap[i];
              if (tokenCount !== undefined) {
                freshTokenMap[i] = tokenCount;
              }
            }

            /** Calculate tokens only for the new prompt message (last message) */
            const promptMessageIndex = state.agentMessages.length - 1;
            if (promptMessageIndex >= this.startIndex) {
              const promptMessage = state.agentMessages[promptMessageIndex];
              freshTokenMap[promptMessageIndex] =
                agentContext.tokenCounter(promptMessage);
            }

            /** Update the agent's token map with instructions added */
            agentContext.updateTokenMapWithInstructions(freshTokenMap);
          }

          /** Temporary state with messages replaced by `agentMessages` */
          const transformedState: t.MultiAgentGraphState = {
            ...state,
            messages: state.agentMessages,
          };
          result = await agentSubgraph.invoke(transformedState, config);
          result = {
            ...result,
            /** Clear agentMessages for next agent */
            agentMessages: [],
          };
        } else {
          result = await agentSubgraph.invoke(state, config);
        }

        /** If agent has both handoff and direct edges, use Command for exclusive routing */
        if (needsCommandRouting) {
          /** Check if a handoff occurred */
          const lastMessage = result.messages[
            result.messages.length - 1
          ] as BaseMessage | null;
          if (
            lastMessage != null &&
            lastMessage.getType() === 'tool' &&
            typeof lastMessage.name === 'string' &&
            lastMessage.name.startsWith(Constants.LC_TRANSFER_TO_)
          ) {
            /** Handoff occurred - extract destination and navigate there exclusively */
            const handoffDest = lastMessage.name.replace(
              Constants.LC_TRANSFER_TO_,
              ''
            );
            return new Command({
              update: result,
              goto: handoffDest,
            });
          } else {
            /** No handoff - proceed with direct edges */
            const directDests = Array.from(directDestinations);
            if (directDests.length === 1) {
              return new Command({
                update: result,
                goto: directDests[0],
              });
            } else if (directDests.length > 1) {
              /** Multiple direct destinations - they'll run in parallel */
              return new Command({
                update: result,
                goto: directDests,
              });
            }
          }
        }

        /** No special routing needed - return state normally */
        return result;
      };

      /** Wrapped agent as a node with its possible destinations */
      builder.addNode(agentId, agentWrapper, {
        ends: Array.from(allDestinations),
      });
    }

    // Add starting edges for all starting nodes
    for (const startNode of this.startingNodes) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      builder.addEdge(START, startNode);
    }

    /**
     * Add direct edges for automatic transitions
     * Group edges by destination to handle fan-in scenarios
     */
    const edgesByDestination = new Map<string, t.GraphEdge[]>();

    for (const edge of this.directEdges) {
      const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
      for (const destination of destinations) {
        if (!edgesByDestination.has(destination)) {
          edgesByDestination.set(destination, []);
        }
        edgesByDestination.get(destination)!.push(edge);
      }
    }

    for (const [destination, edges] of edgesByDestination) {
      /** Checks if this is a fan-in scenario with prompt instructions */
      const edgesWithPrompt = edges.filter(
        (edge) => edge.prompt != null && edge.prompt !== ''
      );

      if (edgesWithPrompt.length > 0) {
        /**
         * Single wrapper node for destination (Fan-in with prompt)
         */
        const wrapperNodeId = `fan_in_${destination}_prompt`;
        /**
         * First edge's `prompt`
         * (they should all be the same for fan-in)
         */
        const prompt = edgesWithPrompt[0].prompt;
        /**
         * First edge's `excludeResults` flag
         * (they should all be the same for fan-in)
         */
        const excludeResults = edgesWithPrompt[0].excludeResults;

        builder.addNode(wrapperNodeId, async (state: t.BaseGraphState) => {
          let promptText: string | undefined;
          let effectiveExcludeResults = excludeResults;

          if (typeof prompt === 'function') {
            promptText = await prompt(state.messages, this.startIndex);
          } else if (prompt != null) {
            if (prompt.includes('{results}')) {
              const resultsMessages = state.messages.slice(this.startIndex);
              const resultsString = getBufferString(resultsMessages);
              const promptTemplate = PromptTemplate.fromTemplate(prompt);
              const result = await promptTemplate.invoke({
                results: resultsString,
              });
              promptText = result.value;
              effectiveExcludeResults =
                excludeResults !== false && promptText !== '';
            } else {
              promptText = prompt;
            }
          }

          if (promptText != null && promptText !== '') {
            if (
              effectiveExcludeResults == null ||
              effectiveExcludeResults === false
            ) {
              return {
                messages: [new HumanMessage(promptText)],
              };
            }

            /** When `excludeResults` is true, use agentMessages channel
             * to pass filtered messages + prompt to the destination agent
             */
            const filteredMessages = state.messages.slice(0, this.startIndex);
            return {
              messages: [new HumanMessage(promptText)],
              agentMessages: messagesStateReducer(filteredMessages, [
                new HumanMessage(promptText),
              ]),
            };
          }

          /** No prompt needed, return empty update */
          return {};
        });

        /** Add edges from all sources to the wrapper, then wrapper to destination */
        for (const edge of edges) {
          const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
          for (const source of sources) {
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            /** @ts-ignore */
            builder.addEdge(source, wrapperNodeId);
          }
        }

        /** Single edge from wrapper to destination */
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        /** @ts-ignore */
        builder.addEdge(wrapperNodeId, destination);
      } else {
        /** No prompt instructions, add direct edges (skip if source uses Command routing) */
        for (const edge of edges) {
          const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
          for (const source of sources) {
            /** Check if this source node has both handoff and direct edges */
            const sourceHandoffEdges = this.handoffEdges.filter((e) => {
              const eSources = Array.isArray(e.from) ? e.from : [e.from];
              return eSources.includes(source);
            });
            const sourceDirectEdges = this.directEdges.filter((e) => {
              const eSources = Array.isArray(e.from) ? e.from : [e.from];
              return eSources.includes(source);
            });

            /** Skip adding edge if source uses Command routing (has both types) */
            if (sourceHandoffEdges.length > 0 && sourceDirectEdges.length > 0) {
              continue;
            }

            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            /** @ts-ignore */
            builder.addEdge(source, destination);
          }
        }
      }
    }

    return builder.compile(this.compileOptions as unknown as never);
  }
}
