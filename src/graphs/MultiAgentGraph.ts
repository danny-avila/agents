import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { ToolMessage, HumanMessage } from '@langchain/core/messages';
import {
  StateGraph,
  Command,
  START,
  END,
  getCurrentTaskInput,
  Annotation,
  messagesStateReducer,
} from '@langchain/langgraph';
import type { ToolRunnableConfig } from '@langchain/core/tools';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { StandardGraph } from './Graph';

/**
 * MultiAgentGraph extends StandardGraph to support dynamic multi-agent workflows
 * with handoffs, fan-in/fan-out, and other composable patterns
 */
export class MultiAgentGraph extends StandardGraph {
  private edges: t.GraphEdge[];
  private startingNodes: Set<string> = new Set();
  private parallelEdges: t.GraphEdge[] = [];
  private handoffEdges: t.GraphEdge[] = [];

  constructor(input: t.MultiAgentGraphInput) {
    super(input);
    this.edges = input.edges;

    // Categorize edges by type
    this.categorizeEdges();

    // Determine starting nodes and create handoff tools
    this.analyzeGraph();
    this.createHandoffTools();
  }

  /**
   * Categorize edges into handoff and parallel types
   */
  private categorizeEdges(): void {
    for (const edge of this.edges) {
      // Default behavior: edges with conditions or explicit 'handoff' type are handoff edges
      // Edges with explicit 'parallel' type or multi-destination without conditions are parallel edges
      if (edge.edgeType === 'parallel') {
        this.parallelEdges.push(edge);
      } else if (edge.edgeType === 'handoff' || edge.condition != null) {
        this.handoffEdges.push(edge);
      } else {
        // Default: single-to-single edges are handoff, single-to-multiple are parallel
        const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];

        if (sources.length === 1 && destinations.length > 1) {
          // Fan-out pattern defaults to parallel
          this.parallelEdges.push(edge);
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
      for (const edge of edges) {
        handoffTools.push(...this.createHandoffToolsForEdge(edge));
      }

      // Add handoff tools to the agent's existing tools
      if (!agentContext.tools) {
        agentContext.tools = [];
      }
      agentContext.tools.push(...handoffTools);

      // Update tool map
      for (const tool of handoffTools) {
        if (!agentContext.toolMap) {
          agentContext.toolMap = new Map();
        }
        agentContext.toolMap.set(tool.name, tool);
      }
    }
  }

  /**
   * Create handoff tools for an edge (handles multiple destinations)
   */
  private createHandoffToolsForEdge(edge: t.GraphEdge): t.GenericTool[] {
    const tools: t.GenericTool[] = [];
    const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];

    // If there's a condition, create a single conditional handoff tool
    if (edge.condition != null) {
      const toolName = 'conditional_transfer';
      const toolDescription =
        edge.description ?? 'Conditionally transfer control based on state';

      tools.push(
        tool(
          async (_, config) => {
            const state = getCurrentTaskInput() as t.BaseGraphState;
            const toolCallId =
              (config as ToolRunnableConfig | undefined)?.toolCall?.id ??
              'unknown';

            // Evaluate condition
            const result = edge.condition!(state);
            let destination: string;

            if (typeof result === 'boolean') {
              // If true, use first destination; if false, don't transfer
              if (!result) return null;
              destination = destinations[0];
            } else if (typeof result === 'string') {
              destination = result;
            } else {
              // Array of destinations - for now, use the first
              destination = Array.isArray(result) ? result[0] : destinations[0];
            }

            const toolMessage = new ToolMessage({
              content: `Conditionally transferred to ${destination}`,
              name: toolName,
              tool_call_id: toolCallId,
            });

            return new Command({
              goto: destination,
              update: { messages: state.messages.concat(toolMessage) },
              graph: Command.PARENT,
            });
          },
          {
            name: toolName,
            schema: z.object({}),
            description: toolDescription,
          }
        )
      );
    } else {
      // Create individual tools for each destination
      for (const destination of destinations) {
        const toolName = `transfer_to_${destination}`;
        const toolDescription =
          edge.description ?? `Transfer control to agent '${destination}'`;

        tools.push(
          tool(
            async (_, config) => {
              const toolCallId =
                (config as ToolRunnableConfig | undefined)?.toolCall?.id ??
                'unknown';
              const toolMessage = new ToolMessage({
                content: `Successfully transferred to ${destination}`,
                name: toolName,
                tool_call_id: toolCallId,
              });

              const state = getCurrentTaskInput() as t.BaseGraphState;

              return new Command({
                goto: destination,
                update: { messages: state.messages.concat(toolMessage) },
                graph: Command.PARENT,
              });
            },
            {
              name: toolName,
              schema: z.object({}),
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
    // This is essentially the same as createAgentNode from StandardGraph
    return this.createAgentNode(agentId);
  }

  /**
   * Create the multi-agent workflow with dynamic handoffs
   */
  override createWorkflow(): t.CompiledStateWorkflow {
    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (...args) => {
          const result = messagesStateReducer(...args);
          this.messages = result;
          return result;
        },
        default: () => [],
      }),
    });

    const builder = new StateGraph(StateAnnotation);

    // Add all agents as complete subgraphs
    for (const [agentId] of this.agentContexts) {
      // Get all possible destinations for this agent
      const handoffDestinations = new Set<string>();
      const parallelDestinations = new Set<string>();

      // Check handoff edges for destinations
      for (const edge of this.handoffEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (sources.includes(agentId) === true) {
          const dests = Array.isArray(edge.to) ? edge.to : [edge.to];
          dests.forEach((dest) => handoffDestinations.add(dest));
        }
      }

      // Check parallel edges for destinations
      for (const edge of this.parallelEdges) {
        const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
        if (sources.includes(agentId) === true) {
          const dests = Array.isArray(edge.to) ? edge.to : [edge.to];
          dests.forEach((dest) => parallelDestinations.add(dest));
        }
      }

      // If agent has handoff destinations, add END to possible ends
      // If agent only has parallel destinations, it naturally ends without explicit END
      const destinations = new Set([...handoffDestinations]);
      if (handoffDestinations.size > 0 || parallelDestinations.size === 0) {
        destinations.add(END);
      }

      // Create the agent subgraph (includes agent + tools)
      const agentSubgraph = this.createAgentSubgraph(agentId);

      // Add the agent as a node with its possible destinations
      builder.addNode(agentId, agentSubgraph, {
        ends: Array.from(destinations),
      });
    }

    // Add starting edges for all starting nodes
    for (const startNode of this.startingNodes) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      /** @ts-ignore */
      builder.addEdge(START, startNode);
    }

    // Add direct edges for parallel execution
    for (const edge of this.parallelEdges) {
      const sources = Array.isArray(edge.from) ? edge.from : [edge.from];
      const destinations = Array.isArray(edge.to) ? edge.to : [edge.to];

      // For each source, add edges to all destinations (fan-out)
      for (const source of sources) {
        for (const destination of destinations) {
          if (
            edge.promptInstructions != null &&
            edge.promptInstructions !== ''
          ) {
            // Create a wrapper node that adds the prompt before the destination
            const wrapperNodeId = `${source}_to_${destination}_prompt`;

            builder.addNode(wrapperNodeId, async (state: t.BaseGraphState) => {
              let promptText: string | undefined;

              if (typeof edge.promptInstructions === 'function') {
                promptText = edge.promptInstructions(state.messages);
              } else {
                promptText = edge.promptInstructions;
              }

              if (promptText != null && promptText !== '') {
                // Return state with the prompt message added
                return {
                  messages: [...state.messages, new HumanMessage(promptText)],
                };
              }

              // No prompt needed, return empty update
              return {};
            });

            // Add edges through the wrapper
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            /** @ts-ignore */
            builder.addEdge(source, wrapperNodeId);
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            /** @ts-ignore */
            builder.addEdge(wrapperNodeId, destination);
          } else {
            // No prompt, direct edge
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            /** @ts-ignore */
            builder.addEdge(source, destination);
          }
        }
      }
    }

    // Compile with options
    return builder.compile(this.compileOptions as unknown as never);
  }

  /**
   * Override createGraphState to return appropriate state for multi-agent
   */
  override createGraphState(): t.GraphStateChannels<t.BaseGraphState> {
    return {
      messages: {
        value: (x: BaseMessage[], y: BaseMessage[]): BaseMessage[] => {
          if (!x.length) {
            this.startIndex = x.length + y.length;
          }
          const current = x.concat(y);
          this.messages = current;
          return current;
        },
        default: () => [],
      },
    };
  }
}
