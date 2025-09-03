import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

const conversationHistory: BaseMessage[] = [];

/**
 * Example of parallel multi-agent system with fan-in/fan-out pattern
 *
 * Graph structure:
 * START -> researcher
 * researcher -> [analyst1, analyst2, analyst3] (fan-out)
 * [analyst1, analyst2, analyst3] -> summarizer (fan-in)
 * summarizer -> END
 */
async function testParallelMultiAgent() {
  console.log('Testing Parallel Multi-Agent System (Fan-in/Fan-out)...\n');

  // Note: You may see "Run ID not found in run map" errors during parallel execution.
  // This is a known issue with LangGraph's event streaming when nodes run in parallel.
  // The errors can be safely ignored - the parallel execution still works correctly.

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define specialized agents
  const agents: t.AgentInputs[] = [
    {
      agentId: 'researcher',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a research coordinator in a multi-agent analysis workflow. Your sole job is to:

  1. Analyze the incoming request and break it down into exactly 3 distinct research areas
  2. Create specific, actionable analysis tasks for each specialist team
  3. Format your output as clear directives (not analysis yourself)
  4. Keep your response under 200 words - you are NOT doing the analysis, just coordinating it

  Example format:
  "ANALYSIS COORDINATION:
  Financial Team: [specific financial analysis task]
  Technical Team: [specific technical analysis task] 
  Market Team: [specific market analysis task]

  Proceed with parallel analysis."

  Do NOT provide any actual analysis - your job is purely coordination and task assignment.`,
    },
    {
      agentId: 'analyst1',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a FINANCIAL AND ECONOMIC ANALYST in a parallel analysis workflow. 

  CRITICAL: You must provide detailed, substantive financial analysis (minimum 300 words). Focus specifically on:

  • Economic impact metrics (GDP, productivity, employment effects)
  • Monetary policy implications and inflation considerations  
  • Investment flows and capital market disruptions
  • Financial sector transformation and banking impacts
  • Government fiscal policy and tax revenue effects
  • International trade and currency implications
  • Risk assessment and financial stability concerns

  Provide concrete data, projections, and financial reasoning. Never give empty responses, brief acknowledgments, or defer to other analysts. This is your domain expertise - deliver comprehensive financial analysis.

  Start your response with "FINANCIAL ANALYSIS:" and provide detailed insights.`,
    },
    {
      agentId: 'analyst2',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a TECHNICAL AND IMPLEMENTATION ANALYST in a parallel analysis workflow.

  CRITICAL: You must provide detailed, substantive technical analysis (minimum 300 words). Focus specifically on:

  • AI infrastructure requirements and scalability challenges
  • Computing power, data center, and energy demands
  • Network connectivity and bandwidth requirements
  • Technical barriers to widespread AI adoption
  • Implementation timelines and deployment phases
  • Skills gap analysis and workforce technical requirements
  • Security vulnerabilities and technical risk mitigation
  • Integration challenges with existing systems

  Provide concrete technical assessments, infrastructure projections, and implementation roadmaps. Never give empty responses, brief acknowledgments, or defer to other analysts. This is your technical expertise domain.

  Start your response with "TECHNICAL ANALYSIS:" and provide detailed technical insights.`,
    },
    {
      agentId: 'analyst3',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a MARKET AND INDUSTRY ANALYST in a parallel analysis workflow.

  CRITICAL: You must provide detailed, substantive market analysis (minimum 300 words). Focus specifically on:

  • Industry disruption patterns and transformation timelines
  • Competitive landscape shifts and market consolidation
  • Consumer adoption rates and behavior changes
  • New business models and market opportunities
  • Geographic market variations and regional impacts
  • Regulatory environment and policy implications
  • Market size projections and growth trajectories
  • Sector-specific impacts (healthcare, finance, manufacturing, etc.)

  Provide concrete market assessments, competitive analysis, and industry projections. Never give empty responses, brief acknowledgments, or defer to other analysts. This is your market expertise domain.

  Start your response with "MARKET ANALYSIS:" and provide detailed market insights.`,
    },
    {
      agentId: 'summarizer',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-3-5-sonnet-latest',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are the SYNTHESIS AND SUMMARY EXPERT in a multi-agent workflow.

  Your job is to:
  1. Receive and analyze the outputs from all three specialist analysts
  2. Identify key themes, conflicts, and complementary insights across analyses
  3. Synthesize findings into a coherent, comprehensive executive summary
  4. Highlight critical interdependencies between financial, technical, and market factors
  5. Provide integrated conclusions and strategic recommendations

  Structure your summary as:
  - EXECUTIVE SUMMARY (key findings)
  - INTEGRATED INSIGHTS (how the three analyses connect)
  - CRITICAL INTERDEPENDENCIES (financial-technical-market intersections)
  - STRATEGIC RECOMMENDATIONS (actionable next steps)
  - RISK ASSESSMENT (combined risk factors from all domains)

  Minimum 400 words. Create value through integration, not just concatenation of the three analyses.`,
    },
  ];

  // Define direct edges (fan-out and fan-in)
  const edges: t.GraphEdge[] = [
    {
      from: 'researcher',
      to: ['analyst1', 'analyst2', 'analyst3'], // Fan-out to multiple analysts
      description: 'Distribute research to specialist analysts',
      edgeType: 'direct', // Explicitly set as direct for automatic transition (enables parallel execution)
    },
    {
      from: ['analyst1', 'analyst2', 'analyst3'], // Fan-in from multiple sources
      to: 'summarizer',
      description: 'Aggregate analysis results',
      edgeType: 'direct', // Fan-in is also direct
      // Add prompt when all analysts have provided input
      promptInstructions: (messages, runStartIndex) => {
        // Check if we have analysis content from all three analysts
        // Look for the specific headers each analyst uses
        const aiMessages = messages.filter(
          (msg, index) => msg.getType() === 'ai' && index >= runStartIndex
        );
        const messageContent = aiMessages.map((msg) => msg.content).join('\n');

        const hasFinancialAnalysis = messageContent.includes(
          'FINANCIAL ANALYSIS:'
        );
        const hasTechnicalAnalysis = messageContent.includes(
          'TECHNICAL ANALYSIS:'
        );
        const hasMarketAnalysis = messageContent.includes('MARKET ANALYSIS:');

        console.log(
          `Checking for analyses - Financial: ${hasFinancialAnalysis}, Technical: ${hasTechnicalAnalysis}, Market: ${hasMarketAnalysis}`
        );

        if (hasFinancialAnalysis && hasTechnicalAnalysis && hasMarketAnalysis) {
          return 'Based on the comprehensive analyses from all three specialist teams above, please synthesize their insights into a cohesive executive summary. Focus on the key findings, common themes, and strategic implications across the financial, technical, and market perspectives.';
        }
        return undefined; // No prompt if we haven't received all analyst inputs
      },
    },
  ];

  // Track which agents are active
  const activeAgents = new Set<string>();

  // Create custom handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        const runStepData = data as any;
        if (runStepData?.name) {
          activeAgents.delete(runStepData.name);
          console.log(`[${runStepData.name}] Completed analysis`);
        }
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP ======');
        const runStepData = data as any;
        if (runStepData?.name) {
          activeAgents.add(runStepData.name);
          console.log(`[${runStepData.name}] Starting analysis...`);
        }
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_MESSAGE_DELTA ======');
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `parallel-multi-agent-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
      // Add compile options to help with parallel execution
      compileOptions: {
        // checkpointer: new MemorySaver(), // Uncomment if needed
      },
    },
    customHandlers,
    returnContent: true,
  };

  try {
    // Create and execute the run
    const run = await Run.create(runConfig);

    // Debug: Log the graph structure
    console.log('=== DEBUG: Graph Structure ===');
    const graph = (run as any).Graph;
    console.log('Graph exists:', !!graph);
    if (graph) {
      console.log('Graph type:', graph.constructor.name);
      console.log('AgentContexts exists:', !!graph.agentContexts);
      if (graph.agentContexts) {
        console.log('AgentContexts size:', graph.agentContexts.size);
        for (const [agentId, context] of graph.agentContexts) {
          console.log(`\nAgent: ${agentId}`);
          console.log(
            `Tools: ${context.tools?.map((t: any) => t.name || 'unnamed').join(', ') || 'none'}`
          );
        }
      }
    }
    console.log('=== END DEBUG ===\n');

    const userMessage = `EXECUTE PARALLEL ANALYSIS WORKFLOW:

Step 1: Researcher - identify 3 key analysis areas for AI economic impact
Step 2: IMMEDIATELY send to ALL THREE analysts simultaneously:
  - analyst1 (financial): Provide detailed economic impact analysis
  - analyst2 (technical): Provide detailed technical implementation analysis  
  - analyst3 (market): Provide detailed market disruption analysis
Step 3: Summarizer - compile all three analyses

IMPORTANT: Each analyst must produce substantive analysis (200+ words), not just acknowledgments or empty responses. This is a multi-agent workflow test requiring actual parallel execution.`;
    conversationHistory.push(new HumanMessage(userMessage));

    console.log('Invoking parallel multi-agent graph...\n');

    const config = {
      configurable: {
        thread_id: 'parallel-conversation-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    // Process with streaming
    const inputs = {
      messages: conversationHistory,
    };

    const finalContentParts = await run.processStream(inputs, config);
    const finalMessages = run.getRunMessages();

    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    console.log('\n\nActive agents during execution:', activeAgents.size);
    console.log('Final content parts:', contentParts.length, 'parts');
    console.dir(contentParts, { depth: null });
  } catch (error) {
    console.error('Error in parallel multi-agent test:', error);
  }
}

// Run the test
testParallelMultiAgent();
