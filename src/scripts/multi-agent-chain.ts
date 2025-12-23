import { config } from 'dotenv';
config();

import {
  HumanMessage,
  BaseMessage,
  getBufferString,
} from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

/**
 * Helper function to create sequential chain edges with buffer string prompts
 *
 * @param agentIds - Array of agent IDs in order of execution
 * @returns Array of edges configured for sequential chain with buffer prompts
 */
function createSequentialChainEdges(agentIds: string[]): t.GraphEdge[] {
  const edges: t.GraphEdge[] = [];

  for (let i = 0; i < agentIds.length - 1; i++) {
    const fromAgent = agentIds[i];
    const toAgent = agentIds[i + 1];

    edges.push({
      from: fromAgent,
      to: toAgent,
      edgeType: 'direct',
      // Use a prompt function to create the buffer string from all previous results
      prompt: (messages: BaseMessage[], startIndex: number) => {
        // Get only the messages from this run (after startIndex)
        const runMessages = messages.slice(startIndex);

        // Create buffer string from run messages
        const bufferString = getBufferString(runMessages);

        // Format the prompt for the next agent
        return `Based on the following conversation and analysis from previous agents, please provide your insights:\n\n${bufferString}\n\nPlease add your specific expertise and perspective to this discussion.`;
      },
      // Critical: exclude previous results so only the prompt is passed
      excludeResults: true,
      description: `Sequential chain from ${fromAgent} to ${toAgent}`,
    });
  }

  return edges;
}

const conversationHistory: BaseMessage[] = [];

/**
 * Example of sequential agent chain mimicking the old chain behavior
 *
 * Graph structure:
 * START -> researcher -> analyst -> reviewer -> summarizer -> END
 *
 * Each agent receives a buffer string of all previous results
 */
async function testSequentialAgentChain() {
  console.log('Testing Sequential Agent Chain (Old Chain Pattern)...\n');

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define four agents with specific roles
  const agents: t.AgentInputs[] = [
    {
      agentId: 'researcher',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a Research Agent specializing in gathering initial information.
      Your role is to:
      1. Identify key aspects of the user's query
      2. List important factors to consider
      3. Provide initial research findings
      
      Format your response with clear sections and bullet points.
      Start with "RESEARCH FINDINGS:" and be thorough but concise.`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'analyst',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are an Analysis Agent that builds upon research findings.
      Your role is to:
      1. Analyze the research provided by the previous agent
      2. Identify patterns, risks, and opportunities
      3. Provide deeper analytical insights
      
      Start with "ANALYSIS:" and structure your response with clear categories.
      Reference specific points from the research when relevant.`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'reviewer',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a Critical Review Agent that evaluates the work done so far.
      Your role is to:
      1. Review the research and analysis from previous agents
      2. Identify any gaps or areas that need more attention
      3. Suggest improvements or additional considerations
      
      Start with "CRITICAL REVIEW:" and be constructive in your feedback.
      Highlight both strengths and areas for improvement.`,
      maxContextTokens: 8000,
    },
    {
      agentId: 'summarizer',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a Summary Agent that creates the final comprehensive output.
      Your role is to:
      1. Synthesize all insights from the researcher, analyst, and reviewer
      2. Create a cohesive, actionable summary
      3. Provide clear recommendations or conclusions
      
      Start with "EXECUTIVE SUMMARY:" followed by key sections.
      End with "KEY RECOMMENDATIONS:" or "CONCLUSIONS:" as appropriate.`,
      maxContextTokens: 8000,
    },
  ];

  // Create sequential chain edges using our helper function
  const agentIds = agents.map((a) => a.agentId);
  const edges = createSequentialChainEdges(agentIds);

  // Track agent progression
  let currentAgent = '';
  const startTime = Date.now();
  let messageCount = 0;

  // Create custom handlers with extensive metadata logging
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_END METADATA ======');
        console.dir(metadata, { depth: null });
        const elapsed = Date.now() - startTime;
        console.log(`⏱️  COMPLETED at ${elapsed}ms`);
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== CHAT_MODEL_START METADATA ======');
        console.dir(metadata, { depth: null });
        const elapsed = Date.now() - startTime;
        console.log(`⏱️  STARTED at ${elapsed}ms`);
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const runStepData = data as any;
        console.log('\n====== ON_RUN_STEP ======');
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        if (runStepData?.name) {
          currentAgent = runStepData.name;
          console.log(`\n→ ${currentAgent} is processing...`);
        }
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const runStepData = data as any;
        console.log('\n====== ON_RUN_STEP_COMPLETED ======');
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        if (runStepData?.name) {
          console.log(`✓ ${runStepData.name} completed`);
        }
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        console.log('\n====== ON_RUN_STEP_DELTA ======');
        console.log('DATA:');
        console.dir(data, { depth: null });
        console.log('METADATA:');
        console.dir(metadata, { depth: null });
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        messageCount++;
        // Only log first few message deltas to avoid spam
        if (messageCount <= 3) {
          console.log('\n====== ON_MESSAGE_DELTA ======');
          console.log('DATA:');
          console.dir(data, { depth: null });
          console.log('METADATA:');
          console.dir(metadata, { depth: null });
        }
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  // Create multi-agent run configuration
  const runConfig: t.RunConfig = {
    runId: `sequential-chain-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    customHandlers,
    returnContent: true,
  };

  try {
    // Create and execute the run
    const run = await Run.create(runConfig);

    // Test with a complex question that benefits from multiple perspectives
    const userMessage =
      'I want to launch a new mobile app for personal finance management. What should I consider?';
    conversationHistory.push(new HumanMessage(userMessage));

    console.log(`User: "${userMessage}"\n`);
    console.log('Starting sequential agent chain...\n');

    const config = {
      configurable: {
        thread_id: 'sequential-chain-1',
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

    console.log('\n\n=== Final Output ===');
    console.log('Sequential chain completed successfully!');
    console.log(`Total content parts: ${contentParts.length}`);

    // Display the buffer accumulation
    console.log('\n=== Agent Outputs (Buffer Accumulation) ===');

    const aiMessages = conversationHistory.filter(
      (msg) => msg._getType() === 'ai'
    );

    aiMessages.forEach((msg, index) => {
      console.log(`\n--- Agent ${index + 1}: ${agentIds[index]} ---`);
      console.log(msg.content);

      // Show buffer preview for next agent (except last)
      if (index < aiMessages.length - 1) {
        const bufferSoFar = getBufferString(aiMessages.slice(0, index + 1));
        console.log(
          `\n[Buffer passed to ${agentIds[index + 1]}]: ${bufferSoFar.slice(0, 150)}...`
        );
      }
    });

    // Demonstrate that each agent built upon previous results
    console.log('\n=== Chain Analysis ===');
    console.log('1. Researcher provided initial findings');
    console.log('2. Analyst built upon research with deeper insights');
    console.log('3. Reviewer critiqued and identified gaps');
    console.log('4. Summarizer synthesized everything into actionable output');
  } catch (error) {
    console.error('Error in sequential agent chain test:', error);
  }
}

// Run the test
testSequentialAgentChain();
