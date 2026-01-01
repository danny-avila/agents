import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler } from '@/events';
import { Providers, GraphEvents } from '@/common';
import { sleep } from '@/utils/run';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];

/**
 * Test parallel handoffs - where an LLM calls multiple transfer tools simultaneously
 *
 * Graph structure:
 * coordinator -> [researcher, writer] (via parallel handoff tools)
 *
 * The coordinator agent has two transfer tools:
 * - transfer_to_researcher
 * - transfer_to_writer
 *
 * When given a task that needs both, it should call both tools in parallel.
 */
async function testParallelHandoffs() {
  console.log(
    'Testing Parallel Handoffs (LLM calling multiple transfers)...\n'
  );

  const { contentParts, aggregateContent } = createContentAggregator();

  const agents: t.AgentInputs[] = [
    {
      agentId: 'coordinator',
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4o-mini',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a COORDINATOR agent. Your job is to delegate tasks to specialized agents.

You have access to two transfer tools:
- transfer_to_researcher: For research and fact-finding tasks
- transfer_to_writer: For content creation and writing tasks

IMPORTANT: When a task requires BOTH research AND writing, you MUST call BOTH transfer tools SIMULTANEOUSLY in the same response. Do not call them sequentially.

For example, if asked to "research and write about X", call both transfers at once to enable parallel work.

When delegating, provide clear instructions to each agent about what they should do.`,
    },
    {
      agentId: 'researcher',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a RESEARCHER. When you receive a task:
1. Acknowledge the handoff
2. Provide concise research findings (100-150 words)
3. Start your response with "📚 RESEARCH FINDINGS:"`,
    },
    {
      agentId: 'writer',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a WRITER. When you receive a task:
1. Acknowledge the handoff
2. Provide creative content (100-150 words)
3. Start your response with "✍️ WRITTEN CONTENT:"`,
    },
  ];

  /**
   * Create handoff edges from coordinator to both researcher and writer.
   * These are separate edges so the LLM sees both transfer tools.
   */
  const edges: t.GraphEdge[] = [
    {
      from: 'coordinator',
      to: 'researcher',
      edgeType: 'handoff',
      description: 'Transfer to researcher for research and fact-finding tasks',
      prompt: 'Research task instructions',
    },
    {
      from: 'coordinator',
      to: 'writer',
      edgeType: 'handoff',
      description: 'Transfer to writer for content creation and writing tasks',
      prompt: 'Writing task instructions',
    },
  ];

  /** Track which agents are active and their timing */
  const activeAgents = new Set<string>();
  const agentTimings: Record<string, { start?: number; end?: number }> = {};
  const startTime = Date.now();

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const nodeName = metadata?.langgraph_node as string;
        if (nodeName) {
          const elapsed = Date.now() - startTime;
          agentTimings[nodeName] = agentTimings[nodeName] || {};
          agentTimings[nodeName].end = elapsed;
          activeAgents.delete(nodeName);
          console.log(`\n⏱️  [${nodeName}] COMPLETED at ${elapsed}ms`);
        }
      },
    },
    [GraphEvents.CHAT_MODEL_START]: {
      handle: (
        _event: string,
        _data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const nodeName = metadata?.langgraph_node as string;
        if (nodeName) {
          const elapsed = Date.now() - startTime;
          /** Store first start time for parallel overlap calculation */
          if (!agentTimings[nodeName]?.start) {
            agentTimings[nodeName] = agentTimings[nodeName] || {};
            agentTimings[nodeName].start = elapsed;
          }
          activeAgents.add(nodeName);
          console.log(`\n⏱️  [${nodeName}] STARTED at ${elapsed}ms`);
          console.log(
            `   Active agents: ${Array.from(activeAgents).join(', ')}`
          );
        }
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
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
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
  };

  const runConfig: t.RunConfig = {
    runId: `parallel-handoffs-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    customHandlers,
    returnContent: true,
  };

  try {
    const run = await Run.create(runConfig);

    /** Prompt designed to trigger parallel handoffs without confusing language */
    const userMessage = `Help me with two topics:
1. The history of the internet
2. A short poem about technology

I need information on both topics.`;

    conversationHistory.push(new HumanMessage(userMessage));

    console.log('User message:', userMessage);
    console.log(
      '\nInvoking multi-agent graph with parallel handoff request...\n'
    );

    const config = {
      configurable: {
        thread_id: 'parallel-handoffs-test-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const inputs = {
      messages: conversationHistory,
    };

    await run.processStream(inputs, config);
    const finalMessages = run.getRunMessages();

    if (finalMessages) {
      conversationHistory.push(...finalMessages);
    }

    /** Analyze parallel execution */
    console.log('\n\n========== TIMING SUMMARY ==========');
    console.log('Available timing keys:', Object.keys(agentTimings));
    for (const [agent, timing] of Object.entries(agentTimings)) {
      const duration =
        timing.end && timing.start ? timing.end - timing.start : 'N/A';
      console.log(
        `${agent}: started=${timing.start}ms, ended=${timing.end}ms, duration=${duration}ms`
      );
    }

    /** Check if researcher and writer ran in parallel (handle key variations) */
    const researcherKey = Object.keys(agentTimings).find((k) =>
      k.includes('researcher')
    );
    const writerKey = Object.keys(agentTimings).find((k) =>
      k.includes('writer')
    );
    const researcherTiming = researcherKey
      ? agentTimings[researcherKey]
      : undefined;
    const writerTiming = writerKey ? agentTimings[writerKey] : undefined;

    if (researcherTiming && writerTiming) {
      const bothStarted = researcherTiming.start && writerTiming.start;
      const bothEnded = researcherTiming.end && writerTiming.end;

      if (bothStarted && bothEnded) {
        const overlap =
          Math.min(researcherTiming.end!, writerTiming.end!) -
          Math.max(researcherTiming.start!, writerTiming.start!);

        if (overlap > 0) {
          console.log(
            `\n✅ PARALLEL HANDOFFS SUCCESSFUL: ${overlap}ms overlap between researcher and writer`
          );
        } else {
          console.log(
            `\n⚠️  SEQUENTIAL EXECUTION: researcher and writer did not overlap`
          );
          console.log(
            `   This may indicate the LLM called transfers sequentially, not in parallel`
          );
        }
      }
    } else {
      console.log(
        '\n⚠️  Not all agents were invoked. Check if handoffs occurred.'
      );
      console.log('   researcher timing:', researcherTiming);
      console.log('   writer timing:', writerTiming);
    }
    console.log('====================================\n');

    console.log('Final content parts:', contentParts.length, 'parts');
    console.dir(contentParts, { depth: null });
    await sleep(3000);
  } catch (error) {
    console.error('Error in parallel handoffs test:', error);
    throw error;
  }
}

testParallelHandoffs();
