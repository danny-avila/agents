import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

/**
 * Test LLM steering quality after handoff with system prompt instructions.
 *
 * Validates that the receiving agent clearly understands:
 * 1. WHO it is (its role/identity)
 * 2. WHAT the task is (instructions from the handoff)
 * 3. WHO transferred control (source agent context)
 *
 * Uses specific, verifiable instructions so we can check the output.
 */
async function testHandoffSteering(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Test: Handoff Steering Quality (System Prompt Instructions)');
  console.log('='.repeat(60));

  const { contentParts, aggregateContent } = createContentAggregator();

  let currentAgent = '';
  const agentResponses: Record<string, string> = {};

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        const runStep = data as t.RunStep;
        if (runStep.agentId) {
          currentAgent = runStep.agentId;
          console.log(`\n[Agent: ${currentAgent}] Processing...`);
        }
        aggregateContent({ event, data: runStep });
      },
    },
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
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        _metadata?: Record<string, unknown>
      ): void => {
        const toolData = data as { name?: string };
        if (toolData?.name?.includes('transfer_to_')) {
          const specialist = toolData.name.replace('lc_transfer_to_', '');
          console.log(`\n  >> Handoff to: ${specialist}`);
        }
      },
    },
  };

  /**
   * Test 1: Basic handoff with specific task instructions
   * The specialist should clearly follow the coordinator's instructions.
   */
  async function test1_basicInstructions(): Promise<void> {
    console.log('\n' + '-'.repeat(60));
    console.log('TEST 1: Basic handoff with specific task instructions');
    console.log('-'.repeat(60));

    const agents: t.AgentInputs[] = [
      {
        agentId: 'coordinator',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a Task Coordinator. When a user makes a request:
1. Analyze what they need
2. Transfer to the specialist with SPECIFIC instructions about what to do

IMPORTANT: Always use the transfer tool. Do not try to do the work yourself.`,
        maxContextTokens: 8000,
      },
      {
        agentId: 'specialist',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a Technical Specialist. You provide detailed technical responses.
When you receive a task, execute it thoroughly. Always identify yourself as the Technical Specialist in your response.`,
        maxContextTokens: 8000,
      },
    ];

    const edges: t.GraphEdge[] = [
      {
        from: 'coordinator',
        to: 'specialist',
        edgeType: 'handoff',
        description: 'Transfer to specialist for detailed work',
        prompt:
          'Provide specific instructions for the specialist about what to analyze or create',
        promptKey: 'instructions',
      },
    ];

    const run = await Run.create({
      runId: `steering-test1-${Date.now()}`,
      graphConfig: { type: 'multi-agent', agents, edges },
      customHandlers,
      returnContent: true,
    });

    const streamConfig: Partial<RunnableConfig> & {
      version: 'v1' | 'v2';
      streamMode: string;
    } = {
      configurable: { thread_id: 'steering-test1' },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const query =
      'Explain the difference between TCP and UDP. I need exactly 3 bullet points for each protocol.';
    console.log(`\nQuery: "${query}"\n`);

    const messages = [new HumanMessage(query)];
    await run.processStream({ messages }, streamConfig);
    const finalMessages = run.getRunMessages();

    console.log('\n--- Specialist Response ---');
    if (finalMessages) {
      for (const msg of finalMessages) {
        if (msg.getType() === 'ai' && typeof msg.content === 'string') {
          console.log(msg.content);
          agentResponses['test1'] = msg.content;
        }
      }
    }

    // Check steering quality
    const response = agentResponses['test1'] || '';
    const mentionsSpecialist =
      response.toLowerCase().includes('specialist') ||
      response.toLowerCase().includes('technical');
    const hasBulletPoints =
      (response.match(/[-•*]\s/g) || []).length >= 4 ||
      (response.match(/\d\./g) || []).length >= 4;
    const mentionsTCP = response.toLowerCase().includes('tcp');
    const mentionsUDP = response.toLowerCase().includes('udp');

    console.log('\n--- Steering Checks ---');
    console.log(
      `  Identifies as specialist: ${mentionsSpecialist ? 'YES' : 'NO'}`
    );
    console.log(`  Has bullet points: ${hasBulletPoints ? 'YES' : 'NO'}`);
    console.log(`  Covers TCP: ${mentionsTCP ? 'YES' : 'NO'}`);
    console.log(`  Covers UDP: ${mentionsUDP ? 'YES' : 'NO'}`);
  }

  /**
   * Test 2: Handoff with very specific formatting instructions
   * Tests whether the receiving agent follows precise instructions from the handoff.
   */
  async function test2_preciseFormatting(): Promise<void> {
    console.log('\n' + '-'.repeat(60));
    console.log('TEST 2: Handoff with precise formatting instructions');
    console.log('-'.repeat(60));

    const agents: t.AgentInputs[] = [
      {
        agentId: 'manager',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a Project Manager. When a user asks about a topic:
1. Transfer to the writer with VERY SPECIFIC formatting instructions
2. Tell the writer to start their response with "REPORT:" and end with "END REPORT"
3. Tell the writer to use exactly 2 paragraphs

CRITICAL: Always transfer to the writer. Do NOT write the report yourself.`,
        maxContextTokens: 8000,
      },
      {
        agentId: 'writer',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a Report Writer. Follow any formatting instructions you receive precisely.
You must follow the exact format requested.`,
        maxContextTokens: 8000,
      },
    ];

    const edges: t.GraphEdge[] = [
      {
        from: 'manager',
        to: 'writer',
        edgeType: 'handoff',
        description: 'Transfer to writer for report creation',
        prompt:
          'Provide specific formatting and content instructions for the writer',
        promptKey: 'instructions',
      },
    ];

    const run = await Run.create({
      runId: `steering-test2-${Date.now()}`,
      graphConfig: { type: 'multi-agent', agents, edges },
      customHandlers,
      returnContent: true,
    });

    const streamConfig: Partial<RunnableConfig> & {
      version: 'v1' | 'v2';
      streamMode: string;
    } = {
      configurable: { thread_id: 'steering-test2' },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const query = 'Write a brief report about cloud computing benefits.';
    console.log(`\nQuery: "${query}"\n`);

    const messages = [new HumanMessage(query)];
    await run.processStream({ messages }, streamConfig);
    const finalMessages = run.getRunMessages();

    console.log('\n--- Writer Response ---');
    if (finalMessages) {
      for (const msg of finalMessages) {
        if (msg.getType() === 'ai' && typeof msg.content === 'string') {
          console.log(msg.content);
          agentResponses['test2'] = msg.content;
        }
      }
    }

    // Check if the writer followed the manager's formatting instructions
    const response = agentResponses['test2'] || '';
    const startsWithReport = response.trimStart().startsWith('REPORT:');
    const endsWithEndReport = response.trimEnd().endsWith('END REPORT');
    const mentionsCloud = response.toLowerCase().includes('cloud');

    console.log('\n--- Steering Checks ---');
    console.log(`  Starts with "REPORT:": ${startsWithReport ? 'YES' : 'NO'}`);
    console.log(
      `  Ends with "END REPORT": ${endsWithEndReport ? 'YES' : 'NO'}`
    );
    console.log(`  Covers cloud computing: ${mentionsCloud ? 'YES' : 'NO'}`);
  }

  /**
   * Test 3: Multi-turn after handoff
   * Tests that identity and context persist across turns.
   */
  async function test3_multiTurn(): Promise<void> {
    console.log('\n' + '-'.repeat(60));
    console.log('TEST 3: Multi-turn conversation after handoff');
    console.log('-'.repeat(60));

    const agents: t.AgentInputs[] = [
      {
        agentId: 'router',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a Router. Transfer all requests to the chef.
When transferring, tell the chef to respond ONLY about Italian cuisine.
CRITICAL: Always transfer. Never answer directly.`,
        maxContextTokens: 8000,
      },
      {
        agentId: 'chef',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are Chef Marco, an Italian cuisine expert.
Always introduce yourself as Chef Marco. Only discuss Italian food.
If asked about non-Italian food, politely redirect to Italian alternatives.`,
        maxContextTokens: 8000,
      },
    ];

    const edges: t.GraphEdge[] = [
      {
        from: 'router',
        to: 'chef',
        edgeType: 'handoff',
        description: 'Transfer to chef',
        prompt: 'Instructions for the chef about how to respond',
        promptKey: 'instructions',
      },
    ];

    const run = await Run.create({
      runId: `steering-test3-${Date.now()}`,
      graphConfig: { type: 'multi-agent', agents, edges },
      customHandlers,
      returnContent: true,
    });

    const streamConfig: Partial<RunnableConfig> & {
      version: 'v1' | 'v2';
      streamMode: string;
    } = {
      configurable: { thread_id: 'steering-test3' },
      streamMode: 'values',
      version: 'v2' as const,
    };

    const conversationHistory: BaseMessage[] = [];

    // Turn 1
    const query1 = 'What is a good pasta recipe?';
    console.log(`\nTurn 1: "${query1}"\n`);
    conversationHistory.push(new HumanMessage(query1));
    await run.processStream({ messages: conversationHistory }, streamConfig);
    const turn1Messages = run.getRunMessages();
    if (turn1Messages) {
      conversationHistory.push(...turn1Messages);
      for (const msg of turn1Messages) {
        if (msg.getType() === 'ai' && typeof msg.content === 'string') {
          console.log(msg.content.substring(0, 300) + '...');
          agentResponses['test3_turn1'] = msg.content;
        }
      }
    }

    // Turn 2 - follow up
    const query2 = 'What about sushi instead?';
    console.log(`\nTurn 2: "${query2}"\n`);
    conversationHistory.push(new HumanMessage(query2));
    await run.processStream({ messages: conversationHistory }, streamConfig);
    const turn2Messages = run.getRunMessages();
    if (turn2Messages) {
      conversationHistory.push(...turn2Messages);
      for (const msg of turn2Messages) {
        if (msg.getType() === 'ai' && typeof msg.content === 'string') {
          console.log(msg.content.substring(0, 300) + '...');
          agentResponses['test3_turn2'] = msg.content;
        }
      }
    }

    const response1 = agentResponses['test3_turn1'] || '';
    const response2 = agentResponses['test3_turn2'] || '';
    const t1Identity =
      response1.toLowerCase().includes('marco') ||
      response1.toLowerCase().includes('chef');
    const t1Italian =
      response1.toLowerCase().includes('italian') ||
      response1.toLowerCase().includes('pasta');
    const t2Redirects =
      response2.toLowerCase().includes('italian') ||
      response2.toLowerCase().includes('instead');

    console.log('\n--- Steering Checks ---');
    console.log(`  Turn 1 - Chef identity: ${t1Identity ? 'YES' : 'NO'}`);
    console.log(`  Turn 1 - Italian focus: ${t1Italian ? 'YES' : 'NO'}`);
    console.log(
      `  Turn 2 - Redirects to Italian: ${t2Redirects ? 'YES' : 'NO'}`
    );
  }

  try {
    await test1_basicInstructions();
    await test2_preciseFormatting();
    await test3_multiTurn();

    console.log('\n\n' + '='.repeat(60));
    console.log('ALL TESTS COMPLETE');
    console.log('='.repeat(60));
    console.log('\nReview the steering checks above.');
    console.log(
      'If the receiving agents consistently follow instructions and maintain identity,'
    );
    console.log('the system prompt injection approach is working correctly.');
  } catch (error) {
    console.error('\nTest failed:', error);
    process.exit(1);
  }
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

testHandoffSteering().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
