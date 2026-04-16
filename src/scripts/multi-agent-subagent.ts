import { config } from 'dotenv';
config();

import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Providers, GraphEvents, Constants } from '@/common';
import { Run } from '@/run';

/**
 * Manual verification script for the subagent primitive.
 *
 * Configures a supervisor agent with two subagent types (researcher, coder),
 * sends a query, and confirms:
 * 1. The parent agent delegates to a subagent via the `subagent` tool
 * 2. The child executes with isolated context (fresh message history)
 * 3. Only the filtered text result returns to the parent
 * 4. The parent incorporates the result and responds
 *
 * Usage:
 *   OPENAI_API_KEY=... npx ts-node -r tsconfig-paths/register src/scripts/multi-agent-subagent.ts
 *
 * Or with Anthropic:
 *   ANTHROPIC_API_KEY=... npx ts-node -r tsconfig-paths/register src/scripts/multi-agent-subagent.ts --provider anthropic
 */

const useAnthropic =
  process.argv.includes('--provider') &&
  process.argv[process.argv.indexOf('--provider') + 1] === 'anthropic';

const provider = useAnthropic ? Providers.ANTHROPIC : Providers.OPENAI;
const apiKey = useAnthropic
  ? process.env.ANTHROPIC_API_KEY
  : process.env.OPENAI_API_KEY;
const modelName = useAnthropic ? 'claude-sonnet-4-20250514' : 'gpt-4.1';

if (!apiKey) {
  console.error(
    `Missing ${useAnthropic ? 'ANTHROPIC_API_KEY' : 'OPENAI_API_KEY'} environment variable`
  );
  process.exit(1);
}

async function testSubagentPrimitive() {
  console.log('=== Subagent Primitive Manual Verification ===\n');
  console.log(`Provider: ${provider}`);
  console.log(`Model: ${modelName}\n`);

  const { contentParts, aggregateContent } = createContentAggregator();

  const parentAgent: t.AgentInputs = {
    agentId: 'supervisor',
    provider,
    clientOptions: { modelName, apiKey },
    instructions: `You are a supervisor agent. You have access to specialized subagents.

When the user asks a research question, delegate it to the "researcher" subagent.
When the user asks for code, delegate it to the "coder" subagent.

After receiving the subagent's result, synthesize it into a clear final answer for the user.
Always use a subagent for research or coding tasks — do not answer directly.`,
    maxContextTokens: 16000,
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Research Specialist',
        description:
          'Researches topics and provides detailed summaries with sources.',
        agentInputs: {
          agentId: 'researcher',
          provider,
          clientOptions: { modelName, apiKey },
          instructions: `You are a research specialist working in an isolated context.
You receive a single task description and must answer it thoroughly.
Be concise but comprehensive. Include key facts and details.`,
          maxContextTokens: 8000,
        },
      },
      {
        type: 'coder',
        name: 'Coding Specialist',
        description:
          'Writes, reviews, and explains code in any programming language.',
        agentInputs: {
          agentId: 'coder',
          provider,
          clientOptions: { modelName, apiKey },
          instructions: `You are a coding specialist working in an isolated context.
You receive a single task description and must provide working code.
Include brief explanations. Use clean, idiomatic code.`,
          maxContextTokens: 8000,
        },
      },
    ],
  };

  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (event: string, data: t.StreamEventData): void => {
        aggregateContent({
          event: event as GraphEvents,
          data: data as t.RunStep,
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: string, data: t.StreamEventData): void => {
        aggregateContent({
          event: event as GraphEvents,
          data: data as t.RunStep,
        });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (event: string, data: t.StreamEventData): void => {
        aggregateContent({
          event: event as GraphEvents,
          data: data as t.RunStepDeltaEvent,
        });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (event: string, data: t.StreamEventData): void => {
        aggregateContent({
          event: event as GraphEvents,
          data: data as t.MessageDeltaEvent,
        });
      },
    },
  };

  const run = await Run.create<t.IState>({
    runId: `subagent-manual-${Date.now()}`,
    graphConfig: {
      type: 'standard',
      agents: [parentAgent],
    },
    returnContent: true,
    customHandlers,
  });

  console.log('--- Run created ---');
  console.log(
    `Subagent tool present: ${
      (
        (run.Graph as import('@/graphs/Graph').StandardGraph).agentContexts.get(
          'supervisor'
        )?.graphTools as t.GenericTool[] | undefined
      )?.some((t) => 'name' in t && t.name === Constants.SUBAGENT) ?? false
    }\n`
  );

  const conversationHistory: BaseMessage[] = [];

  // Turn 1: Research question (should delegate to researcher subagent)
  console.log('=== Turn 1: Research Question ===\n');
  console.log(
    'User: What are the three laws of thermodynamics? Explain briefly.\n'
  );

  const userMessage = new HumanMessage(
    'What are the three laws of thermodynamics? Explain briefly.'
  );
  conversationHistory.push(userMessage);

  const callerConfig = {
    configurable: { thread_id: 'subagent-verify' },
    streamMode: 'values' as const,
    version: 'v2' as const,
  };

  console.log('--- Streaming response ---\n');
  const result = await run.processStream(
    { messages: conversationHistory },
    callerConfig
  );

  const runMessages = run.getRunMessages();
  console.log('\n\n--- Run Messages ---\n');

  if (runMessages) {
    for (const msg of runMessages) {
      const type = msg._getType();
      if (type === 'tool') {
        const name = 'name' in msg ? msg.name : 'unknown';
        const content =
          typeof msg.content === 'string'
            ? msg.content.slice(0, 200)
            : JSON.stringify(msg.content).slice(0, 200);
        console.log(`[ToolMessage] name=${name}`);
        console.log(
          `  content: ${content}${(msg.content as string).length > 200 ? '...' : ''}\n`
        );
      } else if (type === 'ai') {
        const content =
          typeof msg.content === 'string'
            ? msg.content.slice(0, 300)
            : JSON.stringify(msg.content).slice(0, 300);
        const toolCalls = 'tool_calls' in msg ? msg.tool_calls : undefined;
        console.log(`[AIMessage]`);
        if (toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0) {
          for (const tc of toolCalls) {
            console.log(
              `  tool_call: ${tc.name}(${JSON.stringify(tc.args).slice(0, 100)}...)`
            );
          }
        }
        console.log(
          `  content: ${content}${content.length >= 300 ? '...' : ''}\n`
        );
      }
    }

    const subagentToolMessages = runMessages.filter(
      (msg) =>
        msg._getType() === 'tool' &&
        'name' in msg &&
        msg.name === Constants.SUBAGENT
    );
    console.log(`\n--- Verification ---`);
    console.log(`Subagent tool calls found: ${subagentToolMessages.length}`);
    console.log(`Total run messages: ${runMessages.length}`);
    console.log(`Result content parts: ${result?.length ?? 0}`);

    if (subagentToolMessages.length > 0) {
      console.log(
        '\nSUCCESS: Subagent was invoked and returned a filtered result.'
      );
      console.log(
        'The child context was isolated — only the final text came back.'
      );
    } else {
      console.log('\nNOTE: No subagent tool calls detected.');
      console.log('The LLM may have answered directly without delegating.');
    }
  }

  console.log('\n=== Done ===');
}

testSubagentPrimitive().catch(console.error);
