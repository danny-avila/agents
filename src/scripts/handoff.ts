/* eslint-disable no-console */
// src/scripts/handoff.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Run } from '@/run';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { createHandoffTool } from '@/tools/handoff';

const conversationHistory: BaseMessage[] = [];

async function testHandoff(): Promise<void> {
  const { provider } = await getArgs();
  const { aggregateContent } = createContentAggregator();

  const llmConfig = getLLMConfig(provider);

  const customHandlers: Record<string | GraphEvents, t.EventHandler> = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('Message delta event:', event);
        console.dir(data, { depth: null });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        console.log('Run step event:', event);
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        console.log('Run step completed event:', event);
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
  } as unknown as Record<string | GraphEvents, t.EventHandler>;

  const run = await Run.create<t.IState>({
    runId: 'handoff-run-id',
    graphConfig: {
      type: 'supervised',
      llmConfig,
      routerEnabled: true,
      toolEnd: false,
      tools: [createHandoffTool()],
      instructions:
        'You can call the handoff tool to jump between nodes (router/agent/tools).',
    },
    returnContent: true,
    customHandlers,
  });

  // Prompt the model to exercise the handoff tool explicitly
  const userMessage =
    "Call the 'handoff' tool with target='router' and mode='one_way'. After the handoff, briefly reply with: Handoff done.";
  conversationHistory.push(new HumanMessage(userMessage));

  const config_ = {
    configurable: {
      provider: provider as Providers,
      thread_id: 'handoff-conversation',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  await run.processStream({ messages: conversationHistory }, config_);
}

testHandoff().catch((err) => {
  console.error(err);
  process.exit(1);
});
