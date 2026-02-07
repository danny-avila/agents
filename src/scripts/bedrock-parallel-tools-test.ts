import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type { StandardGraph } from '@/graphs';
import * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { GraphEvents, ContentTypes, Providers } from '@/common';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { Calculator } from '@/tools/Calculator';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];
let _contentParts: t.MessageContentComplex[] = [];
const collectedUsage: UsageMetadata[] = [];

async function testParallelToolCalls(): Promise<void> {
  const { contentParts, aggregateContent } = createContentAggregator();
  _contentParts = contentParts as t.MessageContentComplex[];

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    [GraphEvents.CHAT_MODEL_STREAM]: {
      handle: async (
        event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>,
        graph?: unknown
      ): Promise<void> => {
        const chunk = data.chunk as Record<string, unknown> | undefined;
        const tcc = chunk?.tool_call_chunks as
          | Array<{ id?: string; name?: string; index?: number }>
          | undefined;
        if (tcc && tcc.length > 0) {
          console.log(
            `[CHAT_MODEL_STREAM] tool_call_chunks: ${JSON.stringify(tcc.map((c) => ({ id: c.id, name: c.name, index: c.index })))}`
          );
        }
        const handler = new ChatModelStreamHandler();
        return handler.handle(event, data, metadata, graph as StandardGraph);
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        const result = (data as unknown as { result: t.ToolEndEvent }).result;
        console.log(
          `[ON_RUN_STEP_COMPLETED] stepId=${result.id} index=${result.index} type=${result.type} tool=${result.tool_call?.name ?? 'n/a'}`
        );
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: GraphEvents.ON_RUN_STEP, data: t.RunStep) => {
        const toolCalls =
          data.stepDetails.type === 'tool_calls' && data.stepDetails.tool_calls
            ? (
                data.stepDetails.tool_calls as Array<{
                  name?: string;
                  id?: string;
                }>
              )
                .map((tc) => `${tc.name ?? '?'}(${tc.id ?? '?'})`)
                .join(', ')
            : 'none';
        console.log(
          `[ON_RUN_STEP] stepId=${data.id} index=${data.index} type=${data.type} stepIndex=${data.stepIndex} toolCalls=[${toolCalls}]`
        );
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.RunStepDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.MessageDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.ReasoningDeltaEvent
      ) => {
        aggregateContent({ event, data });
      },
    },
  };

  const baseLlmConfig = getLLMConfig(Providers.BEDROCK);

  const llmConfig = {
    ...baseLlmConfig,
    model: 'global.anthropic.claude-opus-4-6-v1',
    maxTokens: 16000,
    additionalModelRequestFields: {
      thinking: { type: 'enabled', budget_tokens: 10000 },
    },
  };

  const run = await Run.create<t.IState>({
    runId: 'bedrock-parallel-tools-test',
    graphConfig: {
      instructions:
        'You are a math assistant. When asked to calculate multiple things, use the calculator tool for ALL of them in parallel. Do NOT chain calculations sequentially.',
      type: 'standard',
      tools: [new Calculator()],
      llmConfig,
    },
    returnContent: true,
    customHandlers: customHandlers as t.RunConfig['customHandlers'],
  });

  const streamConfig = {
    configurable: { thread_id: 'bedrock-parallel-tools-thread' },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage =
    'Calculate these 3 things at the same time using the calculator: 1) 123 * 456, 2) sqrt(144) + 7, 3) 2^10 - 24';
  conversationHistory.push(new HumanMessage(userMessage));

  console.log('Running Bedrock parallel tool calls test...\n');
  console.log(`Prompt: "${userMessage}"\n`);

  const inputs = { messages: [...conversationHistory] };
  await run.processStream(inputs, streamConfig);

  console.log('\n\n========== ANALYSIS ==========\n');

  let toolCallCount = 0;
  const toolCallNames: string[] = [];
  let hasUndefined = false;

  for (let i = 0; i < _contentParts.length; i++) {
    const part = _contentParts[i];
    if (!part) {
      hasUndefined = true;
      console.log(`  [${i}] *** UNDEFINED ***`);
      continue;
    }
    if (part.type === ContentTypes.TOOL_CALL) {
      toolCallCount++;
      const tc = (part as t.ToolCallContent).tool_call;
      const hasData = tc && tc.name;
      if (!hasData) {
        console.log(`  [${i}] TOOL_CALL *** EMPTY ***`);
      } else {
        toolCallNames.push(tc.name ?? '');
        console.log(
          `  [${i}] TOOL_CALL name=${tc.name} id=${tc.id} output=${String(tc.output ?? '').substring(0, 40)}`
        );
      }
    } else if (part.type === ContentTypes.THINK) {
      const think = (part as t.ReasoningContentText).think ?? '';
      console.log(`  [${i}] THINK (${think.length} chars)`);
    } else if (part.type === ContentTypes.TEXT) {
      const text = (part as t.MessageDeltaUpdate).text ?? '';
      console.log(
        `  [${i}] TEXT (${text.length} chars): "${text.substring(0, 80)}..."`
      );
    }
  }

  console.log('\n========== SUMMARY ==========\n');
  console.log(`Total content parts: ${_contentParts.filter(Boolean).length}`);
  console.log(`Tool calls found: ${toolCallCount}`);
  console.log(`Tool call names: [${toolCallNames.join(', ')}]`);
  console.log(`Undefined gaps: ${hasUndefined ? 'YES (BUG)' : 'No'}`);
  console.log(
    `Expected 3 tool calls: ${toolCallCount >= 3 ? 'PASS' : 'FAIL (only ' + toolCallCount + ')'}`
  );
  console.log('\nFull contentParts dump:');
  console.dir(_contentParts, { depth: null });
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});

testParallelToolCalls().catch((err) => {
  console.error(err);
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
