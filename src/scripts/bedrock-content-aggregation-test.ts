import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { createCodeExecutionTool } from '@/tools/CodeExecutor';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { GraphEvents, ContentTypes, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];
let _contentParts: t.MessageContentComplex[] = [];
const collectedUsage: UsageMetadata[] = [];

async function testBedrockContentAggregation(): Promise<void> {
  const instructions =
    'You are a helpful AI assistant with coding capabilities. When answering questions, be thorough in your reasoning.';
  const { contentParts, aggregateContent } = createContentAggregator();
  _contentParts = contentParts as t.MessageContentComplex[];

  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
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
        const tcNames =
          data.delta.tool_calls
            ?.map(
              (tc) =>
                `${tc.name ?? '?'}(args=${(tc.args ?? '').substring(0, 30)}...)`
            )
            .join(', ') ?? 'none';
        console.log(
          `[ON_RUN_STEP_DELTA] stepId=${data.id} type=${data.delta.type} toolCalls=[${tcNames}]`
        );
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.MessageDeltaEvent
      ) => {
        const preview = Array.isArray(data.delta.content)
          ? data.delta.content
              .map(
                (c) =>
                  `${c.type}:"${String((c as Record<string, unknown>).text ?? (c as Record<string, unknown>).think ?? '').substring(0, 40)}"`
              )
              .join(', ')
          : String(data.delta.content).substring(0, 40);
        console.log(
          `[ON_MESSAGE_DELTA] stepId=${data.id} content=[${preview}]`
        );
        aggregateContent({ event, data });
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.ReasoningDeltaEvent
      ) => {
        const preview = Array.isArray(data.delta.content)
          ? data.delta.content
              .map(
                (c) =>
                  `${c.type}:"${String((c as Record<string, unknown>).think ?? '').substring(0, 40)}"`
              )
              .join(', ')
          : '?';
        console.log(
          `[ON_REASONING_DELTA] stepId=${data.id} content=[${preview}]`
        );
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
    runId: 'bedrock-content-aggregation-test',
    graphConfig: {
      instructions,
      type: 'standard',
      tools: [createCodeExecutionTool()],
      llmConfig,
    },
    returnContent: true,
    customHandlers: customHandlers as t.RunConfig['customHandlers'],
  });

  const streamConfig = {
    configurable: {
      thread_id: 'bedrock-content-aggregation-thread',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const userMessage = `im testing edge cases with our code interpreter. i know we can persist files, but what happens when we put them in directories?`;
  conversationHistory.push(new HumanMessage(userMessage));

  console.log('Running Bedrock content aggregation test...\n');
  console.log(`Prompt: "${userMessage}"\n`);

  const inputs = { messages: [...conversationHistory] };
  await run.processStream(inputs, streamConfig);

  console.log('\n\n========== CONTENT PARTS ANALYSIS ==========\n');

  let hasEmptyToolCall = false;
  let hasReasoningOrderIssue = false;

  for (let i = 0; i < _contentParts.length; i++) {
    const part = _contentParts[i];
    if (!part) {
      console.log(`  [${i}] undefined`);
      continue;
    }

    const partType = part.type;
    if (partType === ContentTypes.TOOL_CALL) {
      const tc = (part as t.ToolCallContent).tool_call;
      if (!tc || !tc.name) {
        hasEmptyToolCall = true;
        console.log(`  [${i}] TOOL_CALL *** EMPTY (no tool_call data) ***`);
      } else {
        const outputPreview = tc.output
          ? `output=${(tc.output as string).substring(0, 80)}...`
          : 'no output';
        console.log(`  [${i}] TOOL_CALL name=${tc.name} ${outputPreview}`);
      }
    } else if (partType === ContentTypes.THINK) {
      const think = (part as t.ReasoningContentText).think ?? '';
      console.log(
        `  [${i}] THINK (${think.length} chars): "${think.substring(0, 80)}..."`
      );
    } else if (partType === ContentTypes.TEXT) {
      const text = (part as t.MessageDeltaUpdate).text ?? '';
      console.log(
        `  [${i}] TEXT (${text.length} chars): "${text.substring(0, 80)}..."`
      );
    } else {
      console.log(`  [${i}] ${partType}`);
    }
  }

  /**
   * Check reasoning ordering within a single invocation cycle.
   * A tool_call resets the cycle — text before think across different
   * invocations (e.g., text from invocation 2, think from invocation 3) is valid.
   */
  let lastTextInCycle: number | null = null;
  for (let i = 0; i < _contentParts.length; i++) {
    const part = _contentParts[i];
    if (!part) continue;

    if (part.type === ContentTypes.TOOL_CALL) {
      lastTextInCycle = null;
      continue;
    }

    if (part.type === ContentTypes.TEXT) {
      lastTextInCycle = i;
    } else if (part.type === ContentTypes.THINK && lastTextInCycle !== null) {
      const prevText = _contentParts[lastTextInCycle] as t.MessageDeltaUpdate;
      const thinkContent = (part as t.ReasoningContentText).think ?? '';
      if (
        prevText?.text &&
        prevText.text.trim().length > 5 &&
        thinkContent.length > 0
      ) {
        hasReasoningOrderIssue = true;
        console.log(
          `\n  *** ORDERING ISSUE (same invocation): TEXT at [${lastTextInCycle}] appears before THINK at [${i}]`
        );
        console.log(
          `      Text ends with: "...${prevText.text.substring(prevText.text.length - 60)}"`
        );
        console.log(
          `      Think starts with: "${thinkContent.substring(0, 60)}..."`
        );
      }
    }
  }

  console.log('\n========== SUMMARY ==========\n');
  console.log(`Total content parts: ${_contentParts.filter(Boolean).length}`);
  console.log(
    `Empty tool_call parts: ${hasEmptyToolCall ? 'YES (BUG)' : 'No'}`
  );
  console.log(
    `Reasoning order issues: ${hasReasoningOrderIssue ? 'YES (BUG)' : 'No'}`
  );
  console.log('\nFull contentParts dump:');
  console.dir(_contentParts, { depth: null });
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

testBedrockContentAggregation().catch((err) => {
  console.error(err);
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
