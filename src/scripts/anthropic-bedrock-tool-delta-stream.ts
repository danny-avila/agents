/* eslint-disable no-console */
import { config } from 'dotenv';
config({ path: process.env.DOTENV_CONFIG_PATH });

import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { performance } from 'node:perf_hooks';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { GraphEvents, Providers, StepTypes } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { Run } from '@/run';

type ProviderUnderTest = Providers.ANTHROPIC | Providers.BEDROCK;
type ToolCallDeltaChunk = NonNullable<t.ToolCallDelta['tool_calls']>[number] & {
  function?: {
    name?: unknown;
    arguments?: unknown;
  };
};

const providerArgs = new Set(['anthropic', 'bedrock', 'both']);
const weatherSchema = {
  type: 'object',
  properties: {
    city: {
      type: 'string',
      description: 'City and region to look up.',
    },
    unit: {
      type: 'string',
      enum: ['fahrenheit', 'celsius'],
    },
    detail: {
      type: 'string',
      description: 'Detailed request text.',
    },
  },
  required: ['city', 'unit', 'detail'],
} as const;

function getProviders(): ProviderUnderTest[] {
  const rawArg =
    process.argv.find((arg) => arg.startsWith('--provider='))?.split('=')[1] ??
    process.argv.find((arg) => providerArgs.has(arg));
  if (rawArg === 'anthropic') {
    return [Providers.ANTHROPIC];
  }
  if (rawArg === 'bedrock') {
    return [Providers.BEDROCK];
  }
  return [Providers.ANTHROPIC, Providers.BEDROCK];
}

function preview(value: string): string {
  return value.length > 90 ? `${value.slice(0, 87)}...` : value;
}

function createLookupWeatherTool(): StructuredToolInterface {
  return tool(
    async (input: Record<string, unknown>): Promise<string> =>
      JSON.stringify({
        status: 'ok',
        received: input,
        forecast: 'Clear enough for this streaming test.',
      }),
    {
      name: 'lookup_weather',
      description:
        'Lookup weather for a city with explicit formatting preferences.',
      schema: weatherSchema,
    }
  );
}

function configureLLM(provider: ProviderUnderTest): t.LLMConfig {
  const llmConfig = {
    ...getLLMConfig(provider),
    streaming: true,
    streamUsage: true,
    maxTokens: 512,
    _lc_stream_delay: Number(process.env.LC_STREAM_DELAY_MS ?? 35),
  } as t.LLMConfig;

  if (
    provider === Providers.ANTHROPIC &&
    process.env.ANTHROPIC_TEST_MODEL != null &&
    process.env.ANTHROPIC_TEST_MODEL !== ''
  ) {
    llmConfig.model = process.env.ANTHROPIC_TEST_MODEL;
  }
  if (
    provider === Providers.BEDROCK &&
    process.env.BEDROCK_TEST_MODEL != null &&
    process.env.BEDROCK_TEST_MODEL !== ''
  ) {
    llmConfig.model = process.env.BEDROCK_TEST_MODEL;
  }

  return llmConfig;
}

async function testProvider(provider: ProviderUnderTest): Promise<void> {
  const startedAt = performance.now();
  const nonEmptyArgTimes: number[] = [];
  const argsByIndex = new Map<number, string>();
  const messageInputs: string[] = [];
  const eventCounts = new Map<string, number>();

  const elapsed = (): number => Math.round(performance.now() - startedAt);
  const countEvent = (event: string): void => {
    eventCounts.set(event, (eventCounts.get(event) ?? 0) + 1);
  };

  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.ON_RUN_STEP]: {
      handle: (event: string, data: t.StreamEventData): void => {
        countEvent(event);
        const runStep = data as t.RunStep;
        const toolCalls =
          runStep.stepDetails.type === StepTypes.TOOL_CALLS
            ? runStep.stepDetails.tool_calls
            : undefined;
        console.log(
          `[${provider}] +${elapsed()}ms ${event} step=${runStep.id} type=${
            runStep.stepDetails.type
          } index=${runStep.index}`
        );
        if (toolCalls != null && toolCalls.length > 0) {
          console.dir(toolCalls, { depth: null });
        }
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (event: string, data: t.StreamEventData): void => {
        countEvent(event);
        const runStepDelta = data as t.RunStepDeltaEvent;
        if (runStepDelta.delta.type !== StepTypes.TOOL_CALLS) {
          return;
        }

        for (const rawToolCall of runStepDelta.delta.tool_calls ?? []) {
          const toolCall = rawToolCall as ToolCallDeltaChunk;
          const index = toolCall.index ?? 0;
          const functionArguments =
            typeof toolCall.function?.arguments === 'string'
              ? toolCall.function.arguments
              : '';
          const args =
            typeof toolCall.args === 'string'
              ? toolCall.args
              : functionArguments;
          const previousArgs = argsByIndex.get(index) ?? '';
          const accumulatedArgs = previousArgs + args;
          argsByIndex.set(index, accumulatedArgs);
          if (args !== '') {
            nonEmptyArgTimes.push(elapsed());
          }

          console.log(
            `[${provider}] +${elapsed()}ms ${event} step=${
              runStepDelta.id
            } index=${index} id=${toolCall.id ?? ''} name=${
              toolCall.name ?? toolCall.function?.name ?? ''
            } args=${JSON.stringify(preview(args))} accumulated=${JSON.stringify(
              preview(accumulatedArgs)
            )}`
          );
        }
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (event: string, data: t.StreamEventData): void => {
        countEvent(event);
        const messageDelta = data as t.MessageDeltaEvent;
        const content = messageDelta.delta.content ?? [];
        const inputParts: string[] = [];
        for (const part of content) {
          if (
            typeof part === 'object' &&
            'input' in part &&
            typeof part.input === 'string'
          ) {
            inputParts.push(part.input);
          }
        }
        messageInputs.push(...inputParts);
        console.log(
          `[${provider}] +${elapsed()}ms ${event} step=${
            messageDelta.id
          } contentTypes=${content.map((part) => part.type).join(',')}`
        );
        if (inputParts.length > 0) {
          console.log(
            `[${provider}] +${elapsed()}ms ${event} input=${JSON.stringify(
              inputParts.map(preview)
            )}`
          );
        }
      },
    },
  };

  const run = await Run.create<t.IState>({
    runId: `tool-delta-${provider}-${Date.now()}`,
    graphConfig: {
      type: 'standard',
      llmConfig: configureLLM(provider),
      tools: [createLookupWeatherTool()],
      instructions:
        'You are a test assistant. You must call lookup_weather exactly once before answering.',
      maxContextTokens: 120000,
    },
    customHandlers,
    returnContent: true,
    skipCleanup: true,
  });

  await run.processStream(
    {
      messages: [
        new HumanMessage(
          'Call lookup_weather exactly once for San Francisco, California. Use fahrenheit. In detail, ask for current conditions, wind, humidity, and a one sentence commute note.'
        ),
      ],
    },
    {
      configurable: {
        provider,
        thread_id: `tool-delta-${provider}`,
      },
      version: 'v2',
    }
  );

  const gaps = nonEmptyArgTimes
    .slice(1)
    .map((time, index) => time - nonEmptyArgTimes[index]);
  console.log(`\n[${provider}] summary`);
  console.log('eventCounts:', Object.fromEntries(eventCounts));
  console.log('argsByIndex:', Object.fromEntries(argsByIndex));
  console.log('messageDeltaInputFragments:', messageInputs);
  console.log('nonEmptyArgGapsMs:', gaps);
  console.log('');
}

async function main(): Promise<void> {
  for (const provider of getProviders()) {
    await testProvider(provider);
  }
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
