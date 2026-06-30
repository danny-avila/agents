/**
 * Live Bedrock tool-call truncation verification.
 *
 * Reproduces the production failure where a large tool argument (`content`)
 * gets cut off by the max output token limit, leaving the handler with a
 * well-formed-but-incomplete tool call and no truncation signal.
 *
 * Run with:
 * RUN_BEDROCK_TRUNCATION_LIVE=1 npm test -- bedrock-truncation.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import { z } from 'zod';
import { HumanMessage } from '@langchain/core/messages';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { describe, expect, it, jest } from '@jest/globals';
import type * as t from '@/types';
import { GraphEvents, Providers } from '@/common';
import { ChatModelStreamHandler } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { Run } from '@/run';

const shouldRunLive =
  process.env.RUN_BEDROCK_TRUNCATION_LIVE === '1' &&
  (process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? '') !== '' &&
  (process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ?? '') !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;

const REGION =
  [
    process.env.BEDROCK_AWS_REGION,
    process.env.BEDROCK_AWS_DEFAULT_REGION,
    process.env.AWS_DEFAULT_REGION,
  ].find(
    (value): value is string => typeof value === 'string' && value !== ''
  ) ?? 'us-east-1';
const MODEL = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0';
const MAX_TOKENS = Number(process.env.REPRO_MAX_TOKENS ?? 350);

const schema = z.object({
  path: z.string().describe('Path to write.'),
  content: z.string().describe('Complete file contents.'),
  overwrite: z.boolean().optional(),
});

describeIfLive('Bedrock tool-call truncation (live)', () => {
  jest.setTimeout(120_000);

  it('fails fast with a truncation error instead of looping', async () => {
    const received: Array<{ keys: string[]; contentLen: number | null }> = [];
    const stepToolCalls: string[][] = [];
    let invocations = 0;

    const createFile = new DynamicStructuredTool({
      name: 'create_file',
      description: 'Create a new file. Requires path and content.',
      schema,
      func: async (input: z.infer<typeof schema>) => {
        invocations += 1;
        const hasContent =
          typeof input.content === 'string' && input.content.length > 0;
        received.push({
          keys: Object.keys(input as Record<string, unknown>),
          contentLen: hasContent ? input.content.length : null,
        });
        if (!hasContent) {
          return 'Error: content is required\n Please fix your mistakes.';
        }
        return `Wrote ${input.content.length} bytes to ${input.path}`;
      },
    });

    const llmConfig = {
      provider: Providers.BEDROCK,
      model: MODEL,
      region: REGION,
      credentials: {
        accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
      },
      maxTokens: MAX_TOKENS,
      streaming: true,
      streamUsage: true,
    } as t.LLMConfig;

    const run = await Run.create<t.IState>({
      runId: 'bedrock-truncation-live',
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [createFile],
        instructions: 'You are a ClickHouse assistant. Use tools when asked.',
        maxContextTokens: 89000,
      },
      returnContent: true,
      skipCleanup: true,
      customHandlers: {
        [GraphEvents.TOOL_END]: new ToolEndHandler(
          async (toolEndData: t.ToolEndData) => {
            const out = toolEndData.output as
              | { content?: unknown; name?: string; status?: string }
              | undefined;
            const content =
              typeof out?.content === 'string'
                ? out.content.slice(0, 120)
                : JSON.stringify(out?.content).slice(0, 120);
            // eslint-disable-next-line no-console
            console.log(
              `TOOL_END name=${out?.name} status=${out?.status} content=${content}`
            );
          }
        ),
        [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
        [GraphEvents.ON_RUN_STEP]: {
          handle: (_event: string, data: t.StreamEventData): void => {
            const step = data as unknown as {
              stepDetails?: {
                type?: string;
                tool_calls?: Array<{ name?: string; args?: unknown }>;
              };
            };
            const tcs = step.stepDetails?.tool_calls;
            if ((tcs?.length ?? 0) > 0 && tcs != null) {
              for (const tc of tcs) {
                if (tc.name === 'create_file') {
                  stepToolCalls.push(
                    typeof tc.args === 'string'
                      ? ['<raw-string>']
                      : Object.keys((tc.args ?? {}) as Record<string, unknown>)
                  );
                }
              }
              // eslint-disable-next-line no-console
              console.log(
                'ON_RUN_STEP tool_calls=',
                JSON.stringify(
                  tcs.map((tc) => ({
                    name: tc.name,
                    args:
                      typeof tc.args === 'string'
                        ? tc.args.slice(0, 80)
                        : Object.keys(
                            (tc.args ?? {}) as Record<string, unknown>
                        ),
                  }))
                )
              );
            }
          },
        },
      },
    });

    const prompt =
      'Call create_file once: path="skills/clickhouse-demo/SKILL.md", content=a complete ' +
      '3000-word ClickHouse guide (schema design, MergeTree, partitioning, 10 example queries). ' +
      'Put the entire guide in the content argument.';

    let loopError: string | null = null;
    try {
      await run.processStream(
        { messages: [new HumanMessage(prompt)] },
        {
          configurable: { provider: Providers.BEDROCK, thread_id: 'trunc' },
          version: 'v2',
          recursionLimit: 8,
        }
      );
    } catch (err) {
      loopError = err instanceof Error ? err.message : String(err);
    }

    // eslint-disable-next-line no-console
    console.log('LIVE RESULT', {
      invocations,
      received,
      stepToolCalls,
      loopError,
    });

    // The run fails fast with the actionable truncation error...
    expect(loopError).toMatch(/truncated at the maximum output token limit/i);
    // ...instead of looping until the recursion limit.
    expect(loopError).not.toMatch(/Recursion limit/i);
    // No truncated create_file call ever carried a usable `content` arg.
    expect(stepToolCalls.every((keys) => !keys.includes('content'))).toBe(true);
  });
});
