// src/specs/bedrock-toolless.live.test.ts
/**
 * Live Bedrock verification for the tool-less-destination toolConfig fix.
 *
 * A tool-less agent in a multi-agent graph inherits the prior agent's
 * toolUse/toolResult history. Because it binds no tools, Bedrock's Converse API
 * rejects the request ("The toolConfig field must be defined when using toolUse
 * and toolResult content blocks"). `foldToolBlocksForToollessAgent` folds that
 * history into text so the request becomes valid.
 *
 * Run with:
 * RUN_BEDROCK_LIVE_TESTS=1 BEDROCK_AWS_ACCESS_KEY_ID=... BEDROCK_AWS_SECRET_ACCESS_KEY=... \
 *   BEDROCK_AWS_DEFAULT_REGION=us-west-2 npm test -- bedrock-toolless.live.test.ts --runInBand
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig();

import {
  AIMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { beforeAll, describe, expect, it } from '@jest/globals';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { Providers } from '@/common';
import { initializeModel } from '@/llm/init';
import { foldToolBlocksForToollessAgent } from '@/messages';

const accessKeyId =
  process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? process.env.AWS_ACCESS_KEY_ID;
const secretAccessKey =
  process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ??
  process.env.AWS_SECRET_ACCESS_KEY;

const shouldRunLive =
  process.env.RUN_BEDROCK_LIVE_TESTS === '1' &&
  accessKeyId != null &&
  accessKeyId !== '' &&
  secretAccessKey != null &&
  secretAccessKey !== '';

const describeIfLive = shouldRunLive ? describe : describe.skip;

const MODEL =
  process.env.LIVE_BEDROCK_MODEL ??
  'us.anthropic.claude-sonnet-4-5-20250929-v1:0';
const REGION =
  process.env.BEDROCK_AWS_DEFAULT_REGION ??
  process.env.AWS_REGION ??
  'us-west-2';

/** History a tool-less destination inherits: a completed tool call, then a
 *  follow-up user turn that itself invokes no tool. */
function toollessHistory(): BaseMessage[] {
  return [
    new HumanMessage('Search my files for the roadmap.'),
    new AIMessage({
      content: '',
      tool_calls: [
        {
          id: 'tt_live_1',
          name: 'file_search',
          args: { query: 'roadmap' },
          type: 'tool_call',
        },
      ],
    }),
    new ToolMessage({
      content: 'Found: roadmap.md — Q3 goals and milestones.',
      tool_call_id: 'tt_live_1',
      name: 'file_search',
    }),
    new AIMessage('I found roadmap.md with your Q3 goals.'),
    new HumanMessage('thanks!'),
  ];
}

describeIfLive('Bedrock tool-less destination (live)', () => {
  let clientOptions: t.ClientOptions;

  beforeAll(() => {
    // Force SigV4 with the explicit keys; the Bedrock API-key (bearer) auth
    // scheme otherwise takes precedence in the AWS SDK.
    delete process.env.AWS_BEARER_TOKEN_BEDROCK;
    clientOptions = {
      region: REGION,
      model: MODEL,
      credentials: { accessKeyId, secretAccessKey },
    } as unknown as t.ClientOptions;
  });

  it('rejects inherited tool history when the agent binds no tools', async () => {
    const model = initializeModel({
      provider: Providers.BEDROCK,
      clientOptions,
      tools: undefined,
    });
    await expect(model.invoke(toollessHistory())).rejects.toThrow(
      /toolConfig field must be defined/i
    );
  }, 30000);

  it('succeeds once inherited tool blocks are folded to text', async () => {
    const model = initializeModel({
      provider: Providers.BEDROCK,
      clientOptions,
      tools: undefined,
    });
    const folded = foldToolBlocksForToollessAgent(toollessHistory());
    const res = await model.invoke(folded);
    const text =
      typeof res.content === 'string'
        ? res.content
        : JSON.stringify(res.content);
    expect(text.length).toBeGreaterThan(0);
  }, 30000);
});
