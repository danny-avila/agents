import { config } from 'dotenv';
config();
import { HumanMessage } from '@langchain/core/messages';
import type { AIMessageChunk } from '@langchain/core/messages';
import { concat } from '@langchain/core/utils/stream';
import { CustomChatBedrockConverse } from '@/llm/bedrock';
import { modifyDeltaProperties } from '@/messages/core';
import { Providers } from '@/common';

async function testBedrockMerge(): Promise<void> {
  const model = new CustomChatBedrockConverse({
    model: 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    region: process.env.BEDROCK_AWS_REGION,
    credentials: {
      accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
      secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
    },
    maxTokens: 4000,
    streaming: true,
    streamUsage: true,
    additionalModelRequestFields: {
      thinking: { type: 'enabled', budget_tokens: 2000 },
    },
  });

  const messages = [new HumanMessage('What is 25 * 37? Think step by step.')];

  console.log('Streaming from Bedrock with thinking enabled...\n');

  const stream = await model.stream(messages);
  let finalChunk: AIMessageChunk | undefined;
  let chunkCount = 0;
  let firstTextLogged = false;

  for await (const chunk of stream) {
    chunkCount++;
    const isArr = Array.isArray(chunk.content);
    const isStr = typeof chunk.content === 'string';
    const isTextStr = isStr && (chunk.content as string).length > 0;

    if (!firstTextLogged && isTextStr) {
      console.log(
        `chunk ${chunkCount} (first text): contentType=string, value="${chunk.content}"`
      );
      console.log(
        '  response_metadata:',
        JSON.stringify(chunk.response_metadata)
      );
      firstTextLogged = true;
    }

    if (isArr) {
      const blocks = chunk.content as Array<Record<string, unknown>>;
      const info = blocks.map((b) => ({
        type: b.type,
        hasIndex: 'index' in b,
        index: b.index,
      }));
      console.log(`chunk ${chunkCount}: array content, blocks:`, info);
    }

    finalChunk = finalChunk ? concat(finalChunk, chunk) : chunk;
  }

  console.log(`Total chunks received: ${chunkCount}\n`);

  console.log('=== RAW concat result (before modifyDeltaProperties) ===');
  console.log('content type:', typeof finalChunk!.content);
  if (Array.isArray(finalChunk!.content)) {
    console.log('content array length:', finalChunk!.content.length);
    const types = finalChunk!.content.map((b) =>
      typeof b === 'object' && 'type' in b ? b.type : typeof b
    );
    const typeCounts = types.reduce(
      (acc, t) => {
        acc[t ?? ''] = (acc[t ?? ''] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
    console.log('content block type counts:', typeCounts);
  }

  console.log('\ncontent:');
  console.dir(finalChunk!.content, { depth: null });

  console.log('\n=== lc_kwargs.content ===');
  if (Array.isArray(finalChunk!.lc_kwargs.content)) {
    console.log(
      'lc_kwargs.content length:',
      finalChunk!.lc_kwargs.content.length
    );
  }
  console.dir(finalChunk!.lc_kwargs.content, { depth: null });

  const modified = modifyDeltaProperties(Providers.BEDROCK, finalChunk);
  console.log('\n=== After modifyDeltaProperties ===');
  console.log('content:');
  console.dir(modified!.content, { depth: null });
  console.log('\nlc_kwargs.content:');
  console.dir(modified!.lc_kwargs.content, { depth: null });
}

testBedrockMerge().catch((err) => {
  console.error(err);
  process.exit(1);
});
