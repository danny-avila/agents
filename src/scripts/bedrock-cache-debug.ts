/**
 * Debug script to investigate cache token omission in Bedrock responses.
 *
 * This script:
 * 1. Makes a streaming call to Bedrock and logs the raw metadata event
 * 2. Shows exactly what fields the AWS SDK returns in usage (including cache tokens)
 * 3. Shows what our handleConverseStreamMetadata produces vs what it should produce
 * 4. Makes a multi-turn call to trigger caching and verify cache tokens appear
 */
import { config } from 'dotenv';
config();
import { HumanMessage } from '@langchain/core/messages';
import type { AIMessageChunk } from '@langchain/core/messages';
import { concat } from '@langchain/core/utils/stream';
import {
  ConverseStreamCommand,
  BedrockRuntimeClient,
} from '@aws-sdk/client-bedrock-runtime';
import { CustomChatBedrockConverse } from '@/llm/bedrock';

const region = process.env.BEDROCK_AWS_REGION ?? 'us-east-1';
const credentials = {
  accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
  secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
};

const MODEL_ID = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0';

// A long system prompt to increase likelihood of cache usage
// Bedrock requires minimum 1024 tokens for prompt caching to activate
const SYSTEM_PROMPT = `You are an expert assistant. Here is a large context block to help trigger cache behavior:

${Array(200).fill('This is padding content to make the prompt large enough to trigger Bedrock prompt caching. The minimum requirement for Anthropic models on Bedrock is 1024 tokens in the cached prefix. We need to ensure this prompt is well above that threshold. ').join('')}

When answering, be brief and direct.`;

async function rawSdkCall(): Promise<void> {
  console.log('='.repeat(60));
  console.log('TEST 1: Raw AWS SDK call - inspect metadata.usage directly');
  console.log('='.repeat(60));

  const client = new BedrockRuntimeClient({ region, credentials });

  // First call - should create cache
  // Use cachePoint block to explicitly enable prompt caching
  console.log('\n--- Call 1 (cache write expected) ---');
  const command1 = new ConverseStreamCommand({
    modelId: MODEL_ID,
    system: [{ text: SYSTEM_PROMPT }, { cachePoint: { type: 'default' } }],
    messages: [{ role: 'user', content: [{ text: 'What is 2+2?' }] }],
    inferenceConfig: { maxTokens: 100 },
  });

  const response1 = await client.send(command1);
  if (response1.stream) {
    for await (const event of response1.stream) {
      if (event.metadata != null) {
        console.log('\nRAW metadata event (Call 1):');
        console.dir(event.metadata, { depth: null });
        console.log('\nRAW metadata.usage:');
        console.dir(event.metadata.usage, { depth: null });
        console.log('\nSpecific cache fields:');
        console.log(
          '  cacheReadInputTokens:',
          (event.metadata.usage as Record<string, unknown>)
            ?.cacheReadInputTokens
        );
        console.log(
          '  cacheWriteInputTokens:',
          (event.metadata.usage as Record<string, unknown>)
            ?.cacheWriteInputTokens
        );
      }
    }
  }

  // Second call - should read from cache
  console.log('\n--- Call 2 (cache read expected) ---');
  const command2 = new ConverseStreamCommand({
    modelId: MODEL_ID,
    system: [{ text: SYSTEM_PROMPT }, { cachePoint: { type: 'default' } }],
    messages: [
      { role: 'user', content: [{ text: 'What is 2+2?' }] },
      { role: 'assistant', content: [{ text: '4' }] },
      { role: 'user', content: [{ text: 'And what is 3+3?' }] },
    ],
    inferenceConfig: { maxTokens: 100 },
  });

  const response2 = await client.send(command2);
  if (response2.stream) {
    for await (const event of response2.stream) {
      if (event.metadata != null) {
        console.log('\nRAW metadata event (Call 2):');
        console.dir(event.metadata, { depth: null });
        console.log('\nRAW metadata.usage:');
        console.dir(event.metadata.usage, { depth: null });
        console.log('\nSpecific cache fields:');
        console.log(
          '  cacheReadInputTokens:',
          (event.metadata.usage as Record<string, unknown>)
            ?.cacheReadInputTokens
        );
        console.log(
          '  cacheWriteInputTokens:',
          (event.metadata.usage as Record<string, unknown>)
            ?.cacheWriteInputTokens
        );
      }
    }
  }
}

async function wrapperStreamCallNoCachePoint(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log(
    'TEST 2: CustomChatBedrockConverse stream (NO cachePoint) - check usage_metadata'
  );
  console.log('='.repeat(60));
  console.log('(Without cachePoint, Bedrock does NOT return cache tokens)');

  const model = new CustomChatBedrockConverse({
    model: MODEL_ID,
    region,
    credentials,
    maxTokens: 100,
    streaming: true,
    streamUsage: true,
  });

  console.log('\n--- Wrapper Call (no cachePoint) ---');
  const messages1 = [new HumanMessage(SYSTEM_PROMPT + '\n\nWhat is 2+2?')];
  let finalChunk1: AIMessageChunk | undefined;

  for await (const chunk of await model.stream(messages1)) {
    finalChunk1 = finalChunk1 ? concat(finalChunk1, chunk) : chunk;
  }

  console.log(
    '\nFinal usage_metadata:',
    JSON.stringify(finalChunk1!.usage_metadata)
  );
  console.log('(No cache tokens expected since no cachePoint block was sent)');
}

async function wrapperStreamCallWithCachePoint(): Promise<void> {
  console.log('\n' + '='.repeat(60));
  console.log(
    'TEST 3: Raw SDK with cachePoint -> verify handleConverseStreamMetadata extracts cache tokens'
  );
  console.log('='.repeat(60));

  // We use the raw SDK with cachePoint to trigger caching, then verify
  // that our handleConverseStreamMetadata function properly extracts cache fields
  const { handleConverseStreamMetadata } = await import(
    '@/llm/bedrock/utils/message_outputs'
  );

  const client = new BedrockRuntimeClient({ region, credentials });

  // Call 1 - establish cache
  console.log('\n--- Call 1 (cache write) ---');
  const command1 = new ConverseStreamCommand({
    modelId: MODEL_ID,
    system: [{ text: SYSTEM_PROMPT }, { cachePoint: { type: 'default' } }],
    messages: [{ role: 'user', content: [{ text: 'What is 2+2?' }] }],
    inferenceConfig: { maxTokens: 100 },
  });

  const response1 = await client.send(command1);
  if (response1.stream) {
    for await (const event of response1.stream) {
      if (event.metadata != null) {
        console.log('Raw usage:', JSON.stringify(event.metadata.usage));

        // Test our handler
        const chunk = handleConverseStreamMetadata(event.metadata, {
          streamUsage: true,
        });
        console.log(
          'handleConverseStreamMetadata output usage_metadata:',
          JSON.stringify(chunk.message.usage_metadata)
        );

        const hasDetails =
          chunk.message.usage_metadata?.input_token_details != null;
        console.log(
          `Has input_token_details: ${hasDetails}`,
          hasDetails
            ? JSON.stringify(chunk.message.usage_metadata!.input_token_details)
            : '(MISSING - BUG!)'
        );
      }
    }
  }

  // Call 2 - read from cache
  console.log('\n--- Call 2 (cache read) ---');
  const command2 = new ConverseStreamCommand({
    modelId: MODEL_ID,
    system: [{ text: SYSTEM_PROMPT }, { cachePoint: { type: 'default' } }],
    messages: [
      { role: 'user', content: [{ text: 'What is 2+2?' }] },
      { role: 'assistant', content: [{ text: '4' }] },
      { role: 'user', content: [{ text: 'What is 3+3?' }] },
    ],
    inferenceConfig: { maxTokens: 100 },
  });

  const response2 = await client.send(command2);
  if (response2.stream) {
    for await (const event of response2.stream) {
      if (event.metadata != null) {
        console.log('Raw usage:', JSON.stringify(event.metadata.usage));

        const chunk = handleConverseStreamMetadata(event.metadata, {
          streamUsage: true,
        });
        console.log(
          'handleConverseStreamMetadata output usage_metadata:',
          JSON.stringify(chunk.message.usage_metadata)
        );

        const hasDetails =
          chunk.message.usage_metadata?.input_token_details != null;
        console.log(
          `Has input_token_details: ${hasDetails}`,
          hasDetails
            ? JSON.stringify(chunk.message.usage_metadata!.input_token_details)
            : '(MISSING - BUG!)'
        );
      }
    }
  }
}

async function main(): Promise<void> {
  console.log('Bedrock Cache Token Debug Script');
  console.log(`Model: ${MODEL_ID}`);
  console.log(`Region: ${region}\n`);

  await rawSdkCall();
  await wrapperStreamCallNoCachePoint();
  await wrapperStreamCallWithCachePoint();
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
