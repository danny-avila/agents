#!/usr/bin/env bun

/**
 * Live API confirmation of three suspected bugs:
 *
 * 1. OpenAI Responses-style tool-call ID → Anthropic = 400
 * 2. Anthropic thinking block → OpenAI = ?
 * 3. Empty thinking block → OpenAI = ?
 *
 * Uses cheap models (claude-haiku-4-5, gpt-4o-mini) and minimal token budgets.
 */

import { config } from 'dotenv';
config({ override: true });

import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { CustomAnthropic } from '@/llm/anthropic';
import { ChatOpenAI } from '@/llm/openai';

const ANTHROPIC_KEY = process.env.ANTHROPIC_API_KEY;
const OPENAI_KEY = process.env.OPENAI_API_KEY;

if (!ANTHROPIC_KEY || !OPENAI_KEY) {
  console.error('Missing ANTHROPIC_API_KEY or OPENAI_API_KEY');
  process.exit(1);
}

function header(label: string): void {
  console.log(`\n${'='.repeat(70)}\n${label}\n${'='.repeat(70)}`);
}

async function tryCall<T>(
  fn: () => Promise<T>
): Promise<{ ok: true; result: T } | { ok: false; error: unknown }> {
  try {
    const result = await fn();
    return { ok: true, result };
  } catch (error) {
    return { ok: false, error };
  }
}

function describeError(error: unknown): string {
  if (error == null) return 'unknown';
  const e = error as Record<string, unknown>;
  const status = e.status ?? e.statusCode ?? '?';
  const messageRaw = (e.message as string | undefined) ?? String(error);
  const message =
    messageRaw.length > 800 ? messageRaw.slice(0, 800) + '…' : messageRaw;
  return `status=${status}\n  message=${message}`;
}

// Test 1: send an OpenAI Responses-style tool-call ID to Anthropic
async function testAnthropicRejectsResponsesId(): Promise<void> {
  header('TEST 1: OpenAI Responses-style ID replayed to Anthropic');

  const responsesStyleId =
    'fc_67abc1234def567|call_abc123def456ghi789jkl0mnopqrs';
  console.log(`Tool-call id: ${responsesStyleId}`);

  const claude = new CustomAnthropic({
    modelName: 'claude-haiku-4-5',
    apiKey: ANTHROPIC_KEY,
    maxTokens: 100,
  });

  const tools = [
    {
      name: 'get_weather',
      description: 'Get weather',
      input_schema: {
        type: 'object',
        properties: { location: { type: 'string' } },
        required: ['location'],
      },
    },
  ];

  const history = [
    new HumanMessage('What is the weather in Tokyo?'),
    new AIMessage({
      content: 'Looking that up.',
      tool_calls: [
        {
          id: responsesStyleId,
          name: 'get_weather',
          args: { location: 'Tokyo' },
          type: 'tool_call',
        },
      ],
    }),
    new ToolMessage({
      tool_call_id: responsesStyleId,
      content: '{"temp": 21, "unit": "C"}',
    }),
    new HumanMessage('Now translate that to Fahrenheit.'),
  ];

  const result = await tryCall(async () => {
    const stream = await claude.bindTools(tools).stream(history);
    const chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    return chunks;
  });
  if (result.ok) {
    console.log('Anthropic ACCEPTED the request after normalization.');
    const chunks = result.result as Array<{ content?: unknown }>;
    const text = chunks
      .map((c) =>
        typeof c.content === 'string' ? c.content : JSON.stringify(c.content)
      )
      .join('');
    console.log(`Response sample: ${text.slice(0, 200)}`);
  } else {
    console.log('Anthropic rejected:');
    console.log(`  ${describeError(result.error)}`);
  }
}

// Test 2: send an Anthropic thinking block to OpenAI
async function testOpenAIWithThinkingBlock(): Promise<void> {
  header('TEST 2: Anthropic thinking block replayed to OpenAI');

  const openai = new ChatOpenAI({
    modelName: 'gpt-4o-mini',
    apiKey: OPENAI_KEY,
    maxTokens: 100,
  });

  const history = [
    new HumanMessage('Is 17 prime?'),
    new AIMessage({
      content: [
        {
          type: 'thinking',
          thinking:
            'Let me check: 17 is not divisible by 2, 3, or 5. So 17 is prime.',
          signature:
            'EuYBCkQYAhokAGSEYTYqCfRY8Bz...truncated_signature_blob_for_test...==',
        },
        { type: 'text', text: 'Yes, 17 is prime.' },
      ] as never,
    }),
    new HumanMessage('What about 19?'),
  ];

  const result = await tryCall(() => openai.invoke(history));
  if (result.ok) {
    const r = result.result as { content?: unknown };
    const text =
      typeof r.content === 'string'
        ? r.content
        : JSON.stringify(r.content).slice(0, 200);
    console.log(`OpenAI accepted. Response: ${text}`);
    console.log(
      '  → check whether reasoning text was visible to the model (i.e. did it answer correctly).'
    );
  } else {
    console.log('OpenAI rejected:');
    console.log(`  ${describeError(result.error)}`);
  }
}

// Test 3: send an empty thinking block to OpenAI
async function testOpenAIWithEmptyThinking(): Promise<void> {
  header('TEST 3: Empty thinking block replayed to OpenAI');

  const openai = new ChatOpenAI({
    modelName: 'gpt-4o-mini',
    apiKey: OPENAI_KEY,
    maxTokens: 100,
  });

  const history = [
    new HumanMessage('What is 2+2?'),
    new AIMessage({
      content: [
        { type: 'thinking', thinking: '', signature: 'sig_empty' },
        { type: 'text', text: '4' },
      ] as never,
    }),
    new HumanMessage('And 3+3?'),
  ];

  const result = await tryCall(() => openai.invoke(history));
  if (result.ok) {
    const r = result.result as { content?: unknown };
    const text =
      typeof r.content === 'string'
        ? r.content
        : JSON.stringify(r.content).slice(0, 200);
    console.log(`OpenAI accepted. Response: ${text}`);
  } else {
    console.log('OpenAI rejected:');
    console.log(`  ${describeError(result.error)}`);
  }
}

async function main(): Promise<void> {
  await testAnthropicRejectsResponsesId();
  await testOpenAIWithThinkingBlock();
  await testOpenAIWithEmptyThinking();
  console.log('\nDone.');
}

main().catch((e) => {
  console.error('Fatal:', e);
  process.exit(1);
});
