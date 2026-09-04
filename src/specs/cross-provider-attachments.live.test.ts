/**
 * Live recovery verification for chats created before an agent moved from
 * Bedrock to an OpenAI-compatible endpoint.
 *
 * Run with:
 * CROSS_PROVIDER_ATTACHMENT_BASE_URL=... \
 * CROSS_PROVIDER_ATTACHMENT_API_KEY=... \
 * CROSS_PROVIDER_ATTACHMENT_MODEL=... \
 * npm run test:live:cross-provider-attachments
 */
import { config as dotenvConfig } from 'dotenv';
dotenvConfig({ path: process.env.DOTENV_CONFIG_PATH ?? '.env' });

import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, jest } from '@jest/globals';
import type { ContentBlock } from '@langchain/core/messages';
import type * as t from '@/types';
import { Providers } from '@/common';
import { Run } from '@/run';

const apiKey = process.env.CROSS_PROVIDER_ATTACHMENT_API_KEY;
const configuredBaseURL = process.env.CROSS_PROVIDER_ATTACHMENT_BASE_URL;
const model = process.env.CROSS_PROVIDER_ATTACHMENT_MODEL;
const shouldRunLive =
  process.env.RUN_CROSS_PROVIDER_ATTACHMENT_LIVE_TESTS === '1' &&
  apiKey != null &&
  apiKey !== '' &&
  configuredBaseURL != null &&
  configuredBaseURL !== '' &&
  model != null &&
  model !== '';
const describeIfLive = shouldRunLive ? describe : describe.skip;
const expectedAnswer = '42';

interface PersistedBedrockDocument extends ContentBlock {
  type: 'document';
  document: {
    name: string;
    format: 'csv' | 'xlsx';
    source: {
      bytes: { type: 'Buffer'; data: number[] };
    };
  };
}

function requireLiveValue(
  value: string | undefined,
  name: string
): string {
  if (value == null || value === '') {
    throw new Error(`${name} is required`);
  }
  return value;
}

function normalizeBaseURL(value: string): string {
  return value
    .replace(/\/chat\/completions\/?$/, '')
    .replace(/\/$/, '');
}

function createPersistedDocument(
  name: string,
  format: PersistedBedrockDocument['document']['format']
): PersistedBedrockDocument {
  return {
    type: 'document',
    document: {
      name,
      format,
      source: {
        bytes: { type: 'Buffer', data: [80, 75, 3, 4] },
      },
    },
  };
}

function contentPartsToText(
  content: t.MessageContentComplex[] | undefined
): string {
  const text: string[] = [];
  for (const block of content ?? []) {
    if (block.type === 'text' && typeof block.text === 'string') {
      text.push(block.text);
    }
  }
  return text.join('');
}

describeIfLive('cross-provider attachment recovery live API', () => {
  jest.setTimeout(120_000);

  it('reads extracted CSV/XLSX context without sending persisted Bedrock documents', async () => {
    const requestBodies: string[] = [];
    const nativeFetch = globalThis.fetch.bind(globalThis);
    const capturingFetch: typeof fetch = async (input, init) => {
      if (typeof init?.body === 'string') {
        requestBodies.push(init.body);
      }
      return nativeFetch(input, init);
    };
    const resolvedApiKey = requireLiveValue(
      apiKey,
      'CROSS_PROVIDER_ATTACHMENT_API_KEY'
    );
    const resolvedModel = requireLiveValue(
      model,
      'CROSS_PROVIDER_ATTACHMENT_MODEL'
    );
    const baseURL = normalizeBaseURL(
      requireLiveValue(
        configuredBaseURL,
        'CROSS_PROVIDER_ATTACHMENT_BASE_URL'
      )
    );
    const nonce = `cross-provider-attachment-${Date.now()}`;
    const run = await Run.create({
      runId: nonce,
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: Providers.OPENAI,
          model: resolvedModel,
          modelName: resolvedModel,
          apiKey: resolvedApiKey,
          temperature: 0,
          maxTokens: 32,
          streaming: false,
          streamUsage: false,
          configuration: { baseURL, fetch: capturingFetch },
        },
        instructions:
          'Answer only with the requested numeric total. Do not add punctuation or explanation.',
      },
      returnContent: true,
      skipCleanup: true,
    });
    const historicalMessage = new HumanMessage({
      content: [
        {
          type: 'text',
          text: [
            'Attached document(s):',
            'sales.csv extracted rows: north revenue = 17',
            'forecast.xlsx extracted rows: north forecast = 25',
            `Recovery probe: ${nonce}`,
            'Return the sum of north revenue and north forecast.',
          ].join('\n'),
        },
        createPersistedDocument('sales.csv', 'csv'),
        createPersistedDocument('forecast.xlsx', 'xlsx'),
      ],
    });

    const finalContent = await run.processStream(
      { messages: [historicalMessage] },
      {
        configurable: { thread_id: `${nonce}-thread` },
        version: 'v2',
      }
    );

    const requestBody = requestBodies.at(-1);
    expect(requestBody).toBeDefined();
    expect(requestBody).not.toContain('"type":"document"');
    expect(requestBody).toContain('sales.csv extracted rows');
    expect(requestBody).toContain('forecast.xlsx extracted rows');
    expect(historicalMessage.content).toHaveLength(3);

    const finalText = contentPartsToText(finalContent).trim();
    expect(finalText).toBe(expectedAnswer);
  });
});
