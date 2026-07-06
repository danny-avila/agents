import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { context as otelContext, trace as otelTrace } from '@opentelemetry/api';
import type { SpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { Context } from '@opentelemetry/api';
import type * as t from '@/types';
import { Constants, ContentTypes, Providers, TitleMethod } from '@/common';
import { withLangfuseRuntimeScope } from '@/langfuseRuntimeScope';
import { initializeLangfuseTracing } from '@/instrumentation';
import { traceIdFromSeed } from '@/langfuseRuntimeContext';
import * as providers from '@/llm/providers';
import { Run } from '@/run';

type ProcessorParams = {
  publicKey?: string;
  secretKey?: string;
  baseUrl?: string;
};

type SpanStartRecord = {
  name: string;
  params: ProcessorParams;
  traceId: string;
};

const spanStarts: SpanStartRecord[] = [];
let providerInput:
  | {
      spanProcessors?: SpanProcessor[];
      idGenerator?: {
        generateTraceId: () => string;
        generateSpanId: () => string;
      };
    }
  | undefined;

type MockSpan = {
  name: string;
  attributes: Record<string, unknown>;
  addEvent: jest.Mock;
  end: jest.Mock;
  isRecording: jest.Mock;
  recordException: jest.Mock;
  setAttribute: jest.Mock;
  setAttributes: jest.Mock;
  setStatus: jest.Mock;
  spanContext: jest.Mock;
  updateName: jest.Mock;
};

function isOtelContext(value: unknown): value is Context {
  return (
    value != null &&
    typeof value === 'object' &&
    typeof (value as { getValue?: unknown }).getValue === 'function'
  );
}

function createMockSpan(
  name: string,
  parentContext: Context = otelContext.active()
): MockSpan {
  const parentTraceId = otelTrace.getSpanContext(parentContext)?.traceId;
  const traceId =
    parentTraceId ??
    providerInput?.idGenerator?.generateTraceId() ??
    'trace-id';
  const spanId = providerInput?.idGenerator?.generateSpanId() ?? 'span-id';
  const attributes: Record<string, unknown> = {};
  const span = {} as MockSpan;
  Object.assign(span, {
    name,
    attributes,
    addEvent: jest.fn(),
    end: jest.fn(() => {
      for (const processor of providerInput?.spanProcessors ?? []) {
        processor.onEnd(span as never);
      }
    }),
    isRecording: jest.fn(() => true),
    recordException: jest.fn(),
    setAttribute: jest.fn((key: string, value: unknown) => {
      attributes[key] = value;
      return span;
    }),
    setAttributes: jest.fn((next: Record<string, unknown>) => {
      Object.assign(attributes, next);
      return span;
    }),
    setStatus: jest.fn(),
    spanContext: jest.fn(() => ({
      traceId,
      spanId,
      traceFlags: 1,
    })),
    updateName: jest.fn((nextName: string) => {
      span.name = nextName;
      return span;
    }),
  });

  for (const processor of providerInput?.spanProcessors ?? []) {
    processor.onStart(span as never, parentContext);
  }
  return span;
}

const startSpan = jest.fn((name: string, _options?: unknown, ctx?: unknown) =>
  createMockSpan(name, isOtelContext(ctx) ? ctx : otelContext.active())
);

function getParentContextFromStartActiveSpanArgs(args: unknown[]): Context {
  if (args.length >= 3 && isOtelContext(args[1])) {
    return args[1];
  }
  if (args.length >= 4 && isOtelContext(args[2])) {
    return args[2];
  }
  return otelContext.active();
}

const startActiveSpan = jest.fn((name: string, ...args: unknown[]) => {
  const callback = args[args.length - 1];
  const parentContext = getParentContextFromStartActiveSpanArgs(args);

  if (typeof callback !== 'function') {
    throw new Error('startActiveSpan mock expected a callback');
  }

  const span = createMockSpan(name, parentContext);
  const activeContext = otelTrace.setSpan(parentContext, span as never);
  return otelContext.with(activeContext, () => callback(span));
});

jest.mock('@langfuse/otel', () => ({
  LangfuseSpanProcessor: jest.fn().mockImplementation((params) => ({
    forceFlush: jest.fn(),
    onEnd: jest.fn(),
    onStart: jest.fn((span) => {
      spanStarts.push({
        name: span.name,
        params,
        traceId: span.spanContext().traceId,
      });
    }),
    shutdown: jest.fn(),
  })),
  isDefaultExportSpan: jest.fn(() => false),
}));

jest.mock('@opentelemetry/sdk-trace-base', () => ({
  BasicTracerProvider: jest.fn().mockImplementation((input) => {
    providerInput = input;
    return {
      forceFlush: jest.fn(),
      getTracer: jest.fn(() => ({
        startActiveSpan,
        startSpan,
      })),
      shutdown: jest.fn(),
    };
  }),
}));

const echoTool = tool(async ({ text }) => `echo:${text}`, {
  name: 'echo',
  description: 'Echoes text for routing tests.',
  schema: z.object({ text: z.string() }),
});

const callerConfig: Partial<RunnableConfig> & {
  version: 'v1' | 'v2';
  streamMode: string;
} = {
  configurable: { thread_id: 'routing-thread', user_id: 'routing-user' },
  streamMode: 'values',
  version: 'v2',
};

function tenantLangfuse(tenantId: string): t.LangfuseConfig {
  return {
    enabled: true,
    publicKey: `pk-${tenantId}`,
    secretKey: `sk-${tenantId}`,
    baseUrl: 'https://langfuse.proxy',
    deterministicTraceId: true,
    metadata: { tenantId },
    tags: [`tenant:${tenantId}`],
    toolNodeTracing: { enabled: true },
    toolOutputTracing: { enabled: true },
  };
}

function startsForTenant(tenantId: string): SpanStartRecord[] {
  return spanStarts.filter(
    (record) => record.params.publicKey === `pk-${tenantId}`
  );
}

function expectTenantCredentials(
  starts: SpanStartRecord[],
  tenantId: string
): void {
  expect(starts).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        params: expect.objectContaining({
          publicKey: `pk-${tenantId}`,
          secretKey: `sk-${tenantId}`,
          baseUrl: 'https://langfuse.proxy',
        }),
      }),
    ])
  );
}

function expectNamedSpansUseTraceId({
  starts,
  names,
  traceId,
}: {
  starts: SpanStartRecord[];
  names: string[];
  traceId: string;
}): void {
  for (const name of names) {
    const matching = starts.filter((record) => record.name === name);
    expect(matching).not.toHaveLength(0);
    expect(matching.map((record) => record.traceId)).toEqual(
      expect.arrayContaining([traceId])
    );
    expect(
      matching.filter((record) => record.traceId !== traceId)
    ).toHaveLength(0);
  }
}

function expectOnlyTraceIds(
  starts: SpanStartRecord[],
  allowedTraceIds: string[]
): void {
  const allowed = new Set(allowedTraceIds);
  expect(starts.filter((record) => !allowed.has(record.traceId))).toHaveLength(
    0
  );
}

function expectNoCrossTenantTrace({
  tenantId,
  otherTenantId,
  traceId,
}: {
  tenantId: string;
  otherTenantId: string;
  traceId: string;
}): void {
  expect(
    startsForTenant(otherTenantId).filter(
      (record) => record.traceId === traceId
    )
  ).toHaveLength(0);
  expect(startsForTenant(tenantId)).toEqual(
    expect.arrayContaining([expect.objectContaining({ traceId })])
  );
}

function createAgent(tenantId: string): t.AgentInputs {
  return {
    agentId: 'parent',
    name: `Parent ${tenantId}`,
    provider: Providers.OPENAI,
    clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
    instructions: 'Use tools when asked.',
    maxContextTokens: 8000,
    tools: [echoTool],
    subagentConfigs: [
      {
        type: 'researcher',
        name: 'Researcher',
        description: 'Answers delegated research tasks.',
        agentInputs: {
          agentId: 'researcher',
          name: `Researcher ${tenantId}`,
          provider: Providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'Answer delegated tasks briefly.',
          maxContextTokens: 8000,
        },
      },
    ],
  };
}

async function runTenantFlow(tenantId: string): Promise<void> {
  const runId = `routing-${tenantId}`;
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [createAgent(tenantId)],
    },
    langfuse: tenantLangfuse(tenantId),
    returnContent: true,
    skipCleanup: true,
  });

  const toolCalls: ToolCall[] = [
    {
      id: `call_echo_${tenantId}`,
      name: 'echo',
      args: { text: `hello ${tenantId}` },
      type: 'tool_call',
    },
    {
      id: `call_subagent_${tenantId}`,
      name: Constants.SUBAGENT,
      args: {
        description: `Research ${tenantId}`,
        subagent_type: 'researcher',
      },
      type: 'tool_call',
    },
  ];
  run.Graph?.overrideTestModel(
    [`Using tools for ${tenantId}.`, `Final answer for ${tenantId}.`],
    1,
    toolCalls
  );

  await run.processStream(
    { messages: [new HumanMessage(`Use tools for ${tenantId}`)] },
    {
      ...callerConfig,
      configurable: {
        thread_id: `thread-${tenantId}`,
        user_id: `user-${tenantId}`,
      },
    }
  );

  await run.generateTitle({
    provider: Providers.OPENAI,
    inputText: `Use tools for ${tenantId}`,
    titleMethod: TitleMethod.COMPLETION,
    contentParts: [
      { type: ContentTypes.TEXT, text: `Final answer for ${tenantId}.` },
    ],
    chainOptions: {
      configurable: {
        thread_id: `thread-${tenantId}`,
        user_id: `user-${tenantId}`,
      },
    },
  });
}

const compactingTokenCounter: t.TokenCounter = (message) => {
  if (message._getType() === 'system') {
    return 1;
  }
  const content = message.content;
  return typeof content === 'string'
    ? content.length
    : JSON.stringify(content).length;
};

async function runTenantSummarizationFlow(tenantId: string): Promise<void> {
  const runId = `routing-summary-${tenantId}`;
  const run = await Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'parent',
          name: `Parent ${tenantId}`,
          provider: Providers.OPENAI,
          clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
          instructions: 'Summarize when context is full.',
          maxContextTokens: 120,
          summarizationEnabled: true,
          summarizationConfig: {
            retainRecent: { turns: 0 },
          },
        },
      ],
    },
    langfuse: tenantLangfuse(tenantId),
    tokenCounter: compactingTokenCounter,
    returnContent: true,
    skipCleanup: true,
  });

  run.Graph?.overrideTestModel([`After summary for ${tenantId}.`], 1);

  await run.processStream(
    {
      messages: [
        new HumanMessage(`${tenantId} old context `.repeat(8)),
        new HumanMessage(`${tenantId} more old context `.repeat(8)),
        new HumanMessage(`Continue for ${tenantId}`),
      ],
    },
    {
      ...callerConfig,
      configurable: {
        thread_id: `summary-thread-${tenantId}`,
        user_id: `summary-user-${tenantId}`,
      },
    }
  );
}

describe('Langfuse per-run routing integration', () => {
  let getChatModelClassSpy: jest.SpyInstance;
  const originalGetChatModelClass = providers.getChatModelClass;

  beforeEach(() => {
    jest.clearAllMocks();
    spanStarts.length = 0;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_BASE_URL;
    delete process.env.LANGFUSE_BASEURL;
    getChatModelClassSpy = jest
      .spyOn(providers, 'getChatModelClass')
      .mockImplementation(((provider: Providers) => {
        if (provider === Providers.OPENAI) {
          return class extends FakeListChatModel {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            constructor(_options: any) {
              super({ responses: ['provider response'] });
            }
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
          } as any;
        }
        return originalGetChatModelClass(provider);
      }) as typeof providers.getChatModelClass);
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  it('routes parallel root, model, tool, subagent, and title spans to each run config', async () => {
    await Promise.all([runTenantFlow('tenant-a'), runTenantFlow('tenant-b')]);

    for (const tenantId of ['tenant-a', 'tenant-b']) {
      const starts = startsForTenant(tenantId);
      const otherTenantId = tenantId === 'tenant-a' ? 'tenant-b' : 'tenant-a';
      const runTraceId = traceIdFromSeed(`routing-${tenantId}`);
      const titleTraceId = traceIdFromSeed(`title-routing-${tenantId}`);

      expectTenantCredentials(starts, tenantId);
      expectNamedSpansUseTraceId({
        starts,
        traceId: runTraceId,
        names: [
          `LibreChat Agent: Parent ${tenantId}`,
          'FakeChatModel',
          'tool_batch',
          'subagent',
        ],
      });
      expectNamedSpansUseTraceId({
        starts,
        traceId: titleTraceId,
        names: [`LibreChat Title: Parent ${tenantId}`, 'GenerateTitle'],
      });
      expectNoCrossTenantTrace({
        tenantId,
        otherTenantId,
        traceId: runTraceId,
      });
      expectNoCrossTenantTrace({
        tenantId,
        otherTenantId,
        traceId: titleTraceId,
      });
    }
  });

  it('routes parallel summarization spans to each run config', async () => {
    await Promise.all([
      runTenantSummarizationFlow('tenant-a'),
      runTenantSummarizationFlow('tenant-b'),
    ]);

    for (const tenantId of ['tenant-a', 'tenant-b']) {
      const otherTenantId = tenantId === 'tenant-a' ? 'tenant-b' : 'tenant-a';
      const starts = startsForTenant(tenantId);
      const summaryTraceId = traceIdFromSeed(`routing-summary-${tenantId}`);

      expectTenantCredentials(starts, tenantId);
      expectOnlyTraceIds(starts, [summaryTraceId]);
      expectNamedSpansUseTraceId({
        starts,
        traceId: summaryTraceId,
        names: [
          `LibreChat Agent: Parent ${tenantId}`,
          'summarize=parent',
          'summarization:cache_hit_compaction',
          'FakeChatModel',
        ],
      });
      expectNoCrossTenantTrace({
        tenantId,
        otherTenantId,
        traceId: summaryTraceId,
      });
    }
  });

  it('routes spans from captured OTel context after ALS scope exits', () => {
    const langfuse = tenantLangfuse('tenant-otel');
    initializeLangfuseTracing(langfuse);

    let capturedContext: Context | undefined;
    withLangfuseRuntimeScope({ langfuse }, () => {
      capturedContext = otelContext.active();
    });

    expect(capturedContext).toBeDefined();
    createMockSpan('otel-context-only', capturedContext);

    expect(startsForTenant('tenant-otel')).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'otel-context-only',
          params: expect.objectContaining({
            publicKey: 'pk-tenant-otel',
            secretKey: 'sk-tenant-otel',
            baseUrl: 'https://langfuse.proxy',
          }),
        }),
      ])
    );
  });
});
