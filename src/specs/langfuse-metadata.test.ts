import { CallbackHandler } from '@langfuse/langchain';
import { propagateAttributes } from '@langfuse/tracing';
import { Providers } from '@/common';
import { Run } from '@/run';

jest.mock('@langfuse/langchain', () => ({
  CallbackHandler: jest.fn().mockImplementation(() => ({})),
}));

jest.mock('@langfuse/tracing', () => ({
  ...jest.requireActual('@langfuse/tracing'),
  propagateAttributes: jest.fn((_params, action: () => unknown) => action()),
}));

const MockedCallbackHandler = CallbackHandler as jest.MockedClass<
  typeof CallbackHandler
>;
const MockedPropagateAttributes = propagateAttributes as jest.MockedFunction<
  typeof propagateAttributes
>;

async function createTestRun(
  agentName?: string,
  agentOverrides: Record<string, unknown> = {},
  runOverrides: Record<string, unknown> = {}
): Promise<Run<never>> {
  const run = await Run.create({
    runId: 'test-run-id',
    graphConfig: {
      type: 'standard',
      agents: [
        {
          agentId: 'agent_abc123',
          ...(agentName != null && { name: agentName }),
          provider: Providers.OPENAI,
          clientOptions: { model: 'gpt-4' },
          tools: [],
          ...agentOverrides,
        },
      ],
    },
    ...runOverrides,
  });

  const emptyStream = (async function* (): AsyncGenerator {
    /* no events */
  })();
  run.graphRunnable = { streamEvents: () => emptyStream } as never;

  return run;
}

describe('Langfuse trace metadata includes agentName', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = {
      ...originalEnv,
      LANGFUSE_SECRET_KEY: 'sk-test',
      LANGFUSE_PUBLIC_KEY: 'pk-test',
      LANGFUSE_BASE_URL: 'https://langfuse.test',
    };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('passes agentName in processStream traceMetadata when agent has a name', async () => {
    const run = await createTestRun('DWAINE');
    await run.processStream(
      { messages: [] },
      { configurable: { thread_id: 't1', user_id: 'u1' }, version: 'v2' }
    );

    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
    const ctorArgs = MockedCallbackHandler.mock.calls[0][0];
    expect(ctorArgs?.traceMetadata).toMatchObject({ agentName: 'DWAINE' });
  });

  it('propagates Langfuse identity around processStream observations', async () => {
    const run = await createTestRun('DWAINE');
    await run.processStream(
      { messages: [] },
      {
        configurable: {
          thread_id: 'thread-123',
          user_id: 'user-456',
          requestBody: { parentMessageId: 'parent-789' },
        },
        version: 'v2',
      }
    );

    expect(MockedPropagateAttributes).toHaveBeenCalledTimes(1);
    expect(MockedPropagateAttributes.mock.calls[0][0]).toMatchObject({
      userId: 'user-456',
      sessionId: 'thread-123',
      traceName: 'LibreChat Agent: DWAINE',
      tags: ['librechat', 'agent'],
      metadata: {
        messageId: 'test-run-id',
        parentMessageId: 'parent-789',
        agentId: 'agent_abc123',
        agentName: 'DWAINE',
      },
    });
  });

  it('falls back to agentId when agent has no explicit name', async () => {
    const run = await createTestRun();
    await run.processStream(
      { messages: [] },
      { configurable: { thread_id: 't1', user_id: 'u1' }, version: 'v2' }
    );

    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
    const ctorArgs = MockedCallbackHandler.mock.calls[0][0];
    expect(ctorArgs?.traceMetadata).toMatchObject({
      agentName: 'agent_abc123',
    });
  });

  it('does not create CallbackHandler when Langfuse env vars are missing', async () => {
    delete process.env.LANGFUSE_SECRET_KEY;
    const run = await createTestRun('MAIA');
    await run.processStream(
      { messages: [] },
      { configurable: { thread_id: 't1', user_id: 'u1' }, version: 'v2' }
    );

    expect(MockedCallbackHandler).not.toHaveBeenCalled();
    expect(MockedPropagateAttributes).not.toHaveBeenCalled();
  });

  it('does not create the legacy CallbackHandler when explicit agent config is supplied', async () => {
    const run = await createTestRun('DWAINE', {
      langfuse: {
        enabled: false,
      },
    });
    await run.processStream(
      { messages: [] },
      { configurable: { thread_id: 't1', user_id: 'u1' }, version: 'v2' }
    );

    expect(MockedCallbackHandler).not.toHaveBeenCalled();
  });

  it('uses the nested Langfuse CallbackHandler for redaction-only agent config', async () => {
    const run = await createTestRun('DWAINE', {
      langfuse: {
        toolOutputTracing: { enabled: false },
      },
    });
    await run.processStream(
      { messages: [] },
      { configurable: { thread_id: 't1', user_id: 'u1' }, version: 'v2' }
    );

    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
  });

  it('uses the nested Langfuse CallbackHandler for redaction-only run config', async () => {
    const run = await createTestRun(
      'DWAINE',
      {},
      {
        langfuse: {
          toolOutputTracing: { enabled: false },
        },
      }
    );
    await run.processStream(
      { messages: [] },
      { configurable: { thread_id: 't1', user_id: 'u1' }, version: 'v2' }
    );

    expect(MockedCallbackHandler).toHaveBeenCalledTimes(1);
  });

  it('preserves run-level Langfuse config after graph cleanup for later turns', async () => {
    const langfuse = {
      toolOutputTracing: { enabled: false },
    };
    const run = await createTestRun(
      'DWAINE',
      {},
      {
        langfuse,
      }
    );
    await run.processStream(
      { messages: [] },
      { configurable: { thread_id: 't1', user_id: 'u1' }, version: 'v2' }
    );

    expect(run.Graph?.langfuse).toBe(langfuse);
  });
});
