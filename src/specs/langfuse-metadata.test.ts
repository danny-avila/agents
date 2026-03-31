import { CallbackHandler } from '@langfuse/langchain';
import { Providers } from '@/common';
import { Run } from '@/run';

jest.mock('@langfuse/langchain', () => ({
  CallbackHandler: jest.fn().mockImplementation(() => ({})),
}));

const MockedCallbackHandler = CallbackHandler as jest.MockedClass<
  typeof CallbackHandler
>;

async function createTestRun(agentName?: string): Promise<Run<never>> {
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
        },
      ],
    },
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
  });
});
