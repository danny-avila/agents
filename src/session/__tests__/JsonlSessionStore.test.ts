import { mkdtemp, rm } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { MemorySaver } from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';
import type { Checkpoint, CheckpointMetadata } from '@langchain/langgraph';
import { JsonlSessionStore, createAgentSession } from '@/session';
import * as providers from '@/llm/providers';
import type * as t from '@/types';
import { Run } from '@/run';

type MockRun = {
  processStream: jest.MockedFunction<Run<t.IState>['processStream']>;
  resume: jest.MockedFunction<Run<t.IState>['resume']>;
  getRunMessages: jest.MockedFunction<Run<t.IState>['getRunMessages']>;
  getCalibrationRatio: jest.MockedFunction<
    Run<t.IState>['getCalibrationRatio']
  >;
  getInterrupt: jest.MockedFunction<Run<t.IState>['getInterrupt']>;
  getHaltReason: jest.MockedFunction<Run<t.IState>['getHaltReason']>;
};

function createMockRun(outputText = 'ok'): MockRun {
  return {
    processStream: jest
      .fn<
        ReturnType<Run<t.IState>['processStream']>,
        Parameters<Run<t.IState>['processStream']>
      >()
      .mockResolvedValue([{ type: 'text', text: outputText }]),
    resume: jest
      .fn<
        ReturnType<Run<t.IState>['resume']>,
        Parameters<Run<t.IState>['resume']>
      >()
      .mockResolvedValue([{ type: 'text', text: outputText }]),
    getRunMessages: jest.fn(() => [new AIMessage(outputText)]),
    getCalibrationRatio: jest.fn(() => 1),
    getInterrupt: jest.fn(() => undefined),
    getHaltReason: jest.fn(() => undefined),
  };
}

function mockRunCreate(mockRun: MockRun): t.RunConfig[] {
  const capturedConfigs: t.RunConfig[] = [];
  jest.spyOn(Run, 'create').mockImplementation((async <
    T extends t.BaseGraphState,
  >(
    config: t.RunConfig
  ): Promise<Run<T>> => {
    capturedConfigs.push(config);
    return mockRun as unknown as Run<T>;
  }) as never);
  return capturedConfigs;
}

function getProcessedMessages(mockRun: MockRun): BaseMessage[] {
  expect(mockRun.processStream).toHaveBeenCalled();
  const input = mockRun.processStream.mock.calls[0][0];
  if (!('messages' in input)) {
    throw new Error('Expected processStream to receive message state');
  }
  return input.messages;
}

async function putCheckpoint(params: {
  checkpointer: MemorySaver;
  threadId: string;
  id: string;
  checkpointNs?: string;
}): Promise<void> {
  const checkpoint: Checkpoint = {
    v: 4,
    id: params.id,
    ts: new Date().toISOString(),
    channel_values: {},
    channel_versions: {},
    versions_seen: {},
  };
  const metadata: CheckpointMetadata = {
    source: 'loop',
    step: 0,
    parents: {},
  };
  await params.checkpointer.put(
    {
      configurable: {
        thread_id: params.threadId,
        checkpoint_ns: params.checkpointNs ?? '',
      },
    },
    checkpoint,
    metadata
  );
}

function mockSummarizer(response: string): void {
  jest.spyOn(providers, 'getChatModelClass').mockReturnValue(
    class {
      constructor() {
        return {
          invoke: jest.fn().mockResolvedValue({ content: response }),
        };
      }
    } as never
  );
}

describe('JsonlSessionStore', () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'lc-agent-session-'));
  });

  afterEach(async () => {
    jest.restoreAllMocks();
    await rm(dir, { recursive: true, force: true });
  });

  it('stores messages as an append-only tree and restores the active path', async () => {
    const path = join(dir, 'session.jsonl');
    const store = await JsonlSessionStore.create({
      path,
      cwd: dir,
      sessionId: 'session-a',
    });

    const user = await store.appendMessage(new HumanMessage('hello'));
    const assistant = await store.appendMessage(new AIMessage('hi'));

    const reopened = await JsonlSessionStore.open(path);

    expect(reopened.header.id).toBe('session-a');
    expect(reopened.getLeafEntry()?.id).toBe(assistant.id);
    expect(reopened.getPath().map((entry) => entry.id)).toEqual([
      user.id,
      assistant.id,
    ]);
    expect(reopened.getMessages().map((message) => message.content)).toEqual([
      'hello',
      'hi',
    ]);
  });

  it('branches in place without deleting abandoned children', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'branch.jsonl'),
      cwd: dir,
    });
    const first = await store.appendMessage(new HumanMessage('one'));
    const abandoned = await store.appendMessage(new AIMessage('abandoned'));

    await store.branch(first.id);
    const alternate = await store.appendMessage(new AIMessage('alternate'));

    expect(
      store
        .getChildren(first.id)
        .filter((entry) => entry.type === 'message')
        .map((entry) => entry.id)
        .sort()
    ).toEqual([abandoned.id, alternate.id].sort());
    expect(store.getPath().map((entry) => entry.id)).toEqual([
      first.id,
      alternate.id,
    ]);
  });

  it('clones and forks active paths into new session files', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'source.jsonl'),
      cwd: dir,
    });
    const first = await store.appendMessage(new HumanMessage('first'));
    const second = await store.appendMessage(new AIMessage('second'));

    const clone = await store.clone({ cwd: dir });
    const fork = await store.fork(second.id, { cwd: dir, position: 'before' });

    expect(clone.header.parentSession).toBe(store.path);
    expect(clone.getPath().map((entry) => entry.id)).toEqual([
      first.id,
      second.id,
    ]);
    expect(fork.getPath().map((entry) => entry.id)).toEqual([first.id]);
  });

  it('tracks labels and compaction entries', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'labels.jsonl'),
      cwd: dir,
    });
    const message = await store.appendMessage(new HumanMessage('hello'));

    await store.setLabel(message.id, 'checkpoint');
    const summary = await store.appendEntryForCompaction({
      text: 'summary',
      retainedEntryIds: [message.id],
      summarizedEntryIds: [],
    });
    const compaction = await store.appendCompactionEntry({
      summaryEntryId: summary.id,
      retainedEntryIds: [message.id],
      summarizedEntryIds: [],
    });

    expect(store.getLabel(message.id)).toBe('checkpoint');
    expect(summary.data.text).toBe('summary');
    expect(compaction.data.summaryEntryId).toBe(summary.id);
  });

  it('records LangGraph checkpoint references without moving the active leaf', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'checkpoints.jsonl'),
      cwd: dir,
    });
    const message = await store.appendMessage(new HumanMessage('hello'));

    const checkpoint = await store.appendCheckpoint({
      source: 'run',
      threadId: store.header.id,
      runId: 'run_checkpoint',
      checkpointId: 'checkpoint_1',
      checkpointNs: '',
    });

    expect(checkpoint.data.provider).toBe('langgraph');
    expect(store.getLeafEntry()?.id).toBe(message.id);
    expect(store.getLatestCheckpoint(store.header.id)?.id).toBe(checkpoint.id);
  });

  it('creates high-level sessions with a JSONL store by default', async () => {
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.getSessionStore()?.header.cwd).toBe(dir);
    expect(session.sessionPath).toContain('.jsonl');
  });

  it('shares a session-level LangGraph checkpointer for HITL resume', async () => {
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.getCheckpointer()).toBeInstanceOf(MemorySaver);
  });

  it('preserves a caller-supplied session checkpointer', async () => {
    const checkpointer = new MemorySaver();
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    expect(session.getCheckpointer()).toBe(checkpointer);
  });

  it('injects the session checkpointer and replays JSONL history before checkpoints exist', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('first output');
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));

    await session.run('next');

    expect(capturedConfigs[0].graphConfig.compileOptions?.checkpointer).toBe(
      checkpointer
    );
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['history', 'next']);
  });

  it('uses only new input when LangGraph checkpoint state already exists', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('checkpointed output');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_existing',
    });

    await session.run('fresh turn', { runId: 'run_checkpointed' });

    const checkpoints = session
      .getSessionStore()
      ?.getCheckpoints(session.threadId);
    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['fresh turn']);
    expect(checkpoints?.at(-1)?.data).toMatchObject({
      source: 'run',
      runId: 'run_checkpointed',
      checkpointId: 'checkpoint_existing',
    });
  });

  it('replays JSONL history when the requested checkpoint namespace has no state', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('namespace output');
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_other_namespace',
      checkpointNs: 'other',
    });

    await session.run('fresh turn', {
      config: { configurable: { checkpoint_ns: 'requested' } },
    });

    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['history', 'fresh turn']);
  });

  it('resets stale checkpoint state when branching changes the active JSONL path', async () => {
    const checkpointer = new MemorySaver();
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: { checkpointer },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const first = await store?.appendMessage(new HumanMessage('first'));
    await store?.appendMessage(new AIMessage('second'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_to_reset',
    });

    await session.branch(first?.id ?? '', { position: 'at' });

    const tuple = await checkpointer.getTuple({
      configurable: { thread_id: session.threadId },
    });
    expect(tuple).toBeUndefined();
    expect(store?.getCheckpoints(session.threadId).at(-1)?.data).toMatchObject({
      source: 'reset',
      reason: 'branch',
    });
  });

  it('records run.failed when resumeInterrupt throws', async () => {
    const mockRun = createMockRun('unused');
    mockRun.resume.mockRejectedValue(new Error('resume failed'));
    mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });

    await expect(
      session.resumeInterrupt([{ type: 'approve' }], {
        runId: 'run_resume_failure',
      })
    ).rejects.toThrow('resume failed');

    const events = session
      .getSessionStore()
      ?.getEntries()
      .filter((entry) => entry.type === 'run_event')
      .map((entry) => entry.data.event);
    expect(events).toEqual(['run.started', 'run.failed']);
  });

  it('compacts into a summary plus retained active path', async () => {
    mockSummarizer('summary of old work');
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    await store?.appendMessage(new HumanMessage('old'));
    await store?.appendMessage(new AIMessage('old answer'));
    await store?.appendMessage(new HumanMessage('recent'));

    await session.compact({
      instructions: 'summary of old work',
      retainRecentTurns: 1,
    });

    expect(store?.getMessages().map((message) => message.content)).toEqual([
      'summary of old work',
      'recent',
    ]);
  });

  it('summarizes an abandoned branch before switching in place', async () => {
    mockSummarizer('summary of abandoned branch');
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
      },
    });
    const store = session.getSessionStore();
    const first = await store?.appendMessage(new HumanMessage('first'));
    await store?.appendMessage(new AIMessage('abandoned answer'));

    await session.branch(first?.id ?? '', {
      position: 'at',
      summarizeAbandoned: {
        instructions: 'summarize abandoned branch',
      },
    });

    expect(store?.getLeafEntry()?.id).toBe(first?.id);
    expect(
      store
        ?.getEntries()
        .some(
          (entry) =>
            entry.type === 'summary' &&
            entry.data.text === 'summary of abandoned branch' &&
            entry.data.instructions === 'summarize abandoned branch'
        )
    ).toBe(true);
    expect(
      store?.getEntries().some((entry) => entry.type === 'compaction')
    ).toBe(true);
  });
});
