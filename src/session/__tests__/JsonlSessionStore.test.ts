import { mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import { dirname, join } from 'path';
import { tmpdir } from 'os';
import {
  AIMessage,
  HumanMessage,
  RemoveMessage,
} from '@langchain/core/messages';
import { MemorySaver } from '@langchain/langgraph';
import type { BaseMessage } from '@langchain/core/messages';
import type { Checkpoint, CheckpointMetadata } from '@langchain/langgraph';
import { JsonlSessionStore, createAgentSession } from '@/session';
import { toJsonValue } from '@/session/messageSerialization';
import * as providers from '@/llm/providers';
import { GraphEvents } from '@/common';
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

function getProcessedState(mockRun: MockRun): t.IState {
  expect(mockRun.processStream).toHaveBeenCalled();
  const input = mockRun.processStream.mock.calls[0][0];
  if (!('messages' in input)) {
    throw new Error('Expected processStream to receive message state');
  }
  return input;
}

function getProcessedMessages(mockRun: MockRun): BaseMessage[] {
  return getProcessedState(mockRun).messages;
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

  it('round-trips remove messages in persisted sessions', async () => {
    const path = join(dir, 'remove.jsonl');
    const store = await JsonlSessionStore.create({ path, cwd: dir });

    await store.appendMessage(
      new HumanMessage({ id: 'message-a', content: 'a' })
    );
    await store.appendMessage(new RemoveMessage({ id: 'message-a' }));

    const reopened = await JsonlSessionStore.open(path);
    const messages = reopened.getMessages();

    expect(messages.map((message) => message._getType())).toEqual([
      'human',
      'remove',
    ]);
    expect((messages[1] as RemoveMessage).id).toBe('message-a');
  });

  it('fails when creating a session file that already exists', async () => {
    const path = join(dir, 'existing.jsonl');
    await JsonlSessionStore.create({
      path,
      cwd: dir,
      sessionId: 'session-a',
    });

    await expect(
      JsonlSessionStore.create({
        path,
        cwd: dir,
        sessionId: 'session-b',
      })
    ).rejects.toMatchObject({ code: 'EEXIST' });

    const raw = await readFile(path, 'utf8');
    expect(raw.match(/"type":"session"/g)).toHaveLength(1);
  });

  it('keeps default session roots distinct for similar cwd strings', async () => {
    const cwdA = join(dir, 'foo/bar');
    const cwdB = join(dir, 'foo-bar');
    const storeA = await JsonlSessionStore.create({
      cwd: cwdA,
      sessionId: 'cwd-a',
    });
    const storeB = await JsonlSessionStore.create({
      cwd: cwdB,
      sessionId: 'cwd-b',
    });

    try {
      const [itemsA, itemsB] = await Promise.all([
        JsonlSessionStore.list(cwdA),
        JsonlSessionStore.list(cwdB),
      ]);

      expect(dirname(storeA.path)).not.toBe(dirname(storeB.path));
      expect(itemsA.map((item) => item.id)).toContain('cwd-a');
      expect(itemsA.map((item) => item.id)).not.toContain('cwd-b');
      expect(itemsB.map((item) => item.id)).toContain('cwd-b');
      expect(itemsB.map((item) => item.id)).not.toContain('cwd-a');
    } finally {
      await Promise.all([
        rm(storeA.path, { force: true }),
        rm(storeB.path, { force: true }),
      ]);
    }
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

  it('treats reset checkpoints as latest checkpoint barriers', async () => {
    const store = await JsonlSessionStore.create({
      path: join(dir, 'checkpoint-reset.jsonl'),
      cwd: dir,
    });
    await store.appendCheckpoint({
      source: 'run',
      threadId: store.header.id,
      runId: 'run_before_reset',
      checkpointId: 'checkpoint_before_reset',
    });
    await store.appendCheckpoint({
      source: 'reset',
      threadId: store.header.id,
      reason: 'branch',
    });

    expect(store.getLatestCheckpoint(store.header.id)).toBeUndefined();

    const checkpoint = await store.appendCheckpoint({
      source: 'run',
      threadId: store.header.id,
      runId: 'run_after_reset',
      checkpointId: 'checkpoint_after_reset',
    });

    expect(store.getLatestCheckpoint(store.header.id)?.id).toBe(checkpoint.id);
  });

  it('preserves Error details in JSONL payloads', () => {
    const error = new Error('resume failed');
    const payload = toJsonValue(error);

    expect(payload).toMatchObject({
      name: 'Error',
      message: 'resume failed',
    });
    expect(
      typeof payload === 'object' &&
        payload != null &&
        !Array.isArray(payload) &&
        typeof payload.stack === 'string'
    ).toBe(true);
  });

  it('replaces circular object references in JSONL payloads', () => {
    interface CircularPayload {
      label: string;
      self?: CircularPayload;
      child?: { parent?: CircularPayload };
    }
    const circular: CircularPayload = { label: 'root' };
    circular.self = circular;
    circular.child = { parent: circular };

    expect(toJsonValue(circular)).toMatchObject({
      label: 'root',
      self: '[Circular]',
      child: { parent: '[Circular]' },
    });
  });

  it('replaces circular Error causes in JSONL payloads', () => {
    const error = new Error('request failed');
    Object.defineProperty(error, 'cause', {
      value: error,
      configurable: true,
    });

    expect(toJsonValue(error)).toMatchObject({
      name: 'Error',
      message: 'request failed',
      cause: '[Circular]',
    });
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

  it('surfaces invalid explicit session files instead of replacing them', async () => {
    const sessionPath = join(dir, 'invalid.jsonl');
    await writeFile(sessionPath, 'not jsonl\n', 'utf8');

    await expect(
      createAgentSession({
        cwd: dir,
        sessionPath,
        runId: 'template-run',
        graphConfig: {
          type: 'standard',
          llmConfig: {
            provider: 'openAI' as never,
            model: 'test-model',
          },
          instructions: 'test',
        },
      })
    ).rejects.toThrow('Invalid session file');

    expect(await readFile(sessionPath, 'utf8')).toBe('not jsonl\n');
  });

  it('creates an explicit session path without fuzzy matching existing sessions', async () => {
    const existing = await JsonlSessionStore.create({
      path: join(dir, 'matching-existing.jsonl'),
      cwd: dir,
      sessionId: 'explicit-target-existing',
    });
    await existing.appendMessage(new HumanMessage('existing history'));
    const sessionPath = join(dir, 'explicit-target.jsonl');

    const session = await createAgentSession({
      cwd: dir,
      sessionPath,
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

    expect(session.sessionPath).toBe(sessionPath);
    expect(session.getSessionStore()?.header.id).not.toBe(existing.header.id);
    expect(session.getSessionStore()?.getMessages()).toEqual([]);
  });

  it('preserves non-message state while applying session history', async () => {
    const mockRun = createMockRun('stateful output');
    mockRunCreate(mockRun);
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
    await session.getSessionStore()?.appendMessage(new HumanMessage('history'));
    const input: t.IState & { selectedAgent: string } = {
      messages: [new HumanMessage('fresh')],
      selectedAgent: 'subagent-a',
    };

    await session.run(input);

    const processedState = getProcessedState(mockRun) as t.IState & {
      selectedAgent?: string;
    };
    expect(processedState.selectedAgent).toBe('subagent-a');
    expect(processedState.messages.map((message) => message.content)).toEqual([
      'history',
      'fresh',
    ]);
  });

  it('restores persisted summary token counts into run config', async () => {
    const mockRun = createMockRun('with summary');
    const capturedConfigs = mockRunCreate(mockRun);
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
    await session.getSessionStore()?.appendEntryForCompaction({
      text: 'stored summary',
      tokenCount: 123,
      retainedEntryIds: [],
      summarizedEntryIds: [],
    });

    await session.run('fresh');

    const { graphConfig } = capturedConfigs[0];
    const initialSummary =
      'initialSummary' in graphConfig ? graphConfig.initialSummary : undefined;
    expect(initialSummary).toEqual({
      text: 'stored summary',
      tokenCount: 123,
    });
  });

  it('preserves custom handlers outside the session event adapter set', async () => {
    const mockRun = createMockRun('handled');
    const capturedConfigs = mockRunCreate(mockRun);
    const agentLogHandler: t.EventHandler = { handle: jest.fn() };
    const messageDeltaHandler: t.EventHandler = { handle: jest.fn() };
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
      customHandlers: {
        [GraphEvents.ON_AGENT_LOG]: agentLogHandler,
        [GraphEvents.ON_MESSAGE_DELTA]: messageDeltaHandler,
      },
    });

    await session.run('start');

    expect(capturedConfigs[0].customHandlers?.[GraphEvents.ON_AGENT_LOG]).toBe(
      agentLogHandler
    );
    expect(
      capturedConfigs[0].customHandlers?.[GraphEvents.ON_MESSAGE_DELTA]
    ).not.toBe(messageDeltaHandler);
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

  it('keeps stores and checkpointing optional for high-level sessions', async () => {
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      ephemeral: true,
      checkpointing: false,
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

    expect(session.getSessionStore()).toBeUndefined();
    expect(session.getCheckpointer()).toBeUndefined();
  });

  it('removes graph checkpointers when checkpointing is disabled', async () => {
    const graphCheckpointer = new MemorySaver();
    const mockRun = createMockRun('disabled');
    const capturedConfigs = mockRunCreate(mockRun);
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      checkpointing: false,
      graphConfig: {
        type: 'standard',
        llmConfig: {
          provider: 'openAI' as never,
          model: 'test-model',
        },
        instructions: 'test',
        compileOptions: { checkpointer: graphCheckpointer },
      },
    });

    await session.run('start');

    expect(session.getCheckpointer()).toBeUndefined();
    expect(
      capturedConfigs[0].graphConfig.compileOptions?.checkpointer
    ).toBeUndefined();
  });

  it('reuses the session-level checkpointer across HITL resume', async () => {
    const mockRun = createMockRun('resumed');
    const capturedConfigs = mockRunCreate(mockRun);
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

    await session.run('start');
    await session.resumeInterrupt([]);

    const checkpointer =
      capturedConfigs[0].graphConfig.compileOptions?.checkpointer;
    expect(checkpointer).toBeInstanceOf(MemorySaver);
    expect(session.getCheckpointer()).toBe(checkpointer);
    expect(capturedConfigs[1].graphConfig.compileOptions?.checkpointer).toBe(
      checkpointer
    );
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

  it('does not replay session history when overriding thread id without checkpoint state', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('override output');
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

    await session.run('fresh turn', { threadId: 'thread_override' });

    expect(
      getProcessedMessages(mockRun).map((message) => message.content)
    ).toEqual(['fresh turn']);
    expect(
      session
        .getSessionStore()
        ?.getPath()
        .filter((entry) => entry.type === 'message')
        .map((entry) => entry.data.message.content)
    ).toEqual(['history']);
  });

  it('does not persist resumed override thread messages into the session path', async () => {
    const checkpointer = new MemorySaver();
    const mockRun = createMockRun('override resumed');
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

    await session.resumeInterrupt([], { threadId: 'thread_override' });

    expect(mockRun.resume).toHaveBeenCalledWith(
      [],
      expect.objectContaining({
        configurable: expect.objectContaining({
          thread_id: 'thread_override',
        }),
      })
    );
    expect(
      session
        .getSessionStore()
        ?.getPath()
        .filter((entry) => entry.type === 'message')
        .map((entry) => entry.data.message.content)
    ).toEqual(['history']);
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

  it('looks up the latest checkpoint in the requested namespace', async () => {
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
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_requested_namespace',
      checkpointNs: 'requested',
    });

    await expect(session.getLatestCheckpoint()).resolves.toBeUndefined();
    await expect(
      session.getLatestCheckpoint({ checkpointNs: 'requested' })
    ).resolves.toMatchObject({
      checkpointId: 'checkpoint_requested_namespace',
      checkpointNs: 'requested',
    });
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

  it('keeps checkpoint state when branching to the active JSONL leaf', async () => {
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
    const active = await store?.appendMessage(new HumanMessage('current'));
    await putCheckpoint({
      checkpointer,
      threadId: session.threadId,
      id: 'checkpoint_to_keep',
    });

    await session.branch(active?.id ?? '', { position: 'at' });

    const tuple = await checkpointer.getTuple({
      configurable: { thread_id: session.threadId },
    });
    expect(tuple?.checkpoint.id).toBe('checkpoint_to_keep');
    expect(
      store
        ?.getCheckpoints(session.threadId)
        .some((checkpoint) => checkpoint.data.source === 'reset')
    ).toBe(false);
  });

  it('resets overridden thread checkpoints when branching changes the active path', async () => {
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
      threadId: 'thread_override',
      id: 'checkpoint_override',
    });
    await store?.appendCheckpoint({
      source: 'run',
      threadId: 'thread_override',
      runId: 'run_override',
      checkpointId: 'checkpoint_override',
    });

    await session.branch(first?.id ?? '', { position: 'at' });

    const tuple = await checkpointer.getTuple({
      configurable: { thread_id: 'thread_override' },
    });
    const reset = store
      ?.getCheckpoints('thread_override')
      .find((checkpoint) => checkpoint.data.source === 'reset');
    expect(tuple).toBeUndefined();
    expect(reset?.data.reason).toBe('branch');
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
    const tokenCounter: t.TokenCounter = () => 7;
    const session = await createAgentSession({
      cwd: dir,
      runId: 'template-run',
      tokenCounter,
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
    const summary = store
      ?.getEntries()
      .find((entry) => entry.type === 'summary');
    expect(summary?.data.tokenCount).toBeGreaterThan(0);
  });

  it('records no retained ids when compaction retains zero messages', async () => {
    mockSummarizer('summary of everything');
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
    const user = await store?.appendMessage(new HumanMessage('old'));
    const assistant = await store?.appendMessage(new AIMessage('old answer'));

    await session.compact({ retainRecentTurns: 0 });

    const summary = store
      ?.getEntries()
      .find((entry) => entry.type === 'summary');
    const compaction = store
      ?.getEntries()
      .find((entry) => entry.type === 'compaction');
    expect(summary?.data.retainedEntryIds).toEqual([]);
    expect(summary?.data.summarizedEntryIds).toEqual([user?.id, assistant?.id]);
    expect(compaction?.data.retainedEntryIds).toEqual([]);
  });

  it('carries calibration ratio forward after resumeInterrupt', async () => {
    const mockRun = createMockRun('resumed');
    mockRun.getCalibrationRatio.mockReturnValue(2);
    const capturedConfigs = mockRunCreate(mockRun);
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

    await session.resumeInterrupt([]);
    await session.run('after resume');

    expect(capturedConfigs[1].calibrationRatio).toBe(2);
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
    const abandoned = await store?.appendMessage(
      new AIMessage('abandoned answer')
    );

    await session.branch(first?.id ?? '', {
      position: 'at',
      summarizeAbandoned: {
        instructions: 'summarize abandoned branch',
      },
    });

    const activePath = store?.getPath();
    const summary = activePath?.at(-1);
    expect(activePath?.map((entry) => entry.id)).toEqual([
      first?.id,
      summary?.id,
    ]);
    expect(summary).toMatchObject({
      type: 'summary',
      parentId: first?.id,
      data: {
        text: 'summary of abandoned branch',
        summarizedEntryIds: [abandoned?.id],
        instructions: 'summarize abandoned branch',
      },
    });
    expect(
      store?.getEntries().some((entry) => entry.type === 'compaction')
    ).toBe(true);
  });

  it('summarizes a sibling branch before switching branches', async () => {
    mockSummarizer('summary of sibling branch');
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
    const inactive = await store?.appendMessage(new AIMessage('inactive'));
    await store?.branch(first?.id ?? '');
    const activeSibling = await store?.appendMessage(new AIMessage('active'));

    await session.branch(inactive?.id ?? '', {
      position: 'at',
      summarizeAbandoned: true,
    });

    const activePath = store?.getPath();
    const summary = activePath?.at(-1);
    expect(activePath?.map((entry) => entry.id)).toEqual([
      first?.id,
      inactive?.id,
      summary?.id,
    ]);
    expect(summary).toMatchObject({
      type: 'summary',
      parentId: inactive?.id,
      data: {
        text: 'summary of sibling branch',
        summarizedEntryIds: [activeSibling?.id],
      },
    });
  });
});
