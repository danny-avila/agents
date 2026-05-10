import { mkdtemp, rm } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { JsonlSessionStore, createAgentSession } from '@/session';
import * as providers from '@/llm/providers';

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
