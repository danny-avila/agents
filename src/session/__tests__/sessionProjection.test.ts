import { join } from 'path';
import { tmpdir } from 'os';
import { mkdtemp, rm, appendFile } from 'fs/promises';
import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  RemoveMessage,
} from '@langchain/core/messages';
import type { SessionEntry } from '../types';
import {
  deriveSessionMessages,
  releaseSessionProjection,
} from '../sessionProjection';
import { JsonlSessionStore } from '../JsonlSessionStore';
import { deriveMessages } from '../deriveMessages';

describe('session projection cache', () => {
  let dir: string;
  let store: JsonlSessionStore;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'session-projection-test-'));
    store = await JsonlSessionStore.create({
      cwd: dir,
      path: join(dir, 'session.jsonl'),
    });
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  function expectEquivalent(target = store): void {
    expect(deriveSessionMessages(target)).toEqual(
      deriveMessages(target.getPath())
    );
  }

  it('matches empty, appended, and tool-rich histories with fresh message instances', async () => {
    expectEquivalent();
    const messages = [
      new HumanMessage('Inspect the repository'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call-1', name: 'inspect', args: { path: 'src' } }],
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'result' }],
        tool_call_id: 'call-1',
      }),
      new AIMessage({
        content: 'Done',
        invalid_tool_calls: [{ name: 'bad', args: '{', error: 'invalid' }],
      }),
      new RemoveMessage({ id: 'removed' }),
    ];
    for (const message of messages) {
      await store.appendMessage(message);
      expectEquivalent();
    }
    const first = deriveSessionMessages(store);
    const second = deriveSessionMessages(store);
    expect(first).toEqual(second);
    expect(first.messages).not.toBe(second.messages);
    for (let i = 0; i < first.messages.length; i++) {
      expect(first.messages[i]).not.toBe(second.messages[i]);
    }
    first.messages[0].content = 'run-local edit';
    first.messages[0].id = 'run-local id';
    first.messages.pop();
    expectEquivalent();
  });

  it('preserves nested mutation semantics while recreating each message wrapper', async () => {
    await store.appendMessage(
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'call', name: 'inspect', args: { path: 'old' } }],
      })
    );
    const message = deriveSessionMessages(store).messages[0] as AIMessage;
    message.tool_calls![0].args.path = 'changed';
    expectEquivalent();
  });

  it('does not rebuild the full path for warm reads or append-only continuations', async () => {
    await store.appendMessage(new HumanMessage('start'));
    deriveSessionMessages(store);
    const getPath = jest.spyOn(store, 'getPath');
    for (let i = 0; i < 10; i++) {
      await store.appendRunEvent('run.started');
      await store.appendMessage(new AIMessage(`answer ${i}`));
      await store.appendCheckpoint({
        source: 'resume',
        threadId: 'thread',
        checkpointId: `checkpoint-${i}`,
      });
      deriveSessionMessages(store);
      deriveSessionMessages(store);
    }
    expect(getPath).not.toHaveBeenCalled();
    expectEquivalent();
  });

  it('matches successive summaries, retained messages, and summary mutation isolation', async () => {
    await store.appendMessage(new HumanMessage('old'));
    deriveSessionMessages(store);
    for (let i = 0; i < 3; i++) {
      const summary = await store.appendEntryForCompaction({
        text: `summary ${i}`,
        tokenCount: i + 20,
        retainedEntryIds: [],
        summarizedEntryIds: [],
      });
      expectEquivalent();
      const result = deriveSessionMessages(store);
      result.initialSummary!.text = 'caller edit';
      expectEquivalent();
      await store.appendMessage(new HumanMessage(`retained ${i}`), summary.id);
      await store.appendCompactionEntry({
        summaryEntryId: summary.id,
        retainedEntryIds: [],
        summarizedEntryIds: [],
      });
      await store.appendFadingState(null);
      expectEquivalent();
    }
  });

  it('rebuilds for branches, cleared or missing leaves, and roots with no shared prefix', async () => {
    const first = await store.appendMessage(new HumanMessage('first'));
    const second = await store.appendMessage(new AIMessage('second'));
    deriveSessionMessages(store);
    await store.branch(first.id);
    expectEquivalent();
    await store.appendMessage(new AIMessage('alternate'));
    expectEquivalent();
    await store.setLeaf(second.id);
    expectEquivalent();
    await store.setLeaf(null);
    expectEquivalent();
    await store.setLeaf('missing');
    expectEquivalent();
    await store.appendMessage(new HumanMessage('new root'), null);
    expectEquivalent();
    await store.branch(first.id, { position: 'before' });
    expectEquivalent();
  });

  it('handles leaf selection of non-message entries', async () => {
    const first = await store.appendMessage(new HumanMessage('first'));
    const event = await store.appendRunEvent('run.completed');
    await store.setLeaf(event.id);
    expectEquivalent();
    await store.appendMessage(new HumanMessage('after event'));
    expectEquivalent();
    await store.setLeaf(first.id);
    expectEquivalent();
  });

  it('reopens from disk without reusing another store projection', async () => {
    await store.appendMessage(new HumanMessage('persisted'));
    deriveSessionMessages(store);
    const reopened = await JsonlSessionStore.openPath(store.path);
    expectEquivalent(reopened);
    await reopened.appendMessage(new AIMessage('continued'));
    expectEquivalent(reopened);
    expectEquivalent();
  });

  it('permanently falls back after mutable store exposure, including retained references', async () => {
    const entry = await store.appendMessage(new HumanMessage('original'));
    deriveSessionMessages(store);
    releaseSessionProjection(store);
    const getPath = jest.spyOn(store, 'getPath');
    expectEquivalent();
    entry.data.message.content = 'edited later';
    entry.parentId = null;
    expectEquivalent();
    const state = store.getEntries().at(-1)!;
    if (state.type === 'session_state') {
      state.data.leafId = null;
    }
    expectEquivalent();
    expect(getPath).toHaveBeenCalledTimes(6);
  });

  it.each(['source', 'child'] as const)(
    'disables shared-entry fork caches when the %s is exposed',
    async (exposed) => {
      const entry = await store.appendMessage(new HumanMessage('shared'));
      deriveSessionMessages(store);
      const child = await store.clone();
      try {
        expectEquivalent(child);
        releaseSessionProjection(exposed === 'source' ? store : child);
        entry.id = 'changed-id';
        expectEquivalent();
        expectEquivalent(child);
      } finally {
        await rm(child.path);
      }
    }
  );

  it('inherits an already exposed source when creating a new fork', async () => {
    const entry = await store.appendMessage(new HumanMessage('shared'));
    releaseSessionProjection(store);
    const child = await store.clone();
    try {
      expectEquivalent(child);
      entry.id = 'changed-after-child-read';
      expectEquivalent(child);
    } finally {
      await rm(child.path);
    }
  });

  it('falls back for duplicate entry IDs in legacy logs', async () => {
    const entry = await store.appendMessage(new HumanMessage('first'));
    const duplicate: SessionEntry = {
      ...entry,
      timestamp: '2099-01-01T00:00:00.000Z',
      data: {
        ...entry.data,
        message: { ...entry.data.message, content: 'duplicate' },
      },
    };
    await appendFile(store.path, `${JSON.stringify(duplicate)}\n`);
    const reopened = await JsonlSessionStore.openPath(store.path);
    const getPath = jest.spyOn(reopened, 'getPath');
    expectEquivalent(reopened);
    expect(getPath).toHaveBeenCalledTimes(2);
  });
});
