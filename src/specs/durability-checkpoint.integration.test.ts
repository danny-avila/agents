import { MongoClient } from 'mongodb';
import { HumanMessage } from '@langchain/core/messages';
import { MongoMemoryServer } from 'mongodb-memory-server-core';
import { MongoDBSaver } from '@langchain/langgraph-checkpoint-mongodb';
import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import {
  END,
  START,
  Command,
  Annotation,
  StateGraph,
  interrupt,
} from '@langchain/langgraph';
import type * as t from '@/types';
import { Providers } from '@/common';
import { Run } from '@/run';

/**
 * End-to-end proof against a real Mongo store that the durability default is
 * the storage optimization it claims to be, not just a config flag. A real
 * `Run` with a fake streaming model (no provider API hit) writes through the
 * official MongoDBSaver, and a langgraph interrupt graph exercises the HITL
 * pause/resume boundary. `exit` persists only at the exit/interrupt boundary
 * with no per-superstep writes; `async` checkpoints every superstep.
 */
describe('durability default — real Mongo checkpointer', () => {
  let mongod: MongoMemoryServer;
  let client: MongoClient;

  beforeAll(async () => {
    // First CI run downloads a mongod binary — allow time beyond the 60s default.
    mongod = await MongoMemoryServer.create();
    client = new MongoClient(mongod.getUri());
    await client.connect();
  }, 120000);

  afterAll(async () => {
    await client.close();
    await mongod.stop();
  });

  const counts = async (
    dbName: string
  ): Promise<{ checkpoints: number; writes: number }> => {
    const db = client.db(dbName);
    const [checkpoints, writes] = await Promise.all([
      db.collection('checkpoints').countDocuments({}),
      db.collection('checkpoint_writes').countDocuments({}),
    ]);
    return { checkpoints, writes };
  };

  describe('normal run (real Run + fake model)', () => {
    const runOnce = async (
      dbName: string,
      durability?: t.Durability
    ): Promise<void> => {
      const run = await Run.create<t.IState>({
        runId: `durability-${dbName}`,
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'a',
              provider: Providers.OPENAI,
              clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
              instructions: 'noop',
              maxContextTokens: 8000,
            },
          ],
          compileOptions: {
            checkpointer: new MongoDBSaver({ client, dbName }),
          },
        },
      });
      // Fake streaming model: one assistant turn, no tool calls -> agent -> END.
      run.Graph?.overrideTestModel(['done'], 1);
      await run.processStream(
        { messages: [new HumanMessage('hi')] },
        {
          version: 'v2',
          ...(durability != null ? { durability } : {}),
          configurable: { thread_id: dbName },
        }
      );
    };

    it('exit default writes one checkpoint and no per-superstep writes; async writes more of both', async () => {
      await runOnce('normal_exit');
      await runOnce('normal_async', 'async');

      const exitCounts = await counts('normal_exit');
      const asyncCounts = await counts('normal_async');

      // exit: only the final checkpoint, zero per-superstep pending writes.
      expect(exitCounts).toEqual({ checkpoints: 1, writes: 0 });
      // async: a checkpoint plus pending writes for every superstep.
      expect(asyncCounts.checkpoints).toBeGreaterThan(exitCounts.checkpoints);
      expect(asyncCounts.writes).toBeGreaterThan(exitCounts.writes);
    });
  });

  describe('HITL interrupt (langgraph graph + real saver)', () => {
    const State = Annotation.Root({
      steps: Annotation<string[]>({
        reducer: (a, b) => a.concat(b),
        default: () => [],
      }),
    });

    const drain = async (stream: AsyncIterable<unknown>): Promise<void> => {
      for await (const _event of stream) {
        // Consume the event stream exactly like Run.processStream does.
      }
    };

    const runInterruptGraph = async (
      dbName: string,
      durability: t.Durability
    ): Promise<{
      atPause: { checkpoints: number; writes: number };
      done: { checkpoints: number; writes: number };
    }> => {
      const saver = new MongoDBSaver({ client, dbName });
      const graph = new StateGraph(State)
        .addNode('a', () => ({ steps: ['a'] }))
        .addNode('b', () => {
          interrupt({ ask: 'approve?' });
          return { steps: ['b'] };
        })
        .addNode('c', () => ({ steps: ['c'] }))
        .addEdge(START, 'a')
        .addEdge('a', 'b')
        .addEdge('b', 'c')
        .addEdge('c', END)
        .compile({ checkpointer: saver });

      // durability rides in the 2nd-arg config, exactly as run.ts threads it.
      const cfg = {
        configurable: { thread_id: dbName },
        durability,
        version: 'v2' as const,
      };

      await drain(graph.streamEvents({ steps: [] }, cfg, { raiseError: true }));
      const atPause = await counts(dbName);

      await drain(
        graph.streamEvents(new Command({ resume: 'ok' }), cfg, {
          raiseError: true,
        })
      );
      const done = await counts(dbName);

      return { atPause, done };
    };

    it('exit checkpoints only at the interrupt boundary and final exit; async checkpoints every superstep', async () => {
      const exitRun = await runInterruptGraph('hitl_exit', 'exit');
      const asyncRun = await runInterruptGraph('hitl_async', 'async');

      // exit: one checkpoint at the pause (the interrupt), two once resumed
      // (interrupt boundary + final exit).
      expect(exitRun.atPause.checkpoints).toBe(1);
      expect(exitRun.done.checkpoints).toBe(2);

      // async: a checkpoint for every superstep, strictly more at each stage.
      expect(asyncRun.atPause.checkpoints).toBeGreaterThan(
        exitRun.atPause.checkpoints
      );
      expect(asyncRun.done.checkpoints).toBeGreaterThan(
        exitRun.done.checkpoints
      );
      expect(asyncRun.done.writes).toBeGreaterThan(exitRun.done.writes);
    });
  });

  describe('multiple interrupts (sequential ask-user-questions)', () => {
    const State = Annotation.Root({
      qa: Annotation<string[]>({
        reducer: (a, b) => a.concat(b),
        default: () => [],
      }),
    });

    const drain = async (stream: AsyncIterable<unknown>): Promise<void> => {
      for await (const _event of stream) {
        // Consume the event stream.
      }
    };

    it('checkpoints each interrupt boundary, chains them, and discards none across resumes', async () => {
      const dbName = 'multi_exit';
      const saver = new MongoDBSaver({ client, dbName });
      const graph = new StateGraph(State)
        .addNode('ask1', () => ({ qa: [`Q1=${interrupt({ q: 'Q1' })}`] }))
        .addNode('ask2', () => ({ qa: [`Q2=${interrupt({ q: 'Q2' })}`] }))
        .addNode('done', () => ({ qa: ['done'] }))
        .addEdge(START, 'ask1')
        .addEdge('ask1', 'ask2')
        .addEdge('ask2', 'done')
        .addEdge('done', END)
        .compile({ checkpointer: saver });

      const cfg = {
        configurable: { thread_id: dbName },
        durability: 'exit' as const,
        version: 'v2' as const,
      };

      // First question -> pauses at ask1, one checkpoint at that boundary.
      await drain(graph.streamEvents({ qa: [] }, cfg, { raiseError: true }));
      let state = await graph.getState(cfg);
      expect(state.next).toEqual(['ask1']);
      expect((await counts(dbName)).checkpoints).toBe(1);

      // Answer 1 -> the run continues and the LLM asks a second question.
      // The chain grows to 2: the first interrupt checkpoint is retained as
      // the parent, not discarded or overwritten.
      await drain(
        graph.streamEvents(new Command({ resume: 'A1' }), cfg, {
          raiseError: true,
        })
      );
      state = await graph.getState(cfg);
      expect(state.next).toEqual(['ask2']);
      expect(state.values.qa).toEqual(['Q1=A1']);
      expect((await counts(dbName)).checkpoints).toBe(2);

      // Answer 2 -> run completes; both answers applied, three checkpoints
      // total (two interrupt boundaries + final exit).
      await drain(
        graph.streamEvents(new Command({ resume: 'A2' }), cfg, {
          raiseError: true,
        })
      );
      state = await graph.getState(cfg);
      expect(state.next).toEqual([]);
      expect(state.values.qa).toEqual(['Q1=A1', 'Q2=A2', 'done']);
      expect((await counts(dbName)).checkpoints).toBe(3);
    });
  });
});
