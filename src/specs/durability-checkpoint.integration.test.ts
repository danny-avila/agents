import { MongoClient } from 'mongodb';
import { HumanMessage } from '@langchain/core/messages';
import { MongoMemoryServer } from 'mongodb-memory-server-core';
import { MongoDBSaver } from '@langchain/langgraph-checkpoint-mongodb';
import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import type * as t from '@/types';
import { Providers } from '@/common';
import { Run } from '@/run';

/**
 * End-to-end proof against a real Mongo store: a real `Run` with a fake
 * streaming model (no provider API hit) writes checkpoints through the
 * official MongoDBSaver. The SDK's default `exit` durability persists only
 * the final checkpoint, while an explicit `async` override checkpoints every
 * superstep — the storage-level payoff behind the `processStream` default.
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

  const buildRun = async (dbName: string): Promise<Run<t.IState>> => {
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
        compileOptions: { checkpointer: new MongoDBSaver({ client, dbName }) },
      },
    });
    // Fake streaming model: one assistant turn, no tool calls -> agent -> END.
    run.Graph?.overrideTestModel(['done'], 1);
    return run;
  };

  const countCheckpoints = (dbName: string): Promise<number> =>
    client.db(dbName).collection('checkpoints').countDocuments({});

  it('persists a single checkpoint under the default exit durability', async () => {
    const run = await buildRun('exit_default');
    await run.processStream(
      { messages: [new HumanMessage('hi')] },
      { version: 'v2', configurable: { thread_id: 'thread-exit' } }
    );
    expect(await countCheckpoints('exit_default')).toBe(1);
  });

  it('checkpoints every superstep when the caller overrides durability to async', async () => {
    const run = await buildRun('async_override');
    await run.processStream(
      { messages: [new HumanMessage('hi')] },
      {
        version: 'v2',
        durability: 'async',
        configurable: { thread_id: 'thread-async' },
      }
    );
    expect(await countCheckpoints('async_override')).toBeGreaterThan(1);
  });
});
