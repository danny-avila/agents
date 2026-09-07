/* eslint-disable no-console */
import assert from 'node:assert/strict';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { HumanMessage } from '@langchain/core/messages';
import type { SessionEntry, SessionHeader } from '../session/types';
import { JsonlSessionStore } from '../session/JsonlSessionStore';
import { deriveSessionMessages } from '../session/sessionProjection';
import { deriveMessages } from '../session/deriveMessages';

function fixture(turns: number, compacted: boolean): SessionEntry[] {
  const entries: SessionEntry[] = [];
  let parentId: string | null = null;
  for (let i = 0; i < turns; i++) {
    const timestamp = new Date(1_700_000_000_000 + i).toISOString();
    const id = `message-${i}`;
    if (compacted && i === turns - 40) {
      entries.push({
        type: 'summary',
        id: 'summary',
        parentId,
        timestamp,
        data: {
          text: 'Earlier work is summarized here.',
          tokenCount: 8,
          retainedEntryIds: [],
          summarizedEntryIds: [],
        },
      });
      parentId = 'summary';
    }
    entries.push({
      type: 'message',
      id,
      parentId,
      timestamp,
      data: {
        role: i % 2 === 0 ? 'assistant' : 'tool',
        message:
          i % 2 === 0
            ? {
                messageType: 'ai',
                content: '',
                toolCalls: [
                  { id: `call-${i}`, name: 'inspect', args: { path: 'src' } },
                ],
              }
            : {
                messageType: 'tool',
                content: 'Repository evidence. '.repeat(30),
                toolCallId: `call-${i - 1}`,
              },
      },
    });
    entries.push({
      type: 'session_state',
      id: `state-${i}`,
      parentId: id,
      timestamp,
      data: { leafId: id },
    });
    entries.push({
      type: 'run_event',
      id: `event-${i}`,
      parentId: id,
      timestamp,
      data: { event: 'run.completed' },
    });
    parentId = id;
  }
  return entries;
}

function median(values: number[]): number {
  return values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
}

function measure(run: () => void, iterations: number): number {
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    run();
  }
  return (performance.now() - start) / iterations;
}

async function main(): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), 'bench-session-projection-'));
  try {
    const rows = [];
    for (const compacted of [false, true]) {
      for (const turns of [100, 1_000, 10_000]) {
        const path = join(dir, `${turns}-${compacted}.jsonl`);
        const header: SessionHeader = {
          type: 'session',
          version: 1,
          id: 'benchmark',
          timestamp: '2023-01-01T00:00:00.000Z',
          cwd: dir,
        };
        await writeFile(
          path,
          [header, ...fixture(turns, compacted)]
            .map((entry) => JSON.stringify(entry))
            .join('\n') + '\n'
        );
        const store = await JsonlSessionStore.openPath(path);
        const baseline = () => deriveMessages(store.getPath());
        const cached = () => deriveSessionMessages(store);
        const coldStart = performance.now();
        const first = cached();
        const firstReadMs = performance.now() - coldStart;
        assert.deepStrictEqual(first, baseline());
        const iterations = Math.max(10, Math.floor(10_000 / turns));
        measure(baseline, iterations);
        measure(cached, iterations);
        const before: number[] = [];
        const after: number[] = [];
        for (let round = 0; round < 7; round++) {
          const order =
            round % 2 === 0 ? [baseline, cached] : [cached, baseline];
          for (const run of order) {
            (run === baseline ? before : after).push(measure(run, iterations));
          }
        }
        const beforeMs = median(before);
        const afterMs = median(after);
        rows.push({
          messages: turns,
          compacted,
          firstReadMs: +firstReadMs.toFixed(3),
          fullMs: +beforeMs.toFixed(3),
          warmMs: +afterMs.toFixed(3),
          speedup: +(beforeMs / afterMs).toFixed(2),
        });
        const deltaBefore: number[] = [];
        const deltaAfter: number[] = [];
        for (let turn = 0; turn < 30; turn++) {
          await store.appendMessage(new HumanMessage(`follow-up ${turn}`));
          const order =
            turn % 2 === 0 ? [baseline, cached] : [cached, baseline];
          for (const run of order) {
            (run === baseline ? deltaBefore : deltaAfter).push(measure(run, 1));
          }
        }
        assert.deepStrictEqual(cached(), baseline());
        console.log(
          JSON.stringify({
            messages: turns,
            compacted,
            deltaFullMs: median(deltaBefore),
            deltaCachedMs: median(deltaAfter),
          })
        );
      }
    }
    console.table(rows);
    console.log(
      'Times include fresh message construction; delta times exclude filesystem writes. No model/network latency measured.'
    );
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

main().catch((error: Error) => {
  console.error(error);
  process.exitCode = 1;
});
