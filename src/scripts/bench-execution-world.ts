import { performance } from 'node:perf_hooks';

import type * as t from '@/types';

import { createCloudflareLocalExecutionConfig } from '@/tools/cloudflare/CloudflareSandboxExecutionEngine';
import {
  runPostEditSyntaxCheck,
  _resetSyntaxCheckProbeCacheForTests,
} from '@/tools/local/syntaxCheck';

const BINDINGS = 10;
const REMOTE_LATENCY_MS = 10;

type BenchmarkResult = {
  elapsedMs: number;
  remoteExecCalls: number;
};

function delayedRuntime(counter: { calls: number }): t.CloudflareSandboxRuntime {
  return {
    exec: async (): Promise<t.CloudflareSandboxExecResult> => {
      counter.calls++;
      await new Promise((resolve) => setTimeout(resolve, REMOTE_LATENCY_MS));
      return { exitCode: 0, stdout: '', stderr: '' };
    },
    readFile: async () => '',
    writeFile: async () => undefined,
    mkdir: async () => undefined,
    listFiles: async () => [],
    deleteFile: async () => undefined,
  };
}

async function runBenchmark(reuseWorld: boolean): Promise<BenchmarkResult> {
  _resetSyntaxCheckProbeCacheForTests();
  const counter = { calls: 0 };
  const sandbox = delayedRuntime(counter);
  const sharedConfig: t.CloudflareSandboxExecutionConfig = { sandbox };
  const startedAt = performance.now();

  for (let binding = 0; binding < BINDINGS; binding++) {
    const config = reuseWorld ? sharedConfig : { sandbox };
    const localConfig = createCloudflareLocalExecutionConfig(config);
    await runPostEditSyntaxCheck('/workspace/benchmark.js', localConfig);
  }

  return {
    elapsedMs: performance.now() - startedAt,
    remoteExecCalls: counter.calls,
  };
}

await runBenchmark(false);
await runBenchmark(true);
const before = await runBenchmark(false);
const after = await runBenchmark(true);

if (before.remoteExecCalls !== BINDINGS * 2) {
  throw new Error(`Unexpected uncached call count: ${before.remoteExecCalls}`);
}
if (after.remoteExecCalls !== BINDINGS + 1) {
  throw new Error(`Unexpected cached call count: ${after.remoteExecCalls}`);
}

// eslint-disable-next-line no-console
console.log(
  JSON.stringify({
    bindings: BINDINGS,
    simulatedRemoteLatencyMs: REMOTE_LATENCY_MS,
    beforeMs: Number(before.elapsedMs.toFixed(2)),
    afterMs: Number(after.elapsedMs.toFixed(2)),
    speedup: Number((before.elapsedMs / after.elapsedMs).toFixed(2)),
    remoteExecCallsBefore: before.remoteExecCalls,
    remoteExecCallsAfter: after.remoteExecCalls,
  })
);
