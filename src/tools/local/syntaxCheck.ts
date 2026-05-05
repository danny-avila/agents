/**
 * Per-file syntax check used by `edit_file` / `write_file` to surface
 * obvious errors immediately after the write — strictly cheaper than
 * full LSP integration and catches the bulk of "you broke the file"
 * regressions a vision-less agent loop would otherwise miss until
 * the next call.
 *
 * Each checker is a tiny shell-out (or in-process function) keyed on
 * file extension. Failures are returned as a single short message;
 * the wiring layer decides whether to append it to the tool result
 * advisorily (`auto`) or to throw and force the model to react
 * (`strict`).
 *
 * We deliberately do NOT cover TypeScript here because per-file `tsc`
 * is slow and per-file syntax (without type info) misses most TS
 * errors anyway. Use the project-level `compile_check` tool for that.
 */

import { extname } from 'path';
import type * as t from '@/types';
import {
  getSpawn,
  getWorkspaceFS,
  spawnLocalProcess,
} from './LocalExecutionEngine';

export type SyntaxCheckOutcome =
  | { ok: true }
  | { ok: false; checker: string; output: string };

export type SyntaxChecker = (
  path: string,
  config: t.LocalExecutionConfig
) => Promise<SyntaxCheckOutcome>;

/**
 * Per-backend availability cache for the post-edit syntax-check probe
 * tools (node, python3, bash). Keyed on the *effective spawn backend*
 * — see `getSpawn(config)` in LocalExecutionEngine — so a Run that
 * probes node over Node's child_process can't poison a subsequent Run
 * whose `local.exec.spawn` routes elsewhere (a remote sandbox might
 * have python but not node, etc.).
 *
 * Mirrors the same fix that landed for the ripgrep cache in
 * `LocalCodingTools.ts` after the first round of Codex review.
 * WeakMap keying lets disposed backends GC their entry; the test
 * reset hook re-creates the map.
 */
type ProbeKind = 'hasNode' | 'hasPython' | 'hasBash';
type ProbeCache = Partial<Record<ProbeKind, Promise<boolean>>>;

// Per-backend × per-env cache. Codex P2 #40 — keying by spawn
// backend alone misses env-driven availability changes (e.g. PATH
// loses node between Runs that share the same backend). Same fix
// shape as the ripgrep cache (Codex P1 #34).
let probeCacheByBackend = new WeakMap<
  t.LocalSpawn,
  Map<string, ProbeCache>
>();

function envCacheKey(env: NodeJS.ProcessEnv | undefined): string {
  if (env == null) return '';
  const sorted: Record<string, string | undefined> = {};
  for (const k of Object.keys(env).sort()) {
    sorted[k] = env[k];
  }
  return JSON.stringify(sorted);
}

function cacheFor(
  config: t.LocalExecutionConfig
): ProbeCache {
  const backend = getSpawn(config);
  let envMap = probeCacheByBackend.get(backend);
  if (envMap == null) {
    envMap = new Map();
    probeCacheByBackend.set(backend, envMap);
  }
  const envKey = envCacheKey(config.env);
  let entry = envMap.get(envKey);
  if (entry == null) {
    entry = {};
    envMap.set(envKey, entry);
  }
  return entry;
}

async function probe(
  command: string,
  args: string[],
  cached: ProbeKind,
  config: t.LocalExecutionConfig
): Promise<boolean> {
  const entry = cacheFor(config);
  let probePromise = entry[cached];
  if (probePromise == null) {
    probePromise = spawnLocalProcess(
      command,
      args,
      { ...config, timeoutMs: 5000, sandbox: { enabled: false } },
      { internal: true }
    )
      .then((result) => result != null && result.exitCode === 0)
      .catch(() => false);
    entry[cached] = probePromise;
  }
  return probePromise;
}

/**
 * Test-only reset hook. Clears the per-backend probe cache so tests
 * can swap in mocked spawn backends and reprobe deterministically.
 *
 * @internal Not part of the public SDK surface.
 */
export function _resetSyntaxCheckProbeCacheForTests(): void {
  probeCacheByBackend = new WeakMap();
}

const jsCheck: SyntaxChecker = async (path, config) => {
  if (!(await probe('node', ['--version'], 'hasNode', config))) {
    return { ok: true };
  }
  const result = await spawnLocalProcess(
    'node',
    ['--check', path],
    { ...config, timeoutMs: 5000, sandbox: { enabled: false } },
    { internal: true }
  );
  if (result.exitCode === 0) return { ok: true };
  return {
    ok: false,
    checker: 'node --check',
    output: result.stderr.trim() || result.stdout.trim() || 'syntax error',
  };
};

const pythonCheck: SyntaxChecker = async (path, config) => {
  if (!(await probe('python3', ['--version'], 'hasPython', config))) {
    return { ok: true };
  }
  const program =
    'import py_compile, sys\n' +
    'try:\n' +
    '  py_compile.compile(sys.argv[1], doraise=True)\n' +
    'except py_compile.PyCompileError as e:\n' +
    '  print(e.msg.strip(), file=sys.stderr)\n' +
    '  sys.exit(1)\n';
  const result = await spawnLocalProcess(
    'python3',
    ['-c', program, path],
    { ...config, timeoutMs: 5000, sandbox: { enabled: false } },
    { internal: true }
  );
  if (result.exitCode === 0) return { ok: true };
  return {
    ok: false,
    checker: 'py_compile',
    output: result.stderr.trim() || result.stdout.trim() || 'syntax error',
  };
};

const jsonCheck: SyntaxChecker = async (path, config) => {
  // Route through the configured WorkspaceFS so a Run with a custom
  // `local.exec.fs` (in-memory or remote engine) validates the same
  // file the write_file/edit_file path actually wrote — pre-fix this
  // read went to the host fs and either silently passed (no host
  // file → catch returns undefined → ok: true) or read a different
  // file with the same absolute path. Codex P1 #24.
  const fs = getWorkspaceFS(config);
  const raw = await fs.readFile(path, 'utf8').catch(() => undefined);
  if (raw == null) return { ok: true };
  try {
    JSON.parse(raw);
    return { ok: true };
  } catch (err) {
    return {
      ok: false,
      checker: 'JSON.parse',
      output: (err as Error).message,
    };
  }
};

const bashCheck: SyntaxChecker = async (path, config) => {
  if (!(await probe('bash', ['--version'], 'hasBash', config))) {
    return { ok: true };
  }
  const result = await spawnLocalProcess(
    'bash',
    ['-n', path],
    { ...config, timeoutMs: 5000, sandbox: { enabled: false } },
    { internal: true }
  );
  if (result.exitCode === 0) return { ok: true };
  return {
    ok: false,
    checker: 'bash -n',
    output: result.stderr.trim() || result.stdout.trim() || 'syntax error',
  };
};

const CHECKERS_BY_EXT: Record<string, SyntaxChecker> = {
  '.js': jsCheck,
  '.mjs': jsCheck,
  '.cjs': jsCheck,
  '.jsx': jsCheck,
  '.py': pythonCheck,
  '.pyw': pythonCheck,
  '.json': jsonCheck,
  '.sh': bashCheck,
  '.bash': bashCheck,
};

/**
 * Run the post-edit syntax check for `absolutePath`. Returns
 * `null` when no checker matches the extension (most files), or a
 * `SyntaxCheckOutcome`.
 *
 * Truncates `output` to `maxOutputChars` (default 4096) so a
 * 10MB-of-errors transpiler dump can't blow the model context.
 */
export async function runPostEditSyntaxCheck(
  absolutePath: string,
  config: t.LocalExecutionConfig
): Promise<SyntaxCheckOutcome | null> {
  const ext = extname(absolutePath).toLowerCase();
  const checker = (CHECKERS_BY_EXT as Record<string, SyntaxChecker | undefined>)[ext];
  if (checker == null) return null;
  try {
    const result = await checker(absolutePath, config);
    if (!result.ok) {
      return {
        ok: false,
        checker: result.checker,
        output: result.output.slice(0, 4096),
      };
    }
    return result;
  } catch {
    return null;
  }
}
