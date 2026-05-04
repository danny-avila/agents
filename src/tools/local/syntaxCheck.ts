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
import { readFile } from 'fs/promises';
import type * as t from '@/types';
import { spawnLocalProcess } from './LocalExecutionEngine';

export type SyntaxCheckOutcome =
  | { ok: true }
  | { ok: false; checker: string; output: string };

export type SyntaxChecker = (
  path: string,
  config: t.LocalExecutionConfig
) => Promise<SyntaxCheckOutcome>;

const cache = {
  hasNode: undefined as boolean | undefined,
  hasPython: undefined as boolean | undefined,
  hasBash: undefined as boolean | undefined,
};

async function probe(
  command: string,
  args: string[],
  cached: keyof typeof cache,
  config: t.LocalExecutionConfig
): Promise<boolean> {
  if (cache[cached] !== undefined) {
    return cache[cached] as boolean;
  }
  const result = await spawnLocalProcess(command, args, {
    ...config,
    timeoutMs: 5000,
    sandbox: { enabled: false },
  }).catch(() => undefined);
  const ok = result != null && result.exitCode === 0;
  cache[cached] = ok;
  return ok;
}

export function _resetSyntaxCheckProbeCacheForTests(): void {
  cache.hasNode = undefined;
  cache.hasPython = undefined;
  cache.hasBash = undefined;
}

const jsCheck: SyntaxChecker = async (path, config) => {
  if (!(await probe('node', ['--version'], 'hasNode', config))) {
    return { ok: true };
  }
  const result = await spawnLocalProcess('node', ['--check', path], {
    ...config,
    timeoutMs: 5000,
    sandbox: { enabled: false },
  });
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
    { ...config, timeoutMs: 5000, sandbox: { enabled: false } }
  );
  if (result.exitCode === 0) return { ok: true };
  return {
    ok: false,
    checker: 'py_compile',
    output: result.stderr.trim() || result.stdout.trim() || 'syntax error',
  };
};

const jsonCheck: SyntaxChecker = async (path) => {
  const raw = await readFile(path, 'utf8').catch(() => undefined);
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
  const result = await spawnLocalProcess('bash', ['-n', path], {
    ...config,
    timeoutMs: 5000,
    sandbox: { enabled: false },
  });
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
