import { tmpdir } from 'os';
import { isAbsolute, relative, resolve } from 'path';
import { createHash, randomUUID } from 'crypto';
import { mkdir, realpath, rm, writeFile } from 'fs/promises';
import { createWriteStream } from 'fs';
import { spawn } from 'child_process';
import type { ChildProcess } from 'child_process';
import type { SandboxRuntimeConfig } from '@anthropic-ai/sandbox-runtime';
import { runBashAstChecks, bashAstFindingsToErrors } from './bashAst';
import { nodeWorkspaceFS } from './workspaceFS';
import type { WorkspaceFS } from './workspaceFS';
import type * as t from '@/types';

const DEFAULT_TIMEOUT_MS = 60000;
const DEFAULT_MAX_OUTPUT_CHARS = 200000;
/**
 * Hard cap on total stdout+stderr bytes a child process can stream
 * before we kill its process tree. Independent from `maxOutputChars`
 * (which only affects what the *model* sees) — this is the OOM
 * backstop. Configurable via `local.maxSpawnedBytes`.
 */
const DEFAULT_MAX_SPAWNED_BYTES = 50 * 1024 * 1024;
const DEFAULT_LOCAL_SESSION_ID = 'local';
const DEFAULT_SHELL = process.platform === 'win32' ? 'bash.exe' : 'bash';

// `(?:--\s+)?` before each destructive-target alternation: GNU/BSD
// utilities accept `--` as an end-of-options marker, so `rm -rf -- /`
// is identical in effect to `rm -rf /` but pre-fix it slipped past
// the guard because the regex required the path to follow option
// flags directly. Codex P1 #20.
const dangerousCommandPatterns: ReadonlyArray<RegExp> = [
  /\brm\s+(?:-[^\s]*[rf][^\s]*\s+|-[^\s]*[r][^\s]*\s+-[^\s]*[f][^\s]*\s+)(?:--\s+)?(?:\/|~|\$HOME|\.)\s*(?:$|[;&|])/,
  /\b(?:mkfs|mkswap|fdisk|parted|diskutil)\b/,
  /\bdd\s+[^;&|]*\bof=\/dev\//,
  /\bchmod\s+-R\s+(?:777|a\+w)\s+(?:--\s+)?(?:\/|~|\$HOME|\.)\b/,
  /\bchown\s+-R\s+[^;&|]+\s+(?:--\s+)?(?:\/|~|\$HOME|\.)\b/,
  /:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:/,
];

/**
 * Companion patterns that look for destructive targets *inside*
 * matching quote pairs. These are checked against the ORIGINAL
 * command (not the post-quote-strip `normalized` form), because
 * `stripQuotedContent` blanks the contents of quoted spans —
 * which would otherwise let `rm -rf "/"` and friends slip past
 * `dangerousCommandPatterns`.
 *
 * Kept as a separate list so we don't pay false-positive cost on
 * benign uses like `echo "rm -rf /"` (the print case): each pattern
 * here REQUIRES a quote *around the destructive path argument*, not
 * just a quote *somewhere* in the command. `echo "rm -rf /"` has
 * `/` outside of any quote-pair-around-the-path (the quotes wrap
 * the whole `rm -rf /` text), so it doesn't match here either.
 */
const quotedDestructivePatterns: ReadonlyArray<RegExp> = [
  /\brm\s+(?:-[^\s]*[rf][^\s]*\s+){1,3}(?:--\s+)?["'](?:\/|~|\$\{?HOME\}?|\.)["']/,
  /\bchmod\s+-R\s+(?:777|a\+w)\s+(?:--\s+)?["'](?:\/|~|\$\{?HOME\}?|\.)["']/,
  /\bchown\s+-R\s+[^;&|]+\s+(?:--\s+)?["'](?:\/|~|\$\{?HOME\}?|\.)["']/,
];

/**
 * Catches destructive operations smuggled inside a nested shell or
 * `eval` call, e.g. `bash -lc "rm -rf $HOME"` — the outer command
 * looks benign (`bash -lc "..."`) and the destructive `rm` lives
 * inside the quoted payload that `stripQuotedContent` blanks out.
 * Comprehensive review (manual finding C) flagged this as a real
 * bypass of the otherwise-correct quote-strip-then-match approach.
 *
 * Run against the ORIGINAL command (quotes intact) so the inside of
 * the nested-shell payload is visible. Conservative: matches only
 * the same operation set as `dangerousCommandPatterns` (rm -rf,
 * chmod -R 777, chown -R) when they appear inside a `<shell> -[l]?c
 * "..."` or `eval "..."` payload.
 */
const NESTED_SHELL_PREFIX = '(?:(?:ba|z|da|k)?sh|eval)\\s+(?:-l?c\\s+)?';
const nestedShellDestructivePatterns: ReadonlyArray<RegExp> = [
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\brm\\s+-[^\\s"\']*[rf][^\\s"\']*\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\bchmod\\s+-R\\s+(?:777|a\\+w)\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\bchown\\s+-R\\s+[^;&|]+\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
];

const mutatingCommandPattern =
  /\b(?:rm|mv|cp|touch|mkdir|rmdir|ln|truncate|tee|sed\s+-i|perl\s+-pi|python(?:3)?\s+-c|node\s+-e|npm\s+(?:install|ci|update|publish)|pnpm\s+(?:install|update|publish)|yarn\s+(?:install|add|publish)|git\s+(?:add|commit|checkout|switch|reset|clean|rebase|merge|push|pull|stash|tag|branch)|chmod|chown)\b|(?:^|[^<])>\s*[^&]|\bcat\s+[^|;&]*>\s*/;

type SpawnResult = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  /** Path to the full untruncated stdout/stderr when output exceeded `maxOutputChars`. */
  fullOutputPath?: string;
};

type RuntimeCommand = {
  command: string;
  args: string[];
  fileName: string;
  source?: string;
};

type SandboxRuntimeModule = typeof import('@anthropic-ai/sandbox-runtime');
type SandboxManagerType = SandboxRuntimeModule['SandboxManager'];

let sandboxConfigKey: string | undefined;
let sandboxInitialized = false;
let sandboxRuntimePromise: Promise<SandboxRuntimeModule> | undefined;

export type BashValidationResult = {
  valid: boolean;
  errors: string[];
  warnings: string[];
};

function isToolExecutionConfig(
  config: t.ToolExecutionConfig | t.LocalExecutionConfig
): config is t.ToolExecutionConfig {
  return 'engine' in config || 'local' in config;
}

export function resolveLocalExecutionConfig(
  config?: t.ToolExecutionConfig | t.LocalExecutionConfig
): t.LocalExecutionConfig {
  if (config != null && isToolExecutionConfig(config)) {
    return config.local ?? {};
  }
  return config ?? {};
}

export function getLocalCwd(config?: t.LocalExecutionConfig): string {
  return resolve(config?.workspace?.root ?? config?.cwd ?? process.cwd());
}

/**
 * Resolves the effective workspace boundary: a list of absolute roots
 * that file operations are allowed to touch. The first entry is always
 * the canonical root (`getLocalCwd`); subsequent entries come from
 * `workspace.additionalRoots` when provided.
 *
 * Returns plain absolute paths — callers symlink-resolve when they
 * need realpath equality (see `resolveWorkspacePathSafe`).
 */
export function getWorkspaceRoots(
  config?: t.LocalExecutionConfig
): string[] {
  const root = getLocalCwd(config);
  const extras = config?.workspace?.additionalRoots ?? [];
  if (extras.length === 0) return [root];
  const seen = new Set<string>([root]);
  const out: string[] = [root];
  for (const extra of extras) {
    // Relative `additionalRoots` entries are anchored to the
    // workspace root (so monorepo configs like
    // `additionalRoots: ['../shared']` resolve to a sibling of
    // `root` rather than to `process.cwd()/../shared`, which would
    // mean something completely different on a server with a
    // different cwd).
    const abs = isAbsolute(extra) ? resolve(extra) : resolve(root, extra);
    if (!seen.has(abs)) {
      seen.add(abs);
      out.push(abs);
    }
  }
  return out;
}

/**
 * Pluggable spawn resolver. Honours `local.exec.spawn` first, falls
 * back to the legacy top-level `local.spawn`, then to Node's
 * `child_process.spawn`. Centralised so engine swapping is one knob.
 */
export function getSpawn(
  config?: t.LocalExecutionConfig
): t.LocalSpawn {
  return (config?.exec?.spawn ?? config?.spawn ?? spawn) as t.LocalSpawn;
}

/**
 * Pluggable filesystem resolver. Honours `local.exec.fs`, falls back
 * to the Node-host implementation. A future remote engine supplies
 * its own implementation here and inherits every file-touching tool.
 */
export function getWorkspaceFS(
  config?: t.LocalExecutionConfig
): WorkspaceFS {
  return config?.exec?.fs ?? nodeWorkspaceFS;
}

/**
 * Resolves the workspace boundary for *write* operations. Honours
 * `workspace.allowWriteOutside` (and the deprecated
 * `allowOutsideWorkspace`) by returning `null`, which the path-safety
 * helpers interpret as "skip the write clamp".
 */
export function getWriteRoots(
  config?: t.LocalExecutionConfig
): string[] | null {
  if (
    config?.workspace?.allowWriteOutside === true ||
    config?.allowOutsideWorkspace === true
  ) {
    return null;
  }
  return getWorkspaceRoots(config);
}

/**
 * Resolves the workspace boundary for *read* operations. Honours
 * `workspace.allowReadOutside` (and the deprecated
 * `allowOutsideWorkspace`) by returning `null`.
 */
export function getReadRoots(
  config?: t.LocalExecutionConfig
): string[] | null {
  if (
    config?.workspace?.allowReadOutside === true ||
    config?.allowOutsideWorkspace === true
  ) {
    return null;
  }
  return getWorkspaceRoots(config);
}

export function getLocalSessionId(config?: t.LocalExecutionConfig): string {
  const cwd = getLocalCwd(config);
  const digest = createHash('sha1').update(cwd).digest('hex').slice(0, 12);
  return `${DEFAULT_LOCAL_SESSION_ID}:${digest}`;
}

const missingSandboxRuntimeMessage = [
  'Local sandbox is enabled, but @anthropic-ai/sandbox-runtime is not installed.',
  'Install it with `npm install @anthropic-ai/sandbox-runtime`, or disable local sandboxing with `local.sandbox.enabled: false`.',
].join(' ');

/** Lazy-loads the ESM-only sandbox runtime only when sandboxing is enabled. */
function loadSandboxRuntime(): Promise<SandboxRuntimeModule> {
  sandboxRuntimePromise ??= import('@anthropic-ai/sandbox-runtime');
  return sandboxRuntimePromise;
}

function shouldUseLocalSandbox(config: t.LocalExecutionConfig): boolean {
  return config.sandbox?.enabled === true;
}

let sandboxOffWarned = false;

function maybeWarnSandboxOff(config: t.LocalExecutionConfig): void {
  if (sandboxOffWarned || shouldUseLocalSandbox(config)) {
    return;
  }
  sandboxOffWarned = true;
  // eslint-disable-next-line no-console
  console.warn(
    '[@librechat/agents] Local execution engine is running without ' +
      '@anthropic-ai/sandbox-runtime wrapping. The agent has full access to ' +
      'the host filesystem and network. Set toolExecution.local.sandbox.enabled ' +
      '= true to opt into process sandboxing.'
  );
}

/**
 * Test-only reset hook for the sandbox-off warning latch.
 *
 * @internal Not part of the public SDK surface.
 */
export function _resetLocalEngineWarningsForTests(): void {
  sandboxOffWarned = false;
}

export function truncateLocalOutput(
  value: string,
  maxChars = DEFAULT_MAX_OUTPUT_CHARS
): string {
  if (value.length <= maxChars) {
    return value;
  }
  const head = Math.floor(maxChars * 0.6);
  const tail = maxChars - head;
  const omitted = value.length - maxChars;
  return `${value.slice(0, head)}\n\n[... ${omitted} characters truncated ...]\n\n${value.slice(
    value.length - tail
  )}`;
}

function stripQuotedContent(command: string): string {
  let output = '';
  let quote: '"' | '\'' | '`' | undefined;
  let escaped = false;

  for (let i = 0; i < command.length; i++) {
    const char = command[i];

    if (escaped) {
      escaped = false;
      output += ' ';
      continue;
    }

    if (char === '\\') {
      escaped = true;
      output += ' ';
      continue;
    }

    if (quote != null) {
      if (char === quote) {
        quote = undefined;
      }
      output += ' ';
      continue;
    }

    if (char === '"' || char === '\'' || char === '`') {
      quote = char;
      output += ' ';
      continue;
    }

    if (char === '#') {
      while (i < command.length && command[i] !== '\n') {
        output += ' ';
        i++;
      }
      output += '\n';
      continue;
    }

    output += char;
  }

  return output;
}

export async function validateBashCommand(
  command: string,
  config: t.LocalExecutionConfig = {}
): Promise<BashValidationResult> {
  const errors: string[] = [];
  const warnings: string[] = [];
  const normalized = stripQuotedContent(command);

  if (command.trim() === '') {
    errors.push('Command is empty.');
  }

  if (command.includes('\0')) {
    errors.push('Command contains a NUL byte.');
  }

  if (config.allowDangerousCommands !== true) {
    let blocked = false;
    // Strip-then-match for the bare-form patterns (avoids false
    // positives where the destructive text is buried inside a
    // string the user is just printing).
    for (const pattern of dangerousCommandPatterns) {
      if (pattern.test(normalized)) {
        errors.push('Command matches a destructive command pattern.');
        blocked = true;
        break;
      }
    }
    // Original-form pass for patterns that REQUIRE matching quote
    // pairs around a destructive path. Without this, `rm -rf "/"`
    // and `chmod -R 777 "/"` slip past the strip-then-match pass
    // because their destructive target is inside quotes.
    if (!blocked) {
      for (const pattern of quotedDestructivePatterns) {
        if (pattern.test(command)) {
          errors.push(
            'Command matches a destructive command pattern (quoted target).'
          );
          blocked = true;
          break;
        }
      }
    }
    if (!blocked) {
      for (const pattern of nestedShellDestructivePatterns) {
        if (pattern.test(command)) {
          errors.push(
            'Command matches a destructive command pattern (nested shell payload).'
          );
          break;
        }
      }
    }
  }

  const bashAstMode = config.bashAst ?? 'off';
  if (bashAstMode !== 'off' && config.allowDangerousCommands !== true) {
    const findings = runBashAstChecks(normalized, bashAstMode);
    const split = bashAstFindingsToErrors(findings);
    errors.push(...split.errors);
    warnings.push(...split.warnings);
  }

  if (config.readOnly === true && mutatingCommandPattern.test(normalized)) {
    errors.push('Command appears to mutate files or repository state in read-only local mode.');
  }

  // Use the same shell the actual execution path will use. Hard-coding
  // DEFAULT_SHELL here would reject perfectly valid commands when the
  // host configures `local.shell` to a non-bash binary (or when the
  // runtime doesn't have bash installed at all but does have e.g. zsh).
  const syntaxShell = config.shell ?? DEFAULT_SHELL;
  const syntax = await spawnLocalProcess(
    syntaxShell,
    ['-n', '-c', command],
    {
      ...config,
      timeoutMs: Math.min(config.timeoutMs ?? DEFAULT_TIMEOUT_MS, 5000),
      sandbox: { enabled: false },
    },
    { internal: true }
  ).catch((error: Error): SpawnResult => ({
    stdout: '',
    stderr: error.message,
    exitCode: 1,
    timedOut: false,
  }));

  if (syntax.exitCode !== 0) {
    errors.push(
      syntax.stderr.trim() === ''
        ? 'Command failed shell syntax validation.'
        : `Command failed shell syntax validation: ${syntax.stderr.trim()}`
    );
  }

  if (/\bsudo\b/.test(normalized)) {
    warnings.push('Command requests elevated privileges with sudo.');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

async function ensureSandbox(
  config: t.LocalExecutionConfig,
  cwd: string
): Promise<SandboxManagerType | undefined> {
  if (!shouldUseLocalSandbox(config)) {
    return undefined;
  }

  const runtime = await loadSandboxRuntime().catch((error: Error) => {
    throw new Error(`${missingSandboxRuntimeMessage} Cause: ${error.message}`);
  });

  const runtimeConfig = buildSandboxRuntimeConfig(
    config,
    cwd,
    runtime.getDefaultWritePaths
  );
  const nextKey = JSON.stringify(runtimeConfig);

  if (sandboxInitialized && sandboxConfigKey === nextKey) {
    return runtime.SandboxManager;
  }

  const dependencyCheck = runtime.SandboxManager.checkDependencies();
  if (dependencyCheck.errors.length > 0) {
    if (config.sandbox?.failIfUnavailable === true) {
      throw new Error(
        `Local sandbox requested but unavailable: ${dependencyCheck.errors.join('; ')}`
      );
    }
    return undefined;
  }

  if (sandboxInitialized) {
    await runtime.SandboxManager.reset();
  }

  await runtime.SandboxManager.initialize(runtimeConfig);
  sandboxInitialized = true;
  sandboxConfigKey = nextKey;
  return runtime.SandboxManager;
}

/**
 * Loopback addresses the in-process programmatic-tool bridge listens
 * on (`LocalProgrammaticToolCalling.ts` binds 127.0.0.1). Sandboxed
 * code launched by `run_tools_with_code` / `run_tools_with_bash` HTTPs
 * back to that address — without the entries below, the bridge is
 * silently blocked under sandbox.
 */
const BRIDGE_LOOPBACK_HOSTS = ['127.0.0.1', 'localhost', '::1'] as const;

export function buildSandboxRuntimeConfig(
  config: t.LocalExecutionConfig,
  cwd: string,
  getDefaultWritePaths: () => string[]
): SandboxRuntimeConfig {
  const sandbox = config.sandbox;
  // Seed allowedDomains with loopback so the programmatic-tool bridge
  // works under sandbox. If the host explicitly denied a loopback
  // entry via `deniedDomains`, respect that and skip seeding it.
  const userAllowed = sandbox?.network?.allowedDomains ?? [];
  const denied = new Set(sandbox?.network?.deniedDomains ?? []);
  const seededLoopback = BRIDGE_LOOPBACK_HOSTS.filter(
    (host) => !denied.has(host) && !userAllowed.includes(host)
  );
  const allowedDomains = [...seededLoopback, ...userAllowed];
  // Mirror the file-tools workspace boundary: anything in
  // `additionalRoots` counts as in-workspace, so sandboxed shell/code
  // can write there too. Without this, file_tools can resolve a
  // sibling-root path but `bash`/`execute_code` is denied write
  // access — confusing divergence flagged in Codex P2 #15.
  const workspaceWriteRoots =
    config.workspace?.additionalRoots != null
      ? getWorkspaceRoots(config)
      : [cwd];
  return {
    network: {
      allowedDomains,
      deniedDomains: sandbox?.network?.deniedDomains ?? [],
      ...(sandbox?.network?.allowUnixSockets != null && {
        allowUnixSockets: sandbox.network.allowUnixSockets,
      }),
      ...(sandbox?.network?.allowAllUnixSockets != null && {
        allowAllUnixSockets: sandbox.network.allowAllUnixSockets,
      }),
      ...(sandbox?.network?.allowLocalBinding != null && {
        allowLocalBinding: sandbox.network.allowLocalBinding,
      }),
      ...(sandbox?.network?.allowMachLookup != null && {
        allowMachLookup: sandbox.network.allowMachLookup,
      }),
    },
    filesystem: {
      denyRead: sandbox?.filesystem?.denyRead ?? [],
      allowRead: sandbox?.filesystem?.allowRead,
      allowWrite: sandbox?.filesystem?.allowWrite ?? [
        ...workspaceWriteRoots,
        ...getDefaultWritePaths(),
      ],
      denyWrite: sandbox?.filesystem?.denyWrite ?? [
        '.env',
        '.env.*',
        '.git/config',
        '.git/hooks/**',
      ],
      allowGitConfig: sandbox?.filesystem?.allowGitConfig,
    },
  };
}

/**
 * Internal options for {@link spawnLocalProcess} that we don't want
 * exposed on the public `LocalExecutionConfig` type.
 *
 * @internal
 */
export interface SpawnLocalProcessOptions {
  /**
   * When true, suppress the "sandbox is off" warning AND its latch
   * for this spawn. Use for SDK-internal probes (`bash -n` syntax
   * preflight, `rg --version`, etc.) that intentionally run with
   * the sandbox forced off — the warning is noise for those, and
   * letting the latch flip would hide the warning when a *real*
   * unsandboxed execution happens later in the same process.
   */
  internal?: boolean;
}

export async function spawnLocalProcess(
  command: string,
  args: string[],
  config: t.LocalExecutionConfig = {},
  options?: SpawnLocalProcessOptions
): Promise<SpawnResult> {
  const cwd = getLocalCwd(config);
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxOutputChars = config.maxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS;
  // Streaming caps. Local tools execute arbitrary shell/code, so a noisy
  // command (`yes`, `cat /dev/urandom | base64`, a verbose build) could
  // accumulate gigabytes in memory before hitting the post-close cap.
  // We bound in-memory per-stream and spill the rest to disk; we also
  // hard-kill the child once total streamed bytes pass `maxSpawnedBytes`
  // so a process producing unbounded output gets stopped instead of
  // letting the host OOM.
  const inMemoryCapBytes = maxOutputChars * 2;
  const hardKillBytes =
    config.maxSpawnedBytes ?? DEFAULT_MAX_SPAWNED_BYTES;
  const sandboxManager = await ensureSandbox(config, cwd);
  // Internal probes (validateBashCommand syntax preflight,
  // isRipgrepAvailable, syntax-check probe cache priming) pass
  // `internal: true` so they don't emit a misleading "sandbox is
  // off" warning AND don't flip `sandboxOffWarned = true`. Without
  // this Codex P2 path: a run with `sandbox.enabled: true` would
  // see a false warning from the syntax preflight, and the latch
  // flip would suppress the warning in a *later* truly-unsandboxed
  // run — exactly the scenario operators need to see.
  if (sandboxManager == null && options?.internal !== true) {
    maybeWarnSandboxOff(config);
  }
  let spawnCommand = command;
  let spawnArgs = args;

  if (sandboxManager != null) {
    const rendered = [command, ...args.map(shellQuote)].join(' ');
    const sandboxed = await sandboxManager.wrapWithSandbox(rendered);
    spawnCommand = config.shell ?? DEFAULT_SHELL;
    spawnArgs = ['-lc', sandboxed];
  }

  const launcher = getSpawn(config);
  return new Promise<SpawnResult>((resolveResult, reject) => {
    const child = launcher(spawnCommand, spawnArgs, {
      cwd,
      detached: process.platform !== 'win32',
      env: { ...process.env, ...(config.env ?? {}) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let totalSpawnedBytes = 0;
    let overflowKilled = false;
    let spillStream: import('fs').WriteStream | undefined;
    let spillPath: string | undefined;
    let settled = false;
    let timedOut = false;
    let timeout: NodeJS.Timeout | undefined;

    const ensureSpill = (): void => {
      if (spillStream != null) return;
      // Lazy-open the temp file the first time a stream's in-memory
      // buffer overflows. Seed it with everything we've buffered so
      // the file holds the FULL output (not just the post-cap tail).
      // Uses the static `createWriteStream` import — `require('fs')`
      // would throw `ReferenceError: require is not defined` in ESM
      // consumers (this package ships both `dist/cjs` and `dist/esm`).
      spillPath = resolve(tmpdir(), `lc-local-output-${randomUUID()}.txt`);
      spillStream = createWriteStream(spillPath);
      spillStream.write('===== stdout =====\n');
      spillStream.write(stdout);
      spillStream.write('\n===== stderr =====\n');
      spillStream.write(stderr);
      spillStream.write('\n===== overflow stream begins here =====\n');
    };

    const handleChunk = (
      buf: Buffer,
      kind: 'stdout' | 'stderr'
    ): void => {
      totalSpawnedBytes += buf.length;
      // hardKillBytes <= 0 means "no cap" per the public config contract
      // (see LocalExecutionConfig.maxSpawnedBytes). Skip the kill check
      // entirely in that case so a single byte doesn't terminate the run.
      if (
        hardKillBytes > 0 &&
        totalSpawnedBytes > hardKillBytes &&
        !overflowKilled
      ) {
        overflowKilled = true;
        killProcessTree(child);
        return;
      }
      const current = kind === 'stdout' ? stdout : stderr;
      if (current.length < inMemoryCapBytes) {
        const text = buf.toString('utf8');
        if (kind === 'stdout') stdout += text;
        else stderr += text;
        if (current.length + text.length >= inMemoryCapBytes) {
          ensureSpill();
        }
      } else {
        ensureSpill();
        spillStream!.write(`[${kind}] `);
        spillStream!.write(buf);
      }
    };

    const finish = (result: SpawnResult): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (timeout != null) {
        clearTimeout(timeout);
      }
      const finalize = (): void => {
        const truncated = {
          stdout: truncateLocalOutput(result.stdout, maxOutputChars),
          stderr: truncateLocalOutput(result.stderr, maxOutputChars),
        };
        resolveResult({
          ...result,
          ...truncated,
          ...(spillPath != null ? { fullOutputPath: spillPath } : {}),
        });
      };
      if (spillStream == null) {
        finalize();
        return;
      }
      // Wait for the temp file to flush before reporting the path.
      // Otherwise the model sees `full_output_path: …` for a file
      // that's still being written.
      spillStream.end(() => finalize());
    };

    const fail = (error: Error): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (timeout != null) {
        clearTimeout(timeout);
      }
      if (spillStream != null) {
        spillStream.end();
      }
      reject(error);
    };

    if (timeoutMs > 0) {
      timeout = setTimeout(() => {
        timedOut = true;
        killProcessTree(child);
      }, timeoutMs);
    }

    child.stdout?.on('data', (chunk: Buffer) => {
      handleChunk(chunk, 'stdout');
    });

    child.stderr?.on('data', (chunk: Buffer) => {
      handleChunk(chunk, 'stderr');
    });

    child.on('error', fail);

    child.on('close', (exitCode) => {
      finish({ stdout, stderr, exitCode, timedOut });
    });
  });
}

export async function executeLocalBash(
  command: string,
  config: t.LocalExecutionConfig = {}
): Promise<SpawnResult> {
  const validation = await validateBashCommand(command, config);
  if (!validation.valid) {
    throw new Error(validation.errors.join('\n'));
  }
  const shell = config.shell ?? DEFAULT_SHELL;
  return spawnLocalProcess(shell, ['-lc', command], config);
}

/**
 * Variant of `executeLocalBash` that exposes `args` as positional
 * shell parameters (`$1`, `$2`, …). Mirrors what the other runtimes
 * do in `getRuntimeCommand`. Uses the standard `bash -c <code> --
 * arg0 arg1 …` form: the `--` becomes `$0`, then `args[0]` is `$1`
 * and so on. Same AST validation as the no-args path.
 *
 * Used by both the `execute_code`/`lang:'bash'` path AND the
 * `bash_tool` factory so the schema's `args` contract works
 * identically in both surfaces.
 */
export async function executeLocalBashWithArgs(
  command: string,
  args: readonly string[],
  config: t.LocalExecutionConfig = {}
): Promise<SpawnResult> {
  const validation = await validateBashCommand(command, config);
  if (!validation.valid) {
    throw new Error(validation.errors.join('\n'));
  }
  const shell = config.shell ?? DEFAULT_SHELL;
  return spawnLocalProcess(
    shell,
    ['-lc', command, '--', ...args],
    config
  );
}

export async function executeLocalCode(
  input: {
    lang: string;
    code: string;
    args?: string[];
  },
  config: t.LocalExecutionConfig = {}
): Promise<SpawnResult> {
  if (input.lang === 'bash') {
    // Append `args` as positional parameters via the standard
    // `bash -c <code> -- <args...>` form so `$1`, `$2`, … inside
    // `code` resolve correctly. Honours the same args contract the
    // other runtimes (py, js, …) already support.
    if (input.args != null && input.args.length > 0) {
      return executeLocalBashWithArgs(input.code, input.args, config);
    }
    return executeLocalBash(input.code, config);
  }

  const tempDir = resolve(tmpdir(), `lc-local-${randomUUID()}`);
  await mkdir(tempDir, { recursive: true });

  try {
    const runtime = getRuntimeCommand(input.lang, tempDir, input.code, input.args);
    if (runtime.source != null) {
      await writeFile(resolve(tempDir, runtime.fileName), runtime.source, 'utf8');
    }
    return await spawnLocalProcess(runtime.command, runtime.args, config);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

function getRuntimeCommand(
  lang: string,
  tempDir: string,
  code: string,
  args: string[] = []
): RuntimeCommand {
  const fileFor = (name: string): string => resolve(tempDir, name);

  switch (lang) {
  case 'py':
    return {
      command: 'python3',
      args: [fileFor('main.py'), ...args],
      fileName: 'main.py',
      source: code,
    };
  case 'js':
    return {
      command: 'node',
      args: [fileFor('main.js'), ...args],
      fileName: 'main.js',
      source: code,
    };
  case 'ts':
    return {
      command: 'npx',
      args: ['--no-install', 'tsx', fileFor('main.ts'), ...args],
      fileName: 'main.ts',
      source: code,
    };
  case 'php':
    return {
      command: 'php',
      args: [fileFor('main.php'), ...args],
      fileName: 'main.php',
      source: code,
    };
  case 'go':
    return {
      command: 'go',
      args: ['run', fileFor('main.go'), ...args],
      fileName: 'main.go',
      source: code,
    };
  case 'rs':
    return {
      command: configShell(),
      args: [
        '-lc',
        `rustc ${shellQuote(fileFor('main.rs'))} -o ${shellQuote(
          fileFor('main-rs')
        )} && ${shellQuote(fileFor('main-rs'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.rs',
      source: code,
    };
  case 'c':
    return {
      command: configShell(),
      args: [
        '-lc',
        `cc ${shellQuote(fileFor('main.c'))} -o ${shellQuote(
          fileFor('main-c')
        )} && ${shellQuote(fileFor('main-c'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.c',
      source: code,
    };
  case 'cpp':
    return {
      command: configShell(),
      args: [
        '-lc',
        `c++ ${shellQuote(fileFor('main.cpp'))} -o ${shellQuote(
          fileFor('main-cpp')
        )} && ${shellQuote(fileFor('main-cpp'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.cpp',
      source: code,
    };
  case 'java':
    return {
      command: configShell(),
      args: [
        '-lc',
        `javac ${shellQuote(fileFor('Main.java'))} && java -cp ${shellQuote(
          tempDir
        )} Main ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'Main.java',
      source: code,
    };
  case 'r':
    return {
      command: 'Rscript',
      args: [fileFor('main.R'), ...args],
      fileName: 'main.R',
      source: code,
    };
  case 'd':
    return {
      command: configShell(),
      args: [
        '-lc',
        `dmd ${shellQuote(fileFor('main.d'))} -of=${shellQuote(
          fileFor('main-d')
        )} && ${shellQuote(fileFor('main-d'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.d',
      source: code,
    };
  case 'f90':
    return {
      command: configShell(),
      args: [
        '-lc',
        `gfortran ${shellQuote(fileFor('main.f90'))} -o ${shellQuote(
          fileFor('main-f90')
        )} && ${shellQuote(fileFor('main-f90'))} ${args.map(shellQuote).join(' ')}`,
      ],
      fileName: 'main.f90',
      source: code,
    };
  default:
    throw new Error(`Unsupported local runtime: ${lang}`);
  }
}

function configShell(): string {
  return process.platform === 'win32' ? 'bash.exe' : 'bash';
}

function killProcessTree(child: ChildProcess): void {
  if (child.pid == null) {
    return;
  }
  try {
    if (process.platform === 'win32') {
      child.kill('SIGTERM');
      return;
    }
    process.kill(-child.pid, 'SIGTERM');
  } catch {
    child.kill('SIGTERM');
  }
}

export function shellQuote(value: string): string {
  if (value === '') {
    return '\'\'';
  }
  if (/^[A-Za-z0-9_/:=.,@%+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replace(/'/g, '\'\\\'\'')}'`;
}

export function resolveWorkspacePath(
  filePath: string,
  config: t.LocalExecutionConfig = {},
  intent: 'read' | 'write' = 'write'
): string {
  const cwd = getLocalCwd(config);
  const absolutePath = isAbsolute(filePath) ? resolve(filePath) : resolve(cwd, filePath);

  const roots = intent === 'write' ? getWriteRoots(config) : getReadRoots(config);
  if (roots == null) return absolutePath; // explicit allow-outside

  if (absolutePath === cwd || isInsideAnyRoot(absolutePath, roots)) {
    return absolutePath;
  }
  throw new Error(`Path is outside the local workspace: ${filePath}`);
}

function isInsideAnyRoot(absolutePath: string, roots: string[]): boolean {
  for (const root of roots) {
    if (absolutePath === root) return true;
    const rel = relative(root, absolutePath);
    if (!rel.startsWith('..') && !isAbsolute(rel)) return true;
  }
  return false;
}

async function realpathOrSelf(absolutePath: string): Promise<string> {
  try {
    return await realpath(absolutePath);
  } catch {
    return absolutePath;
  }
}

/**
 * Resolves the realpath of `absolutePath`, falling back to the nearest
 * existing ancestor when the target itself does not yet exist (so the
 * containment check still works for `write_file` to a brand-new path).
 */
async function realpathOfPathOrAncestor(absolutePath: string): Promise<string> {
  let current = absolutePath;
  let suffix = '';
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  while (true) {
    try {
      const real = await realpath(current);
      return suffix === '' ? real : resolve(real, suffix);
    } catch {
      const parent = resolve(current, '..');
      if (parent === current) {
        return absolutePath;
      }
      const base = current.slice(parent.length + 1);
      suffix = suffix === '' ? base : `${base}/${suffix}`;
      current = parent;
    }
  }
}

/**
 * Resolves a workspace path AND follows any symlinks before checking
 * containment, so a symlink inside the workspace pointing outside is
 * rejected even though the lexical path looks safe. Handles paths that
 * don't yet exist (e.g. write_file targets) by realpath-resolving the
 * nearest existing ancestor and re-attaching the unresolved suffix.
 */
export async function resolveWorkspacePathSafe(
  filePath: string,
  config: t.LocalExecutionConfig = {},
  intent: 'read' | 'write' = 'write'
): Promise<string> {
  const lexical = resolveWorkspacePath(filePath, config, intent);
  const roots = intent === 'write' ? getWriteRoots(config) : getReadRoots(config);
  if (roots == null) {
    return lexical;
  }
  const realRoots = await Promise.all(roots.map(realpathOrSelf));
  const realPath = await realpathOfPathOrAncestor(lexical);
  if (isInsideAnyRoot(realPath, realRoots)) {
    return lexical;
  }
  throw new Error(
    `Path is outside the local workspace (symlink escape): ${filePath}`
  );
}
