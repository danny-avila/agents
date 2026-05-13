import { tmpdir } from 'os';
import { isAbsolute, relative, resolve } from 'path';
import { createHash, randomUUID } from 'crypto';
import { mkdir, realpath, rm, writeFile } from 'fs/promises';
import { createWriteStream, unlinkSync } from 'fs';
import { spawn } from 'child_process';
import type { ChildProcess } from 'child_process';
import {
  validateBashCommandStatic,
  type BashValidationResult,
} from './bashValidation';
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

// Regex bodies for destructive/quoted/nested-shell/positional-arg
// command shapes live in `./bashValidation.ts` so the same set is
// reusable from the remote BashExecutor without pulling in
// `LocalExecutionEngine`'s node-only deps. `validateBashCommand`
// below is now a thin wrapper that adds the `bash -n` syntax
// preflight on top of `validateBashCommandStatic`.

type SpawnResult = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  /**
   * True when the process was force-killed because total streamed bytes
   * exceeded `maxSpawnedBytes`. Distinct from `timedOut`. Without this
   * flag, callers (`bash_tool`, `execute_code`, etc.) would see a
   * SIGKILL'd process with `exitCode: null` and treat it as success
   * (Codex P1 — runaway commands like `yes` or noisy builds silently
   * looked successful even though their output was truncated).
   */
  overflowKilled?: boolean;
  /**
   * Signal name (e.g. `'SIGKILL'`, `'SIGSEGV'`) when the process was
   * terminated by a signal. Distinct from the overflow-kill path:
   * this captures `kill -9 $$` from inside the script, native crashes,
   * OS OOM killer, etc. Without this, signal-killed processes
   * reported `exitCode: null` and looked like clean runs (Codex P2 —
   * generalization of the overflow-kill fix). When present, the
   * exitCode field is also synthesized to `128 + signum` per the
   * POSIX convention so non-null-exit-code consumers see a failure.
   */
  signal?: string;
  /** Path to the full untruncated stdout/stderr when output exceeded `maxOutputChars`. */
  fullOutputPath?: string;
};

/**
 * POSIX convention: `128 + signum` when a process is killed by a
 * signal. Maps the common signals; unknown ones default to 1 so the
 * caller still sees a non-zero (failed) exit. Only used when Node's
 * `close` event reports `exitCode === null` (true signal kill).
 */
const SIGNAL_TO_EXIT_CODE: Record<string, number> = {
  SIGHUP: 129,
  SIGINT: 130,
  SIGQUIT: 131,
  SIGILL: 132,
  SIGTRAP: 133,
  SIGABRT: 134,
  SIGBUS: 135,
  SIGFPE: 136,
  SIGKILL: 137,
  SIGUSR1: 138,
  SIGSEGV: 139,
  SIGUSR2: 140,
  SIGPIPE: 141,
  SIGALRM: 142,
  SIGTERM: 143,
};
function exitCodeForSignal(signal: string | null): number {
  if (signal == null) return 1;
  return SIGNAL_TO_EXIT_CODE[signal] ?? 1;
}

type RuntimeCommand = {
  command: string;
  args: string[];
  fileName: string;
  source?: string;
};

type SandboxManagerType = {
  checkDependencies(): { errors: string[] };
  initialize(config: BuiltSandboxRuntimeConfig): Promise<void>;
  reset(): Promise<void>;
  wrapWithSandbox(command: string): Promise<string>;
};

type SandboxRuntimeModule = {
  getDefaultWritePaths(): string[];
  SandboxManager: SandboxManagerType;
};

let sandboxConfigKey: string | undefined;
let sandboxInitialized = false;
let sandboxRuntimePromise: Promise<SandboxRuntimeModule> | undefined;

/**
 * Registry of spill files this Node process has created. The spill
 * exists so the model can `cat $fullOutputPath` to inspect overflow
 * output across subsequent tool calls — its lifetime must outlast
 * the originating spawn, but should NOT outlast the Node process
 * (otherwise CI runners and dev machines accumulate orphan files in
 * `tmpdir()` forever; the GiB-scale tmp leak that prompted this fix).
 *
 * On normal process exit, `unlinkAllSpillFiles()` synchronously
 * deletes every tracked path. `_resetLocalEngineSpillFilesForTests`
 * lets test suites trigger the same cleanup in `afterEach`/`afterAll`
 * so a long-running Jest worker doesn't accumulate files between
 * unrelated test files.
 *
 * Unclean process exits (SIGKILL, native crash, OS OOM killer) still
 * leak — same as today. The registry is a best-effort cleanup, not
 * a guarantee.
 */
const trackedSpillPaths = new Set<string>();
let spillExitHandlerInstalled = false;

function unlinkAllSpillFiles(): void {
  for (const p of trackedSpillPaths) {
    try {
      unlinkSync(p);
    } catch {
      /* file may already be gone; cleanup is best-effort */
    }
  }
  trackedSpillPaths.clear();
}

function ensureSpillExitHandler(): void {
  if (spillExitHandlerInstalled) return;
  spillExitHandlerInstalled = true;
  process.on('exit', unlinkAllSpillFiles);
}

/**
 * Test-only: synchronously delete every tracked spill file and
 * forget them. Pair with `_resetLocalEngineWarningsForTests` in
 * `afterEach`/`afterAll` for full state reset.
 *
 * @internal Not part of the public SDK surface.
 */
export function _resetLocalEngineSpillFilesForTests(): void {
  unlinkAllSpillFiles();
}

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
export function getWorkspaceRoots(config?: t.LocalExecutionConfig): string[] {
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
export function getSpawn(config?: t.LocalExecutionConfig): t.LocalSpawn {
  return (config?.exec?.spawn ?? config?.spawn ?? spawn) as t.LocalSpawn;
}

/**
 * Pluggable filesystem resolver. Honours `local.exec.fs`, falls back
 * to the Node-host implementation. A future remote engine supplies
 * its own implementation here and inherits every file-touching tool.
 */
export function getWorkspaceFS(config?: t.LocalExecutionConfig): WorkspaceFS {
  return config?.exec?.fs ?? nodeWorkspaceFS;
}

/**
 * Resolves the workspace boundary for *write* operations. Honours
 * `workspace.allowWriteOutside` (and the deprecated
 * `allowOutsideWorkspace`) by returning `null`, which the path-safety
 * helpers interpret as "skip the write clamp".
 */
export function getWriteRoots(
  config: t.LocalExecutionConfig = {}
): string[] | null {
  // Granular flag wins over the legacy one when explicitly set
  // (true OR false) — otherwise a host tightening access during
  // migration (`allowOutsideWorkspace: true, workspace.
  // allowWriteOutside: false`) would still get the loose behavior
  // because the legacy flag short-circuited the OR. Codex P1 #36.
  const granular = config.workspace?.allowWriteOutside;
  if (granular === true) return null;
  if (granular === false) return getWorkspaceRoots(config);
  if (config.allowOutsideWorkspace === true) return null;
  return getWorkspaceRoots(config);
}

/**
 * Resolves the workspace boundary for *read* operations. Honours
 * `workspace.allowReadOutside` (and the deprecated
 * `allowOutsideWorkspace`) by returning `null`.
 */
export function getReadRoots(
  config: t.LocalExecutionConfig = {}
): string[] | null {
  // Same precedence as getWriteRoots: granular flag is authoritative
  // when set, legacy flag is the fallback. Codex P1 #36.
  const granular = config.workspace?.allowReadOutside;
  if (granular === true) return null;
  if (granular === false) return getWorkspaceRoots(config);
  if (config.allowOutsideWorkspace === true) return null;
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
const sandboxRuntimePackage = '@anthropic-ai/sandbox-runtime';

/** Lazy-loads the ESM-only sandbox runtime only when sandboxing is enabled. */
function loadSandboxRuntime(): Promise<SandboxRuntimeModule> {
  sandboxRuntimePromise ??= import(
    sandboxRuntimePackage
  ) as Promise<SandboxRuntimeModule>;
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

/**
 * Local-engine bash validation: synchronous regex + heuristic pass
 * (via {@link validateBashCommandStatic}) followed by a `bash -n`
 * syntax preflight. Returns the historical `{ valid, errors,
 * warnings }` shape so existing call sites (`executeLocalBash`,
 * `executeLocalBashWithArgs`, `CompileCheckTool`, …) keep working
 * unchanged.
 *
 * Remote callers that can't spawn a local bash should use
 * {@link validateBashCommandStatic} or {@link validateBashCommandHardFloor}
 * from `./bashValidation` directly.
 */
export async function validateBashCommand(
  command: string,
  config: t.LocalExecutionConfig = {},
  args?: readonly string[]
): Promise<BashValidationResult> {
  const staticResult = validateBashCommandStatic(command, args, {
    readOnly: config.readOnly === true,
    bashAst: config.bashAst ?? 'off',
    allowDangerousCommands: config.allowDangerousCommands === true,
  });
  const errors = [...staticResult.errors];
  const warnings = [...staticResult.warnings];

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
  ).catch(
    (error: Error): SpawnResult => ({
      stdout: '',
      stderr: error.message,
      exitCode: 1,
      timedOut: false,
    })
  );

  if (syntax.exitCode !== 0) {
    errors.push(
      syntax.stderr.trim() === ''
        ? 'Command failed shell syntax validation.'
        : `Command failed shell syntax validation: ${syntax.stderr.trim()}`
    );
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

/**
 * Structural shape of the sandbox-runtime config we hand to
 * `SandboxManager.initialize()`. Intentionally NOT typed as the peer
 * `SandboxRuntimeConfig` from `@anthropic-ai/sandbox-runtime`: that
 * package is an OPTIONAL peer dep, and exporting a function whose
 * return type references it would make our generated `.d.ts` import
 * a module the consumer may not have installed (Codex P1 #22 — type-
 * checking would fail with `Cannot find module
 * '@anthropic-ai/sandbox-runtime'` for any host that doesn't enable
 * local sandboxing). The shape here is a structural subset; assignable
 * to the real `SandboxRuntimeConfig` at the one runtime call site.
 *
 * @internal
 */
export interface BuiltSandboxRuntimeConfig {
  network: {
    allowedDomains: string[];
    deniedDomains: string[];
    allowUnixSockets?: string[];
    allowAllUnixSockets?: boolean;
    allowLocalBinding?: boolean;
    allowMachLookup?: string[];
  };
  filesystem: {
    denyRead: string[];
    allowRead?: string[];
    allowWrite: string[];
    denyWrite: string[];
    allowGitConfig?: boolean;
  };
}

export function buildSandboxRuntimeConfig(
  config: t.LocalExecutionConfig,
  cwd: string,
  getDefaultWritePaths: () => string[]
): BuiltSandboxRuntimeConfig {
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
  const hardKillBytes = config.maxSpawnedBytes ?? DEFAULT_MAX_SPAWNED_BYTES;
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
      // Track for process-exit cleanup. The spill file outlives the
      // spawn (so the model can `cat` it in a follow-up call) but
      // not the Node process — without this registry, every overflow
      // ever produced by every Run on this machine accumulates in
      // tmpdir forever. Lazy `process.on('exit')` install avoids
      // adding a global listener until a spill actually happens.
      trackedSpillPaths.add(spillPath);
      ensureSpillExitHandler();
      spillStream = createWriteStream(spillPath);
      spillStream.write('===== stdout =====\n');
      spillStream.write(stdout);
      spillStream.write('\n===== stderr =====\n');
      spillStream.write(stderr);
      spillStream.write('\n===== overflow stream begins here =====\n');
    };

    const handleChunk = (buf: Buffer, kind: 'stdout' | 'stderr'): void => {
      // Once the overflow guard has tripped, drop EVERY subsequent
      // chunk on the floor. Pre-fix this was the GB-scale tmp-leak
      // bug: the first overflow chunk returned, but subsequent chunks
      // (from a fast-output child like `yes` that ignores SIGTERM)
      // kept falling through to `ensureSpill()` and writing into the
      // temp file for the full 2-second SIGTERM→SIGKILL escalation
      // window — easily 5–20 GiB at typical `yes` rates. We still
      // tally `totalSpawnedBytes` so the post-kill summary is
      // accurate, but no bytes go to memory or disk after the kill.
      totalSpawnedBytes += buf.length;
      if (overflowKilled) {
        return;
      }
      // hardKillBytes <= 0 means "no cap" per the public config contract
      // (see LocalExecutionConfig.maxSpawnedBytes). Skip the kill check
      // entirely in that case so a single byte doesn't terminate the run.
      if (hardKillBytes > 0 && totalSpawnedBytes > hardKillBytes) {
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

    child.stdout.on('data', (chunk: Buffer) => {
      handleChunk(chunk, 'stdout');
    });

    child.stderr.on('data', (chunk: Buffer) => {
      handleChunk(chunk, 'stderr');
    });

    child.on('error', fail);

    child.on('close', (exitCode, signal) => {
      // Synthesize a non-zero exit code whenever the process exited
      // by signal — Node reports `exitCode: null` in that case and
      // the formatter only prints non-null exit codes, so signal
      // kills (overflow guard, `kill -9 $$` from inside the script,
      // native crashes, OS OOM killer, …) would otherwise look like
      // successful runs (Codex P1 + Codex P2). Overflow path keeps
      // its 137 (SIGKILL) for compatibility; other signals map per
      // POSIX `128 + signum`.
      let finalExit: number | null = exitCode;
      if (finalExit == null) {
        if (overflowKilled) {
          finalExit = 137;
        } else if (signal != null) {
          finalExit = exitCodeForSignal(signal);
        }
      }
      finish({
        stdout,
        stderr,
        exitCode: finalExit,
        timedOut,
        ...(overflowKilled ? { overflowKilled: true } : {}),
        ...(signal != null ? { signal } : {}),
      });
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
/**
 * Variant of `executeLocalBash` that exposes `args` as positional
 * shell parameters (`$1`, `$2`, …). The positional-arg
 * protected-target check (e.g. `command: 'rm -rf "$1"', args: ['/']`)
 * lives in `validateBashCommandStatic` — we thread `args` through to
 * the validator so the bypass-via-positional case is caught up front.
 */
export async function executeLocalBashWithArgs(
  command: string,
  args: readonly string[],
  config: t.LocalExecutionConfig = {}
): Promise<SpawnResult> {
  const validation = await validateBashCommand(command, config, args);
  if (!validation.valid) {
    throw new Error(validation.errors.join('\n'));
  }
  const shell = config.shell ?? DEFAULT_SHELL;
  return spawnLocalProcess(shell, ['-lc', command, '--', ...args], config);
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
    const runtime = getRuntimeCommand(
      input.lang,
      tempDir,
      input.code,
      input.args,
      config.shell
    );
    if (runtime.source != null) {
      await writeFile(
        resolve(tempDir, runtime.fileName),
        runtime.source,
        'utf8'
      );
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
  args: string[] = [],
  // Override for the shell used by compile-style runtimes (`rs`,
  // `c`, `cpp`, `java`, `d`, `f90`). Threads `local.shell` so a host
  // that doesn't have bash (or wants `/bin/sh` / zsh) can still
  // execute these languages — Codex P2 #29: the bare-bash hardcode
  // mirrored the same gap that Codex P1 #6 fixed for the syntax
  // preflight, but had been missed for these runtime invocations.
  shellOverride?: string
): RuntimeCommand {
  const fileFor = (name: string): string => resolve(tempDir, name);
  const shell = shellOverride ?? configShell();

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
      command: shell,
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
      command: shell,
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
      command: shell,
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
      command: shell,
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
      command: shell,
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
      command: shell,
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

/**
 * How long after SIGTERM we wait before escalating to SIGKILL. A
 * cooperative process gets a graceful chance to flush + clean up;
 * a process that ignores or traps SIGTERM (`trap '' TERM`) gets
 * killed unconditionally so timeoutMs / maxSpawnedBytes can't be
 * defeated by a hostile script. Codex P1 #28 — pre-fix the spawn
 * promise would never resolve in that case and the entire tool run
 * would hang past the advertised timeout.
 */
const SIGKILL_ESCALATION_MS = 2000;

function sigterm(child: ChildProcess): void {
  if (child.pid == null) return;
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

function sigkill(child: ChildProcess): void {
  if (child.pid == null) return;
  if (child.exitCode != null || child.signalCode != null) return;
  try {
    if (process.platform === 'win32') {
      child.kill('SIGKILL');
      return;
    }
    process.kill(-child.pid, 'SIGKILL');
  } catch {
    try {
      child.kill('SIGKILL');
    } catch {
      /* already dead */
    }
  }
}

function killProcessTree(child: ChildProcess): void {
  sigterm(child);
  // Escalate to SIGKILL if the child is still alive after the grace
  // window. Use unref() so the timer doesn't keep the Node process
  // alive past the parent's natural exit.
  const escalation = setTimeout(() => sigkill(child), SIGKILL_ESCALATION_MS);
  escalation.unref();
  child.once('close', () => clearTimeout(escalation));
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
  const absolutePath = isAbsolute(filePath)
    ? resolve(filePath)
    : resolve(cwd, filePath);

  const roots =
    intent === 'write' ? getWriteRoots(config) : getReadRoots(config);
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

type RealpathFn = (p: string) => Promise<string>;

async function realpathOrSelf(
  absolutePath: string,
  realpathImpl: RealpathFn = realpath
): Promise<string> {
  try {
    return await realpathImpl(absolutePath);
  } catch {
    return absolutePath;
  }
}

/**
 * Resolves the realpath of `absolutePath`, falling back to the nearest
 * existing ancestor when the target itself does not yet exist (so the
 * containment check still works for `write_file` to a brand-new path).
 *
 * Codex P2 #38: takes the realpath impl as a parameter so callers
 * can route through `WorkspaceFS.realpath` when a custom engine is
 * configured. Pre-fix, host `fs/promises.realpath` would fail on a
 * remote/in-memory FS path and silently fall back to lexical
 * containment, leaving the symlink-escape clamp ineffective on
 * non-default engines.
 */
async function realpathOfPathOrAncestor(
  absolutePath: string,
  realpathImpl: RealpathFn = realpath
): Promise<string> {
  let current = absolutePath;
  let suffix = '';
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  while (true) {
    try {
      const real = await realpathImpl(current);
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
  const roots =
    intent === 'write' ? getWriteRoots(config) : getReadRoots(config);
  if (roots == null) {
    return lexical;
  }
  // Route realpath through the configured WorkspaceFS so a custom
  // engine (in-memory, remote) gets the same symlink-escape clamp
  // the host-fs path gets. Codex P2 #38: pre-fix the host realpath
  // would fail on a non-default FS path and silently fall back to
  // lexical containment, leaving the clamp ineffective.
  const fsRealpath: RealpathFn = (p) => getWorkspaceFS(config).realpath(p);
  const realRoots = await Promise.all(
    roots.map((r) => realpathOrSelf(r, fsRealpath))
  );
  const realPath = await realpathOfPathOrAncestor(lexical, fsRealpath);
  if (isInsideAnyRoot(realPath, realRoots)) {
    return lexical;
  }
  throw new Error(
    `Path is outside the local workspace (symlink escape): ${filePath}`
  );
}
