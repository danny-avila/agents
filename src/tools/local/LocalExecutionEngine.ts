import { tmpdir } from 'os';
import { isAbsolute, relative, resolve } from 'path';
import { createHash, randomUUID } from 'crypto';
import { mkdir, rm, writeFile } from 'fs/promises';
import { spawn } from 'child_process';
import type { ChildProcess } from 'child_process';
import type { SandboxRuntimeConfig } from '@anthropic-ai/sandbox-runtime';
import type * as t from '@/types';

const DEFAULT_TIMEOUT_MS = 60000;
const DEFAULT_MAX_OUTPUT_CHARS = 200000;
const DEFAULT_LOCAL_SESSION_ID = 'local';
const DEFAULT_SHELL = process.platform === 'win32' ? 'bash.exe' : 'bash';

const dangerousCommandPatterns: ReadonlyArray<RegExp> = [
  /\brm\s+(?:-[^\s]*[rf][^\s]*\s+|-[^\s]*[r][^\s]*\s+-[^\s]*[f][^\s]*\s+)(?:\/|~|\$HOME|\.)\s*(?:$|[;&|])/,
  /\b(?:mkfs|mkswap|fdisk|parted|diskutil)\b/,
  /\bdd\s+[^;&|]*\bof=\/dev\//,
  /\bchmod\s+-R\s+(?:777|a\+w)\s+(?:\/|~|\$HOME|\.)\b/,
  /\bchown\s+-R\s+[^;&|]+\s+(?:\/|~|\$HOME|\.)\b/,
  /:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:/,
];

const mutatingCommandPattern =
  /\b(?:rm|mv|cp|touch|mkdir|rmdir|ln|truncate|tee|sed\s+-i|perl\s+-pi|python(?:3)?\s+-c|node\s+-e|npm\s+(?:install|ci|update|publish)|pnpm\s+(?:install|update|publish)|yarn\s+(?:install|add|publish)|git\s+(?:add|commit|checkout|switch|reset|clean|rebase|merge|push|pull|stash|tag|branch)|chmod|chown)\b|(?:^|[^<])>\s*[^&]|\bcat\s+[^|;&]*>\s*/;

type SpawnResult = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
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
  return resolve(config?.cwd ?? process.cwd());
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
    for (const pattern of dangerousCommandPatterns) {
      if (pattern.test(normalized)) {
        errors.push('Command matches a destructive command pattern.');
        break;
      }
    }
  }

  if (config.readOnly === true && mutatingCommandPattern.test(normalized)) {
    errors.push('Command appears to mutate files or repository state in read-only local mode.');
  }

  const syntax = await spawnLocalProcess(DEFAULT_SHELL, ['-n', '-c', command], {
    ...config,
    timeoutMs: Math.min(config.timeoutMs ?? DEFAULT_TIMEOUT_MS, 5000),
    sandbox: { enabled: false },
  }).catch((error: Error): SpawnResult => ({
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

function buildSandboxRuntimeConfig(
  config: t.LocalExecutionConfig,
  cwd: string,
  getDefaultWritePaths: () => string[]
): SandboxRuntimeConfig {
  const sandbox = config.sandbox;
  return {
    network: {
      allowedDomains: sandbox?.network?.allowedDomains ?? [],
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
        cwd,
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

export async function spawnLocalProcess(
  command: string,
  args: string[],
  config: t.LocalExecutionConfig = {}
): Promise<SpawnResult> {
  const cwd = getLocalCwd(config);
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const maxOutputChars = config.maxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS;
  const sandboxManager = await ensureSandbox(config, cwd);
  let spawnCommand = command;
  let spawnArgs = args;

  if (sandboxManager != null) {
    const rendered = [command, ...args.map(shellQuote)].join(' ');
    const sandboxed = await sandboxManager.wrapWithSandbox(rendered);
    spawnCommand = config.shell ?? DEFAULT_SHELL;
    spawnArgs = ['-lc', sandboxed];
  }

  return new Promise<SpawnResult>((resolveResult, reject) => {
    const child = spawn(spawnCommand, spawnArgs, {
      cwd,
      detached: process.platform !== 'win32',
      env: { ...process.env, ...(config.env ?? {}) },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    let timedOut = false;
    let timeout: NodeJS.Timeout | undefined;

    const finish = (result: SpawnResult): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (timeout != null) {
        clearTimeout(timeout);
      }
      resolveResult({
        ...result,
        stdout: truncateLocalOutput(result.stdout, maxOutputChars),
        stderr: truncateLocalOutput(result.stderr, maxOutputChars),
      });
    };

    const fail = (error: Error): void => {
      if (settled) {
        return;
      }
      settled = true;
      if (timeout != null) {
        clearTimeout(timeout);
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
      stdout += chunk.toString('utf8');
    });

    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
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

export async function executeLocalCode(
  input: {
    lang: string;
    code: string;
    args?: string[];
  },
  config: t.LocalExecutionConfig = {}
): Promise<SpawnResult> {
  if (input.lang === 'bash') {
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
  config: t.LocalExecutionConfig = {}
): string {
  const cwd = getLocalCwd(config);
  const absolutePath = isAbsolute(filePath) ? resolve(filePath) : resolve(cwd, filePath);

  if (config.allowOutsideWorkspace === true) {
    return absolutePath;
  }

  const relativePath = relative(cwd, absolutePath);
  if (
    absolutePath !== cwd &&
    (relativePath.startsWith('..') || isAbsolute(relativePath))
  ) {
    throw new Error(`Path is outside the local workspace: ${filePath}`);
  }

  return absolutePath;
}
