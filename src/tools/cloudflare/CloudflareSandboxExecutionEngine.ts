import { EventEmitter } from 'events';
import { PassThrough } from 'stream';
import { posix as path } from 'path';
import type { ChildProcessWithoutNullStreams } from 'child_process';
import type { WriteFileOptions, MakeDirectoryOptions, Stats } from 'fs';
import type { FileHandle } from 'fs/promises';
import type * as t from '@/types';
import {
  LOCAL_SPAWN_TIMEOUT_MS,
  validateBashCommand,
} from '@/tools/local/LocalExecutionEngine';
import type { WorkspaceFS, ReaddirEntry } from '@/tools/local/workspaceFS';

const DEFAULT_WORKSPACE_ROOT = '/workspace';
const DEFAULT_TIMEOUT_MS = 60000;
const DEFAULT_MAX_OUTPUT_CHARS = 200000;
const PROTECTED_TARGET_ARG_RE = /^(?:\/|~|\$\{?HOME\}?|\.)(?:\/?\.?\*|\/)?$/;
const DESTRUCTIVE_OP_IN_COMMAND_RE =
  /\b(?:rm\s+-[^\s]*[rf]|chmod\s+-R|chown\s+-R)\b/;

type SpawnResult = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
};

type RuntimeCommand = {
  fileName: string;
  source?: string;
  command: string;
};

type SandboxRuntimeContext = {
  sandbox: t.CloudflareSandboxRuntime;
  workspaceRoot: string;
  env?: Record<string, string | undefined>;
  timeoutMs: number;
  maxOutputChars: number;
  shell: string;
};

const sandboxFactoryCache = new WeakMap<
  t.CloudflareSandboxExecutionConfig,
  Promise<t.CloudflareSandboxRuntime>
>();

function normalizeWorkspaceRoot(workspaceRoot: string): string {
  const normalized = path.normalize(workspaceRoot);
  return normalized === '/' ? normalized : normalized.replace(/\/+$/, '');
}

export function getCloudflareWorkspaceRoot(
  config?: t.CloudflareSandboxExecutionConfig
): string {
  return normalizeWorkspaceRoot(
    config?.workspaceRoot ?? DEFAULT_WORKSPACE_ROOT
  );
}

export async function resolveCloudflareSandbox(
  config: t.CloudflareSandboxExecutionConfig
): Promise<t.CloudflareSandboxRuntime> {
  const sandbox = config.sandbox;
  if (typeof sandbox !== 'function') {
    return sandbox;
  }
  let cached = sandboxFactoryCache.get(config);
  if (cached == null) {
    cached = Promise.resolve()
      .then(() => sandbox())
      .catch((error: unknown) => {
        sandboxFactoryCache.delete(config);
        throw error;
      });
    sandboxFactoryCache.set(config, cached);
  }
  return cached;
}

async function getRuntimeContext(
  config: t.CloudflareSandboxExecutionConfig
): Promise<SandboxRuntimeContext> {
  return {
    sandbox: await resolveCloudflareSandbox(config),
    workspaceRoot: getCloudflareWorkspaceRoot(config),
    env: config.env,
    timeoutMs: config.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    maxOutputChars: config.maxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS,
    shell: config.shell ?? 'bash',
  };
}

function toSandboxPath(filePath: string, workspaceRoot: string): string {
  const raw = filePath === '' ? '.' : filePath;
  const root = normalizeWorkspaceRoot(workspaceRoot);
  const resolved = raw.startsWith('/')
    ? path.normalize(raw)
    : path.resolve(root, raw);
  if (root === '/') {
    return resolved;
  }
  if (resolved === root || resolved.startsWith(`${root}/`)) {
    return resolved;
  }
  throw new Error(
    `Path is outside the Cloudflare sandbox workspace: ${filePath}`
  );
}

function quote(value: string): string {
  if (value === '') {
    return '\'\'';
  }
  if (/^[A-Za-z0-9_/:=.,@%+-]+$/.test(value)) {
    return value;
  }
  return `'${value.replace(/'/g, '\'\\\'\'')}'`;
}

function withInSandboxTimeout(command: string, timeoutMs: number): string {
  const timeoutSeconds = Math.max(1, Math.ceil(timeoutMs / 1000));
  return `timeout -k 2s ${timeoutSeconds}s ${command}`;
}

function outerTimeoutMs(timeoutMs: number): number {
  return timeoutMs + 5000;
}

function isInSandboxTimeoutExit(exitCode: number | null): boolean {
  return exitCode === 124 || exitCode === 137;
}

function truncateOutput(value: string, maxChars: number): string {
  if (maxChars <= 0 || value.length <= maxChars) {
    return value;
  }
  const head = Math.max(Math.floor(maxChars / 2), 0);
  const tail = Math.max(maxChars - head, 0);
  return `${value.slice(0, head)}\n...[truncated ${value.length - maxChars} chars]...\n${value.slice(value.length - tail)}`;
}

async function readStream(stream: ReadableStream<Uint8Array>): Promise<Buffer> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks.map((chunk) => Buffer.from(chunk)));
}

async function normalizeReadFileContent(
  result: t.CloudflareSandboxReadFileResult
): Promise<Buffer> {
  if (typeof result === 'string') {
    return Buffer.from(result, 'utf8');
  }
  if (Buffer.isBuffer(result)) {
    return result;
  }
  if (result instanceof Uint8Array) {
    return Buffer.from(result);
  }
  const content = result.content;
  if (typeof content === 'string') {
    if (result.encoding === 'base64') {
      return Buffer.from(content, 'base64');
    }
    return Buffer.from(content, 'utf8');
  }
  if (Buffer.isBuffer(content)) {
    return content;
  }
  if (content instanceof Uint8Array) {
    return Buffer.from(content);
  }
  return readStream(content);
}

function bytesToStream(bytes: Uint8Array): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller): void {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

function normalizeWriteFileContent(content: string | Buffer | Uint8Array): {
  content: string | ReadableStream<Uint8Array>;
  options?: { encoding?: string };
} {
  if (typeof content === 'string') {
    return { content, options: { encoding: 'utf8' } };
  }
  return { content: bytesToStream(content) };
}

function createStats(info: {
  size?: number;
  type?: t.CloudflareSandboxFileInfo['type'];
}): Stats {
  const type = info.type ?? 'file';
  const now = new Date();
  return {
    size: info.size ?? 0,
    isFile: () => type === 'file',
    isDirectory: () => type === 'directory',
    isSymbolicLink: () => type === 'symlink',
    isBlockDevice: () => false,
    isCharacterDevice: () => false,
    isFIFO: () => false,
    isSocket: () => false,
    dev: 0,
    ino: 0,
    mode: 0,
    nlink: 1,
    uid: 0,
    gid: 0,
    rdev: 0,
    blksize: 0,
    blocks: 0,
    atimeMs: now.getTime(),
    mtimeMs: now.getTime(),
    ctimeMs: now.getTime(),
    birthtimeMs: now.getTime(),
    atime: now,
    mtime: now,
    ctime: now,
    birthtime: now,
  } as Stats;
}

function normalizeFileList(
  result: t.CloudflareSandboxListFilesResult
): t.CloudflareSandboxFileInfo[] {
  return Array.isArray(result) ? result : result.files;
}

function entryNameFor(
  info: t.CloudflareSandboxFileInfo,
  parentPath: string
): string {
  if (info.name !== '') {
    return info.name.includes('/') ? path.basename(info.name) : info.name;
  }
  if (info.absolutePath != null && info.absolutePath !== '') {
    return path.basename(info.absolutePath);
  }
  if (info.relativePath != null && info.relativePath !== '') {
    return path.basename(info.relativePath);
  }
  return path.basename(parentPath);
}

function entryAbsolutePath(
  info: t.CloudflareSandboxFileInfo,
  parentPath: string
): string {
  if (info.absolutePath != null && info.absolutePath !== '') {
    return path.normalize(info.absolutePath);
  }
  if (info.relativePath != null && info.relativePath !== '') {
    return path.resolve(parentPath, info.relativePath);
  }
  return path.resolve(parentPath, info.name);
}

function createDirent(info: t.CloudflareSandboxFileInfo): ReaddirEntry {
  return {
    name: entryNameFor(info, ''),
    isFile: () => (info.type ?? 'file') === 'file',
    isDirectory: () => info.type === 'directory',
    isSymbolicLink: () => info.type === 'symlink',
  };
}

async function findChildInfo(
  sandbox: t.CloudflareSandboxRuntime,
  filePath: string
): Promise<t.CloudflareSandboxFileInfo | undefined> {
  const parent = path.dirname(filePath);
  const basename = path.basename(filePath);
  const entries = normalizeFileList(
    await sandbox.listFiles(parent, { includeHidden: true })
  );
  return entries.find((entry) => {
    const absolute = entryAbsolutePath(entry, parent);
    return absolute === filePath || entryNameFor(entry, parent) === basename;
  });
}

export function createCloudflareWorkspaceFS(
  config: t.CloudflareSandboxExecutionConfig
): WorkspaceFS {
  const workspaceRoot = getCloudflareWorkspaceRoot(config);

  const fs: WorkspaceFS = {
    readFile: (async (filePath: string, encoding?: 'utf8') => {
      const sandbox = await resolveCloudflareSandbox(config);
      const resolved = toSandboxPath(filePath, workspaceRoot);
      const buffer = await normalizeReadFileContent(
        await sandbox.readFile(resolved, encoding ? { encoding } : undefined)
      );
      return encoding != null ? buffer.toString(encoding) : buffer;
    }) as WorkspaceFS['readFile'],
    writeFile: async (
      filePath: string,
      content: string | Buffer,
      _options?: WriteFileOptions
    ) => {
      const sandbox = await resolveCloudflareSandbox(config);
      const resolved = toSandboxPath(filePath, workspaceRoot);
      const normalized = normalizeWriteFileContent(content);
      await sandbox.writeFile(resolved, normalized.content, normalized.options);
    },
    stat: async (filePath: string) => {
      const sandbox = await resolveCloudflareSandbox(config);
      const resolved = toSandboxPath(filePath, workspaceRoot);
      if (resolved === workspaceRoot) {
        const entries = normalizeFileList(
          await sandbox.listFiles(resolved, { includeHidden: true })
        );
        return createStats({ size: entries.length, type: 'directory' });
      }
      const info = await findChildInfo(sandbox, resolved);
      if (info != null) {
        return createStats({ size: info.size, type: info.type });
      }
      try {
        const entries = normalizeFileList(
          await sandbox.listFiles(resolved, { includeHidden: true })
        );
        return createStats({ size: entries.length, type: 'directory' });
      } catch {
        const buffer = await normalizeReadFileContent(
          await sandbox.readFile(resolved)
        );
        return createStats({ size: buffer.length, type: 'file' });
      }
    },
    readdir: (async (filePath: string, options?: { withFileTypes: true }) => {
      const sandbox = await resolveCloudflareSandbox(config);
      const resolved = toSandboxPath(filePath, workspaceRoot);
      const entries = normalizeFileList(
        await sandbox.listFiles(resolved, { includeHidden: true })
      );
      if (options?.withFileTypes === true) {
        return entries.map(createDirent);
      }
      return entries.map((entry) => entryNameFor(entry, resolved));
    }) as WorkspaceFS['readdir'],
    mkdir: async (filePath: string, options?: MakeDirectoryOptions) => {
      const sandbox = await resolveCloudflareSandbox(config);
      await sandbox.mkdir(toSandboxPath(filePath, workspaceRoot), {
        recursive: options?.recursive,
      });
    },
    realpath: async (filePath: string) =>
      toSandboxPath(filePath, workspaceRoot),
    unlink: async (filePath: string) => {
      const sandbox = await resolveCloudflareSandbox(config);
      await sandbox.deleteFile(toSandboxPath(filePath, workspaceRoot));
    },
    open: async (filePath: string, _flags: 'r') => {
      const sandbox = await resolveCloudflareSandbox(config);
      const resolved = toSandboxPath(filePath, workspaceRoot);
      const buffer = await normalizeReadFileContent(
        await sandbox.readFile(resolved)
      );
      return {
        read: async (
          target: Buffer,
          offset: number,
          length: number,
          position: number
        ) => {
          const start = Math.max(position, 0);
          const slice = buffer.subarray(start, start + length);
          slice.copy(target, offset);
          return { bytesRead: slice.length, buffer: target };
        },
        close: async () => undefined,
      } as unknown as FileHandle;
    },
  };

  return fs;
}

function createCloudflareSpawn(
  config: t.CloudflareSandboxExecutionConfig
): t.LocalSpawn {
  return (command, args, options) => {
    const stdout = new PassThrough();
    const stderr = new PassThrough();
    const child = new EventEmitter() as ChildProcessWithoutNullStreams;
    const abortController = new AbortController();
    const state = { closed: false };
    const closeOnce = (
      exitCode: number | null,
      signal: NodeJS.Signals | null
    ): void => {
      if (state.closed) {
        return;
      }
      state.closed = true;
      stdout.end();
      stderr.end();
      Object.assign(child, {
        exitCode,
        signalCode: signal,
      });
      child.emit('close', exitCode, signal);
    };
    Object.assign(child, {
      stdout,
      stderr,
      stdin: new PassThrough(),
      stdio: [null, stdout, stderr],
      killed: false,
      exitCode: null,
      signalCode: null,
      pid: undefined,
      kill: (signal: NodeJS.Signals = 'SIGTERM') => {
        Object.assign(child, { killed: true, signalCode: signal });
        abortController.abort();
        closeOnce(null, signal);
        return true;
      },
    });

    void (async (): Promise<void> => {
      const ctx = await getRuntimeContext(config);
      const rendered = [command, ...args].map(quote).join(' ');
      const spawnTimeoutMs = (
        options as {
          [LOCAL_SPAWN_TIMEOUT_MS]?: number;
        }
      )[LOCAL_SPAWN_TIMEOUT_MS];
      const timeoutMs =
        typeof spawnTimeoutMs === 'number' && Number.isFinite(spawnTimeoutMs)
          ? spawnTimeoutMs
          : ctx.timeoutMs;
      const timedCommand = withInSandboxTimeout(rendered, timeoutMs);
      const cwd =
        options.cwd == null ? ctx.workspaceRoot : options.cwd.toString();
      if (state.closed) {
        return;
      }
      const execOptions: t.CloudflareSandboxExecOptions = {
        cwd,
        env: ctx.env,
        timeout: outerTimeoutMs(timeoutMs),
      };
      if (ctx.sandbox.supportsExecSignal === true) {
        execOptions.signal = abortController.signal;
      }
      try {
        const result = await ctx.sandbox.exec(timedCommand, execOptions);
        if (state.closed) {
          return;
        }
        if (result.stdout) stdout.write(result.stdout);
        if (result.stderr) stderr.write(result.stderr);
        closeOnce(result.exitCode, null);
      } catch (error) {
        if (state.closed) {
          return;
        }
        stderr.write((error as Error).message);
        closeOnce(1, null);
      }
    })();

    return child;
  };
}

export function createCloudflareLocalExecutionConfig(
  config: t.CloudflareSandboxExecutionConfig
): t.LocalExecutionConfig {
  const workspaceRoot = getCloudflareWorkspaceRoot(config);
  return {
    cwd: workspaceRoot,
    workspace: { root: workspaceRoot },
    exec: {
      spawn: createCloudflareSpawn(config),
      fs: createCloudflareWorkspaceFS(config),
      sandboxed: true,
    },
    shell: config.shell ?? 'bash',
    timeoutMs: config.timeoutMs,
    maxOutputChars: config.maxOutputChars,
    env: config.env,
    includeCodingTools: config.includeCodingTools,
    compileCheck: config.compileCheck,
    readOnly: config.readOnly,
    allowDangerousCommands: config.allowDangerousCommands,
    bashAst: config.bashAst,
    fileCheckpointing: config.fileCheckpointing,
    maxReadBytes: config.maxReadBytes,
    attachReadAttachments: config.attachReadAttachments,
    maxAttachmentBytes: config.maxAttachmentBytes,
    postEditSyntaxCheck: config.postEditSyntaxCheck,
  };
}

export async function validateCloudflareBashCommand(
  command: string,
  args: readonly string[],
  config: t.CloudflareSandboxExecutionConfig
): Promise<void> {
  const localConfig = createCloudflareLocalExecutionConfig(config);
  const validation = await validateBashCommand(command, localConfig);
  if (!validation.valid) {
    throw new Error(validation.errors.join('\n'));
  }

  if (
    args.length > 0 &&
    config.allowDangerousCommands !== true &&
    DESTRUCTIVE_OP_IN_COMMAND_RE.test(command)
  ) {
    const offending = args.find((arg) => PROTECTED_TARGET_ARG_RE.test(arg));
    if (offending !== undefined) {
      throw new Error(
        `Command matches a destructive command pattern (protected target "${offending}" passed via positional arg).`
      );
    }
  }
}

export async function executeCloudflareBash(
  command: string,
  config: t.CloudflareSandboxExecutionConfig,
  args: readonly string[] = []
): Promise<SpawnResult> {
  await validateCloudflareBashCommand(command, args, config);
  const ctx = await getRuntimeContext(config);
  const shellCommand =
    args.length > 0
      ? `${ctx.shell} -lc ${quote(command)} -- ${args.map(quote).join(' ')}`
      : `${ctx.shell} -lc ${quote(command)}`;
  const result = await ctx.sandbox.exec(
    withInSandboxTimeout(shellCommand, ctx.timeoutMs),
    {
      cwd: ctx.workspaceRoot,
      env: ctx.env,
      timeout: outerTimeoutMs(ctx.timeoutMs),
    }
  );
  return {
    stdout: truncateOutput(result.stdout, ctx.maxOutputChars),
    stderr: truncateOutput(result.stderr, ctx.maxOutputChars),
    exitCode: result.exitCode,
    timedOut: isInSandboxTimeoutExit(result.exitCode),
  };
}

function runtimeForCode(
  lang: string,
  tempDir: string,
  code: string,
  args: string[] = [],
  shell = 'bash'
): RuntimeCommand {
  const fileFor = (name: string): string => path.join(tempDir, name);
  const argText = args.map(quote).join(' ');
  switch (lang) {
  case 'py':
  case 'python':
    return {
      fileName: 'main.py',
      source: code,
      command: `python3 ${quote(fileFor('main.py'))} ${argText}`,
    };
  case 'js':
  case 'javascript':
    return {
      fileName: 'main.js',
      source: code,
      command: `node ${quote(fileFor('main.js'))} ${argText}`,
    };
  case 'ts':
  case 'typescript':
    return {
      fileName: 'main.ts',
      source: code,
      command: `npx --no-install tsx ${quote(fileFor('main.ts'))} ${argText}`,
    };
  case 'php':
    return {
      fileName: 'main.php',
      source: code,
      command: `php ${quote(fileFor('main.php'))} ${argText}`,
    };
  case 'go':
    return {
      fileName: 'main.go',
      source: code,
      command: `go run ${quote(fileFor('main.go'))} ${argText}`,
    };
  case 'rs':
    return {
      fileName: 'main.rs',
      source: code,
      command: `${shell} -lc ${quote(
        `rustc ${quote(fileFor('main.rs'))} -o ${quote(fileFor('main-rs'))} && ${quote(fileFor('main-rs'))} ${argText}`
      )}`,
    };
  case 'c':
    return {
      fileName: 'main.c',
      source: code,
      command: `${shell} -lc ${quote(
        `cc ${quote(fileFor('main.c'))} -o ${quote(fileFor('main-c'))} && ${quote(fileFor('main-c'))} ${argText}`
      )}`,
    };
  case 'cpp':
    return {
      fileName: 'main.cpp',
      source: code,
      command: `${shell} -lc ${quote(
        `c++ ${quote(fileFor('main.cpp'))} -o ${quote(fileFor('main-cpp'))} && ${quote(fileFor('main-cpp'))} ${argText}`
      )}`,
    };
  case 'java':
    return {
      fileName: 'Main.java',
      source: code,
      command: `${shell} -lc ${quote(
        `javac ${quote(fileFor('Main.java'))} && java -cp ${quote(tempDir)} Main ${argText}`
      )}`,
    };
  case 'r':
    return {
      fileName: 'main.R',
      source: code,
      command: `Rscript ${quote(fileFor('main.R'))} ${argText}`,
    };
  case 'd':
    return {
      fileName: 'main.d',
      source: code,
      command: `${shell} -lc ${quote(
        `dmd ${quote(fileFor('main.d'))} -of=${quote(fileFor('main-d'))} && ${quote(fileFor('main-d'))} ${argText}`
      )}`,
    };
  case 'f90':
    return {
      fileName: 'main.f90',
      source: code,
      command: `${shell} -lc ${quote(
        `gfortran ${quote(fileFor('main.f90'))} -o ${quote(fileFor('main-f90'))} && ${quote(fileFor('main-f90'))} ${argText}`
      )}`,
    };
  case 'bash':
  case 'sh':
    return {
      fileName: 'main.sh',
      source: code,
      command: `${shell} -lc ${quote(code)} -- ${argText}`,
    };
  default:
    throw new Error(`Unsupported Cloudflare sandbox runtime: ${lang}`);
  }
}

export async function executeCloudflareCode(
  input: { lang: string; code: string; args?: string[] },
  config: t.CloudflareSandboxExecutionConfig
): Promise<SpawnResult> {
  if (input.lang === 'bash' || input.lang === 'sh') {
    return executeCloudflareBash(input.code, config, input.args ?? []);
  }
  const ctx = await getRuntimeContext(config);
  const id = globalThis.crypto.randomUUID();
  const tempDir = path.join(ctx.workspaceRoot, '.lc-exec', id);
  const runtime = runtimeForCode(
    input.lang,
    tempDir,
    input.code,
    input.args,
    ctx.shell
  );
  await ctx.sandbox.mkdir(tempDir, { recursive: true });
  if (runtime.source != null) {
    await ctx.sandbox.writeFile(
      path.join(tempDir, runtime.fileName),
      runtime.source,
      {
        encoding: 'utf8',
      }
    );
  }
  try {
    const result = await ctx.sandbox.exec(
      withInSandboxTimeout(runtime.command, ctx.timeoutMs),
      {
        cwd: ctx.workspaceRoot,
        env: ctx.env,
        timeout: outerTimeoutMs(ctx.timeoutMs),
      }
    );
    return {
      stdout: truncateOutput(result.stdout, ctx.maxOutputChars),
      stderr: truncateOutput(result.stderr, ctx.maxOutputChars),
      exitCode: result.exitCode,
      timedOut: isInSandboxTimeoutExit(result.exitCode),
    };
  } finally {
    await ctx.sandbox
      .exec(`rm -rf ${quote(tempDir)}`, {
        cwd: ctx.workspaceRoot,
        env: ctx.env,
        timeout: 10000,
      })
      .catch(() => undefined);
  }
}

export function formatCloudflareOutput(
  result: SpawnResult,
  cwd: string
): string {
  let formatted = '';
  if (result.stdout !== '') {
    formatted += `stdout:\n${result.stdout}\n`;
  } else {
    formatted += 'stdout: Empty. Ensure you\'re writing output explicitly.\n';
  }
  if (result.stderr !== '') {
    formatted += `stderr:\n${result.stderr}\n`;
  }
  if (result.exitCode != null && result.exitCode !== 0) {
    formatted += `exit_code: ${result.exitCode}\n`;
  }
  if (result.timedOut) {
    formatted += 'timed_out: true\n';
  }
  formatted += `working_directory: ${cwd}`;
  return formatted.trim();
}
