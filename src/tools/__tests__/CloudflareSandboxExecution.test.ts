import type * as t from '@/types';
import { Constants } from '@/common';
import { spawnLocalProcess } from '../local/LocalExecutionEngine';
import { resolveLocalToolsForBinding } from '../local/resolveLocalExecutionTools';
import {
  createCloudflareWorkspaceFS,
  createCloudflareLocalExecutionConfig,
  executeCloudflareBash,
  executeCloudflareCode,
} from '../cloudflare/CloudflareSandboxExecutionEngine';
import { createCloudflareBridgeRuntime } from '../cloudflare/CloudflareBridgeRuntime';
import {
  createCloudflareBashProgrammaticToolCallingTool,
  createCloudflareProgrammaticToolCallingTool,
} from '../cloudflare/CloudflareProgrammaticToolCalling';

function sseResponse(events: string): Response {
  return new Response(events, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

function sseExit(exitCode = 0): Response {
  return sseResponse(`event: exit\ndata: {"exit_code":${exitCode}}\n\n`);
}

function bodyText(body: BodyInit | null | undefined): string {
  if (typeof body === 'string') {
    return body;
  }
  if (body == null) {
    return '';
  }
  if (body instanceof Uint8Array) {
    return Buffer.from(body).toString('utf8');
  }
  return String(body);
}

function createRuntime(
  overrides: Partial<t.CloudflareSandboxRuntime> = {}
): t.CloudflareSandboxRuntime {
  return {
    exec: async () => ({
      exitCode: 0,
      stdout: '',
      stderr: '',
    }),
    readFile: async () => '',
    writeFile: async () => ({ ok: true }),
    mkdir: async () => ({ ok: true }),
    listFiles: async () => [],
    deleteFile: async () => ({ ok: true }),
    ...overrides,
  };
}

describe('Cloudflare sandbox execution backend', () => {
  it('normalizes trailing workspace slashes before clamping paths', async () => {
    const readPaths: string[] = [];
    const fs = createCloudflareWorkspaceFS({
      workspaceRoot: '/workspace/',
      sandbox: createRuntime({
        readFile: async (filePath) => {
          readPaths.push(filePath);
          return 'ok';
        },
      }),
    });

    await expect(fs.readFile('file.txt', 'utf8')).resolves.toBe('ok');
    expect(readPaths).toEqual(['/workspace/file.txt']);
  });

  it('allows root workspace paths', async () => {
    const readPaths: string[] = [];
    const fs = createCloudflareWorkspaceFS({
      workspaceRoot: '/',
      sandbox: createRuntime({
        readFile: async (filePath) => {
          readPaths.push(filePath);
          return 'ok';
        },
      }),
    });

    await expect(fs.readFile('tmp/file.txt', 'utf8')).resolves.toBe('ok');
    expect(readPaths).toEqual(['/tmp/file.txt']);
  });

  it('stats the workspace root without listing its parent directory', async () => {
    const listPaths: string[] = [];
    const fs = createCloudflareWorkspaceFS({
      workspaceRoot: '/workspace/',
      sandbox: createRuntime({
        listFiles: async (filePath) => {
          listPaths.push(filePath);
          return [{ name: 'src', type: 'directory' }];
        },
      }),
    });

    const stats = await fs.stat('.');

    expect(stats.isDirectory()).toBe(true);
    expect(listPaths).toEqual(['/workspace']);
  });

  it('does not pass AbortSignal to Cloudflare spawn exec options', async () => {
    let resolveExecCalled!: () => void;
    const execCalled = new Promise<void>((resolve) => {
      resolveExecCalled = resolve;
    });
    let receivedOptions: t.CloudflareSandboxExecOptions | undefined;
    const sandbox = createRuntime({
      exec: (_command, options) => {
        receivedOptions = options;
        resolveExecCalled();
        return new Promise<t.CloudflareSandboxExecResult>(() => undefined);
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      timeoutMs: 50,
      workspaceRoot: '/workspace',
    });

    const resultPromise = spawnLocalProcess(
      'bash',
      ['-lc', 'sleep 10'],
      config
    );
    await execCalled;
    const result = await resultPromise;

    expect(receivedOptions).not.toHaveProperty('signal');
    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(143);
  });

  it('passes AbortSignal to signal-aware runtimes and aborts it on kill', async () => {
    let resolveExecCalled!: () => void;
    const execCalled = new Promise<void>((resolve) => {
      resolveExecCalled = resolve;
    });
    let receivedSignal: AbortSignal | undefined;
    let abortEvents = 0;
    const sandbox = createRuntime({
      supportsExecSignal: true,
      exec: (_command, options) => {
        receivedSignal = options?.signal;
        receivedSignal?.addEventListener('abort', () => {
          abortEvents += 1;
        });
        resolveExecCalled();
        return new Promise<t.CloudflareSandboxExecResult>(() => undefined);
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      timeoutMs: 50,
      workspaceRoot: '/workspace',
    });

    const resultPromise = spawnLocalProcess(
      'bash',
      ['-lc', 'sleep 10'],
      config
    );
    await execCalled;
    const result = await resultPromise;

    expect(receivedSignal).toBeDefined();
    expect(receivedSignal?.aborted).toBe(true);
    expect(abortEvents).toBe(1);
    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(143);
  });

  it('does not start remote exec when killed before async sandbox resolution finishes', async () => {
    let execCalls = 0;
    let resolveSandbox!: (runtime: t.CloudflareSandboxRuntime) => void;
    const sandboxPromise = new Promise<t.CloudflareSandboxRuntime>(
      (resolve) => {
        resolveSandbox = resolve;
      }
    );
    const config = createCloudflareLocalExecutionConfig({
      sandbox: () => sandboxPromise,
      timeoutMs: 10,
      workspaceRoot: '/workspace',
    });

    const result = await spawnLocalProcess('bash', ['-lc', 'sleep 10'], config);
    resolveSandbox(
      createRuntime({
        exec: async () => {
          execCalls += 1;
          return { exitCode: 0, stdout: '', stderr: '' };
        },
      })
    );
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(143);
    expect(execCalls).toBe(0);
  });

  it('memoizes sandbox factory results per config object', async () => {
    let calls = 0;
    const runtime = createRuntime({
      readFile: async () => 'ok',
      writeFile: async () => ({ ok: true }),
    });
    const config = {
      workspaceRoot: '/workspace',
      sandbox: async (): Promise<t.CloudflareSandboxRuntime> => {
        calls += 1;
        return runtime;
      },
    };
    const fs = createCloudflareWorkspaceFS(config);

    await fs.readFile('a.txt', 'utf8');
    await fs.writeFile('b.txt', 'ok', 'utf8');

    expect(calls).toBe(1);
  });

  it('wraps direct bash commands with an in-sandbox timeout', async () => {
    let execCommand = '';
    let execTimeout: number | undefined;
    let calls = 0;
    const sandbox = createRuntime({
      exec: async (command, options) => {
        calls += 1;
        execCommand = command;
        execTimeout = options?.timeout;
        return {
          exitCode: calls === 1 ? 0 : 124,
          stdout: 'ok',
          stderr: '',
        };
      },
    });

    const result = await executeCloudflareBash('echo ok', {
      sandbox,
      workspaceRoot: '/workspace',
      timeoutMs: 1500,
    });

    expect(execCommand).toContain('timeout -k 2s 2s bash -lc');
    expect(execTimeout).toBe(6500);
    expect(result.timedOut).toBe(true);
  });

  it('passes call-specific timeouts to the Cloudflare spawn wrapper', async () => {
    let execCommand = '';
    let execTimeout: number | undefined;
    const sandbox = createRuntime({
      exec: async (command, options) => {
        execCommand = command;
        execTimeout = options?.timeout;
        return {
          exitCode: 0,
          stdout: 'ok',
          stderr: '',
        };
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      workspaceRoot: '/workspace',
      timeoutMs: 1000,
    });

    await expect(
      spawnLocalProcess('bash', ['-lc', 'echo ok'], {
        ...config,
        timeoutMs: 120000,
      })
    ).resolves.toMatchObject({ exitCode: 0, timedOut: false });

    expect(execCommand).toContain('timeout -k 2s 120s bash -lc');
    expect(execTimeout).toBe(125000);
  });

  it('marks Cloudflare code execution timeouts', async () => {
    const sandbox = createRuntime({
      exec: async (command) => ({
        exitCode: command.startsWith('rm -rf') ? 0 : 124,
        stdout: '',
        stderr: '',
      }),
    });

    const result = await executeCloudflareCode(
      { lang: 'py', code: 'print("slow")' },
      {
        sandbox,
        workspaceRoot: '/workspace',
        timeoutMs: 1000,
      }
    );

    expect(result.timedOut).toBe(true);
  });

  it('forwards only explicit Cloudflare env vars to sandbox exec', async () => {
    let execEnv: Record<string, string | undefined> | undefined;
    const sandbox = createRuntime({
      exec: async (_command, options) => {
        execEnv = options?.env;
        return {
          exitCode: 0,
          stdout: 'ok',
          stderr: '',
        };
      },
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      workspaceRoot: '/workspace',
      env: { SAFE_FOR_SANDBOX: 'yes' },
    });

    await spawnLocalProcess('bash', ['-lc', 'echo ok'], config);

    expect(execEnv).toEqual({ SAFE_FOR_SANDBOX: 'yes' });
  });

  it('injects read-only and workspace guards into Python programmatic tools', async () => {
    let source = '';
    const sandbox = createRuntime({
      writeFile: async (_path, content) => {
        source = String(content);
        return { ok: true };
      },
      exec: async () => ({
        exitCode: 0,
        stdout: 'done',
        stderr: '',
      }),
    });
    const programmatic = createCloudflareProgrammaticToolCallingTool({
      sandbox,
      workspaceRoot: '/workspace',
      readOnly: true,
      shell: '/bin/sh',
    });

    await programmatic.invoke({
      code: 'await write_file("x.txt", "blocked")',
      lang: 'py',
    });

    expect(source).toContain('READ_ONLY = True');
    expect(source).toContain('SHELL = "/bin/sh"');
    expect(source).toContain('[SHELL, "-lc", command, "--"]');
    expect(source).toContain('_assert_writable("write_file")');
    expect(source).toContain('if _is_within_workspace(resolved):');
    expect(source).toContain('_validate_bash_command(command, args=args)');
  });

  it('clamps programmatic timeouts before sandbox execution', async () => {
    const timeouts: Array<number | undefined> = [];
    const sandbox = createRuntime({
      writeFile: async () => ({ ok: true }),
      exec: async (_command, options) => {
        timeouts.push(options?.timeout);
        return {
          exitCode: 0,
          stdout: 'done',
          stderr: '',
        };
      },
    });
    const programmatic = createCloudflareProgrammaticToolCallingTool({
      sandbox,
      workspaceRoot: '/workspace',
    });

    await programmatic.invoke({
      code: 'print("ok")',
      lang: 'py',
      timeout: 300000,
    });

    expect(timeouts[0]).toBe(305000);
  });

  it('injects bash validation into bash programmatic tools', async () => {
    let execCommand = '';
    const sandbox = createRuntime({
      exec: async (command) => {
        execCommand = command;
        return {
          exitCode: 0,
          stdout: 'done',
          stderr: '',
        };
      },
    });
    const programmatic = createCloudflareBashProgrammaticToolCallingTool({
      sandbox,
      workspaceRoot: '/workspace',
      shell: '/bin/sh',
    });

    await programmatic.invoke({
      code: 'printf "%s\\n" "ok"',
    });

    expect(execCommand).toContain('const ALLOW_DANGEROUS_COMMANDS = false;');
    expect(execCommand).toContain('const SHELL = "/bin/sh";');
    expect(execCommand).toContain('cp.spawn(SHELL, ["-lc", command, "--"');
    expect(execCommand).toContain(
      'function validateBashCommand(command, args)'
    );
    expect(execCommand).toContain('validateBashCommand(command, args);');
  });

  it('uses root-safe path containment in programmatic helpers', async () => {
    let pythonSource = '';
    const pythonSandbox = createRuntime({
      writeFile: async (_path, content) => {
        pythonSource = String(content);
        return { ok: true };
      },
      exec: async () => ({
        exitCode: 0,
        stdout: 'done',
        stderr: '',
      }),
    });
    const pythonProgrammatic = createCloudflareProgrammaticToolCallingTool({
      sandbox: pythonSandbox,
      workspaceRoot: '/',
    });

    await pythonProgrammatic.invoke({
      code: 'print("ok")',
      lang: 'py',
    });

    expect(pythonSource).toContain('WORKSPACE = "/"');
    expect(pythonSource).toContain(
      'return os.path.commonpath([root, resolved]) == root'
    );

    let bashCommand = '';
    const bashSandbox = createRuntime({
      exec: async (command) => {
        bashCommand = command;
        return {
          exitCode: 0,
          stdout: 'done',
          stderr: '',
        };
      },
    });
    const bashProgrammatic = createCloudflareBashProgrammaticToolCallingTool({
      sandbox: bashSandbox,
      workspaceRoot: '/',
    });

    await bashProgrammatic.invoke({
      code: 'printf "%s\\n" "ok"',
    });

    expect(bashCommand).toContain('const WORKSPACE = "/";');
    expect(bashCommand).toContain(
      'const relative = path.relative(root, resolved);'
    );
    expect(bashCommand).toContain(
      'relative.startsWith("..") || path.isAbsolute(relative)'
    );
  });

  it('enforces Cloudflare codingToolNames as an allowlist', () => {
    const tools = resolveLocalToolsForBinding({
      tools: [
        { name: Constants.EXECUTE_CODE } as t.GenericTool,
        { name: Constants.BASH_TOOL } as t.GenericTool,
      ],
      toolExecution: {
        engine: 'cloudflare-sandbox',
        cloudflare: {
          sandbox: createRuntime(),
          codingToolNames: [Constants.BASH_TOOL],
        },
      },
    }) as t.GenericTool[];
    const names = tools.map((toolDef) => toolDef.name);

    expect(names).toContain(Constants.BASH_TOOL);
    expect(names).not.toContain(Constants.EXECUTE_CODE);
  });
});

describe('Cloudflare bridge runtime', () => {
  it('preserves caller-provided sandbox ids', async () => {
    const urls: string[] = [];
    const fetchImpl: typeof fetch = async (input) => {
      urls.push(input.toString());
      if (input.toString().endsWith('/exec')) {
        return sseExit();
      }
      throw new Error(`Unexpected URL: ${input.toString()}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'user-123',
      fetch: fetchImpl,
    });

    await expect(runtime.getSandboxId()).resolves.toBe('user-123');
    await runtime.exec('true');

    expect(urls).toEqual(['https://bridge.example/v1/sandbox/user-123/exec']);
  });

  it('retries sandbox creation after a transient create failure', async () => {
    let createCalls = 0;
    const fetchImpl: typeof fetch = async (input) => {
      const url = input.toString();
      if (url.endsWith('/sandbox')) {
        createCalls += 1;
        if (createCalls === 1) {
          return new Response('try again', { status: 503 });
        }
        return Response.json({ id: 'retryid' });
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      fetch: fetchImpl,
    });

    await expect(runtime.getSandboxId()).rejects.toThrow('503');
    await expect(runtime.getSandboxId()).resolves.toBe('retryid');
    expect(createCalls).toBe(2);
  });

  it('fails exec when the SSE stream ends before an exit event', async () => {
    const fetchImpl: typeof fetch = async (input) => {
      const url = input.toString();
      if (url.endsWith('/exec')) {
        const stdout = Buffer.from('partial').toString('base64');
        return sseResponse(`event: stdout\ndata: ${stdout}\n\n`);
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'abc',
      fetch: fetchImpl,
    });

    await expect(runtime.exec('echo partial')).rejects.toThrow(
      'closed before an exit event'
    );
  });

  it('prunes hidden directory trees when includeHidden is disabled', async () => {
    let command = '';
    const fetchImpl: typeof fetch = async (input, init) => {
      const url = input.toString();
      if (url.endsWith('/exec')) {
        const body = JSON.parse(bodyText(init?.body)) as { argv?: string[] };
        command = body.argv?.[2] ?? '';
        return sseExit();
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'abc',
      fetch: fetchImpl,
    });

    await runtime.listFiles('/workspace', {
      recursive: true,
      includeHidden: false,
    });

    expect(command).toContain('-prune');
    expect(command).toContain('\\( -name \'.*\' -prune \\) -o');
  });

  it('fails bridge listFiles for non-directory targets', async () => {
    const fetchImpl: typeof fetch = async (input) => {
      const url = input.toString();
      if (url.endsWith('/exec')) {
        const stderr = Buffer.from('not a directory').toString('base64');
        return sseResponse(
          `event: stderr\ndata: ${stderr}\n\nevent: exit\ndata: {"exit_code":20}\n\n`
        );
      }
      throw new Error(`Unexpected URL: ${url}`);
    };
    const runtime = createCloudflareBridgeRuntime({
      baseURL: 'https://bridge.example',
      sandboxId: 'abc',
      fetch: fetchImpl,
    });

    await expect(runtime.listFiles('/workspace/file.txt')).rejects.toThrow(
      'not a directory'
    );
  });
});
