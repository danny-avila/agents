import type * as t from '@/types';
import { spawnLocalProcess } from '../local/LocalExecutionEngine';
import {
  createCloudflareWorkspaceFS,
  createCloudflareLocalExecutionConfig,
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

  it('aborts remote exec when the local timeout kills the spawn wrapper', async () => {
    let signal: AbortSignal | undefined;
    const sandbox = createRuntime({
      exec: (_command, options) =>
        new Promise<t.CloudflareSandboxExecResult>((_resolve, reject) => {
          signal = options?.signal;
          signal?.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    });
    const config = createCloudflareLocalExecutionConfig({
      sandbox,
      timeoutMs: 10,
      workspaceRoot: '/workspace',
    });

    const result = await spawnLocalProcess('bash', ['-lc', 'sleep 10'], config);

    expect(signal?.aborted).toBe(true);
    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(143);
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
    });

    await programmatic.invoke({
      code: 'await write_file("x.txt", "blocked")',
      lang: 'py',
    });

    expect(source).toContain('READ_ONLY = True');
    expect(source).toContain('_assert_writable("write_file")');
    expect(source).toContain('if _is_within_workspace(resolved):');
    expect(source).toContain('_validate_bash_command(command, args=args)');
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
    });

    await programmatic.invoke({
      code: 'printf "%s\\n" "ok"',
    });

    expect(execCommand).toContain('const ALLOW_DANGEROUS_COMMANDS = false;');
    expect(execCommand).toContain(
      'function validateBashCommand(command, args)'
    );
    expect(execCommand).toContain('validateBashCommand(command, args);');
  });
});

describe('Cloudflare bridge runtime', () => {
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
});
