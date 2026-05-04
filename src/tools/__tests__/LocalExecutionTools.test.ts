import { z } from 'zod';
import { tmpdir } from 'os';
import { join } from 'path';
import { spawnSync } from 'child_process';
import {
  mkdtemp,
  rm,
  symlink,
  writeFile as fsWriteFile,
  readFile as fsReadFile,
} from 'fs/promises';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, afterEach, beforeEach, jest } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { Constants } from '@/common';
import { ToolNode } from '../ToolNode';
import {
  validateBashCommand,
  _resetLocalEngineWarningsForTests,
} from '../local/LocalExecutionEngine';
import { resolveLocalToolsForBinding } from '../local/resolveLocalExecutionTools';
import {
  createLocalCodingToolBundle,
  _resetRipgrepCacheForTests,
} from '../local/LocalCodingTools';
import { runBashAstChecks } from '../local/bashAst';
import { LocalFileCheckpointerImpl } from '../local/FileCheckpointer';

const hasPython3 = spawnSync('python3', ['--version']).status === 0;

const tempDirs: string[] = [];

async function createTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'lc-local-tools-'));
  tempDirs.push(dir);
  return dir;
}

function createRemoteBashStub(): StructuredToolInterface {
  return tool(
    async () => 'remote bash should not run',
    {
      name: Constants.BASH_TOOL,
      description: 'Remote bash stub',
      schema: z.object({ command: z.string() }),
    }
  ) as unknown as StructuredToolInterface;
}

function messagesFromResult(
  result: ToolMessage[] | { messages: ToolMessage[] }
): ToolMessage[] {
  return Array.isArray(result) ? result : result.messages;
}

function aiMessageWithToolCall(
  name: string,
  args: Record<string, string | number | boolean>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [
      {
        id: `call_${name}`,
        name,
        args,
      },
    ],
  });
}

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true }))
  );
});

describe('local execution tools', () => {
  it('blocks clearly destructive bash commands by default', async () => {
    const result = await validateBashCommand('rm -rf /');

    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('destructive command pattern');
  });

  it('replaces a configured remote bash tool when local mode is enabled', async () => {
    const cwd = await createTempDir();
    const node = new ToolNode({
      tools: [createRemoteBashStub()],
      toolExecution: {
        engine: 'local',
        local: {
          cwd,
          includeCodingTools: false,
        },
      },
    });

    const result = await node.invoke({
      messages: [
        aiMessageWithToolCall(Constants.BASH_TOOL, {
          command: 'printf local-mode',
        }),
      ],
    });

    const [message] = messagesFromResult(result as { messages: ToolMessage[] });
    expect(String(message.content)).toContain('local-mode');
    expect(String(message.content)).not.toContain('remote bash should not run');
  });

  it('auto-binds the local coding suite in local mode', () => {
    const tools = resolveLocalToolsForBinding({
      toolExecution: { engine: 'local' },
    }) as t.GenericTool[];
    const names = tools.map((localTool) => localTool.name);

    expect(names).toEqual(
      expect.arrayContaining([
        Constants.EXECUTE_CODE,
        Constants.BASH_TOOL,
        Constants.READ_FILE,
        'write_file',
        'edit_file',
        'grep_search',
        'glob_search',
        'list_directory',
      ])
    );
  });

  it('updates existing code tool bindings when auto-binding is disabled', () => {
    const [bashTool] = resolveLocalToolsForBinding({
      tools: [createRemoteBashStub()],
      toolExecution: {
        engine: 'local',
        local: { includeCodingTools: false },
      },
    }) as t.GenericTool[];

    expect(bashTool.name).toBe(Constants.BASH_TOOL);
    expect(bashTool.description).toContain('local machine');
  });

  it('can call local coding tools from local programmatic execution', async () => {
    if (!hasPython3) {
      return;
    }

    const cwd = await createTempDir();
    const node = new ToolNode({
      tools: [],
      toolExecution: {
        engine: 'local',
        local: { cwd },
      },
    });

    const result = await node.invoke({
      messages: [
        aiMessageWithToolCall(Constants.PROGRAMMATIC_TOOL_CALLING, {
          lang: 'py',
          code: [
            'await write_file(file_path="ptc.txt", content="from local ptc")',
            'contents = await read_file(file_path="ptc.txt")',
            'print(contents)',
          ].join('\n'),
        }),
      ],
    });

    const [message] = messagesFromResult(result as { messages: ToolMessage[] });
    expect(String(message.content)).toContain('from local ptc');
  });

  it('can run bash orchestration through run_tools_with_code in local mode', async () => {
    if (!hasPython3) {
      return;
    }

    const cwd = await createTempDir();
    const node = new ToolNode({
      tools: [],
      toolExecution: {
        engine: 'local',
        local: { cwd },
      },
    });

    const result = await node.invoke({
      messages: [
        aiMessageWithToolCall(Constants.PROGRAMMATIC_TOOL_CALLING, {
          code: [
            'write_file \'{"file_path":"bash-ptc.txt","content":"from bash ptc"}\'',
            'read_file \'{"file_path":"bash-ptc.txt"}\'',
          ].join('\n'),
        }),
      ],
    });

    const [message] = messagesFromResult(result as { messages: ToolMessage[] });
    expect(String(message.content)).toContain('from bash ptc');
  });
});

describe('local engine bashAst', () => {
  it('flags command substitution in auto mode', () => {
    const findings = runBashAstChecks('echo $(whoami)', 'auto');
    expect(findings.some((f) => f.code === 'cmd-subst-dollar-paren')).toBe(true);
  });

  it('escalates command substitution to deny in strict mode', () => {
    const findings = runBashAstChecks('echo $(whoami)', 'strict');
    const subst = findings.find((f) => f.code === 'cmd-subst-dollar-paren');
    expect(subst?.severity).toBe('deny');
  });

  it('always denies /proc/<pid>/environ access', () => {
    const findings = runBashAstChecks('cat /proc/1/environ', 'auto');
    expect(findings.some((f) => f.code === 'proc-environ-read' && f.severity === 'deny')).toBe(true);
  });

  it('never produces findings when off', () => {
    const findings = runBashAstChecks('echo $(whoami)', 'off');
    expect(findings).toHaveLength(0);
  });

  it('blocks bash commands with a deny finding via validateBashCommand', async () => {
    const result = await validateBashCommand('cat /proc/1/environ', {
      bashAst: 'auto',
    });
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('proc-environ-read');
  });
});

describe('local engine sandbox-off warning', () => {
  let warnSpy: jest.SpiedFunction<typeof console.warn>;

  beforeEach(() => {
    _resetLocalEngineWarningsForTests();
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('warns once when running without sandbox', async () => {
    await validateBashCommand('echo hi');
    await validateBashCommand('echo bye');
    const sandboxOffMessages = warnSpy.mock.calls.filter((call) =>
      String(call[0]).includes('without @anthropic-ai/sandbox-runtime')
    );
    expect(sandboxOffMessages).toHaveLength(1);
  });
});

describe('LocalFileCheckpointer', () => {
  it('snapshots and restores existing files', async () => {
    const dir = await createTempDir();
    const file = join(dir, 'a.txt');
    await fsWriteFile(file, 'original', 'utf8');

    const cp = new LocalFileCheckpointerImpl();
    await cp.captureBeforeWrite(file);

    await fsWriteFile(file, 'modified', 'utf8');
    expect(await fsReadFile(file, 'utf8')).toBe('modified');

    const restored = await cp.rewind();
    expect(restored).toBe(1);
    expect(await fsReadFile(file, 'utf8')).toBe('original');
  });

  it('deletes files that did not exist before the run', async () => {
    const dir = await createTempDir();
    const file = join(dir, 'new.txt');

    const cp = new LocalFileCheckpointerImpl();
    await cp.captureBeforeWrite(file);
    await fsWriteFile(file, 'should-be-removed', 'utf8');

    await cp.rewind();
    await expect(fsReadFile(file, 'utf8')).rejects.toThrow();
  });

  it('rewinds tools created via createLocalCodingToolBundle', async () => {
    const cwd = await createTempDir();
    const bundle = createLocalCodingToolBundle({
      cwd,
      fileCheckpointing: true,
    });
    expect(bundle.checkpointer).toBeDefined();

    const writeTool = bundle.tools.find((tool_) => tool_.name === 'write_file');
    expect(writeTool).toBeDefined();
    await writeTool!.invoke({ file_path: 'cp.txt', content: 'first' });
    await writeTool!.invoke({ file_path: 'cp.txt', content: 'second' });

    const restored = await bundle.checkpointer!.rewind();
    expect(restored).toBe(1);
    await expect(fsReadFile(join(cwd, 'cp.txt'), 'utf8')).rejects.toThrow();
  });
});

describe('local read tool guards', () => {
  it('refuses to read files containing NUL bytes', async () => {
    const cwd = await createTempDir();
    const binary = join(cwd, 'binary.bin');
    await fsWriteFile(binary, Buffer.from([0x00, 0x01, 0x02]));

    const bundle = createLocalCodingToolBundle({ cwd });
    const readTool = bundle.tools.find((t_) => t_.name === Constants.READ_FILE);
    const result = await readTool!.invoke({ file_path: 'binary.bin' });
    expect(String(result)).toContain('binary file');
  });

  it('returns a stub instead of OOMing on huge files', async () => {
    const cwd = await createTempDir();
    const big = join(cwd, 'big.txt');
    await fsWriteFile(big, 'x'.repeat(2048));

    const bundle = createLocalCodingToolBundle({
      cwd,
      maxReadBytes: 1024,
    });
    const readTool = bundle.tools.find((t_) => t_.name === Constants.READ_FILE);
    const result = await readTool!.invoke({ file_path: 'big.txt' });
    expect(String(result)).toContain('exceeds the 1024-byte read cap');
  });

  it('rejects symlink escapes', async () => {
    const cwd = await createTempDir();
    const outside = await createTempDir();
    const secret = join(outside, 'secret.txt');
    await fsWriteFile(secret, 'top-secret', 'utf8');
    await symlink(outside, join(cwd, 'escape'));

    const bundle = createLocalCodingToolBundle({ cwd });
    const readTool = bundle.tools.find((t_) => t_.name === Constants.READ_FILE);
    await expect(
      readTool!.invoke({ file_path: 'escape/secret.txt' })
    ).rejects.toThrow(/symlink escape/);
  });
});

describe('local programmatic bridge auth', () => {
  it('rejects unauthenticated requests to the local bridge', async () => {
    if (!hasPython3) {
      return;
    }
    const cwd = await createTempDir();
    const node = new ToolNode({
      tools: [],
      toolExecution: {
        engine: 'local',
        local: { cwd },
      },
    });

    const result = await node.invoke({
      messages: [
        aiMessageWithToolCall(Constants.PROGRAMMATIC_TOOL_CALLING, {
          lang: 'py',
          code: [
            'import os, json, urllib.request, urllib.error',
            'url = os.environ["BRIDGE_PROBE_URL"] if "BRIDGE_PROBE_URL" in os.environ else __LIBRECHAT_TOOL_BRIDGE',
            'body = json.dumps({"name":"read_file","input":{"file_path":"x"}}).encode("utf-8")',
            'try:',
            '  req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"}, method="POST")',
            '  urllib.request.urlopen(req, timeout=5)',
            '  print("LEAK")',
            'except urllib.error.HTTPError as e:',
            '  print(f"AUTH={e.code}")',
          ].join('\n'),
        }),
      ],
    });

    const [message] = messagesFromResult(result as { messages: ToolMessage[] });
    expect(String(message.content)).toContain('AUTH=401');
    expect(String(message.content)).not.toContain('LEAK');
  });
});

describe('local edit fuzzy matching', () => {
  it('falls back to line-trimmed when trailing whitespace differs', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'a.ts');
    // Real file has trailing whitespace on every line.
    await fsWriteFile(
      file,
      'function greet(name: string) {  \n  return `Hello, ${name}!`;  \n}\n',
      'utf8'
    );

    const bundle = createLocalCodingToolBundle({ cwd });
    const editTool = bundle.tools.find((tt) => tt.name === 'edit_file');
    const result = await editTool!.invoke({
      file_path: 'a.ts',
      // LLM emits a trailing-whitespace-stripped version.
      old_text:
        'function greet(name: string) {\n  return `Hello, ${name}!`;\n}',
      new_text:
        'function greet(name: string) {\n  return `Hi, ${name}!`;\n}',
    });
    expect(String(result)).toContain('strategies: line-trimmed');
    const after = await fsReadFile(file, 'utf8');
    expect(after).toContain('Hi, ${name}!');
  });

  it('falls back to indentation-flexible when LLM strips leading indent', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'a.ts');
    await fsWriteFile(
      file,
      'class Foo {\n    method() {\n        return 1;\n    }\n}\n',
      'utf8'
    );

    const bundle = createLocalCodingToolBundle({ cwd });
    const editTool = bundle.tools.find((tt) => tt.name === 'edit_file');
    const result = await editTool!.invoke({
      file_path: 'a.ts',
      // LLM stripped the 4-space indent
      old_text: 'method() {\n    return 1;\n}',
      new_text: 'method() {\n    return 42;\n}',
    });
    expect(String(result)).toMatch(
      /strategies: (indentation-flexible|whitespace-normalized)/
    );
    const after = await fsReadFile(file, 'utf8');
    expect(after).toContain('return 42;');
  });

  it('returns a unified diff in the tool result', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'a.txt');
    await fsWriteFile(file, 'first\nsecond\nthird\n', 'utf8');
    const bundle = createLocalCodingToolBundle({ cwd });
    const editTool = bundle.tools.find((tt) => tt.name === 'edit_file');
    const result = await editTool!.invoke({
      file_path: 'a.txt',
      old_text: 'second',
      new_text: 'SECOND',
    });
    const text = String(result);
    expect(text).toContain('Diff:');
    expect(text).toContain('-second');
    expect(text).toContain('+SECOND');
  });

  it('preserves CRLF line endings on edit', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'a.txt');
    await fsWriteFile(file, 'one\r\ntwo\r\nthree\r\n', 'utf8');
    const bundle = createLocalCodingToolBundle({ cwd });
    const editTool = bundle.tools.find((tt) => tt.name === 'edit_file');
    await editTool!.invoke({
      file_path: 'a.txt',
      old_text: 'two',
      new_text: 'TWO',
    });
    const raw = await fsReadFile(file, 'utf8');
    expect(raw).toBe('one\r\nTWO\r\nthree\r\n');
  });

  it('preserves UTF-8 BOM on overwrite', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'a.txt');
    const BOM = '﻿';
    await fsWriteFile(file, BOM + 'hello\n', 'utf8');
    const bundle = createLocalCodingToolBundle({ cwd });
    const writeTool = bundle.tools.find((tt) => tt.name === 'write_file');
    await writeTool!.invoke({ file_path: 'a.txt', content: 'goodbye\n' });
    const raw = await fsReadFile(file, 'utf8');
    expect(raw.startsWith(BOM)).toBe(true);
    expect(raw.slice(1)).toBe('goodbye\n');
  });
});

describe('local search fallback', () => {
  beforeEach(() => {
    _resetRipgrepCacheForTests();
  });

  it('finds matches via the Node fallback when ripgrep is missing', async () => {
    const cwd = await createTempDir();
    await fsWriteFile(join(cwd, 'a.ts'), 'const needle = 42;\n', 'utf8');
    await fsWriteFile(join(cwd, 'b.ts'), 'const haystack = 1;\n', 'utf8');

    const bundle = createLocalCodingToolBundle({
      cwd,
      env: { PATH: '/nonexistent' },
    });
    const grepTool = bundle.tools.find((t_) => t_.name === 'grep_search');
    const result = await grepTool!.invoke({ pattern: 'needle' });
    expect(String(result)).toContain('a.ts');
    expect(String(result)).toContain('needle');
  });
});
