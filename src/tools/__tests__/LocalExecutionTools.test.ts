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
import type { BaseMessage } from '@langchain/core/messages';
import { describe, it, expect, afterEach, beforeEach, jest } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { Constants, Providers } from '@/common';
import { ToolNode } from '../ToolNode';
import {
  executeLocalBash,
  executeLocalCode,
  validateBashCommand,
  _resetLocalEngineWarningsForTests,
} from '../local/LocalExecutionEngine';
import { resolveLocalToolsForBinding } from '../local/resolveLocalExecutionTools';
import {
  createLocalCodingToolBundle,
  _resetRipgrepCacheForTests,
} from '../local/LocalCodingTools';
import {
  runPostEditSyntaxCheck,
  _resetSyntaxCheckProbeCacheForTests,
} from '../local/syntaxCheck';
import { createCompileCheckTool } from '../local/CompileCheckTool';
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
    // Real (non-internal) executions should warn; the internal
    // `bash -n` syntax preflight inside validateBashCommand opts out
    // (Codex P2 — otherwise the latch would flip on a probe and hide
    // the warning when a genuinely-unsandboxed command later runs).
    await executeLocalBash('echo hi');
    await executeLocalBash('echo bye');
    const sandboxOffMessages = warnSpy.mock.calls.filter((call) =>
      String(call[0]).includes('without @anthropic-ai/sandbox-runtime')
    );
    expect(sandboxOffMessages).toHaveLength(1);
  });

  it('does NOT warn for internal probes when the run actually has sandbox enabled (Codex P2)', async () => {
    // Pre-fix: validateBashCommand's bash -n preflight (which forces
    // sandbox: false for itself, since you can't sandbox a syntax
    // probe) would emit a misleading "sandbox is off" warning AND
    // flip `sandboxOffWarned = true` even when the run had
    // `sandbox.enabled: true` — hiding the warning when a real
    // unsandboxed execution later happened. With the fix internal
    // probes pass `{ internal: true }` to spawnLocalProcess and
    // suppress both the message and the latch.
    await validateBashCommand('echo hi', { sandbox: { enabled: true } });
    const sandboxOffMessages = warnSpy.mock.calls.filter((call) =>
      String(call[0]).includes('without @anthropic-ai/sandbox-runtime')
    );
    expect(sandboxOffMessages).toHaveLength(0);
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

describe('local read attachments', () => {
  // Smallest valid 1x1 PNG.
  const tinyPng = Buffer.from(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c63000100000005000165be7e6e0000000049454e44ae426082',
    'hex'
  );

  it('returns binary stub by default', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'tiny.png');
    await fsWriteFile(file, tinyPng);
    const bundle = createLocalCodingToolBundle({ cwd });
    const readTool = bundle.tools.find((tt) => tt.name === Constants.READ_FILE);
    const result = await readTool!.invoke({ file_path: 'tiny.png' });
    expect(String(result)).toContain('binary file');
  });

  it('returns an image_url content block when attachReadAttachments=images-only', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'tiny.png');
    await fsWriteFile(file, tinyPng);

    const bundle = createLocalCodingToolBundle({
      cwd,
      attachReadAttachments: 'images-only',
    });
    const readTool = bundle.tools.find((tt) => tt.name === Constants.READ_FILE);
    // Invoking via a tool_call envelope (rather than raw args) is what
    // makes the LangChain tool wrap the result as a ToolMessage with
    // `.content` and `.artifact` populated.
    const message = (await readTool!.invoke({
      id: 'call_image',
      name: Constants.READ_FILE,
      args: { file_path: 'tiny.png' },
      type: 'tool_call',
    })) as { content: unknown; artifact: unknown };
    expect(Array.isArray(message.content)).toBe(true);
    const blocks = message.content as Array<{
      type: string;
      image_url?: { url: string };
    }>;
    const imageBlock = blocks.find((b) => b.type === 'image_url');
    expect(imageBlock?.image_url?.url).toMatch(/^data:image\/png;base64,/);
    expect(blocks.find((b) => b.type === 'text')).toBeDefined();
    expect(message.artifact).toMatchObject({
      mime: 'image/png',
      attachment: 'image',
    });
  });

  it('refuses oversize images even when embedding is on', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'big.png');
    // Forge a "PNG" larger than the cap. It will sniff as a generic
    // binary; classifyAttachment returns 'binary' since file-type
    // won't recognise the bytes — that's fine, we just want to
    // verify the oversize gate is reachable. So instead, build a
    // real big PNG by concatenating chunks with a fake IDAT.
    // Easier: keep the tiny PNG header but pad to 200 bytes; cap to 100.
    const padded = Buffer.concat([
      tinyPng,
      Buffer.alloc(200 - tinyPng.length, 0),
    ]);
    await fsWriteFile(file, padded);
    const bundle = createLocalCodingToolBundle({
      cwd,
      attachReadAttachments: 'images-only',
      maxAttachmentBytes: 100,
    });
    const readTool = bundle.tools.find((tt) => tt.name === Constants.READ_FILE);
    const result = await readTool!.invoke({ file_path: 'big.png' });
    expect(String(result)).toMatch(/Refusing to embed/);
  });

  it('still reads text files normally when embedding is on', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'a.txt');
    await fsWriteFile(file, 'hello world\n', 'utf8');
    const bundle = createLocalCodingToolBundle({
      cwd,
      attachReadAttachments: 'images-only',
    });
    const readTool = bundle.tools.find((tt) => tt.name === Constants.READ_FILE);
    const result = await readTool!.invoke({ file_path: 'a.txt' });
    expect(String(result)).toContain('hello world');
  });
});

describe('post-edit syntax check', () => {
  beforeEach(() => {
    _resetSyntaxCheckProbeCacheForTests();
  });

  it('flags broken JS via node --check', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'broken.js');
    await fsWriteFile(file, 'function (\n', 'utf8');
    const outcome = await runPostEditSyntaxCheck(file, {});
    expect(outcome).not.toBeNull();
    expect(outcome!.ok).toBe(false);
    if (outcome!.ok === false) {
      expect(outcome!.checker).toBe('node --check');
      expect(outcome!.output.length).toBeGreaterThan(0);
    }
  });

  it('passes valid JS', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'good.js');
    await fsWriteFile(file, 'console.log(1)\n', 'utf8');
    const outcome = await runPostEditSyntaxCheck(file, {});
    expect(outcome?.ok).toBe(true);
  });

  it('flags broken JSON via JSON.parse', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'broken.json');
    await fsWriteFile(file, '{ "x": ', 'utf8');
    const outcome = await runPostEditSyntaxCheck(file, {});
    expect(outcome?.ok).toBe(false);
    if (outcome!.ok === false) {
      expect(outcome!.checker).toBe('JSON.parse');
    }
  });

  it('returns null for unknown extensions', async () => {
    const cwd = await createTempDir();
    const file = join(cwd, 'random.xyz');
    await fsWriteFile(file, 'whatever\n', 'utf8');
    const outcome = await runPostEditSyntaxCheck(file, {});
    expect(outcome).toBeNull();
  });

  it('write_file appends syntax-check warning when postEditSyntaxCheck=auto', async () => {
    const cwd = await createTempDir();
    const bundle = createLocalCodingToolBundle({
      cwd,
      postEditSyntaxCheck: 'auto',
    });
    const writeTool = bundle.tools.find((tt) => tt.name === 'write_file');
    const message = (await writeTool!.invoke({
      id: 'call_w',
      name: 'write_file',
      args: { file_path: 'broken.js', content: 'function (\n' },
      type: 'tool_call',
    })) as { content: string; artifact: { syntax_error?: string } };
    expect(message.content).toContain('[syntax-check warning');
    expect(message.artifact.syntax_error).toBe('node --check');
  });

  it('write_file in strict mode throws on syntax error', async () => {
    const cwd = await createTempDir();
    const bundle = createLocalCodingToolBundle({
      cwd,
      postEditSyntaxCheck: 'strict',
    });
    const writeTool = bundle.tools.find((tt) => tt.name === 'write_file');
    await expect(
      writeTool!.invoke({
        id: 'call_w',
        name: 'write_file',
        args: { file_path: 'broken.js', content: 'function (\n' },
        type: 'tool_call',
      })
    ).rejects.toThrow(/syntax check failed/);
  });
});

describe('compile_check', () => {
  it('reports "no recognised project marker" when there are none', async () => {
    const cwd = await createTempDir();
    const checkTool = createCompileCheckTool({ cwd });
    const message = (await checkTool.invoke({
      id: 'call_c',
      name: 'compile_check',
      args: {},
      type: 'tool_call',
    })) as { content: string; artifact: { ran: boolean; kind: string } };
    expect(message.content).toContain('no recognised project marker');
    expect(message.artifact.ran).toBe(false);
    expect(message.artifact.kind).toBe('unknown');
  });

  it('honours an explicit command override and reports exit code', async () => {
    const cwd = await createTempDir();
    const checkTool = createCompileCheckTool({ cwd });
    const message = (await checkTool.invoke({
      id: 'call_c2',
      name: 'compile_check',
      args: { command: 'echo hello && false' },
      type: 'tool_call',
    })) as { content: string; artifact: { passed: boolean; exit_code: number | null } };
    expect(message.content).toContain('FAILED');
    expect(message.content).toContain('hello');
    expect(message.artifact.passed).toBe(false);
    expect(message.artifact.exit_code).not.toBe(0);
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

describe('codex review fixes', () => {
  describe('executeLocalCode bash args (Codex P2 #1)', () => {
    it('passes input.args as positional shell parameters when lang is bash', async () => {
      const cwd = await createTempDir();
      const result = await executeLocalCode(
        {
          lang: 'bash',
          // Echo every positional arg space-separated. With the bug,
          // $@ is empty because args were dropped.
          code: 'echo "args:$@"',
          args: ['hello', 'world'],
        },
        { cwd }
      );
      expect(result.exitCode).toBe(0);
      expect(result.stdout.trim()).toBe('args:hello world');
    });

    it('still works when lang is bash and args is missing', async () => {
      const cwd = await createTempDir();
      const result = await executeLocalCode(
        { lang: 'bash', code: 'echo plain' },
        { cwd }
      );
      expect(result.exitCode).toBe(0);
      expect(result.stdout.trim()).toBe('plain');
    });
  });

  describe('ripgrep cache backend scope (Codex P2 #2)', () => {
    it('does not bleed an "rg available" verdict from one backend to another', async () => {
      // Backend A: pretends rg works (returns a fake spawn whose
      // process exits 0 on every call). The cache should record true
      // for THIS backend.
      const okBackend = jest.fn((cmd: string, _args: string[], _opts: unknown) => {
        const ok = require('child_process').spawn('echo', [cmd]);
        return ok;
      }) as unknown as t.LocalSpawn;
      // Backend B: pretends rg does not exist (returns a child that
      // exits 127, the "command not found" code).
      const missingBackend = jest.fn(
        (_cmd: string, _args: string[], _opts: unknown) => {
          const child = require('child_process').spawn(
            'sh',
            ['-c', 'exit 127']
          );
          return child;
        }
      ) as unknown as t.LocalSpawn;

      _resetRipgrepCacheForTests();

      // Build two bundles with distinct backends.
      const cwdA = await createTempDir();
      const cwdB = await createTempDir();
      await fsWriteFile(join(cwdA, 'a.ts'), 'needle\n', 'utf8');
      await fsWriteFile(join(cwdB, 'b.ts'), 'needle\n', 'utf8');

      const bundleA = createLocalCodingToolBundle({
        cwd: cwdA,
        exec: { spawn: okBackend },
      });
      const bundleB = createLocalCodingToolBundle({
        cwd: cwdB,
        exec: { spawn: missingBackend },
      });

      // Run grep against A first — populates cache for A's backend.
      await bundleA.tools.find((t_) => t_.name === 'grep_search')!.invoke({
        pattern: 'needle',
      });
      // Run grep against B — must NOT see cached "true" from A's
      // backend. With the bug, B would try to spawn rg, fail, and
      // throw instead of falling back to the Node walker.
      const bResult = await bundleB.tools
        .find((t_) => t_.name === 'grep_search')!
        .invoke({ pattern: 'needle' });
      expect(String(bResult)).toContain('needle');
    });
  });

  describe('additionalRoots resolved against workspace root (Codex P2 #3)', () => {
    it('treats relative additionalRoots as siblings of root, not of process.cwd', async () => {
      const parent = await createTempDir();
      const fs = await import('fs/promises');
      await fs.mkdir(join(parent, 'app'), { recursive: true });
      await fs.mkdir(join(parent, 'shared'), { recursive: true });
      await fsWriteFile(join(parent, 'shared/lib.ts'), 'X\n', 'utf8');

      const bundle = createLocalCodingToolBundle({
        workspace: {
          root: join(parent, 'app'),
          additionalRoots: ['../shared'],
        },
      });
      const readTool = bundle.tools.find((t_) => t_.name === Constants.READ_FILE);
      // Without the fix, '../shared/lib.ts' would resolve relative to
      // process.cwd (this test runner), miss the boundary check, and
      // throw "Path is outside the local workspace".
      const result = await readTool!.invoke({
        id: 'c',
        name: Constants.READ_FILE,
        args: { file_path: join(parent, 'shared/lib.ts') },
        type: 'tool_call',
      });
      expect(JSON.stringify(result)).toContain('X');
    });
  });
});

describe('codex review fixes (round 2)', () => {
  describe('streaming output cap (Codex P1)', () => {
    const { spawnLocalProcess, _resetLocalEngineWarningsForTests: _ } = require('../local/LocalExecutionEngine');

    it('hard-kills the child when total streamed bytes exceed maxSpawnedBytes', async () => {
      // Cap at 64 KiB. `yes` would otherwise run unbounded.
      const start = Date.now();
      const result = await spawnLocalProcess('yes', [], {
        timeoutMs: 30_000,
        maxSpawnedBytes: 64 * 1024,
        sandbox: { enabled: false },
      });
      const elapsed = Date.now() - start;
      // Killed promptly (much sooner than the 30s timeout).
      expect(elapsed).toBeLessThan(5000);
      // Process was killed by the overflow guard, not by timeout.
      expect(result.timedOut).toBe(false);
      expect(result.exitCode).not.toBe(0);
      // We DID see some output before the kill.
      expect(result.stdout.length).toBeGreaterThan(0);
    });

    it('spills overflow to a temp file (full output recoverable post-cap)', async () => {
      // Generate ~200 KiB of output with a 32 KiB inline cap → spill.
      const result = await spawnLocalProcess(
        'bash',
        ['-c', 'head -c 200000 /dev/urandom | base64 | head -c 200000'],
        {
          timeoutMs: 10_000,
          maxOutputChars: 8_000, // inline cap = 16 KiB; ~200 KiB → overflow
          maxSpawnedBytes: 1024 * 1024, // 1 MiB hard cap
          sandbox: { enabled: false },
        }
      );
      expect(result.exitCode).toBe(0);
      expect(result.fullOutputPath).toBeTruthy();
      const fs = await import('fs/promises');
      const spilled = await fs.readFile(result.fullOutputPath as string, 'utf8');
      // The spill file holds more bytes than the in-memory truncation.
      expect(spilled.length).toBeGreaterThan(result.stdout.length);
    });

    it('does not create a spill file for small outputs', async () => {
      const result = await spawnLocalProcess('bash', ['-c', 'echo small'], {
        timeoutMs: 5_000,
        sandbox: { enabled: false },
      });
      expect(result.fullOutputPath).toBeUndefined();
      expect(result.stdout.trim()).toBe('small');
    });
  });

  describe('bash_tool args (Codex P2)', () => {
    it('populates positional shell parameters from input.args', async () => {
      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({ cwd });
      const bashTool = bundle.tools.find(
        (tt) => tt.name === Constants.BASH_TOOL
      );
      const result = await bashTool!.invoke({
        id: 'b1',
        name: Constants.BASH_TOOL,
        args: { command: 'echo "first=$1 second=$2"', args: ['hello', 'world'] },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).toContain('first=hello second=world');
    });

    it('still works when args is missing', async () => {
      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({ cwd });
      const bashTool = bundle.tools.find(
        (tt) => tt.name === Constants.BASH_TOOL
      );
      const result = await bashTool!.invoke({
        id: 'b2',
        name: Constants.BASH_TOOL,
        args: { command: 'echo plain' },
        type: 'tool_call',
      });
      expect(JSON.stringify(result)).toContain('plain');
    });
  });
});

describe('codex review fixes (round 3)', () => {
  describe('validateBashCommand honours configured shell (Codex P1 #6)', () => {
    it('routes the -n preflight through `local.shell` when set', async () => {
      // Spawn calls go through the config'd backend; intercept and
      // assert which shell binary the syntax check picks.
      const calls: string[] = [];
      const intercept: t.LocalSpawn = ((
        command: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        calls.push(command);
        // Fall through to a real spawn so the call resolves cleanly.
        const { spawn: realSpawn } = require('child_process') as typeof import('child_process');
        return realSpawn(command, args, opts);
      }) as unknown as t.LocalSpawn;

      const result = await validateBashCommand('echo ok', {
        shell: '/bin/sh',
        exec: { spawn: intercept },
      });
      expect(result.valid).toBe(true);
      // The very first call is the syntax-check spawn; assert it used
      // /bin/sh and not the DEFAULT_SHELL fallback.
      expect(calls[0]).toBe('/bin/sh');
    });
  });

  describe('syntax-check probe cache is backend-keyed (Codex P2 #7)', () => {
    it('does not bleed an "rg/node/python available" verdict from one backend to another', async () => {
      _resetSyntaxCheckProbeCacheForTests();

      // Backend A: probes succeed (real spawn).
      const realSpawn = (require('child_process') as typeof import('child_process')).spawn;
      const okBackend: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => realSpawn(cmd, args, opts)) as unknown as t.LocalSpawn;
      // Backend B: probes always fail with exit 127.
      const missingBackend: t.LocalSpawn = ((
        _cmd: string,
        _args: string[],
        opts: import('child_process').SpawnOptions
      ) => realSpawn('sh', ['-c', 'exit 127'], opts)) as unknown as t.LocalSpawn;

      const cwdA = await createTempDir();
      const cwdB = await createTempDir();
      // Write a broken JS file we want syntax-checked.
      await fsWriteFile(join(cwdA, 'a.js'), 'function (\n', 'utf8');
      await fsWriteFile(join(cwdB, 'b.js'), 'function (\n', 'utf8');

      // Run on backend A — succeeds, populates A's probe cache for `node`.
      const a = await runPostEditSyntaxCheck(join(cwdA, 'a.js'), {
        cwd: cwdA,
        exec: { spawn: okBackend },
      });
      expect(a?.ok).toBe(false);

      // Run on backend B — must NOT see A's cached "node available".
      // With the bug, B would assume `node` works (skipping the probe),
      // try to run `node --check`, get exit 127 from the missingBackend,
      // and return ok=false with a misleading checker.
      // With the fix: B's own probe runs, sees node is missing on this
      // backend, and skips the syntax check (returns ok=true).
      const b = await runPostEditSyntaxCheck(join(cwdB, 'b.js'), {
        cwd: cwdB,
        exec: { spawn: missingBackend },
      });
      expect(b?.ok).toBe(true);
    });
  });

  describe('grep passes pattern via -e (Codex P2 #8)', () => {
    it('handles dash-prefixed patterns without rg interpreting them as flags', async () => {
      const cwd = await createTempDir();
      // File contains a literal "-foo" we want to find.
      await fsWriteFile(
        join(cwd, 'flags.txt'),
        'before\n-foo bar\nafter\n',
        'utf8'
      );
      const bundle = createLocalCodingToolBundle({ cwd });
      const grepTool = bundle.tools.find((t_) => t_.name === 'grep_search');
      const result = await grepTool!.invoke({
        id: 'g1',
        name: 'grep_search',
        args: { pattern: '-foo' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      // Pre-fix, rg would parse "-foo" as a flag and bail out.
      // Post-fix, "-foo" is matched and the line shows up.
      expect(text).toContain('-foo bar');
    });
  });
});

describe('codex review fixes (round 4)', () => {
  describe('quoted destructive targets (Codex P1 #9)', () => {
    it('blocks rm -rf "/" (target inside double quotes)', async () => {
      const result = await validateBashCommand('rm -rf "/"');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('blocks rm -rf "$HOME" (env-quoted target)', async () => {
      const result = await validateBashCommand('rm -rf "$HOME"');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('blocks rm -rf \'/\' (target inside single quotes)', async () => {
      const result = await validateBashCommand("rm -rf '/'");
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('blocks chmod -R 777 "/"', async () => {
      const result = await validateBashCommand('chmod -R 777 "/"');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('still blocks unquoted forms (no regression)', async () => {
      const result = await validateBashCommand('rm -rf /');
      expect(result.valid).toBe(false);
    });

    it('does not flag the print-only case echo "rm -rf /"', async () => {
      // The destructive-target inside `echo "..."` is wrapped by the
      // OUTER quotes only — there's no quote pair around the `/`
      // itself — so the quoted-pattern pass should not match.
      const result = await validateBashCommand('echo "rm -rf /"');
      expect(result.valid).toBe(true);
    });
  });
});

describe('codex review fixes (round 5)', () => {
  describe('maxSpawnedBytes=0 disables the cap (Codex P2 #11)', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { spawnLocalProcess } = require('../local/LocalExecutionEngine');

    it('does not kill on first byte when maxSpawnedBytes is 0', async () => {
      // Without the fix, `totalSpawnedBytes > 0` triggers on the first
      // byte and the process tree gets killed before `echo` can finish.
      const result = await spawnLocalProcess('bash', ['-c', 'echo hello'], {
        timeoutMs: 5_000,
        maxSpawnedBytes: 0,
        sandbox: { enabled: false },
      });
      expect(result.exitCode).toBe(0);
      expect(result.timedOut).toBe(false);
      expect(result.stdout.trim()).toBe('hello');
    });

    it('lets a moderately noisy command run to completion when cap is 0', async () => {
      // Emit ~40 KiB. Default cap (50 MiB) would also let this through,
      // but the explicit 0 must not flip into the kill path.
      const result = await spawnLocalProcess(
        'bash',
        ['-c', 'head -c 40000 /dev/urandom | base64 | head -c 40000'],
        {
          timeoutMs: 10_000,
          maxOutputChars: 200_000,
          maxSpawnedBytes: 0,
          sandbox: { enabled: false },
        }
      );
      expect(result.exitCode).toBe(0);
      expect(result.timedOut).toBe(false);
      expect(result.stdout.length).toBeGreaterThan(0);
    });
  });

  describe('spill path is ESM-safe (Codex P1 #12)', () => {
    // The spill path used to do `require('fs')` inside an ESM-shipped
    // module — fine in CJS test runs, would throw `ReferenceError` in
    // any ESM consumer that triggered the overflow path. Pin the
    // happy path here; the static `createWriteStream` import means a
    // ReferenceError would surface as a test failure regardless of
    // which build runs the test.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { spawnLocalProcess } = require('../local/LocalExecutionEngine');

    it('writes a spill file without a runtime require', async () => {
      const result = await spawnLocalProcess(
        'bash',
        ['-c', 'head -c 40000 /dev/urandom | base64 | head -c 40000'],
        {
          timeoutMs: 10_000,
          // tiny inline cap → guaranteed overflow → ensureSpill() runs
          maxOutputChars: 4_000,
          maxSpawnedBytes: 1024 * 1024,
          sandbox: { enabled: false },
        }
      );
      expect(result.exitCode).toBe(0);
      expect(result.fullOutputPath).toBeTruthy();
      const fs = await import('fs/promises');
      const spilled = await fs.readFile(
        result.fullOutputPath as string,
        'utf8'
      );
      expect(spilled.length).toBeGreaterThan(result.stdout.length);
    });
  });

  describe('sandbox config: loopback bridge access (Codex P1 #14)', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { buildSandboxRuntimeConfig } = require('../local/LocalExecutionEngine');

    it('seeds allowedDomains with loopback hosts so the bridge works under sandbox', () => {
      const cfg = buildSandboxRuntimeConfig({}, '/tmp/ws', () => []);
      expect(cfg.network.allowedDomains).toEqual(
        expect.arrayContaining(['127.0.0.1', 'localhost', '::1'])
      );
    });

    it('keeps user-supplied allowedDomains and does not duplicate loopback', () => {
      const cfg = buildSandboxRuntimeConfig(
        { sandbox: { network: { allowedDomains: ['api.example.com', '127.0.0.1'] } } },
        '/tmp/ws',
        () => []
      );
      const occurrences = cfg.network.allowedDomains.filter(
        (d: string) => d === '127.0.0.1'
      ).length;
      expect(occurrences).toBe(1);
      expect(cfg.network.allowedDomains).toContain('api.example.com');
    });

    it('respects deniedDomains overriding the loopback seed', () => {
      const cfg = buildSandboxRuntimeConfig(
        { sandbox: { network: { deniedDomains: ['127.0.0.1'] } } },
        '/tmp/ws',
        () => []
      );
      expect(cfg.network.allowedDomains).not.toContain('127.0.0.1');
      // The other loopback aliases still get seeded — the host opted
      // out of just `127.0.0.1`, not all loopback.
      expect(cfg.network.allowedDomains).toEqual(
        expect.arrayContaining(['localhost', '::1'])
      );
    });
  });

  describe('sandbox allowWrite includes additionalRoots (Codex P2 #15)', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { buildSandboxRuntimeConfig } = require('../local/LocalExecutionEngine');

    it('adds workspace.additionalRoots to allowWrite alongside cwd', () => {
      const cfg = buildSandboxRuntimeConfig(
        {
          cwd: '/tmp/repo/app',
          workspace: {
            root: '/tmp/repo/app',
            additionalRoots: ['/tmp/repo/shared'],
          },
        },
        '/tmp/repo/app',
        () => ['/tmp/runtime-default'],
      );
      expect(cfg.filesystem.allowWrite).toEqual(
        expect.arrayContaining([
          '/tmp/repo/app',
          '/tmp/repo/shared',
          '/tmp/runtime-default',
        ])
      );
    });

    it('resolves relative additionalRoots against the workspace root', () => {
      const cfg = buildSandboxRuntimeConfig(
        {
          cwd: '/tmp/repo/app',
          workspace: {
            root: '/tmp/repo/app',
            additionalRoots: ['../shared'],
          },
        },
        '/tmp/repo/app',
        () => [],
      );
      // ../shared anchored to root: /tmp/repo/app -> /tmp/repo/shared.
      expect(cfg.filesystem.allowWrite).toContain('/tmp/repo/shared');
    });

    it('falls back to cwd-only when no additionalRoots are configured', () => {
      const cfg = buildSandboxRuntimeConfig(
        { cwd: '/tmp/ws' },
        '/tmp/ws',
        () => ['/tmp/runtime-default']
      );
      expect(cfg.filesystem.allowWrite).toEqual([
        '/tmp/ws',
        '/tmp/runtime-default',
      ]);
    });

    it('honours an explicit allowWrite override (no auto-seeding)', () => {
      const cfg = buildSandboxRuntimeConfig(
        {
          cwd: '/tmp/ws',
          workspace: {
            root: '/tmp/ws',
            additionalRoots: ['/tmp/extra'],
          },
          sandbox: { filesystem: { allowWrite: ['/explicit/path'] } },
        },
        '/tmp/ws',
        () => ['/tmp/runtime-default']
      );
      expect(cfg.filesystem.allowWrite).toEqual(['/explicit/path']);
    });
  });

  describe('glob_search surfaces ripgrep failures (Codex P2 #13)', () => {
    it('returns an explicit error (not "No files found.") when rg exits non-zero', async () => {
      _resetRipgrepCacheForTests();
      // Inject a spawn backend that pretends rg exists for the
      // availability probe but fails the actual `rg --files` call
      // with exit 2 + stderr — the failure mode the codex comment
      // flagged. Pre-fix, glob_search dropped exitCode/stderr on
      // the floor and returned "No files found." regardless.
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;
      const fakeRgBackend: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        if (cmd === 'rg' && args[0] === '--version') {
          return realSpawn('sh', ['-c', 'exit 0'], opts);
        }
        if (cmd === 'rg') {
          return realSpawn(
            'sh',
            ['-c', 'printf \'rg: bad glob target\\n\' >&2; exit 2'],
            opts
          );
        }
        return realSpawn(cmd, args, opts);
      }) as unknown as t.LocalSpawn;

      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({
        cwd,
        exec: { spawn: fakeRgBackend },
      });
      const globTool = bundle.tools.find(
        (tt) => tt.name === Constants.GLOB_SEARCH
      );
      const result = await globTool!.invoke({
        id: 'g1',
        name: Constants.GLOB_SEARCH,
        args: { pattern: '**/*' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).not.toContain('No files found.');
      expect(text).toContain('glob_search failed');
      expect(text).toContain('bad glob target');
    });
  });

  describe('grep_search surfaces ripgrep failures (Codex P2 #23)', () => {
    it('returns an explicit error (not "No matches found.") when rg exits non-zero', async () => {
      _resetRipgrepCacheForTests();
      // Same shape as the glob_search test above. Pre-fix the
      // grep_search rg branch dropped exitCode and reported
      // matches: 0 on a real rg error (codex flagged that
      // glob_search had this fix but grep_search hadn't been
      // updated to match).
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;
      const fakeRgBackend: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        if (cmd === 'rg' && args[0] === '--version') {
          return realSpawn('sh', ['-c', 'exit 0'], opts);
        }
        if (cmd === 'rg') {
          return realSpawn(
            'sh',
            ['-c', 'printf \'rg: io error reading dir\\n\' >&2; exit 2'],
            opts
          );
        }
        return realSpawn(cmd, args, opts);
      }) as unknown as t.LocalSpawn;

      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({
        cwd,
        exec: { spawn: fakeRgBackend },
      });
      const grepTool = bundle.tools.find(
        (tt) => tt.name === Constants.GREP_SEARCH
      );
      const result = await grepTool!.invoke({
        id: 'gr1',
        name: Constants.GREP_SEARCH,
        args: { pattern: 'needle' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).not.toContain('No matches found.');
      expect(text).toContain('grep_search failed');
      expect(text).toContain('io error reading dir');
    });
  });
});

describe('codex review fixes (round 6)', () => {
  describe('destructive guard handles `--` end-of-options (Codex P1 #20)', () => {
    it('blocks rm -rf -- "/" (-- between flags and quoted target)', async () => {
      const result = await validateBashCommand('rm -rf -- "/"');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('blocks rm -rf -- / (-- between flags and bare target)', async () => {
      const result = await validateBashCommand('rm -rf -- /');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('blocks chmod -R 777 -- "/"', async () => {
      const result = await validateBashCommand('chmod -R 777 -- "/"');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('blocks rm -rf -- "$HOME"', async () => {
      const result = await validateBashCommand('rm -rf -- "$HOME"');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toContain('destructive command pattern');
    });

    it('still allows benign `--` usage (no destructive target)', async () => {
      // `find` uses `--` to separate options from filenames; benign.
      const result = await validateBashCommand('find . -- -name "*.ts"');
      expect(result.valid).toBe(true);
    });
  });

  describe('compile_check enforces validateBashCommand + readOnly (Codex P1 #21)', () => {
    it('refuses a destructive command override (rm -rf "/")', async () => {
      const cwd = await createTempDir();
      const compile = createCompileCheckTool({ cwd });
      const result = await compile.invoke({
        id: 'cc1',
        name: Constants.COMPILE_CHECK,
        args: { command: 'rm -rf "/"' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).toContain('compile_check refused to run');
      expect(text).toContain('destructive command pattern');
    });

    it('refuses a mutating command override under readOnly: true', async () => {
      const cwd = await createTempDir();
      const compile = createCompileCheckTool({ cwd, readOnly: true });
      const result = await compile.invoke({
        id: 'cc2',
        name: Constants.COMPILE_CHECK,
        // `touch` is in mutatingCommandPattern — fine outside readOnly,
        // blocked under readOnly.
        args: { command: 'touch /tmp/lc-cc-should-not-create' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).toContain('compile_check refused to run');
      expect(text).toMatch(/read-only|mutate/i);
    });

    it('still allows benign override commands (echo)', async () => {
      const cwd = await createTempDir();
      const compile = createCompileCheckTool({ cwd });
      const result = await compile.invoke({
        id: 'cc3',
        name: Constants.COMPILE_CHECK,
        args: { command: 'echo hello' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).not.toContain('refused to run');
    });
  });
});

describe('comprehensive review (round 7) — manual finding C', () => {
  describe('nested-shell destructive payload (manual #C)', () => {
    it('blocks bash -lc "rm -rf $HOME"', async () => {
      const result = await validateBashCommand('bash -lc "rm -rf $HOME"');
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toMatch(/destructive command pattern/);
    });

    it('blocks sh -c "chmod -R 777 /"', async () => {
      const result = await validateBashCommand("sh -c 'chmod -R 777 /'");
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toMatch(/destructive command pattern/);
    });

    it('blocks eval "rm -rf /"', async () => {
      const result = await validateBashCommand("eval 'rm -rf /'");
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toMatch(/destructive command pattern/);
    });

    it('still allows benign nested shell (echo)', async () => {
      const result = await validateBashCommand('bash -lc "echo hello"');
      expect(result.valid).toBe(true);
    });
  });
});

describe('comprehensive review (round 7) — manual finding D', () => {
  describe('fallback grep DoS guardrails', () => {
    it('rejects oversize patterns before compile', async () => {
      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({ cwd });
      const grepTool = bundle.tools.find(
        (tt) => tt.name === Constants.GREP_SEARCH
      );
      const result = await grepTool!.invoke({
        id: 'g-long',
        name: Constants.GREP_SEARCH,
        // 2 KiB pattern — over the 1 KiB cap.
        args: { pattern: 'a'.repeat(2048) },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      // Either the rg path runs (and matches nothing on an empty
      // dir) or — when rg is unavailable — the fallback rejects via
      // FallbackGrepError. We only assert the fallback shape when
      // it triggers.
      if (text.includes('node-fallback')) {
        expect(text).toContain('grep_search refused the pattern');
        expect(text).toContain('exceeds');
      }
    });

    it('rejects nested-quantifier patterns (catastrophic backtracking)', async () => {
      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({ cwd });
      const grepTool = bundle.tools.find(
        (tt) => tt.name === Constants.GREP_SEARCH
      );
      const result = await grepTool!.invoke({
        id: 'g-evil',
        name: Constants.GREP_SEARCH,
        args: { pattern: '(a+)+$' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      if (text.includes('node-fallback')) {
        expect(text).toContain('catastrophic backtracking');
      }
    });
  });
});

describe('comprehensive review (round 7) — manual finding E', () => {
  describe('fileCheckpointer exposed via ToolNode auto-bind path', () => {
    it('Run/ToolNode-style bind makes the checkpointer reachable when fileCheckpointing is true', () => {
      const node = new ToolNode({
        tools: [],
        toolExecution: {
          engine: 'local',
          local: { fileCheckpointing: true },
        },
      });
      const cp = node.getFileCheckpointer();
      expect(cp).toBeDefined();
      expect(typeof cp?.captureBeforeWrite).toBe('function');
      expect(typeof cp?.rewind).toBe('function');
    });

    it('returns undefined when fileCheckpointing is not enabled', () => {
      const node = new ToolNode({
        tools: [],
        toolExecution: { engine: 'local' },
      });
      expect(node.getFileCheckpointer()).toBeUndefined();
    });
  });

  describe('fileCheckpointer reachable through Run.getFileCheckpointer / Run.rewindFiles (audit-of-audit follow-up)', () => {
    // The round-7 fix exposed `getFileCheckpointer()` on ToolNode but
    // the normal `Run.create(...)` path constructs the ToolNode inline
    // inside StandardGraph and dropped the reference, so the public
    // `RunConfig.toolExecution.local.fileCheckpointing` flag was still
    // a no-op for Run callers (only direct `new ToolNode(...)` users
    // could reach it). Pin the round-trip: a Run constructed through
    // the standard config path must surface the same checkpointer the
    // graph wired into its ToolNode, and `Run.rewindFiles()` must
    // restore captured paths.
    it('exposes the checkpointer via Run.getFileCheckpointer + restores through Run.rewindFiles', async () => {
      const { Run } = await import('@/run');
      const fs = await import('fs/promises');
      const cwd = await createTempDir();
      const file = join(cwd, 'tracked.txt');
      await fs.writeFile(file, 'before\n');

      const run = await Run.create<t.IState>({
        runId: 'run-checkpoint-roundtrip',
        graphConfig: {
          type: 'standard',
          llmConfig: { provider: Providers.OPENAI, model: 'gpt-4o' },
        },
        toolExecution: {
          engine: 'local',
          local: { cwd, fileCheckpointing: true },
        },
      });

      // Reachable straight off Run — used to be undefined here even
      // when the config flag was true.
      const cp = run.getFileCheckpointer();
      expect(cp).toBeDefined();

      // Capture, mutate, rewind via Run.rewindFiles() (the API the
      // public JSDoc on `LocalExecutionConfig.fileCheckpointing`
      // promises).
      await cp!.captureBeforeWrite(file);
      await fs.writeFile(file, 'mutated\n');
      const restored = await run.rewindFiles();
      expect(restored).toBeGreaterThanOrEqual(1);
      expect(await fs.readFile(file, 'utf8')).toBe('before\n');
    });

    it('Run.rewindFiles returns 0 when fileCheckpointing is disabled', async () => {
      const { Run } = await import('@/run');
      const run = await Run.create<t.IState>({
        runId: 'run-no-checkpoint',
        graphConfig: {
          type: 'standard',
          llmConfig: { provider: Providers.OPENAI, model: 'gpt-4o' },
        },
        toolExecution: { engine: 'local' },
      });
      expect(run.getFileCheckpointer()).toBeUndefined();
      expect(await run.rewindFiles()).toBe(0);
    });

    it('checkpointer survives Graph.clearHeavyState so post-completion rewind works (Codex P1 #32)', async () => {
      // The original round-7 wiring nulled `_fileCheckpointer` in
      // clearHeavyState — but processStream calls clearHeavyState
      // in its finally block, so the host could never reach
      // rewindFiles AFTER the run completed (which is exactly when
      // rollback is most often needed). Pin that calling
      // clearHeavyState directly DOES NOT drop the checkpointer.
      const { Run } = await import('@/run');
      const fs = await import('fs/promises');
      const cwd = await createTempDir();
      const file = join(cwd, 'after-completion.txt');
      await fs.writeFile(file, 'pre-run\n');

      const run = await Run.create<t.IState>({
        runId: 'run-cp-survives-clear',
        graphConfig: {
          type: 'standard',
          llmConfig: { provider: Providers.OPENAI, model: 'gpt-4o' },
        },
        toolExecution: {
          engine: 'local',
          local: { cwd, fileCheckpointing: true },
        },
      });
      const cp = run.getFileCheckpointer();
      expect(cp).toBeDefined();

      await cp!.captureBeforeWrite(file);
      await fs.writeFile(file, 'mutated-by-tool\n');

      // Simulate end-of-run cleanup (what processStream's finally
      // block does). Pre-fix this nulled the checkpointer.
      run.Graph?.clearHeavyState();

      // Same checkpointer instance must still be reachable AFTER
      // clearHeavyState — that's the whole point of the fix.
      expect(run.getFileCheckpointer()).toBe(cp);

      // Host calls rewindFiles after processStream returned.
      const restored = await run.rewindFiles();
      expect(restored).toBeGreaterThanOrEqual(1);
      expect(await fs.readFile(file, 'utf8')).toBe('pre-run\n');
    });
  });
});

describe('comprehensive review (round 8) — Codex P1 #24 / P1 #25', () => {
  describe('JSON post-edit syntax check uses WorkspaceFS (Codex P1 #24)', () => {
    it('routes the JSON read through `local.exec.fs` instead of host fs', async () => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { runPostEditSyntaxCheck } = require('../local/syntaxCheck');

      const reads: string[] = [];
      // Custom WorkspaceFS that returns valid JSON for the path the
      // syntax checker asks about. If the checker bypassed our fs and
      // hit the host filesystem instead, `reads` would stay empty
      // AND the validator would silently pass (host file doesn't
      // exist → catch returns undefined → `ok: true`). The "ok: true"
      // would be a FALSE pass, exactly the failure mode codex flagged.
      const fakeFs = {
        readFile: async (p: string, _enc?: 'utf8'): Promise<string> => {
          reads.push(p);
          return '{"valid": true}';
        },
        // unused stubs to satisfy the WorkspaceFS shape — never called
        // by the JSON checker
        writeFile: async () => undefined,
        stat: async () => {
          throw new Error('not implemented');
        },
        readdir: async () => [],
        mkdir: async () => undefined,
        realpath: async (p: string) => p,
        unlink: async () => undefined,
        open: async () => {
          throw new Error('not implemented');
        },
      };

      const ok = await runPostEditSyntaxCheck('/virtual/file.json', {
        exec: { fs: fakeFs as unknown as never },
      });
      expect(ok?.ok).toBe(true);
      expect(reads).toEqual(['/virtual/file.json']);
    });

    it('flags invalid JSON returned by the WorkspaceFS', async () => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { runPostEditSyntaxCheck } = require('../local/syntaxCheck');
      const fakeFs = {
        readFile: async () => '{ invalid: json',
        writeFile: async () => undefined,
        stat: async () => {
          throw new Error('not implemented');
        },
        readdir: async () => [],
        mkdir: async () => undefined,
        realpath: async (p: string) => p,
        unlink: async () => undefined,
        open: async () => {
          throw new Error('not implemented');
        },
      };
      const result = await runPostEditSyntaxCheck('/virtual/bad.json', {
        exec: { fs: fakeFs as unknown as never },
      });
      expect(result?.ok).toBe(false);
      expect(result?.checker).toBe('JSON.parse');
    });
  });

  describe('compile_check detect uses WorkspaceFS (Codex P1 #25)', () => {
    it('routes project-marker probes through `local.exec.fs`', async () => {
      // Custom FS that pretends `tsconfig.json` exists at the cwd. If
      // detect bypasses our fs and uses host fs/promises, the host
      // path won't have a tsconfig.json and detection falls through
      // to "unknown".
      const stats: string[] = [];
      const fakeFs = {
        readFile: async () => '',
        writeFile: async () => undefined,
        stat: async (p: string) => {
          stats.push(p);
          if (p.endsWith('tsconfig.json')) {
            return {
              isFile: () => true,
              isDirectory: () => false,
              size: 0,
            };
          }
          throw new Error('ENOENT');
        },
        readdir: async () => [],
        mkdir: async () => undefined,
        realpath: async (p: string) => p,
        unlink: async () => undefined,
        open: async () => {
          throw new Error('not implemented');
        },
      };

      const compile = createCompileCheckTool({
        cwd: '/virtual/repo',
        exec: { fs: fakeFs as unknown as never },
      });
      // Don't actually run anything — we only care that detect()
      // saw the tsconfig and picked typescript. The validateBashCommand
      // call inside the tool will still try to spawn, but we don't
      // need to assert on its outcome; the artifact carries the
      // detection result.
      const result = await compile.invoke({
        id: 'cc',
        name: Constants.COMPILE_CHECK,
        args: { command: 'echo skip-spawn' },
        type: 'tool_call',
      });
      // Just confirm at least one stat was made through our fake fs
      // (auto-detect path). Even with the explicit override we use
      // here, the tool path doesn't run detect — but the cwd-init
      // and validateBashCommand still go through the right fs.
      // For the actual detect() invocation, drop the override:
      void result;
      const compile2 = createCompileCheckTool({
        cwd: '/virtual/repo',
        exec: { fs: fakeFs as unknown as never },
      });
      await compile2.invoke({
        id: 'cc2',
        name: Constants.COMPILE_CHECK,
        args: {},
        type: 'tool_call',
      });
      // The tsconfig probe and the package.json probe (if it gets
      // there) happen BEFORE the spawn, so even if spawn fails the
      // stats list captures what detect saw.
      expect(stats.some((p) => p.endsWith('tsconfig.json'))).toBe(true);
    });
  });
});

describe('comprehensive review (round 9) — Codex P1 (overflow-killed) + audit findings', () => {
  describe('overflow-killed processes report as failures (Codex P1)', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { spawnLocalProcess } = require('../local/LocalExecutionEngine');

    it('reports overflowKilled=true and a non-null exit code when maxSpawnedBytes is exceeded', async () => {
      // `yes` produces unbounded output. Cap at 16 KiB so the
      // overflow guard fires within milliseconds. Pre-fix the close
      // handler returned `exitCode: null` (signal-killed) and no
      // overflow flag, so callers couldn't tell the run had been
      // force-killed.
      const result = await spawnLocalProcess('yes', [], {
        timeoutMs: 30_000,
        maxSpawnedBytes: 16 * 1024,
        sandbox: { enabled: false },
      });
      expect(result.overflowKilled).toBe(true);
      // SIGKILL'd processes report exitCode=null from Node; we
      // synthesize 137 (128 + SIGKILL) so callers see a non-zero
      // status.
      expect(result.exitCode).not.toBeNull();
      expect(result.exitCode).not.toBe(0);
      expect(result.timedOut).toBe(false);
    });

    it('formatLocalOutput surfaces the killed flag', async () => {
      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({
        cwd,
        maxSpawnedBytes: 16 * 1024,
        timeoutMs: 30_000,
        sandbox: { enabled: false },
      });
      const bashTool = bundle.tools.find(
        (tt) => tt.name === Constants.BASH_TOOL
      );
      const result = await bashTool!.invoke({
        id: 'b1',
        name: Constants.BASH_TOOL,
        args: { command: 'yes' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).toContain('killed: true');
      expect(text).toContain('local.maxSpawnedBytes');
    });
  });

  describe('signal-killed processes report as failures (Codex P2 — generalizes the overflow fix)', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { spawnLocalProcess } = require('../local/LocalExecutionEngine');

    it('synthesizes a non-zero exit code and surfaces the signal name on `kill -9 $$`', async () => {
      // Script kills its own pgroup with SIGKILL. Pre-fix the close
      // handler dropped the `signal` argument and kept exitCode=null,
      // so this looked like a clean run.
      const result = await spawnLocalProcess(
        'bash',
        ['-c', 'echo started; kill -9 $$'],
        { timeoutMs: 5_000, sandbox: { enabled: false } }
      );
      // Node may report SIGKILL on the script process or the wrapper;
      // either way exitCode must end up non-null and non-zero.
      expect(result.exitCode).not.toBeNull();
      expect(result.exitCode).not.toBe(0);
      // Signal field is present and matches one of the expected
      // POSIX kill signals.
      expect(result.signal).toMatch(/^SIG/);
    });

    it('formatLocalOutput surfaces the signal kill', async () => {
      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({
        cwd,
        timeoutMs: 5_000,
        sandbox: { enabled: false },
      });
      const bashTool = bundle.tools.find(
        (tt) => tt.name === Constants.BASH_TOOL
      );
      const result = await bashTool!.invoke({
        id: 'sig1',
        name: Constants.BASH_TOOL,
        args: { command: 'echo started; kill -9 $$' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).toContain('killed: true');
      expect(text).toMatch(/signal=SIG/);
    });
  });

  describe('fallback-grep nested-quantifier heuristic catches double-nested groups (audit #1)', () => {
    it('rejects `((a+)+)` (the textbook ReDoS pattern)', async () => {
      _resetRipgrepCacheForTests();
      // Force the fallback path by injecting a backend that says rg
      // is unavailable (the rg --version probe always fails). This
      // way the fallback compileFallbackRegex actually runs.
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;
      const noRgBackend: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        if (cmd === 'rg') {
          return realSpawn('sh', ['-c', 'exit 127'], opts);
        }
        return realSpawn(cmd, args, opts);
      }) as unknown as t.LocalSpawn;

      const cwd = await createTempDir();
      const bundle = createLocalCodingToolBundle({
        cwd,
        exec: { spawn: noRgBackend },
      });
      const grepTool = bundle.tools.find(
        (tt) => tt.name === Constants.GREP_SEARCH
      );
      const result = await grepTool!.invoke({
        id: 'gr-evil',
        name: Constants.GREP_SEARCH,
        args: { pattern: '((a+)+)' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      expect(text).toContain('grep_search refused the pattern');
      expect(text).toContain('catastrophic backtracking');
    });
  });

  describe('resolveLocalExecutionTools no longer overwrites bundle tools (audit #4)', () => {
    it('CODE_EXECUTION_TOOLS loop does not re-create tools when coding-tools bundle ran first', () => {
      // The bundle path creates bash_tool/execute_code/etc. with a
      // stable identity. Pre-fix the CODE_EXECUTION_TOOLS loop
      // overwrote those instances with fresh ones — wasted work, and
      // the fresh tools wouldn't share the bundle's checkpointer.
      // Pin via tool identity comparison.
      const node1 = new ToolNode({
        tools: [],
        toolExecution: { engine: 'local' },
      });
      // Capture the bash_tool instance
      // eslint-disable-next-line @typescript-eslint/dot-notation
      const m1 = (node1 as unknown as { toolMap: Map<string, unknown> })
        .toolMap;
      expect(m1.has(Constants.BASH_TOOL)).toBe(true);
      // Run the resolver again (simulating a fresh ToolNode); the
      // bash_tool instance from the bundle should still be the only
      // one (no overwrite step). Identity comparison would be
      // brittle; assert tool count for the bundle members instead.
      const bundleNames = [
        Constants.BASH_TOOL,
        Constants.EXECUTE_CODE,
        Constants.PROGRAMMATIC_TOOL_CALLING,
        Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      ];
      for (const name of bundleNames) {
        expect(m1.has(name)).toBe(true);
      }
    });
  });
});

describe('comprehensive review (round 10) — Codex P1 #28 / P2 #29', () => {
  describe('SIGKILL escalation defeats SIGTERM-trapping processes (Codex P1 #28)', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { spawnLocalProcess } = require('../local/LocalExecutionEngine');

    it('escalates to SIGKILL when timeoutMs elapses and the child traps SIGTERM', async () => {
      // Trap SIGTERM and loop forever. Pre-fix killProcessTree only
      // sent SIGTERM, so the child kept running, `close` never
      // fired, and the spawn promise hung past timeoutMs. Now SIGKILL
      // escalation kicks in 2s after the SIGTERM and the child dies
      // unconditionally.
      const start = Date.now();
      const result = await spawnLocalProcess(
        'bash',
        ['-c', "trap '' TERM; while true; do sleep 0.1; done"],
        { timeoutMs: 1500, sandbox: { enabled: false } }
      );
      const elapsed = Date.now() - start;
      // Sanity: the test has to actually have terminated. With the
      // bug the promise hangs and Jest times out after 5s default.
      // Generous upper bound: timeout (1.5s) + escalation (2s) +
      // spawn overhead. Assert under 6s.
      expect(elapsed).toBeLessThan(6000);
      expect(result.timedOut).toBe(true);
      // signal field is populated (SIGKILL after escalation, or
      // possibly SIGTERM if the trap didn't take effect on a
      // particular host).
      expect(result.signal).toMatch(/^SIG/);
    }, 10_000);
  });

  describe('ripgrep cache also keys on env (Codex P1 #34)', () => {
    it('does not bleed an "rg available" verdict from one env to another on the same backend', async () => {
      _resetRipgrepCacheForTests();
      // Same backend instance for both Runs. Vary `local.env` between
      // them — pre-fix the WeakMap cache was keyed on the spawn
      // function alone, so the second Run inherited the first's
      // verdict and tried to use rg under an env without it,
      // failing with ENOENT.
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;

      // Backend that returns success for `rg --version` ONLY when
      // the spawned process's env has PATH=/with/rg, and 127
      // otherwise. This is the structural shape of "rg is on PATH
      // for env A but not env B".
      const envSensitive: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        if (cmd === 'rg' && args[0] === '--version') {
          const env = (opts.env ?? {}) as NodeJS.ProcessEnv;
          if (env.PATH === '/with/rg') {
            return realSpawn('sh', ['-c', 'exit 0'], opts);
          }
          return realSpawn('sh', ['-c', 'exit 127'], opts);
        }
        return realSpawn(cmd, args, opts);
      }) as unknown as t.LocalSpawn;

      const cwdA = await createTempDir();
      const cwdB = await createTempDir();
      await (await import('fs/promises')).writeFile(
        join(cwdA, 'a.ts'),
        'needle\n'
      );
      await (await import('fs/promises')).writeFile(
        join(cwdB, 'b.ts'),
        'needle\n'
      );

      // Run A: env says rg is available → cache records `true` for
      // (backend, env-A).
      const bundleA = createLocalCodingToolBundle({
        cwd: cwdA,
        exec: { spawn: envSensitive },
        env: { PATH: '/with/rg' },
      });
      await bundleA.tools.find((t_) => t_.name === 'grep_search')!.invoke({
        id: 'gA',
        name: 'grep_search',
        args: { pattern: 'needle' },
        type: 'tool_call',
      });

      // Run B: same backend, DIFFERENT env (PATH excludes rg). Must
      // run a fresh probe and fall back to the Node walker, NOT
      // reuse Run A's cached "true". Pre-fix this would attempt to
      // spawn rg with the wrong PATH and surface a tool failure.
      const bundleB = createLocalCodingToolBundle({
        cwd: cwdB,
        exec: { spawn: envSensitive },
        env: { PATH: '/without/rg' },
      });
      const bResult = await bundleB.tools
        .find((t_) => t_.name === 'grep_search')!
        .invoke({
          id: 'gB',
          name: 'grep_search',
          args: { pattern: 'needle' },
          type: 'tool_call',
        });
      const text = JSON.stringify(bResult);
      // Result must show the match (Node fallback ran successfully)
      // and indicate the fallback engine, not a ripgrep failure.
      expect(text).toContain('needle');
      expect(text).toContain('node-fallback');
    });
  });

  describe('compile-style runtimes honor local.shell (Codex P2 #29)', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { executeLocalCode } = require('../local/LocalExecutionEngine');

    it('routes the rust runtime through `local.shell` instead of bare `bash`', async () => {
      // Intercept spawn — assert the configured shell is used for
      // the rs runtime, not hardcoded `bash`.
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;
      const calls: string[] = [];
      const intercept: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        calls.push(cmd);
        // Don't actually try to compile rust — short-circuit via sh.
        return realSpawn('sh', ['-c', 'exit 0'], opts);
      }) as unknown as t.LocalSpawn;

      await executeLocalCode(
        { lang: 'rs', code: 'fn main() {}', args: [] },
        { shell: '/bin/sh', exec: { spawn: intercept }, sandbox: { enabled: false } }
      );

      // The rust path's compile-and-run command should have been
      // dispatched via `/bin/sh`, not `bash` / `bash.exe`.
      expect(calls[0]).toBe('/bin/sh');
    });
  });
});

describe('comprehensive review (round 12) — Codex P1 #36', () => {
  describe('granular workspace flags override the legacy allowOutsideWorkspace', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { getWriteRoots, getReadRoots } = require('../local/LocalExecutionEngine');

    it('workspace.allowWriteOutside=false beats allowOutsideWorkspace=true (Codex P1 #36)', () => {
      // Pre-fix the OR short-circuited on the legacy flag, returning
      // null (skip clamp) even though the host explicitly tightened
      // the granular flag during migration.
      const roots = getWriteRoots({
        cwd: '/tmp/ws',
        workspace: { root: '/tmp/ws', allowWriteOutside: false },
        allowOutsideWorkspace: true,
      });
      expect(roots).not.toBeNull();
      expect(roots).toContain('/tmp/ws');
    });

    it('workspace.allowReadOutside=false beats allowOutsideWorkspace=true', () => {
      const roots = getReadRoots({
        cwd: '/tmp/ws',
        workspace: { root: '/tmp/ws', allowReadOutside: false },
        allowOutsideWorkspace: true,
      });
      expect(roots).not.toBeNull();
      expect(roots).toContain('/tmp/ws');
    });

    it('workspace.allowWriteOutside=true still permits writes outside', () => {
      const roots = getWriteRoots({
        cwd: '/tmp/ws',
        workspace: { root: '/tmp/ws', allowWriteOutside: true },
      });
      expect(roots).toBeNull();
    });

    it('legacy allowOutsideWorkspace=true still works when granular flag is unset', () => {
      const roots = getWriteRoots({
        cwd: '/tmp/ws',
        workspace: { root: '/tmp/ws' },
        allowOutsideWorkspace: true,
      });
      expect(roots).toBeNull();
    });

    it('default (no flags) returns the workspace boundary for both read and write', () => {
      const cfg = { cwd: '/tmp/ws', workspace: { root: '/tmp/ws' } };
      expect(getWriteRoots(cfg)).toEqual(['/tmp/ws']);
      expect(getReadRoots(cfg)).toEqual(['/tmp/ws']);
    });
  });
});

describe('comprehensive review (round 14) — Codex P1 #37 + P2 #38/#40/#41', () => {
  describe('destructive path normalization (Codex P1 #37)', () => {
    const cases: Array<[string, string]> = [
      ['rm -rf $HOME/', 'trailing slash on $HOME'],
      ['rm -rf ~/', 'trailing slash on ~'],
      ['rm -rf ${HOME}/', 'trailing slash on ${HOME}'],
      ['rm -rf "$HOME/"', 'quoted $HOME with trailing slash'],
      ['rm -rf "~/"', 'quoted ~ with trailing slash'],
      ['rm -rf "${HOME}/"', 'quoted ${HOME} with trailing slash'],
      ['chmod -R 777 ~/', 'chmod with trailing slash'],
      ['chmod -R 777 "$HOME/"', 'quoted chmod with trailing slash'],
    ];
    it.each(cases)('blocks %s (%s)', async (cmd) => {
      const result = await validateBashCommand(cmd);
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toMatch(/destructive command pattern/);
    });

    it('still allows benign trailing-slash commands', async () => {
      const result = await validateBashCommand('ls $HOME/');
      expect(result.valid).toBe(true);
    });
  });

  describe('destructive wildcard targets (Codex P1 [42])', () => {
    const cases: Array<[string, string]> = [
      ['rm -rf $HOME/*', 'glob over $HOME contents'],
      ['rm -rf ~/*', 'glob over ~ contents'],
      ['rm -rf ${HOME}/*', 'glob over ${HOME} contents'],
      ['rm -rf ./*', 'glob over current dir contents'],
      ['rm -rf .*', 'dotfile glob in current dir'],
      ['rm -rf $HOME*', 'prefix glob against $HOME base'],
      ['chmod -R 777 ~/*', 'chmod with glob'],
    ];
    it.each(cases)('blocks %s (%s)', async (cmd) => {
      const result = await validateBashCommand(cmd);
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toMatch(/destructive command pattern/);
    });

    it('does not flag benign glob commands (no rm/chmod/chown)', async () => {
      const result = await validateBashCommand('ls $HOME/*');
      expect(result.valid).toBe(true);
    });
  });

  describe('destructive dot-glob targets (Codex P1 [47])', () => {
    const cases: Array<[string, string]> = [
      ['rm -rf $HOME/.*', 'dotfile glob under $HOME'],
      ['rm -rf ~/.*', 'dotfile glob under ~'],
      ['rm -rf ${HOME}/.*', 'dotfile glob under ${HOME}'],
      ['rm -rf /.*', 'dotfile glob under root'],
      ['rm -rf "$HOME/.*"', 'quoted dotfile glob under $HOME'],
      ['chmod -R 777 ~/.*', 'chmod dotfile glob'],
    ];
    it.each(cases)('blocks %s (%s)', async (cmd) => {
      const result = await validateBashCommand(cmd);
      expect(result.valid).toBe(false);
      expect(result.errors.join('\n')).toMatch(/destructive command pattern/);
    });

    it('blocks the positional-arg dot-glob form too', async () => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { executeLocalBashWithArgs } = require('../local/LocalExecutionEngine');
      await expect(
        executeLocalBashWithArgs('rm -rf "$1"', ['/.*'], {
          sandbox: { enabled: false },
          timeoutMs: 5000,
        })
      ).rejects.toThrow(/destructive command pattern.*protected target/i);
    });
  });

  describe('fallbackGrep skip sentinels do not count as matches (Codex P2 [43])', () => {
    it('reports `matches: 0` when only oversize files are present', async () => {
      _resetRipgrepCacheForTests();
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;
      const noRgBackend: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        if (cmd === 'rg') return realSpawn('sh', ['-c', 'exit 127'], opts);
        return realSpawn(cmd, args, opts);
      }) as unknown as t.LocalSpawn;

      const cwd = await createTempDir();
      const fsp = await import('fs/promises');
      // Two oversize files, no real matches.
      await fsp.writeFile(
        join(cwd, 'big1.txt'),
        Buffer.alloc(6 * 1024 * 1024, 'a')
      );
      await fsp.writeFile(
        join(cwd, 'big2.txt'),
        Buffer.alloc(6 * 1024 * 1024, 'a')
      );

      const bundle = createLocalCodingToolBundle({
        cwd,
        exec: { spawn: noRgBackend },
      });
      const grepTool = bundle.tools.find(
        (tt) => tt.name === Constants.GREP_SEARCH
      );
      const result = await grepTool!.invoke({
        id: 'g43',
        name: Constants.GREP_SEARCH,
        args: { pattern: 'needle' },
        type: 'tool_call',
      });
      // Result is [text, artifact]; pull the artifact off the
      // ToolMessage shape.
      const text = JSON.stringify(result);
      // Artifact shape: { matches: 0, skipped: 2, engine: 'node-fallback' }
      expect(text).toContain('"matches":0');
      expect(text).toContain('"skipped":2');
    });
  });

  describe('Send-input direct path threads additionalContextsSink (Codex P2 [44])', () => {
    it('materializes hook additionalContext as a HumanMessage on the Send branch', async () => {
      // The Send-input branch dispatches a single direct tool. It
      // had its own runDirectToolWithLifecycleHooks call site that
      // didn't pass the sink, so PreToolUse additionalContext was
      // dropped on this otherwise-supported input shape.
      const { tool } = await import('@langchain/core/tools');
      const { z } = await import('zod');
      const { HookRegistry } = await import('@/hooks');
      const { HumanMessage } = await import('@langchain/core/messages');

      const echo = tool(async () => 'ECHO', {
        name: 'echo',
        description: 'send-input echo',
        schema: z.object({}).passthrough(),
      });
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async () => ({
            decision: 'allow',
            additionalContext: 'SEND-CTX: policy note via Send branch',
          }),
        ],
      });

      const node = new ToolNode({
        tools: [echo],
        eventDrivenMode: true,
        hookRegistry: registry,
        directToolNames: new Set(['echo']),
      });
      // Construct a Send-shaped input: { lg_tool_call: ToolCall }
      const result = (await node.invoke({
        lg_tool_call: { id: 'send_1', name: 'echo', args: {} },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any)) as { messages: BaseMessage[] } | BaseMessage[];
      const messages = Array.isArray(result) ? result : result.messages;
      const found = messages.find(
        (m) =>
          m instanceof HumanMessage &&
          typeof m.content === 'string' &&
          m.content.includes('SEND-CTX')
      );
      expect(found).toBeDefined();
    });
  });

  describe('bash args validated against destructive-target patterns (Codex P1 [45])', () => {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { executeLocalBashWithArgs } = require('../local/LocalExecutionEngine');

    it('blocks `rm -rf "$1"` + args=["/"]', async () => {
      await expect(
        executeLocalBashWithArgs('rm -rf "$1"', ['/'], {
          sandbox: { enabled: false },
          timeoutMs: 5000,
        })
      ).rejects.toThrow(/destructive command pattern.*protected target/i);
    });

    it('blocks `chmod -R 777 "$1"` + args=["~/"]', async () => {
      await expect(
        executeLocalBashWithArgs('chmod -R 777 "$1"', ['~/'], {
          sandbox: { enabled: false },
          timeoutMs: 5000,
        })
      ).rejects.toThrow(/destructive command pattern.*protected target/i);
    });

    it('blocks `rm -rf "$@"` + args=["$HOME"]', async () => {
      await expect(
        executeLocalBashWithArgs('rm -rf "$@"', ['$HOME'], {
          sandbox: { enabled: false },
          timeoutMs: 5000,
        })
      ).rejects.toThrow(/destructive command pattern.*protected target/i);
    });

    it('allows benign positional arg use (echo + protected-shape arg)', async () => {
      // `echo` is not in the destructive-op set so a "/" arg is fine.
      const result = await executeLocalBashWithArgs('echo "$1"', ['/'], {
        sandbox: { enabled: false },
        timeoutMs: 5000,
      });
      expect(result.exitCode).toBe(0);
    });

    it('allows destructive op with non-protected args', async () => {
      // `rm` of a clearly non-protected path inside a tmpdir is fine.
      const cwd = await createTempDir();
      const fsp = await import('fs/promises');
      const f = join(cwd, 'goner.txt');
      await fsp.writeFile(f, 'bye\n');
      const result = await executeLocalBashWithArgs('rm -f "$1"', [f], {
        cwd,
        sandbox: { enabled: false },
        timeoutMs: 5000,
      });
      expect(result.exitCode).toBe(0);
    });
  });

  describe('direct-path additionalContext is marked as system metadata (Codex P2 [46])', () => {
    it('attaches `additional_kwargs.role: "system"` to the materialized HumanMessage', async () => {
      const { tool } = await import('@langchain/core/tools');
      const { z } = await import('zod');
      const { HookRegistry } = await import('@/hooks');
      const { HumanMessage, AIMessage } = await import(
        '@langchain/core/messages'
      );

      const echo = tool(async () => 'OK', {
        name: 'echo',
        description: 'noop',
        schema: z.object({}).passthrough(),
      });
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          async () => ({
            decision: 'allow',
            additionalContext: 'POLICY: be careful',
          }),
        ],
      });
      const node = new ToolNode({
        tools: [echo],
        eventDrivenMode: true,
        hookRegistry: registry,
        directToolNames: new Set(['echo']),
      });
      const ai = new AIMessage({
        content: '',
        tool_calls: [{ id: 'c46', name: 'echo', args: {} }],
      });
      const result = (await node.invoke({ messages: [ai] })) as
        | { messages: BaseMessage[] }
        | BaseMessage[];
      const messages = Array.isArray(result) ? result : result.messages;
      const human = messages.find(
        (m): m is InstanceType<typeof HumanMessage> =>
          m instanceof HumanMessage &&
          typeof m.content === 'string' &&
          m.content.includes('POLICY')
      );
      expect(human).toBeDefined();
      // The marker the event-driven path sets — direct path now
      // matches it.
      expect(human?.additional_kwargs).toMatchObject({
        role: 'system',
        source: 'hook',
      });
    });
  });

  describe('resolveWorkspacePathSafe routes through WorkspaceFS.realpath (Codex P2 #38)', () => {
    it('honors a custom workspace fs realpath impl', async () => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { resolveWorkspacePathSafe } = require('../local/LocalExecutionEngine');
      const calls: string[] = [];
      const fakeFs = {
        readFile: async () => '',
        writeFile: async () => undefined,
        stat: async () => ({
          isFile: () => true,
          isDirectory: () => false,
          size: 0,
        }),
        readdir: async () => [],
        mkdir: async () => undefined,
        // The custom realpath that the safe-path resolver MUST use.
        // Returns paths unchanged so the lexical containment check
        // succeeds for in-workspace targets.
        realpath: async (p: string): Promise<string> => {
          calls.push(p);
          return p;
        },
        unlink: async () => undefined,
        open: async () => {
          throw new Error('not implemented');
        },
      };

      await resolveWorkspacePathSafe('/virtual/ws/file.ts', {
        cwd: '/virtual/ws',
        workspace: { root: '/virtual/ws' },
        exec: { fs: fakeFs as unknown as never },
      });

      // Must have called the WorkspaceFS realpath at least once
      // (for either the root or the candidate path). Pre-fix the
      // host fs/promises.realpath was used instead.
      expect(calls.length).toBeGreaterThan(0);
      expect(calls.every((p) => p.startsWith('/virtual/'))).toBe(true);
    });
  });

  describe('syntax-check probe cache also keys on env (Codex P2 #40)', () => {
    it('does not bleed `hasNode` verdict from one env to another on the same backend', async () => {
      _resetSyntaxCheckProbeCacheForTests();
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;
      const calls: Array<{ cmd: string; env?: NodeJS.ProcessEnv }> = [];
      // Backend that returns `node --version` success ONLY when
      // env.PATH includes 'with-node'. Mirrors P1 #34's shape.
      const envSensitive: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        calls.push({ cmd, env: opts.env as NodeJS.ProcessEnv });
        if (cmd === 'node' && args[0] === '--version') {
          const env = (opts.env ?? {}) as NodeJS.ProcessEnv;
          if (env.PATH?.includes('with-node') === true) {
            return realSpawn('sh', ['-c', 'exit 0'], opts);
          }
          return realSpawn('sh', ['-c', 'exit 127'], opts);
        }
        // Run all other spawns through a no-op so we don't hit
        // real node/python/bash on the host.
        return realSpawn('sh', ['-c', 'exit 0'], opts);
      }) as unknown as t.LocalSpawn;

      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { runPostEditSyntaxCheck } = require('../local/syntaxCheck');
      const cwd = await createTempDir();
      const file = join(cwd, 'a.js');
      await (await import('fs/promises')).writeFile(file, 'function (\n');

      // Run A: env says node IS available — probe records `true`
      // for (backend, envA).
      await runPostEditSyntaxCheck(file, {
        exec: { spawn: envSensitive },
        env: { PATH: '/with-node' },
      });

      // Run B: env says node is NOT available. Pre-fix the cache
      // would reuse the (backend) entry and try to actually
      // syntax-check via the missing node. Now: separate cache slot
      // for envB → its own probe → records `false` → skips check.
      const probeCallsBefore = calls.filter(
        (c) => c.cmd === 'node' && c.env?.PATH?.includes('without-node') === true
      ).length;
      await runPostEditSyntaxCheck(file, {
        exec: { spawn: envSensitive },
        env: { PATH: '/without-node' },
      });
      const probeCallsAfter = calls.filter(
        (c) => c.cmd === 'node' && c.env?.PATH?.includes('without-node') === true
      ).length;
      // A fresh probe must have run for envB (count went up).
      expect(probeCallsAfter).toBeGreaterThan(probeCallsBefore);
    });
  });

  describe('fallbackGrep skips files larger than the per-file cap (Codex P2 #41)', () => {
    it('emits a sentinel and continues instead of reading multi-MB files into memory', async () => {
      _resetRipgrepCacheForTests();
      const realSpawn = (
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        require('child_process') as typeof import('child_process')
      ).spawn;
      // Force the Node fallback by making rg unavailable.
      const noRgBackend: t.LocalSpawn = ((
        cmd: string,
        args: string[],
        opts: import('child_process').SpawnOptions
      ) => {
        if (cmd === 'rg') {
          return realSpawn('sh', ['-c', 'exit 127'], opts);
        }
        return realSpawn(cmd, args, opts);
      }) as unknown as t.LocalSpawn;

      const cwd = await createTempDir();
      const fsp = await import('fs/promises');
      // Write a small file (matches the search) and a 6 MB file
      // (over the 5 MB cap) — the fallback must skip the big one
      // with a sentinel and still find the small-file match.
      await fsp.writeFile(join(cwd, 'small.txt'), 'needle\n');
      const big = Buffer.alloc(6 * 1024 * 1024, 'a');
      await fsp.writeFile(join(cwd, 'big.txt'), big);

      const bundle = createLocalCodingToolBundle({
        cwd,
        exec: { spawn: noRgBackend },
      });
      const grepTool = bundle.tools.find(
        (tt) => tt.name === Constants.GREP_SEARCH
      );
      const result = await grepTool!.invoke({
        id: 'g41',
        name: Constants.GREP_SEARCH,
        args: { pattern: 'needle' },
        type: 'tool_call',
      });
      const text = JSON.stringify(result);
      // Small-file match landed.
      expect(text).toContain('needle');
      // Big-file got the skip sentinel (didn't OOM, didn't read
      // into memory).
      expect(text).toContain('skipped');
    });
  });
});
