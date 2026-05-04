import { z } from 'zod';
import { tmpdir } from 'os';
import { join } from 'path';
import { spawnSync } from 'child_process';
import { mkdtemp, rm } from 'fs/promises';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { Constants } from '@/common';
import { ToolNode } from '../ToolNode';
import { validateBashCommand } from '../local/LocalExecutionEngine';
import { resolveLocalToolsForBinding } from '../local/resolveLocalExecutionTools';

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
