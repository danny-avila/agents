import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import type * as t from '@/types';

/* Mirrors codeapi's `/exec/programmatic` guard, which rejects an empty tool
 * manifest outright: `if (!tools || tools.length === 0) -> 400`. Plain `/exec`
 * has no such requirement. */
type CapturedRequest = { url: string; body: Record<string, unknown> };
const requests: CapturedRequest[] = [];

jest.mock('node-fetch', () => ({
  __esModule: true,
  default: async (url: string, init?: { body?: string }) => {
    const body = JSON.parse(init?.body ?? '{}') as { tools?: unknown[] };
    requests.push({ url, body });

    if (url.endsWith('/exec/programmatic')) {
      const { tools } = body;
      if (!Array.isArray(tools) || tools.length === 0) {
        return {
          ok: false,
          status: 400,
          text: async () =>
            '{"error":"Missing required field: tools (must be a non-empty array)"}',
        };
      }
      return {
        ok: true,
        json: async () => ({
          status: 'completed',
          session_id: 'programmatic-session',
          stdout: 'programmatic',
          stderr: '',
          files: [],
        }),
      };
    }

    return {
      ok: true,
      json: async () => ({
        session_id: 'plain-session',
        stdout: 'plain',
        stderr: '',
        files: [],
      }),
    };
  },
}));

import { createBashProgrammaticToolCallingTool } from '../BashProgrammaticToolCalling';
import {
  createProgrammaticToolCallingTool,
  wrapPythonForPlainExecution,
} from '../ProgrammaticToolCalling';
import { deriveReferencedToolDefs } from '../toolIdentifiers';
import {
  createProgrammaticToolRegistry,
  createGetWeatherTool,
  createCalculatorTool,
} from '@/test/mockTools';

const BASE_URL = 'https://code.example.test';

const allToolDefs = Array.from(createProgrammaticToolRegistry().values());
const toolMap: t.ToolMap = new Map([
  ['get_weather', createGetWeatherTool()],
  ['calculator', createCalculatorTool()],
]);

type ToolResult = {
  content: string;
  artifact?: t.ProgrammaticExecutionArtifact;
};

function invokeBash(
  args: Record<string, unknown>,
  toolCallExtras: Record<string, unknown> = {}
) {
  return createBashProgrammaticToolCallingTool({ baseUrl: BASE_URL }).invoke(
    {
      name: 'run_tools_with_bash',
      args,
      id: 'call-1',
      type: 'tool_call',
      toolMap,
      toolDefs: allToolDefs,
      ...toolCallExtras,
    } as never,
    {} as never
  );
}

function invokePython(
  args: Record<string, unknown>,
  toolCallExtras: Record<string, unknown> = {}
) {
  return createProgrammaticToolCallingTool({ baseUrl: BASE_URL }).invoke(
    {
      name: 'run_tools_with_code',
      args,
      id: 'call-1',
      type: 'tool_call',
      toolMap,
      toolDefs: allToolDefs,
      ...toolCallExtras,
    } as never,
    {} as never
  );
}

describe('programmatic calls that reference no tools', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('routes an explicitly empty bash manifest to plain /exec', async () => {
    const result = (await invokeBash({
      code: 'ls /mnt/data',
      tool_manifest: [],
    })) as ToolResult;

    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
    expect(requests[0].body).toMatchObject({
      lang: 'bash',
      code: 'ls /mnt/data',
    });
    expect(requests[0].body.tools).toBeUndefined();
    expect(result.content).toContain('plain');
  });

  it('sends raw code on the plain path, without the replay `$!` guard', async () => {
    await invokeBash({ code: 'set -u\necho hi', tool_manifest: [] });

    expect(requests[0].body.code).toBe('set -u\necho hi');
  });

  it('routes an explicitly empty python manifest to plain /exec', async () => {
    await invokePython({
      code: 'import os\nprint(os.listdir("/mnt/data"))',
      tool_manifest: [],
    });

    expect(requests).toHaveLength(1);
    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
    expect(requests[0].body).toMatchObject({ lang: 'py' });
  });

  it('carries session, files and runtime hint onto the plain path', async () => {
    const files: t.CodeEnvFile[] = [
      {
        kind: 'user',
        id: 'file-1',
        resource_id: 'user-1',
        name: 'data.csv',
        storage_session_id: 'prior',
      },
    ];

    const result = (await invokeBash(
      { code: 'wc -l /mnt/data/data.csv', tool_manifest: [] },
      {
        session_id: 'prior',
        _injected_files: files,
        _runtime_session_hint: 'hint-1',
      }
    )) as ToolResult;

    expect(requests[0].body).toMatchObject({
      session_id: 'prior',
      files,
      runtime_session_hint: 'hint-1',
    });
    expect(result.artifact?.session_id).toBe('plain-session');
  });
});

describe('omitted manifest with direct-only tools configured', () => {
  const disallowedToolDefs: t.LCTool[] = [{ name: 'send_email' }];

  beforeEach(() => {
    requests.length = 0;
  });

  it('derives an empty manifest for tool-free code and runs it', async () => {
    const result = (await invokeBash(
      { code: 'jq ".[] | .name" /mnt/data/saved.json' },
      { disallowedToolDefs }
    )) as ToolResult;

    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
    expect(result.content).toContain('plain');
  });

  it('derives the referenced tool and keeps the programmatic path', async () => {
    await invokeBash(
      { code: 'raw=$(get_weather \'{"city":"SF"}\'); echo "$raw"' },
      { disallowedToolDefs }
    );

    expect(requests[0].url).toBe(`${BASE_URL}/exec/programmatic`);
    expect(
      (requests[0].body.tools as t.LCTool[]).map((def) => def.name)
    ).toEqual(['get_weather']);
  });

  it('still refuses a manifest that names a direct-only tool', async () => {
    await expect(
      invokeBash(
        { code: 'send_email \'{}\'', tool_manifest: ['send_email'] },
        { disallowedToolDefs }
      )
    ).rejects.toThrow('not marked for code_execution');

    expect(requests).toHaveLength(0);
  });

  it('rejects a derived reference to a direct-only tool', async () => {
    await expect(
      invokeBash({ code: 'send_email \'{}\'' }, { disallowedToolDefs })
    ).rejects.toThrow('not marked for code_execution');

    expect(requests).toHaveLength(0);
  });
});

describe('unchanged behavior when no direct-only tools exist', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('still sends the full manifest for an omitted tool_manifest', async () => {
    await invokeBash({ code: 'ls /mnt/data' });

    expect(requests[0].url).toBe(`${BASE_URL}/exec/programmatic`);
    expect(
      (requests[0].body.tools as t.LCTool[]).map((def) => def.name)
    ).toEqual(allToolDefs.map((def) => def.name));
  });
});

describe('deriveReferencedToolDefs', () => {
  it('returns an empty set rather than falling back to every tool', () => {
    expect(deriveReferencedToolDefs(allToolDefs, 'ls -la', 'bash')).toEqual([]);
  });

  it('matches bash invocations by word boundary', () => {
    expect(
      deriveReferencedToolDefs(
        allToolDefs,
        'calculator \'{"a":1}\'',
        'bash'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('requires a call expression for python', () => {
    expect(
      deriveReferencedToolDefs(allToolDefs, '# calculator is unused', 'python')
    ).toEqual([]);
    expect(
      deriveReferencedToolDefs(allToolDefs, 'calculator(a=1)', 'python').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });
});

describe('python plain route keeps the programmatic contract', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('wraps top-level await so it stays valid python', async () => {
    await invokePython({
      code: 'import asyncio\nawait asyncio.sleep(0)\nprint("done")',
      tool_manifest: [],
    });

    const sent = requests[0].body.code as string;
    expect(sent).toContain('async def __user_main__():');
    expect(sent).toContain('    await asyncio.sleep(0)');
    expect(sent).toContain('asyncio.run(__user_main__())');
  });

  it('leaves bash source untouched', async () => {
    await invokeBash({ code: 'echo hi', tool_manifest: [] });

    expect(requests[0].body.code).toBe('echo hi');
  });

  it('preserves blank lines when indenting', () => {
    expect(wrapPythonForPlainExecution('a = 1\n\nprint(a)')).toContain(
      '    a = 1\n\n    print(a)'
    );
  });
});

describe('tool context is still required to conclude no tools are needed', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('still reports a missing toolMap when nothing was injected', async () => {
    await expect(
      createBashProgrammaticToolCallingTool({ baseUrl: BASE_URL }).invoke({
        code: 'get_weather \'{"city":"SF"}\'',
      } as never)
    ).rejects.toThrow('No toolMap provided');

    expect(requests).toHaveLength(0);
  });

  it('runs tool-free code when only disallowed defs were injected', async () => {
    await createBashProgrammaticToolCallingTool({ baseUrl: BASE_URL }).invoke(
      {
        name: 'run_tools_with_bash',
        args: { code: 'ls /mnt/data' },
        id: 'call-1',
        type: 'tool_call',
        disallowedToolDefs: [{ name: 'send_email' }],
      } as never,
      {} as never
    );

    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
  });
});

describe('ambiguous normalized identifiers', () => {
  const colliding: t.LCTool[] = [
    { name: 'report-tool', allowed_callers: ['code_execution'] },
    { name: 'report_tool', allowed_callers: ['code_execution'] },
  ];

  it('keeps every colliding definition rather than picking by order', () => {
    expect(
      deriveReferencedToolDefs(colliding, 'report_tool \'{}\'', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['report-tool', 'report_tool']);
  });

  it('does not widen selection for an unreferenced collision', () => {
    expect(deriveReferencedToolDefs(colliding, 'ls -la', 'bash')).toEqual([]);
  });
});
