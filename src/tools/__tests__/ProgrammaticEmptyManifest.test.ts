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
import { selectProgrammaticTools } from '../ProgrammaticCallerPolicy';
import { createPythonProgram as createCloudflarePythonProgramForTest } from '../cloudflare/CloudflareProgrammaticToolCalling';
import {
  createProgrammaticToolCallingTool,
  wrapPythonForPlainExecution,
} from '../ProgrammaticToolCalling';
import {
  deriveReferencedToolDefs,
  findNormalizedIdentifierCollision,
  stripNonCodeText,
} from '../toolIdentifiers';
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

  it('ignores a python name that only appears in a comment', () => {
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

  it('surfaces every colliding definition rather than picking by order', () => {
    expect(
      deriveReferencedToolDefs(colliding, 'report_tool \'{}\'', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['report-tool', 'report_tool']);
  });

  it('reports the pair that collapses to one identifier', () => {
    expect(findNormalizedIdentifierCollision(colliding, 'bash')).toEqual({
      first: 'report-tool',
      second: 'report_tool',
      identifier: 'report_tool',
    });
  });

  it('rejects an ambiguous derived selection', () => {
    expect(() =>
      selectProgrammaticTools({
        code: 'report_tool \'{}\'',
        runtime: 'bash',
        allowedToolDefs: colliding,
        disallowedToolDefs: [{ name: 'send_email' }],
        programmaticToolName: 'run_tools_with_bash',
      })
    ).toThrow('cannot tell which one the code means');
  });

  it('rejects an ambiguous explicit manifest the same way', () => {
    expect(() =>
      selectProgrammaticTools({
        code: 'report_tool \'{}\'',
        runtime: 'bash',
        requestedToolNames: ['report-tool', 'report_tool'],
        allowedToolDefs: colliding,
        programmaticToolName: 'run_tools_with_bash',
      })
    ).toThrow('cannot tell which one the code means');
  });

  it('accepts an unambiguous manifest naming one of the pair', () => {
    expect(
      selectProgrammaticTools({
        code: 'report_tool \'{}\'',
        runtime: 'bash',
        requestedToolNames: ['report-tool'],
        allowedToolDefs: colliding,
        programmaticToolName: 'run_tools_with_bash',
      }).map((def) => def.name)
    ).toEqual(['report-tool']);
  });

  it('does not widen selection for an unreferenced collision', () => {
    expect(deriveReferencedToolDefs(colliding, 'ls -la', 'bash')).toEqual([]);
  });
});

describe('prose is not mistaken for a call', () => {
  const colliding: t.LCTool[] = [
    { name: 'write_file', allowed_callers: ['code_execution'] },
  ];

  it('ignores python comments and string literals', () => {
    expect(
      deriveReferencedToolDefs(colliding, 'print("write_file(")', 'python')
    ).toEqual([]);
    expect(
      deriveReferencedToolDefs(colliding, '# calls write_file(x)', 'python')
    ).toEqual([]);
  });

  it('ignores bash comments and quoted text', () => {
    expect(
      deriveReferencedToolDefs(colliding, 'echo \'write_file\'', 'bash')
    ).toEqual([]);
    expect(
      deriveReferencedToolDefs(colliding, '# write_file here', 'bash')
    ).toEqual([]);
  });

  it('still matches a real invocation', () => {
    expect(
      deriveReferencedToolDefs(colliding, 'write_file(path="a")', 'python').map(
        (def) => def.name
      )
    ).toEqual(['write_file']);
  });

  it('keeps code that follows a docstring', () => {
    expect(stripNonCodeText('"""doc"""\nwrite_file(1)', 'python')).toContain(
      'write_file(1)'
    );
  });
});

describe('interpolating constructs stay visible to derivation', () => {
  const tools: t.LCTool[] = [
    { name: 'calculator', allowed_callers: ['code_execution'] },
  ];

  it('keeps a python f-string expression', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'print(f"{await calculator(1)}")',
        'python'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('keeps a bash command substitution', () => {
    expect(
      deriveReferencedToolDefs(tools, 'echo "$(calculator \'{}\')"', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });

  it('still blanks a non-interpolating python literal', () => {
    expect(stripNonCodeText('x = b"calculator("', 'python')).not.toContain(
      'calculator'
    );
  });

  it('still blanks bash single quotes and comments', () => {
    expect(stripNonCodeText('echo \'calculator\'', 'bash')).not.toContain(
      'calculator'
    );
    expect(stripNonCodeText('ls # calculator', 'bash')).not.toContain(
      'calculator'
    );
  });
});

describe('timeout on the plain route', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('forwards the clamped timeout so it applies once /exec honors it', async () => {
    await invokeBash({ code: 'ls /mnt/data', tool_manifest: [], timeout: 5000 });

    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
    expect(requests[0].body.timeout).toBe(5000);
  });
});

describe('f-string literals are prose, replacement fields are code', () => {
  const tools: t.LCTool[] = [
    { name: 'write_file', allowed_callers: ['code_execution'] },
  ];

  it('ignores a tool name in the literal text of an f-string', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'print(f"example: write_file(path)")',
        'python'
      )
    ).toEqual([]);
  });

  it('still sees a call inside a replacement field', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'print(f"{await write_file(path=1)}")',
        'python'
      ).map((def) => def.name)
    ).toEqual(['write_file']);
  });

  it('treats doubled braces as literal text', () => {
    expect(
      deriveReferencedToolDefs(tools, 'print(f"{{write_file(x)}}")', 'python')
    ).toEqual([]);
  });
});

describe('nested replacement fields', () => {
  const tools: t.LCTool[] = [
    { name: 'calculator', allowed_callers: ['code_execution'] },
  ];

  it('keeps a call whose arguments contain a dict literal', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'print(f"{await calculator(payload={\'x\': 1})}")',
        'python'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('keeps a call beside a separate replacement field', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'print(f"{name}: {await calculator({\'a\': [1, 2]})}")',
        'python'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('keeps an unterminated field rather than dropping a call', () => {
    expect(
      deriveReferencedToolDefs(tools, 'x = f"{calculator(1)"', 'python').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });

  it('still blanks doubled braces around prose', () => {
    expect(
      deriveReferencedToolDefs(tools, 'print(f"{{calculator(x)}}")', 'python')
    ).toEqual([]);
  });
});

describe('cloudflare python withholds unselected native tools', () => {
  const build = (names: string[]): string =>
    createCloudflarePythonProgramForTest(
      'print(1)',
      names.map((name) => ({ name })),
      { shell: 'bash' } as never,
      '/workspace'
    );

  it('pops every native tool the selection does not include', () => {
    const program = build(['read_file']);

    expect(program).toContain('globals().pop(_lc_withheld, None)');
    expect(program).toContain('"write_file"');
    expect(program).not.toMatch(/_lc_withheld in \[[^\]]*"read_file"/);
  });

  it('withholds all of them when nothing is selected', () => {
    const program = build([]);

    expect(program).toContain('"read_file"');
    expect(program).toContain('"execute_code"');
  });

  it('emits no withdrawal when every native tool is selected', () => {
    const program = build([
      'read_file',
      'write_file',
      'edit_file',
      'grep_search',
      'glob_search',
      'list_directory',
      'compile_check',
      'bash_tool',
      'execute_code',
    ]);

    expect(program).not.toContain('_lc_withheld');
  });
});

describe('quoting inside scanned regions', () => {
  const tools: t.LCTool[] = [
    { name: 'calculator', allowed_callers: ['code_execution'] },
  ];

  it('keeps a call after a quoted brace in a replacement field', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'print(f"{\'}\' + await calculator()}")',
        'python'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('keeps a command substitution after a literal hash in double quotes', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'echo "label # $(calculator \'{}\')"',
        'bash'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('still treats a real bash comment as prose', () => {
    expect(deriveReferencedToolDefs(tools, 'ls # calculator', 'bash')).toEqual(
      []
    );
    expect(
      deriveReferencedToolDefs(tools, 'echo \'calculator\'', 'bash')
    ).toEqual([]);
  });

  it('does not treat a hash inside a word as a comment', () => {
    expect(
      deriveReferencedToolDefs(tools, 'calculator#1', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });
});

describe('bash comments after control operators', () => {
  const tools: t.LCTool[] = [
    { name: 'write_file', allowed_callers: ['code_execution'] },
  ];

  it('treats a comment straight after a semicolon as prose', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'echo ok;# write_file is unavailable',
        'bash'
      )
    ).toEqual([]);
  });

  it('treats a comment after a pipe or ampersand as prose', () => {
    expect(
      deriveReferencedToolDefs(tools, 'echo ok &# write_file here', 'bash')
    ).toEqual([]);
    expect(
      deriveReferencedToolDefs(tools, 'echo ok |# write_file here', 'bash')
    ).toEqual([]);
  });

  it('still keeps a call that follows a control operator', () => {
    expect(
      deriveReferencedToolDefs(tools, 'echo ok; write_file \'{}\'', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['write_file']);
  });
});

describe('single-quoted words in command position', () => {
  const tools: t.LCTool[] = [
    { name: 'calculator', allowed_callers: ['code_execution'] },
  ];

  it('keeps a quoted command name', () => {
    expect(
      deriveReferencedToolDefs(tools, '\'calculator\' \'{}\'', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });

  it('keeps a quoted command name after a separator', () => {
    expect(
      deriveReferencedToolDefs(tools, 'echo ok; \'calculator\' \'{}\'', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });

  it('still treats a quoted argument as prose', () => {
    expect(
      deriveReferencedToolDefs(tools, 'echo \'calculator\'', 'bash')
    ).toEqual([]);
  });
});

describe('an injected empty tool context still runs tool-free code', () => {
  beforeEach(() => {
    requests.length = 0;
  });

  it('routes to plain exec when ToolNode injects no nested tools', async () => {
    await createBashProgrammaticToolCallingTool({ baseUrl: BASE_URL }).invoke(
      {
        name: 'run_tools_with_bash',
        args: { code: 'ls /mnt/data' },
        id: 'call-1',
        type: 'tool_call',
        toolMap: new Map(),
        toolDefs: [],
      } as never,
      {} as never
    );

    expect(requests[0].url).toBe(`${BASE_URL}/exec`);
  });
});

describe('assignment prefixes keep command position', () => {
  const tools: t.LCTool[] = [
    { name: 'calculator', allowed_callers: ['code_execution'] },
  ];

  it('keeps a quoted command after an assignment prefix', () => {
    expect(
      deriveReferencedToolDefs(tools, 'FOO=bar \'calculator\' \'{}\'', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });

  it('keeps an unquoted command after several prefixes', () => {
    expect(
      deriveReferencedToolDefs(tools, 'A=1 B=2 calculator \'{}\'', 'bash').map(
        (def) => def.name
      )
    ).toEqual(['calculator']);
  });

  it('does not treat a later quoted argument as a command', () => {
    expect(
      deriveReferencedToolDefs(tools, 'FOO=bar echo \'calculator\'', 'bash')
    ).toEqual([]);
  });
});

describe('aliased and higher-order tool references', () => {
  const tools: t.LCTool[] = [
    { name: 'calculator', allowed_callers: ['code_execution'] },
  ];

  it('selects a tool bound to another name', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'fn = calculator\nawait fn(a=1)',
        'python'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('selects a tool passed to a higher-order call', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'results = [await f(a=1) for f in (calculator,)]',
        'python'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('still ignores the name in a comment or literal', () => {
    expect(
      deriveReferencedToolDefs(tools, '# calculator unused', 'python')
    ).toEqual([]);
    expect(
      deriveReferencedToolDefs(tools, 'x = "calculator"', 'python')
    ).toEqual([]);
  });
});

describe('literals inside replacement fields are data', () => {
  const tools: t.LCTool[] = [
    { name: 'write_file', allowed_callers: ['code_execution'] },
  ];

  it('ignores a quoted tool name inside a replacement field', () => {
    expect(
      deriveReferencedToolDefs(tools, 'print(f"{\'write_file\'}")', 'python')
    ).toEqual([]);
  });

  it('still sees the call around a quoted argument', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'print(f"{await write_file(path=\'a\')}")',
        'python'
      ).map((def) => def.name)
    ).toEqual(['write_file']);
  });
});

describe('bash control keywords keep command position', () => {
  const tools: t.LCTool[] = [
    { name: 'calculator', allowed_callers: ['code_execution'] },
  ];

  it('keeps a quoted command after if', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'if \'calculator\' \'{}\'; then echo ok; fi',
        'bash'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('keeps a quoted command after while and do', () => {
    expect(
      deriveReferencedToolDefs(
        tools,
        'while \'calculator\' \'{}\'; do echo ok; done',
        'bash'
      ).map((def) => def.name)
    ).toEqual(['calculator']);
  });

  it('still treats a quoted argument after a keyword as prose', () => {
    expect(
      deriveReferencedToolDefs(tools, 'if echo \'calculator\'; then :; fi', 'bash')
    ).toEqual([]);
  });
});
