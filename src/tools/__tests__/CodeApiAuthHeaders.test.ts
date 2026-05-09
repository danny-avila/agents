import fetch from 'node-fetch';
import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import type { RequestInit } from 'node-fetch';
import type * as t from '@/types';
import {
  createCodeExecutionTool,
  resolveCodeApiAuthHeaders,
} from '../CodeExecutor';
import { createBashExecutionTool } from '../BashExecutor';
import {
  createProgrammaticToolCallingTool,
  fetchSessionFiles,
  makeRequest,
} from '../ProgrammaticToolCalling';
import { createBashProgrammaticToolCallingTool } from '../BashProgrammaticToolCalling';

jest.mock('node-fetch', () => ({
  __esModule: true,
  default: jest.fn(),
}));

type FetchMock = jest.MockedFunction<
  (url: unknown, init?: unknown) => Promise<unknown>
>;

const fetchMock = fetch as unknown as FetchMock;

function jsonResponse(body: unknown): unknown {
  return {
    ok: true,
    json: jest.fn(async () => body),
    text: jest.fn(async () => JSON.stringify(body)),
  };
}

function completedResponse(stdout = 'ok'): unknown {
  return jsonResponse({
    status: 'completed',
    session_id: 'session_123',
    stdout,
  });
}

function errorResponse(status: number, body: string): unknown {
  return {
    ok: false,
    status,
    text: jest.fn(async () => body),
  };
}

const toolDefs = [
  {
    name: 'lookup_user',
    description: 'Lookup a user',
    parameters: {
      type: 'object',
      properties: {},
    },
  },
] as unknown as t.LCTool[];

function toolMap(): t.ToolMap {
  return new Map([
    [
      'lookup_user',
      {
        name: 'lookup_user',
        invoke: jest.fn(async () => ({ id: 'user_123' })),
      },
    ],
  ]) as unknown as t.ToolMap;
}

describe('CodeAPI auth header injection', () => {
  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(completedResponse());
  });

  it('resolves static and dynamic auth header params', async () => {
    await expect(
      resolveCodeApiAuthHeaders({ Authorization: 'Bearer static' })
    ).resolves.toEqual({
      Authorization: 'Bearer static',
    });
    await expect(
      resolveCodeApiAuthHeaders(async () => ({
        Authorization: 'Bearer dynamic',
      }))
    ).resolves.toEqual({
      Authorization: 'Bearer dynamic',
    });
    await expect(resolveCodeApiAuthHeaders()).resolves.toEqual({});
  });

  it('keeps the no-auth request path unchanged', async () => {
    await makeRequest('https://code.example.com/exec/programmatic', {
      code: 'print(1)',
    });

    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/exec/programmatic',
      expect.objectContaining({
        headers: expect.not.objectContaining({
          Authorization: expect.any(String),
        }),
      })
    );
  });

  it('forwards Authorization for direct code execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ session_id: 'session_123', stdout: '1\n' })
    );
    const tool = createCodeExecutionTool({
      authHeaders: async () => ({ Authorization: 'Bearer code-token' }),
    });

    await tool.invoke({ lang: 'py', code: 'print(1)' });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer code-token',
        }),
      })
    );
    expect(
      JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)
    ).not.toHaveProperty('authHeaders');
  });

  it('forwards Authorization for bash execution', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ session_id: 'session_123', stdout: '1\n' })
    );
    const tool = createBashExecutionTool({
      authHeaders: { Authorization: 'Bearer bash-token' },
    });

    await tool.invoke({ command: 'echo 1' });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer bash-token',
        }),
      })
    );
    expect(
      JSON.parse((fetchMock.mock.calls[0]?.[1] as RequestInit).body as string)
    ).not.toHaveProperty('authHeaders');
  });

  it('includes the CodeAPI endpoint and response body on direct execution failures', async () => {
    fetchMock.mockResolvedValueOnce(errorResponse(404, 'Cannot POST /exec'));
    const tool = createBashExecutionTool();

    await expect(tool.invoke({ command: 'echo 1' })).rejects.toThrow(
      /CodeAPI request failed: POST .*\/exec returned 404, body: Cannot POST \/exec/
    );
  });

  it('forwards Authorization on programmatic initial and continuation requests', async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          status: 'tool_call_required',
          continuation_token: 'continue_123',
          tool_calls: [{ id: 'call_1', name: 'lookup_user', input: {} }],
        })
      )
      .mockResolvedValueOnce(completedResponse('done'));

    const tool = createProgrammaticToolCallingTool({
      authHeaders: () => ({ Authorization: 'Bearer ptc-token' }),
    });

    await tool.invoke(
      { code: 'result = await lookup_user()\nprint(result)' },
      {
        toolCall: {
          name: 'programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(fetchMock).toHaveBeenCalledTimes(2);
    for (const call of fetchMock.mock.calls) {
      expect(call[1]).toEqual(
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer ptc-token',
          }),
        })
      );
    }
  });

  it('forwards Authorization for bash programmatic requests', async () => {
    const tool = createBashProgrammaticToolCallingTool({
      authHeaders: { Authorization: 'Bearer bash-ptc-token' },
    });

    await tool.invoke(
      { code: 'lookup_user "{}"' },
      {
        toolCall: {
          name: 'bash_programmatic_code_execution',
          args: {},
          toolMap: toolMap(),
          toolDefs,
        },
      }
    );

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer bash-ptc-token',
        }),
      })
    );
  });

  it('fetches session files with the CodeAPI resource scope and auth headers', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse([
        {
          id: 'file-1',
          resource_id: 'skill-1',
          storage_session_id: 'session_123',
          name: 'skill/file.txt',
          kind: 'skill',
          version: 7,
        },
      ])
    );

    const files = await fetchSessionFiles(
      'https://code.example.com',
      'session_123',
      { kind: 'skill', id: 'skill-1', version: 7 },
      undefined,
      { Authorization: 'Bearer files-token' }
    );

    expect(files).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/files/session_123?detail=full&kind=skill&id=skill-1&version=7',
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer files-token',
        }),
      })
    );
  });

  it('fetches scoped session files with auth headers and no proxy placeholder', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    await fetchSessionFiles(
      'https://code.example.com',
      'session_123',
      { kind: 'skill', id: 'skill-1', version: 7 },
      { Authorization: 'Bearer scoped-files-token' }
    );

    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/files/session_123?detail=full&kind=skill&id=skill-1&version=7',
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer scoped-files-token',
        }),
      })
    );
  });

  it('preserves the legacy fetchSessionFiles proxy/auth argument order', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse([
        {
          name: 'session_123/file-1.txt',
          metadata: { 'original-filename': 'file.txt' },
        },
      ])
    );

    const files = await fetchSessionFiles(
      'https://code.example.com',
      'session_123',
      '',
      { Authorization: 'Bearer legacy-files-token' }
    );

    expect(files).toEqual([
      {
        storage_session_id: 'session_123',
        kind: 'user',
        id: 'file-1',
        resource_id: 'file-1',
        name: 'file.txt',
      },
    ]);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://code.example.com/files/session_123?detail=full',
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: 'Bearer legacy-files-token',
        }),
      })
    );
  });
});
