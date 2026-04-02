import { describe, it, expect, beforeEach, jest } from '@jest/globals';

/**
 * Mock node-fetch so we can intercept the POST to /exec
 * and inspect the request body without hitting any real server.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockFetch = jest.fn<any>();
jest.mock('node-fetch', () => ({
  __esModule: true,
  default: mockFetch,
}));

/**
 * Stub the env-variable helper so createCodeExecutionTool picks up an API key
 * and a deterministic base URL without touching real environment variables.
 */
jest.mock('@langchain/core/utils/env', () => ({
  getEnvironmentVariable: (key: string): string | undefined => {
    if (key === 'LIBRECHAT_CODE_API_KEY') {
      return 'test-api-key';
    }
    if (key === 'LIBRECHAT_CODE_BASEURL') {
      return 'https://mock-code-api.test/v1';
    }
    return undefined;
  },
}));

import { createCodeExecutionTool } from '../CodeExecutor';

/** Helper: build a fake successful /exec response */
function fakeExecResponse(overrides: Record<string, unknown> = {}): {
  ok: boolean;
  json: () => Promise<Record<string, unknown>>;
} {
  return {
    ok: true,
    json: async (): Promise<Record<string, unknown>> => ({
      session_id: 'resp-session-1',
      stdout: 'hello\n',
      stderr: '',
      ...overrides,
    }),
  };
}

/**
 * Extract the parsed JSON body that was sent to /exec from the mock call.
 */
function capturedPostBody(callIndex: number = 0): Record<string, unknown> {
  const call = mockFetch.mock.calls[callIndex] as [string, { body: string }];
  return JSON.parse(call[1].body) as Record<string, unknown>;
}

describe('CodeExecutor params.files fallback priority', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('includes params.files in POST when toolCall has neither session_id nor _injected_files', async () => {
    mockFetch.mockResolvedValueOnce(fakeExecResponse());

    const paramsFiles = [
      { id: 'pf1', name: 'primed.csv', session_id: 'upload-sess' },
    ];

    const codeTool = createCodeExecutionTool({ files: paramsFiles });

    // Invoke with a config whose toolCall carries no session context
    await codeTool.invoke({ lang: 'py', code: 'print("hi")' }, {
      toolCall: { id: 'call_1', name: 'execute_code', args: {} },
    } as never);

    const body = capturedPostBody();
    expect(body.files).toEqual(paramsFiles);
  });

  it('_injected_files takes priority over params.files', async () => {
    mockFetch.mockResolvedValueOnce(fakeExecResponse());

    const paramsFiles = [
      { id: 'pf1', name: 'primed.csv', session_id: 'upload-sess' },
    ];
    const injectedFiles = [
      { id: 'if1', name: 'session-file.csv', session_id: 'prev-sess' },
    ];

    const codeTool = createCodeExecutionTool({ files: paramsFiles });

    await codeTool.invoke({ lang: 'py', code: 'print("hi")' }, {
      toolCall: {
        id: 'call_2',
        name: 'execute_code',
        args: {},
        _injected_files: injectedFiles,
      },
    } as never);

    const body = capturedPostBody();
    expect(body.files).toEqual(injectedFiles);
    expect(body.files).not.toEqual(paramsFiles);
  });

  it('session_id fetch fallback is used when params.files is absent', async () => {
    // First call: GET /files/<session_id> (the fallback fetch)
    const fileFetchResponse = {
      ok: true,
      json: async (): Promise<Record<string, unknown>[]> => [
        {
          name: 'sess-abc/f1.csv',
          metadata: { 'original-filename': 'data.csv' },
        },
      ],
    };

    // Second call: POST /exec
    mockFetch
      .mockResolvedValueOnce(fileFetchResponse)
      .mockResolvedValueOnce(fakeExecResponse());

    // No params.files provided
    const codeTool = createCodeExecutionTool({});

    await codeTool.invoke({ lang: 'py', code: 'print("hi")' }, {
      toolCall: {
        id: 'call_3',
        name: 'execute_code',
        args: {},
        session_id: 'sess-abc',
      },
    } as never);

    // Should have made two fetch calls: GET /files then POST /exec
    expect(mockFetch).toHaveBeenCalledTimes(2);

    const body = capturedPostBody(1);
    expect(body.files).toEqual([
      { session_id: 'sess-abc', id: 'f1', name: 'data.csv' },
    ]);
  });

  it('does not set files when none are available from any source', async () => {
    mockFetch.mockResolvedValueOnce(fakeExecResponse());

    const codeTool = createCodeExecutionTool({});

    await codeTool.invoke({ lang: 'py', code: 'print("hi")' }, {
      toolCall: { id: 'call_4', name: 'execute_code', args: {} },
    } as never);

    const body = capturedPostBody();
    expect(body.files).toBeUndefined();
  });
});
