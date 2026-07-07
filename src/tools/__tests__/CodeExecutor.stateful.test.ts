import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { RunnableConfig } from '@langchain/core/runnables';

/* CodeExecutor imports the default export of node-fetch; capture POST bodies. */
const fetchCalls: Array<Record<string, unknown>> = [];
jest.mock('node-fetch', () => ({
  __esModule: true,
  default: async (_url: string, init?: { body?: string }) => {
    fetchCalls.push(JSON.parse(init?.body ?? '{}'));
    return {
      ok: true,
      json: async () => ({
        session_id: 's1',
        stdout: 'ok',
        stderr: '',
        files: [],
      }),
    };
  },
}));

import {
  CODE_ARTIFACT_PATH_GUIDANCE,
  CodeExecutionToolDescription,
  StatefulCodeExecutionToolDescription,
  buildCodeExecutionToolDescription,
  buildCodeExecutionToolSchema,
  createCodeExecutionTool,
} from '../CodeExecutor';

describe('CodeExecutor stateful description', () => {
  it('selects the stateful description only when statefulSessions is on', () => {
    expect(buildCodeExecutionToolDescription()).toBe(
      CodeExecutionToolDescription
    );
    expect(buildCodeExecutionToolDescription({ statefulSessions: false })).toBe(
      CodeExecutionToolDescription
    );
    expect(buildCodeExecutionToolDescription({ statefulSessions: true })).toBe(
      StatefulCodeExecutionToolDescription
    );
  });

  it('hedges the stateful wording and keeps /mnt/data as the durable store', () => {
    const d = StatefulCodeExecutionToolDescription;
    expect(d).toContain('usually share one runtime');
    expect(d).toContain('may be reset at any time');
    expect(d).toContain('MUST be written to /mnt/data');
    expect(d).toContain(CODE_ARTIFACT_PATH_GUIDANCE);
  });

  it('adjusts the code-param note per mode', () => {
    const stateless =
      buildCodeExecutionToolSchema().properties.code.description;
    const stateful = buildCodeExecutionToolSchema({ statefulSessions: true })
      .properties.code.description;
    expect(stateless).toContain('variables and imports don\'t persist');
    expect(stateful).toContain('typically still defined');
    expect(stateful).toContain('may reset between calls');
  });
});

describe('CodeExecutor runtime_session_hint wire field', () => {
  afterEach(() => {
    fetchCalls.length = 0;
  });

  async function run(
    toolCall: Record<string, unknown>,
    params: Record<string, unknown> = {}
  ): Promise<Record<string, unknown>[]> {
    const codeTool = createCodeExecutionTool(params);
    await codeTool.invoke({ lang: 'py', code: 'print(1)' }, {
      toolCall,
    } as unknown as RunnableConfig);
    return fetchCalls;
  }

  it('sends runtime_session_hint when ToolNode injected it', async () => {
    const calls = await run({ _runtime_session_hint: 'conv-42' });
    expect(calls[0].runtime_session_hint).toBe('conv-42');
  });

  it('omits runtime_session_hint when not injected', async () => {
    const calls = await run({});
    expect('runtime_session_hint' in calls[0]).toBe(false);
  });

  it('does not leak the statefulSessions factory flag onto the wire', async () => {
    const calls = await run({}, { statefulSessions: true });
    expect('statefulSessions' in calls[0]).toBe(false);
  });

  it('strips a model-supplied runtime_session_hint from the raw args', async () => {
    const codeTool = createCodeExecutionTool({});
    await codeTool.invoke(
      { lang: 'py', code: 'print(1)', runtime_session_hint: 'model-picked' },
      { toolCall: {} } as unknown as RunnableConfig
    );
    expect('runtime_session_hint' in fetchCalls[0]).toBe(false);
  });

  it('lets only the ToolNode-injected hint reach the wire, ignoring model args', async () => {
    const codeTool = createCodeExecutionTool({});
    await codeTool.invoke(
      { lang: 'py', code: 'print(1)', runtime_session_hint: 'model-picked' },
      {
        toolCall: { _runtime_session_hint: 'host-conv' },
      } as unknown as RunnableConfig
    );
    expect(fetchCalls[0].runtime_session_hint).toBe('host-conv');
  });
});
