import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { ToolNode } from '../ToolNode';
import { Constants } from '@/common';

/**
 * Creates a mock execute_code tool that captures the toolCall config it receives.
 * Returns a content_and_artifact response with configurable session/files.
 */
function createMockCodeTool(options: {
  capturedConfigs: Record<string, unknown>[];
  artifact?: t.CodeExecutionArtifact;
}): StructuredToolInterface {
  const { capturedConfigs, artifact } = options;
  const defaultArtifact: t.CodeExecutionArtifact = {
    session_id: 'new-session-123',
    files: [],
  };

  return tool(
    async (_input, config) => {
      capturedConfigs.push({ ...(config.toolCall ?? {}) });
      return ['stdout:\nhello world\n', artifact ?? defaultArtifact];
    },
    {
      name: Constants.EXECUTE_CODE,
      description: 'Execute code in a sandbox',
      schema: z.object({ lang: z.string(), code: z.string() }),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  ) as unknown as StructuredToolInterface;
}

function createAIMessageWithCodeCall(callId: string): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [
      {
        id: callId,
        name: Constants.EXECUTE_CODE,
        args: { lang: 'python', code: 'print("hello")' },
      },
    ],
  });
}

describe('ToolNode code execution session management', () => {
  describe('session injection via runTool (direct execution)', () => {
    it('injects session_id and _injected_files when session has files', async () => {
      const capturedConfigs: Record<string, unknown>[] = [];
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'prev-session-abc',
        files: [
          { id: 'file1', name: 'data.csv', session_id: 'prev-session-abc' },
          { id: 'file2', name: 'chart.png', session_id: 'prev-session-abc' },
        ],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs });
      const toolNode = new ToolNode({ tools: [mockTool], sessions });

      const aiMsg = createAIMessageWithCodeCall('call_1');
      await toolNode.invoke({ messages: [aiMsg] });

      expect(capturedConfigs).toHaveLength(1);
      expect(capturedConfigs[0].session_id).toBe('prev-session-abc');
      expect(capturedConfigs[0]._injected_files).toEqual([
        { session_id: 'prev-session-abc', id: 'file1', name: 'data.csv' },
        { session_id: 'prev-session-abc', id: 'file2', name: 'chart.png' },
      ]);
    });

    it('injects session_id even when session has no tracked files', async () => {
      const capturedConfigs: Record<string, unknown>[] = [];
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'prev-session-no-files',
        files: [],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs });
      const toolNode = new ToolNode({ tools: [mockTool], sessions });

      const aiMsg = createAIMessageWithCodeCall('call_2');
      await toolNode.invoke({ messages: [aiMsg] });

      expect(capturedConfigs).toHaveLength(1);
      expect(capturedConfigs[0].session_id).toBe('prev-session-no-files');
      expect(capturedConfigs[0]._injected_files).toBeUndefined();
    });

    it('does not inject session context when no session exists', async () => {
      const capturedConfigs: Record<string, unknown>[] = [];
      const sessions: t.ToolSessionMap = new Map();

      const mockTool = createMockCodeTool({ capturedConfigs });
      const toolNode = new ToolNode({ tools: [mockTool], sessions });

      const aiMsg = createAIMessageWithCodeCall('call_3');
      await toolNode.invoke({ messages: [aiMsg] });

      expect(capturedConfigs).toHaveLength(1);
      expect(capturedConfigs[0].session_id).toBeUndefined();
      expect(capturedConfigs[0]._injected_files).toBeUndefined();
    });

    it('preserves per-file session_id for multi-session files', async () => {
      const capturedConfigs: Record<string, unknown>[] = [];
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'session-B',
        files: [
          { id: 'f1', name: 'old.csv', session_id: 'session-A' },
          { id: 'f2', name: 'new.png', session_id: 'session-B' },
        ],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs });
      const toolNode = new ToolNode({ tools: [mockTool], sessions });

      const aiMsg = createAIMessageWithCodeCall('call_4');
      await toolNode.invoke({ messages: [aiMsg] });

      const files = capturedConfigs[0]._injected_files as t.CodeEnvFile[];
      expect(files[0].session_id).toBe('session-A');
      expect(files[1].session_id).toBe('session-B');
    });
  });

  describe('getCodeSessionContext (via dispatchToolEvents request building)', () => {
    it('builds session context with files for event-driven requests', () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'evt-session',
        files: [{ id: 'ef1', name: 'out.parquet', session_id: 'evt-session' }],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const context = (
        toolNode as unknown as { getCodeSessionContext: () => unknown }
      ).getCodeSessionContext();

      expect(context).toEqual({
        session_id: 'evt-session',
        files: [{ session_id: 'evt-session', id: 'ef1', name: 'out.parquet' }],
      });
    });

    it('builds session context without files when session has no tracked files', () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'evt-session-empty',
        files: [],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const context = (
        toolNode as unknown as { getCodeSessionContext: () => unknown }
      ).getCodeSessionContext();

      expect(context).toEqual({ session_id: 'evt-session-empty' });
    });

    it('returns undefined when no session exists', () => {
      const sessions: t.ToolSessionMap = new Map();

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const context = (
        toolNode as unknown as { getCodeSessionContext: () => unknown }
      ).getCodeSessionContext();

      expect(context).toBeUndefined();
    });
  });

  describe('storeCodeSessionFromResults (session storage from artifacts)', () => {
    it('stores session with files from code execution results', () => {
      const sessions: t.ToolSessionMap = new Map();

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const storeMethod = (
        toolNode as unknown as {
          storeCodeSessionFromResults: (
            results: t.ToolExecuteResult[],
            requests: t.ToolCallRequest[]
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc1',
            content: 'output',
            artifact: {
              session_id: 'new-sess',
              files: [{ id: 'f1', name: 'result.csv' }],
            },
            status: 'success',
          },
        ],
        [{ id: 'tc1', name: Constants.EXECUTE_CODE, args: {} }]
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      expect(stored).toBeDefined();
      expect(stored.session_id).toBe('new-sess');
      expect(stored.files).toHaveLength(1);
      expect(stored.files![0]).toEqual(
        expect.objectContaining({
          id: 'f1',
          name: 'result.csv',
          session_id: 'new-sess',
        })
      );
    });

    it('stores session_id even when Code API returns no files', () => {
      const sessions: t.ToolSessionMap = new Map();

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const storeMethod = (
        toolNode as unknown as {
          storeCodeSessionFromResults: (
            results: t.ToolExecuteResult[],
            requests: t.ToolCallRequest[]
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc2',
            content: 'stdout:\nSaved parquet\n',
            artifact: { session_id: 'parquet-session', files: [] },
            status: 'success',
          },
        ],
        [{ id: 'tc2', name: Constants.EXECUTE_CODE, args: {} }]
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      expect(stored).toBeDefined();
      expect(stored.session_id).toBe('parquet-session');
      expect(stored.files).toEqual([]);
    });

    it('merges new files with existing session, replacing same-name files', () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'old-sess',
        files: [
          { id: 'f1', name: 'data.csv', session_id: 'old-sess' },
          { id: 'f2', name: 'chart.png', session_id: 'old-sess' },
        ],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const storeMethod = (
        toolNode as unknown as {
          storeCodeSessionFromResults: (
            results: t.ToolExecuteResult[],
            requests: t.ToolCallRequest[]
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc3',
            content: 'output',
            artifact: {
              session_id: 'new-sess',
              files: [{ id: 'f3', name: 'chart.png' }],
            },
            status: 'success',
          },
        ],
        [{ id: 'tc3', name: Constants.EXECUTE_CODE, args: {} }]
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      expect(stored.session_id).toBe('new-sess');
      expect(stored.files).toHaveLength(2);

      const csvFile = stored.files!.find((f) => f.name === 'data.csv');
      expect(csvFile!.session_id).toBe('old-sess');

      const chartFile = stored.files!.find((f) => f.name === 'chart.png');
      expect(chartFile!.id).toBe('f3');
      expect(chartFile!.session_id).toBe('new-sess');
    });

    it('preserves existing files when new execution has no files', () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'old-sess',
        files: [{ id: 'f1', name: 'data.csv', session_id: 'old-sess' }],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const storeMethod = (
        toolNode as unknown as {
          storeCodeSessionFromResults: (
            results: t.ToolExecuteResult[],
            requests: t.ToolCallRequest[]
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc4',
            content: 'stdout:\nno files generated\n',
            artifact: { session_id: 'new-sess', files: [] },
            status: 'success',
          },
        ],
        [{ id: 'tc4', name: Constants.EXECUTE_CODE, args: {} }]
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      expect(stored.session_id).toBe('new-sess');
      expect(stored.files).toHaveLength(1);
      expect(stored.files![0].name).toBe('data.csv');
    });

    it('ignores non-code-execution tool results', () => {
      const sessions: t.ToolSessionMap = new Map();

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const storeMethod = (
        toolNode as unknown as {
          storeCodeSessionFromResults: (
            results: t.ToolExecuteResult[],
            requests: t.ToolCallRequest[]
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc5',
            content: 'search results',
            artifact: { session_id: 'should-not-store' },
            status: 'success',
          },
        ],
        [{ id: 'tc5', name: 'web_search', args: {} }]
      );

      expect(sessions.has(Constants.EXECUTE_CODE)).toBe(false);
    });

    it('ignores error results', () => {
      const sessions: t.ToolSessionMap = new Map();

      const mockTool = createMockCodeTool({ capturedConfigs: [] });
      const toolNode = new ToolNode({
        tools: [mockTool],
        sessions,
        eventDrivenMode: true,
      });

      const storeMethod = (
        toolNode as unknown as {
          storeCodeSessionFromResults: (
            results: t.ToolExecuteResult[],
            requests: t.ToolCallRequest[]
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc6',
            content: '',
            artifact: {
              session_id: 'error-session',
              files: [{ id: 'f1', name: 'x' }],
            },
            status: 'error',
            errorMessage: 'execution failed',
          },
        ],
        [{ id: 'tc6', name: Constants.EXECUTE_CODE, args: {} }]
      );

      expect(sessions.has(Constants.EXECUTE_CODE)).toBe(false);
    });
  });
});
