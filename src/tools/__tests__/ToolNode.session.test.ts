import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type * as t from '@/types';
import { ToolNode } from '../ToolNode';
import { Constants } from '@/common';
import * as events from '@/utils/events';

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
    it('injects session ids (both names) and _injected_files when session has files', async () => {
      const capturedConfigs: Record<string, unknown>[] = [];
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'prev-session-abc',
        files: [
          {
            id: 'file1',
            name: 'data.csv',
            storage_session_id: 'prev-session-abc',
          },
          {
            id: 'file2',
            name: 'chart.png',
            storage_session_id: 'prev-session-abc',
          },
        ],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs });
      const toolNode = new ToolNode({ tools: [mockTool], sessions });

      const aiMsg = createAIMessageWithCodeCall('call_1');
      await toolNode.invoke({ messages: [aiMsg] });

      expect(capturedConfigs).toHaveLength(1);
      /* Both names injected so pre- and post-rename consumers see the
       * field they expect. */
      expect(capturedConfigs[0].session_id).toBe('prev-session-abc');
      expect(capturedConfigs[0]._injected_files).toEqual([
        {
          id: 'file1',
          /* `resource_id` defaults to `id` (the storage uuid) when
           * the input ref didn't supply it — informational for
           * `kind: 'user'` since codeapi derives sessionKey from
           * auth context. The real LC priming chain DOES supply
           * resource_id explicitly; the default is a back-compat
           * landing for inputs that haven't been updated. */
          resource_id: 'file1',
          name: 'data.csv',
          storage_session_id: 'prev-session-abc',
          kind: 'user',
        },
        {
          id: 'file2',
          resource_id: 'file2',
          name: 'chart.png',
          storage_session_id: 'prev-session-abc',
          kind: 'user',
        },
      ]);
    });

    it('injects session ids even when session has no tracked files', async () => {
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

    it('preserves per-file storage_session_id for multi-session files', async () => {
      const capturedConfigs: Record<string, unknown>[] = [];
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'session-B',
        files: [
          {
            id: 'f1',
            name: 'old.csv',
            storage_session_id: 'session-A',
          },
          {
            id: 'f2',
            name: 'new.png',
            storage_session_id: 'session-B',
          },
        ],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs });
      const toolNode = new ToolNode({ tools: [mockTool], sessions });

      const aiMsg = createAIMessageWithCodeCall('call_4');
      await toolNode.invoke({ messages: [aiMsg] });

      const files = capturedConfigs[0]._injected_files as t.CodeEnvFile[];
      expect(files[0].storage_session_id).toBe('session-A');
      expect(files[1].storage_session_id).toBe('session-B');
    });

    it('forwards per-file kind and version for mixed-kind sessions', async () => {
      const capturedConfigs: Record<string, unknown>[] = [];
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'session-A',
        files: [
          {
            id: 'skill-123',
            name: 'demo/SKILL.md',
            storage_session_id: 'session-A',
            kind: 'skill',
            version: 7,
          },
          {
            id: 'user-file',
            name: 'attachment.csv',
            storage_session_id: 'session-B',
          },
        ],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const mockTool = createMockCodeTool({ capturedConfigs });
      const toolNode = new ToolNode({ tools: [mockTool], sessions });

      const aiMsg = createAIMessageWithCodeCall('call_5');
      await toolNode.invoke({ messages: [aiMsg] });

      const files = capturedConfigs[0]._injected_files as t.CodeEnvFile[];
      expect(files).toEqual([
        {
          id: 'skill-123',
          /* Default fallback when input doesn't supply `resource_id`.
           * Production LC sets it explicitly to the skill's `_id`. */
          resource_id: 'skill-123',
          name: 'demo/SKILL.md',
          storage_session_id: 'session-A',
          kind: 'skill',
          version: 7,
        },
        {
          id: 'user-file',
          resource_id: 'user-file',
          name: 'attachment.csv',
          storage_session_id: 'session-B',
          kind: 'user',
        },
      ]);
    });
  });

  describe('getCodeSessionContext (via dispatchToolEvents request building)', () => {
    it('builds session context with files for event-driven requests', () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'evt-session',
        files: [
          {
            id: 'ef1',
            name: 'out.parquet',
            storage_session_id: 'evt-session',
          },
        ],
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
        files: [
          {
            id: 'ef1',
            /* `resource_id` defaults to `id` when input doesn't carry
             * separate provenance — see toInjectedFileRef. */
            resource_id: 'ef1',
            name: 'out.parquet',
            storage_session_id: 'evt-session',
            kind: 'user',
          },
        ],
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

      expect(context).toEqual({
        session_id: 'evt-session-empty',
      });
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

    it('forwards per-file kind and version to event-driven request context', () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'evt-session',
        files: [
          {
            id: 'skill-abc',
            name: 'demo/SKILL.md',
            storage_session_id: 'evt-session',
            kind: 'skill',
            version: 3,
          },
          {
            id: 'usr1',
            name: 'data.csv',
            storage_session_id: 'evt-session',
          },
        ],
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
        files: [
          {
            id: 'skill-abc',
            /* Default fallback when input doesn't supply `resource_id`. */
            resource_id: 'skill-abc',
            name: 'demo/SKILL.md',
            storage_session_id: 'evt-session',
            kind: 'skill',
            version: 3,
          },
          {
            id: 'usr1',
            resource_id: 'usr1',
            name: 'data.csv',
            storage_session_id: 'evt-session',
            kind: 'user',
          },
        ],
      });
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
            requestMap: Map<string, t.ToolCallRequest>
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
        new Map([
          ['tc1', { id: 'tc1', name: Constants.EXECUTE_CODE, args: {} }],
        ])
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
          storage_session_id: 'new-sess',
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
            requestMap: Map<string, t.ToolCallRequest>
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
        new Map([
          ['tc2', { id: 'tc2', name: Constants.EXECUTE_CODE, args: {} }],
        ])
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
          { id: 'f1', name: 'data.csv', storage_session_id: 'old-sess' },
          { id: 'f2', name: 'chart.png', storage_session_id: 'old-sess' },
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
            requestMap: Map<string, t.ToolCallRequest>
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
        new Map([
          ['tc3', { id: 'tc3', name: Constants.EXECUTE_CODE, args: {} }],
        ])
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      expect(stored.session_id).toBe('new-sess');
      expect(stored.files).toHaveLength(2);

      const csvFile = stored.files!.find((f) => f.name === 'data.csv');
      expect(csvFile!.storage_session_id).toBe('old-sess');

      const chartFile = stored.files!.find((f) => f.name === 'chart.png');
      expect(chartFile!.id).toBe('f3');
      expect(chartFile!.storage_session_id).toBe('new-sess');
    });

    it('preserves prior kind/resource_id/version when worker echoes inherited file (skill 403 regression)', () => {
      /**
       * Regression for the codeapi `session_key_mismatch` 403 that
       * fires on the second `/exec` after a successful skill prime.
       *
       * Worker `inherited: true` echoes carry only
       * `(id, name, storage_session_id)` — the sandbox doesn't know
       * the resource identity (`kind`, `resource_id`, `version`),
       * which was signed at upload time. If `updateCodeSession`
       * replaces the prior entry verbatim from the echo, the next
       * `_injected_files` derivation reads `kind: undefined` and
       * `toInjectedFileRef` falls back to `kind: 'user'` +
       * `resource_id: file.id`. Codeapi then resolves
       * `legacy:user:<authContext.userId>`, which doesn't match the
       * cached `legacy:skill:<skillId>:v:<v>` set at upload, and
       * authorization 403s.
       *
       * The merge must overlay the echo onto the prior entry by
       * `(storage_session_id, id)` so identity survives.
       */
      const SKILL_ID = '69dcf561f37f717858d4d072';
      const SKILL_VERSION = 59;
      const STORAGE_SESSION = 'p28pbmz0ejTZ8MMEkObN8';
      const FILE_ID = 'JqnzC4f6gpirzW0yq_obR';

      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: STORAGE_SESSION,
        files: [
          {
            id: FILE_ID,
            resource_id: SKILL_ID,
            name: 'pptx/editing.md',
            storage_session_id: STORAGE_SESSION,
            kind: 'skill',
            version: SKILL_VERSION,
          },
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
            requestMap: Map<string, t.ToolCallRequest>
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      /* Worker echo shape — exactly what codeapi sends on
       * `inherited: true` files. No kind, no resource_id, no version. */
      storeMethod(
        [
          {
            toolCallId: 'tc-skill-rerun',
            content: 'output',
            artifact: {
              session_id: 'exec-sess-2',
              files: [
                {
                  id: FILE_ID,
                  name: 'pptx/editing.md',
                  storage_session_id: STORAGE_SESSION,
                  inherited: true,
                } as unknown as t.FileRefs[number],
              ],
            },
            status: 'success',
          },
        ],
        new Map([
          [
            'tc-skill-rerun',
            { id: 'tc-skill-rerun', name: Constants.EXECUTE_CODE, args: {} },
          ],
        ])
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      const merged = stored.files!.find((f) => f.id === FILE_ID);
      expect(merged).toBeDefined();
      /* Identity preserved from prior entry: */
      expect(merged!.kind).toBe('skill');
      expect(merged!.resource_id).toBe(SKILL_ID);
      expect((merged as { version?: number }).version).toBe(SKILL_VERSION);
      /* Echo-owned fields propagated: */
      expect((merged as { inherited?: boolean }).inherited).toBe(true);
    });

    it('uses fresh kind defaults for genuinely new files (no prior entry to merge from)', () => {
      /**
       * The merge keys on `(storage_session_id, id)`, not `name`.
       * A file the worker emits for the first time (typical
       * code-output / generated artifact) has no prior to inherit
       * identity from — it lands as `kind: 'user'` via the standard
       * fallback in `toInjectedFileRef`. Locks the contract that
       * the merge doesn't accidentally re-tag user output as skill.
       */
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'old-sess',
        files: [
          {
            id: 'skill-f1',
            resource_id: 'skill-id-1',
            name: 'pptx/editing.md',
            storage_session_id: 'skill-storage',
            kind: 'skill',
            version: 7,
          },
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
            requestMap: Map<string, t.ToolCallRequest>
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc-output',
            content: 'output',
            artifact: {
              session_id: 'output-sess',
              files: [
                {
                  id: 'fresh-output-id',
                  name: 'generated-chart.png',
                  storage_session_id: 'output-sess',
                } as unknown as t.FileRefs[number],
              ],
            },
            status: 'success',
          },
        ],
        new Map([
          [
            'tc-output',
            { id: 'tc-output', name: Constants.EXECUTE_CODE, args: {} },
          ],
        ])
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      const fresh = stored.files!.find((f) => f.id === 'fresh-output-id');
      expect(fresh).toBeDefined();
      /* No prior to inherit from — kind/resource_id/version absent
       * on the entry; toInjectedFileRef will default kind to 'user'
       * downstream. */
      expect(fresh!.kind).toBeUndefined();
      expect(fresh!.resource_id).toBeUndefined();
      expect((fresh as { version?: number }).version).toBeUndefined();
      /* Skill file untouched. */
      const skillFile = stored.files!.find((f) => f.id === 'skill-f1');
      expect(skillFile!.kind).toBe('skill');
    });

    it('preserves existing files when new execution has no files', () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'old-sess',
        files: [{ id: 'f1', name: 'data.csv', storage_session_id: 'old-sess' }],
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
            requestMap: Map<string, t.ToolCallRequest>
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
        new Map([
          ['tc4', { id: 'tc4', name: Constants.EXECUTE_CODE, args: {} }],
        ])
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
            requestMap: Map<string, t.ToolCallRequest>
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc5',
            content: 'search results',
            artifact: { storage_session_id: 'should-not-store' },
            status: 'success',
          },
        ],
        new Map([['tc5', { id: 'tc5', name: 'web_search', args: {} }]])
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
            requestMap: Map<string, t.ToolCallRequest>
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
        new Map([
          ['tc6', { id: 'tc6', name: Constants.EXECUTE_CODE, args: {} }],
        ])
      );

      expect(sessions.has(Constants.EXECUTE_CODE)).toBe(false);
    });

    it('preserves per-file storage session_id (not overwritten with the exec session_id)', () => {
      /**
       * Regression: the codeapi worker reports `artifact.session_id` (EXEC
       * session — torn down post-run) and per-file `session_id` (STORAGE
       * session where the file lives). Stomping the storage id with the
       * exec id silently 404s every follow-up tool call within the same
       * run because `_injected_files` carry the wrong path on the next
       * `/exec`. The worker tries to mount `<exec_session>/<id>` against
       * file-server, gets 404, mounts nothing — `cat /mnt/data/foo.txt`
       * → "No such file or directory".
       */
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
            requestMap: Map<string, t.ToolCallRequest>
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc-storage',
            content: 'output',
            artifact: {
              /* EXEC session — transient, torn down after this run */
              session_id: 'exec-session-123',
              files: [
                /* STORAGE session — persistent file-server bucket prefix */
                {
                  id: 'f1',
                  name: 'sentinel.txt',
                  storage_session_id: 'storage-session-A',
                },
                {
                  id: 'f2',
                  name: 'data.csv',
                  storage_session_id: 'storage-session-B',
                },
              ],
            },
            status: 'success',
          },
        ],
        new Map([
          [
            'tc-storage',
            { id: 'tc-storage', name: Constants.EXECUTE_CODE, args: {} },
          ],
        ])
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      /* The session-level id is the (latest) exec id — fine for tracking
         "what session ran last" — but per-file storage ids must survive.
         After the rename, both names appear at the top level (exec) and
         on each file (storage). */
      expect(stored.session_id).toBe('exec-session-123');
      expect(stored.files).toHaveLength(2);
      expect(stored.files![0]).toEqual({
        id: 'f1',
        name: 'sentinel.txt',
        storage_session_id: 'storage-session-A',
      });
      expect(stored.files![1]).toEqual({
        id: 'f2',
        name: 'data.csv',
        storage_session_id: 'storage-session-B',
      });
    });

    it('falls back to exec session_id only when per-file session_id is absent (older worker payloads)', () => {
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
            requestMap: Map<string, t.ToolCallRequest>
          ) => void;
        }
      ).storeCodeSessionFromResults.bind(toolNode);

      storeMethod(
        [
          {
            toolCallId: 'tc-mixed',
            content: 'output',
            artifact: {
              session_id: 'exec-mixed',
              files: [
                /* Mix: one file with storage id, one without (older payload). */
                {
                  id: 'f1',
                  name: 'fresh.csv',
                  storage_session_id: 'storage-fresh',
                },
                { id: 'f2', name: 'legacy.csv' },
              ],
            },
            status: 'success',
          },
        ],
        new Map([
          [
            'tc-mixed',
            { id: 'tc-mixed', name: Constants.EXECUTE_CODE, args: {} },
          ],
        ])
      );

      const stored = sessions.get(
        Constants.EXECUTE_CODE
      ) as t.CodeSessionContext;
      expect(stored.files![0].storage_session_id).toBe('storage-fresh');
      /* Fallback only when the per-file id is missing — the fallback
       * value is the exec session id. */
      expect(stored.files![1].storage_session_id).toBe('exec-mixed');
    });
  });

  describe('codeSessionContext emission gate (event-driven request building)', () => {
    /**
     * Captures the `ToolExecuteBatchRequest` dispatched on ON_TOOL_EXECUTE so
     * we can assert which `request.name`s receive `codeSessionContext`. Returns
     * the captured requests; resolves the dispatched event with empty results
     * to let `dispatchToolEvents` complete.
     */
    function captureBatchRequests(): {
      capturedRequests: t.ToolCallRequest[];
      } {
      const capturedRequests: t.ToolCallRequest[] = [];
      jest
        .spyOn(events, 'safeDispatchCustomEvent')
        .mockImplementation(async (_event, data) => {
          const batch = data as t.ToolExecuteBatchRequest;
          if (Array.isArray(batch.toolCalls)) {
            capturedRequests.push(...batch.toolCalls);
          }
          if (typeof batch.resolve === 'function') {
            batch.resolve(
              batch.toolCalls.map((tc) => ({
                toolCallId: tc.id,
                content: '',
                status: 'success' as const,
              }))
            );
          }
        });
      return { capturedRequests };
    }

    const createDummyTool = (name: string): StructuredToolInterface =>
      tool(async () => 'ok', {
        name,
        description: 'dummy',
        schema: z.object({ x: z.string().optional() }),
      }) as unknown as StructuredToolInterface;

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('attaches codeSessionContext to read_file requests so the host can fall back to the code-env sandbox', async () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'rf-session',
        files: [
          {
            id: 'rf1',
            name: 'data.csv',
            storage_session_id: 'rf-session',
          },
        ],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const { capturedRequests } = captureBatchRequests();

      const toolNode = new ToolNode({
        tools: [createDummyTool(Constants.READ_FILE)],
        sessions,
        eventDrivenMode: true,
        toolCallStepIds: new Map([['call_rf', 'step_rf']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_rf',
            name: Constants.READ_FILE,
            args: { file_path: '/mnt/data/data.csv' },
          },
        ],
      });

      await toolNode.invoke({ messages: [aiMsg] });

      expect(capturedRequests).toHaveLength(1);
      expect(capturedRequests[0].name).toBe(Constants.READ_FILE);
      expect(capturedRequests[0].codeSessionContext).toEqual({
        session_id: 'rf-session',
        files: [
          {
            id: 'rf1',
            /* Default fallback for inputs that don't carry separate
             * provenance — see toInjectedFileRef. */
            resource_id: 'rf1',
            name: 'data.csv',
            storage_session_id: 'rf-session',
            kind: 'user',
          },
        ],
      });
    });

    it('does not attach codeSessionContext to read_file when no session exists yet', async () => {
      const { capturedRequests } = captureBatchRequests();

      const toolNode = new ToolNode({
        tools: [createDummyTool(Constants.READ_FILE)],
        sessions: new Map(),
        eventDrivenMode: true,
        toolCallStepIds: new Map([['call_rf2', 'step_rf2']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_rf2',
            name: Constants.READ_FILE,
            args: { file_path: 'some-skill/notes.md' },
          },
        ],
      });

      await toolNode.invoke({ messages: [aiMsg] });

      expect(capturedRequests).toHaveLength(1);
      expect(capturedRequests[0].name).toBe(Constants.READ_FILE);
      expect(capturedRequests[0].codeSessionContext).toBeUndefined();
    });

    it('does not attach codeSessionContext to unrelated tools', async () => {
      const sessions: t.ToolSessionMap = new Map();
      sessions.set(Constants.EXECUTE_CODE, {
        session_id: 'unrelated-session',
        files: [],
        lastUpdated: Date.now(),
      } satisfies t.CodeSessionContext);

      const { capturedRequests } = captureBatchRequests();

      const toolNode = new ToolNode({
        tools: [createDummyTool('web_search')],
        sessions,
        eventDrivenMode: true,
        toolCallStepIds: new Map([['call_ws', 'step_ws']]),
      });

      const aiMsg = new AIMessage({
        content: '',
        tool_calls: [{ id: 'call_ws', name: 'web_search', args: { x: 'q' } }],
      });

      await toolNode.invoke({ messages: [aiMsg] });

      expect(capturedRequests).toHaveLength(1);
      expect(capturedRequests[0].name).toBe('web_search');
      expect(capturedRequests[0].codeSessionContext).toBeUndefined();
    });
  });
});
