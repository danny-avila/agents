import { describe, expect, it } from '@jest/globals';
import type * as t from '@/types';
import {
  seedAgentInitialSessions,
  seedRunInitialSessions,
} from '@/utils/toolSessions';
import { Constants } from '@/common';

function session(id: string, fileId: string): t.ToolSessionContext {
  return {
    session_id: id,
    files: [{ id: fileId, name: `${fileId}.txt` }],
    lastUpdated: 1,
  };
}

describe('tool session partition seeding', () => {
  it('copies a legacy run seed into each custom agent partition', () => {
    const legacy = session('legacy-storage', 'input');
    const sessions: t.ToolSessionMap = new Map();

    seedRunInitialSessions({
      sessions,
      initialSessions: new Map([[Constants.EXECUTE_CODE, legacy]]),
      agents: [
        {},
        { codeSessionKey: 'execute_code:stateful:user-1' },
        { codeSessionKey: 'execute_code:stateful:agent-2' },
      ],
    });

    expect([...sessions.keys()]).toEqual([
      Constants.EXECUTE_CODE,
      'execute_code:stateful:user-1',
      'execute_code:stateful:agent-2',
    ]);
    expect(sessions.get('execute_code:stateful:user-1')).toEqual({
      ...legacy,
      files: [
        {
          ...legacy.files![0],
          storage_session_id: 'legacy-storage',
        },
      ],
    });
    expect(sessions.get('execute_code:stateful:user-1')).not.toBe(legacy);
    expect(sessions.get('execute_code:stateful:agent-2')).not.toBe(legacy);
  });

  it('keeps an explicit custom run seed instead of replacing it with legacy state', () => {
    const key = 'execute_code:stateful:user-1';
    const explicit = session('stateful-storage', 'stateful-input');
    const sessions: t.ToolSessionMap = new Map();

    seedRunInitialSessions({
      sessions,
      initialSessions: new Map([
        [Constants.EXECUTE_CODE, session('legacy-storage', 'legacy-input')],
        [key, explicit],
      ]),
      agents: [{ codeSessionKey: key }],
    });

    expect(sessions.get(key)).toEqual({
      ...explicit,
      files: [
        {
          ...explicit.files![0],
          storage_session_id: 'stateful-storage',
        },
      ],
    });
  });

  it('remaps child legacy seeds to their owning agent partitions', () => {
    const firstKey = 'execute_code:stateful:user-1';
    const secondKey = 'execute_code:stateful:agent-2';
    const sessions: t.ToolSessionMap = new Map();

    seedAgentInitialSessions(sessions, [
      {
        codeSessionKey: firstKey,
        initialSessions: new Map([
          [Constants.EXECUTE_CODE, session('first-storage', 'first-input')],
        ]),
      },
      {
        codeSessionKey: secondKey,
        initialSessions: new Map([
          [Constants.EXECUTE_CODE, session('second-storage', 'second-input')],
        ]),
      },
    ]);

    expect(sessions.has(Constants.EXECUTE_CODE)).toBe(false);
    expect(sessions.get(firstKey)?.files).toEqual([
      expect.objectContaining({
        id: 'first-input',
        storage_session_id: 'first-storage',
      }),
    ]);
    expect(sessions.get(secondKey)?.files).toEqual([
      expect.objectContaining({
        id: 'second-input',
        storage_session_id: 'second-storage',
      }),
    ]);
  });

  it('keeps an explicit child partition seed instead of merging the legacy fallback', () => {
    const key = 'execute_code:stateful:user-1';
    const explicit = session('stateful-storage', 'stateful-input');
    const sessions: t.ToolSessionMap = new Map();

    seedAgentInitialSessions(sessions, [
      {
        codeSessionKey: key,
        initialSessions: new Map([
          [Constants.EXECUTE_CODE, session('legacy-storage', 'legacy-input')],
          [key, explicit],
        ]),
      },
    ]);

    expect(sessions.get(key)).toEqual({
      ...explicit,
      files: [
        {
          ...explicit.files![0],
          storage_session_id: 'stateful-storage',
        },
      ],
    });
  });
});
