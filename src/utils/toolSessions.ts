import type * as t from '@/types';
import { Constants } from '@/common';

type CodeSessionAgent = Pick<t.AgentInputs, 'codeSessionKey' | 'initialSessions'>;

function cloneToolSessionContext(
  context: t.ToolSessionContext
): t.ToolSessionContext {
  return {
    ...context,
    ...(context.files == null
      ? {}
      : {
        files: context.files.map((file) => ({
          ...file,
          storage_session_id:
            file.storage_session_id ?? context.session_id,
        })),
      }),
  };
}

function mergeToolSessionContext(
  sessions: t.ToolSessionMap,
  key: string,
  context: t.ToolSessionContext
): void {
  const existing = sessions.get(key);
  if (existing == null) {
    sessions.set(key, cloneToolSessionContext(context));
    return;
  }
  if (context.files == null || context.files.length === 0) {
    return;
  }

  const seenFiles = new Set(
    existing.files?.map(
      (file) =>
        `${file.storage_session_id ?? existing.session_id}\0${file.id}`
    ) ?? []
  );
  const files = existing.files == null ? [] : [...existing.files];
  for (const file of context.files) {
    const storageSessionId = file.storage_session_id ?? context.session_id;
    const fileKey = `${storageSessionId}\0${file.id}`;
    if (seenFiles.has(fileKey)) {
      continue;
    }
    seenFiles.add(fileKey);
    files.push({ ...file, storage_session_id: storageSessionId });
  }
  sessions.set(key, { ...existing, files });
}

/**
 * Seeds a top-level graph while preserving the legacy run-wide session map.
 * A legacy `execute_code` seed is copied into every custom agent partition
 * unless the host already supplied an explicit seed for that partition.
 */
export function seedRunInitialSessions(args: {
  sessions: t.ToolSessionMap;
  initialSessions: t.ToolSessionMap;
  agents: Iterable<Pick<t.AgentInputs, 'codeSessionKey'>>;
}): void {
  const { sessions, initialSessions, agents } = args;
  for (const [key, context] of initialSessions) {
    mergeToolSessionContext(sessions, key, context);
  }

  const legacyCodeSession = initialSessions.get(Constants.EXECUTE_CODE);
  if (legacyCodeSession == null) {
    return;
  }
  for (const agent of agents) {
    const key = agent.codeSessionKey ?? Constants.EXECUTE_CODE;
    if (
      key === Constants.EXECUTE_CODE ||
      initialSessions.has(key) ||
      sessions.has(key)
    ) {
      continue;
    }
    sessions.set(key, cloneToolSessionContext(legacyCodeSession));
  }
}

/**
 * Seeds an isolated child graph from each selected agent. Legacy code-session
 * entries are remapped to the owning agent's partition before they are merged.
 */
export function seedAgentInitialSessions(
  sessions: t.ToolSessionMap,
  agents: Iterable<CodeSessionAgent>
): void {
  for (const agent of agents) {
    if (agent.initialSessions == null) {
      continue;
    }
    const codeSessionKey = agent.codeSessionKey ?? Constants.EXECUTE_CODE;
    const hasExplicitCodeSession =
      codeSessionKey !== Constants.EXECUTE_CODE &&
      agent.initialSessions.has(codeSessionKey);
    for (const [toolName, context] of agent.initialSessions) {
      if (toolName === Constants.EXECUTE_CODE && hasExplicitCodeSession) {
        continue;
      }
      const key =
        toolName === Constants.EXECUTE_CODE ? codeSessionKey : toolName;
      mergeToolSessionContext(sessions, key, context);
    }
  }
}
