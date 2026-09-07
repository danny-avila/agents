import type { DerivedSessionMessages } from './deriveMessages';
import type { SessionEntry, SessionStateEntry } from './types';
import type { JsonlSessionStore } from './JsonlSessionStore';
import { deriveMessages } from './deriveMessages';

interface SessionProjection {
  entries: readonly SessionEntry[];
  ownership: { exposed: boolean };
  indexedCount: number;
  byId: Map<string, SessionEntry>;
  lastState?: SessionStateEntry;
  lastMessage?: SessionEntry;
  leaf?: SessionEntry;
  projected: SessionEntry[];
}

const projections = new WeakMap<JsonlSessionStore, SessionProjection>();

/** Internal: only AgentSession's unexposed stores may reuse log topology. */
export function initializeSessionProjection(
  store: JsonlSessionStore,
  entries: readonly SessionEntry[]
): void {
  projections.set(store, {
    entries,
    ownership: { exposed: false },
    indexedCount: 0,
    byId: new Map(),
    projected: [],
  });
}

/** Forks share entry references, so exposing either disables the whole family. */
export function shareSessionProjectionOwnership(
  source: JsonlSessionStore,
  target: JsonlSessionStore
): void {
  const sourceProjection = projections.get(source);
  const targetProjection = projections.get(target);
  if (sourceProjection && targetProjection) {
    targetProjection.ownership = sourceProjection.ownership;
  }
}

export function releaseSessionProjection(store: JsonlSessionStore): void {
  const projection = projections.get(store);
  if (projection) {
    projection.ownership.exposed = true;
    projection.byId.clear();
    projection.projected = [];
    projection.leaf = undefined;
  }
}

function indexAppends(projection: SessionProjection): boolean {
  for (
    ;
    projection.indexedCount < projection.entries.length;
    projection.indexedCount++
  ) {
    const entry = projection.entries[projection.indexedCount];
    if (projection.byId.has(entry.id)) {
      return false;
    }
    projection.byId.set(entry.id, entry);
    if (entry.type === 'session_state') {
      projection.lastState = entry;
    } else if (entry.type === 'message' || entry.type === 'summary') {
      projection.lastMessage = entry;
    }
  }
  return true;
}

function projectLeaf(projection: SessionProjection): void {
  const leafId = projection.lastState
    ? projection.lastState.data.leafId
    : projection.lastMessage?.id;
  const leaf =
    leafId != null && leafId !== '' ? projection.byId.get(leafId) : undefined;
  if (leaf === projection.leaf) {
    return;
  }
  const delta: SessionEntry[] = [];
  let current = leaf;
  while (current && current !== projection.leaf) {
    if (current.type === 'message' || current.type === 'summary') {
      delta.push(current);
    }
    if (current.type === 'summary') {
      current = undefined;
      break;
    }
    current =
      current.parentId == null
        ? undefined
        : projection.byId.get(current.parentId);
  }
  if (!current) {
    projection.projected = [];
  }
  for (let i = delta.length - 1; i >= 0; i--) {
    projection.projected.push(delta[i]);
  }
  projection.leaf = leaf;
}

/** Reuses log topology, but materializes fresh mutable messages for every run. */
export function deriveSessionMessages(
  store: JsonlSessionStore | undefined
): DerivedSessionMessages {
  if (!store) {
    return { messages: [] };
  }
  const projection = projections.get(store);
  if (!projection || projection.ownership.exposed) {
    releaseSessionProjection(store);
    return deriveMessages(store.getPath());
  }
  if (!indexAppends(projection)) {
    releaseSessionProjection(store);
    return deriveMessages(store.getPath());
  }
  projectLeaf(projection);
  return deriveMessages(projection.projected);
}
