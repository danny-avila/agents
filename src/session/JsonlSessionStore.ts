import { homedir } from 'os';
import { createHash } from 'crypto';
import { basename, dirname, isAbsolute, join, resolve } from 'path';
import {
  access,
  appendFile,
  mkdir,
  readFile,
  readdir,
  stat,
  writeFile,
} from 'fs/promises';
import type {
  CreateSessionFileOptions,
  SessionBranchOptions,
  SessionEntry,
  SessionForkOptions,
  SessionHeader,
  SessionLabelEntry,
  SessionListItem,
  SessionMessageEntry,
  SessionCheckpointEntry,
  SessionCompactionEntry,
  SessionRunEventEntry,
  SessionSummaryEntry,
  SessionStateEntry,
  SessionTreeNode,
} from './types';
import type { BaseMessage } from '@langchain/core/messages';
import { SystemMessage } from '@langchain/core/messages';
import { createEntryId, createSessionId, createTimestamp } from './ids';
import {
  deserializeMessage,
  getMessageRole,
  serializeMessage,
  toJsonValue,
} from './messageSerialization';

const SESSION_VERSION = 1;
const DEFAULT_SESSION_ROOT = join(
  homedir(),
  '.librechat',
  'agents',
  'sessions'
);

function sanitizeCwd(cwd: string): string {
  const normalized = resolve(cwd);
  return createHash('sha256').update(normalized).digest('hex');
}

function createSessionPath(options: CreateSessionFileOptions): string {
  const sessionId = options.sessionId ?? createSessionId();
  const fileName = `${new Date().toISOString().replace(/[:.]/g, '-')}_${sessionId}.jsonl`;
  return join(DEFAULT_SESSION_ROOT, sanitizeCwd(options.cwd), fileName);
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function parseLine(line: string): SessionHeader | SessionEntry | undefined {
  if (line.trim() === '') {
    return undefined;
  }
  try {
    const parsed = JSON.parse(line) as SessionHeader | SessionEntry;
    if (parsed.type === 'session') {
      return parsed;
    }
    if ('id' in parsed && 'parentId' in parsed && 'data' in parsed) {
      return parsed;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

function createHeader(options: CreateSessionFileOptions): SessionHeader {
  return {
    type: 'session',
    version: SESSION_VERSION,
    id: options.sessionId ?? createSessionId(),
    timestamp: createTimestamp(),
    cwd: resolve(options.cwd),
    ...(options.name != null && options.name !== ''
      ? { name: options.name }
      : {}),
    ...(options.parentSession != null && options.parentSession !== ''
      ? { parentSession: options.parentSession }
      : {}),
  };
}

function sortEntries(entries: SessionEntry[]): SessionEntry[] {
  return [...entries].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

export class JsonlSessionStore {
  readonly path: string;
  readonly header: SessionHeader;
  private entries: SessionEntry[];

  private constructor(params: {
    path: string;
    header: SessionHeader;
    entries: SessionEntry[];
  }) {
    this.path = params.path;
    this.header = params.header;
    this.entries = sortEntries(params.entries);
  }

  static getDefaultRoot(): string {
    return DEFAULT_SESSION_ROOT;
  }

  static async create(
    options: CreateSessionFileOptions & { path?: string }
  ): Promise<JsonlSessionStore> {
    const sessionId = options.sessionId ?? createSessionId();
    const path =
      options.path != null && options.path !== ''
        ? resolve(options.path)
        : createSessionPath({ ...options, sessionId });
    const header = createHeader({ ...options, sessionId });
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, `${JSON.stringify(header)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
    return new JsonlSessionStore({ path, header, entries: [] });
  }

  static async open(pathOrId: string): Promise<JsonlSessionStore> {
    const path = await JsonlSessionStore.resolvePath(pathOrId);
    return JsonlSessionStore.openPath(path);
  }

  static async openPath(path: string): Promise<JsonlSessionStore> {
    const resolved = resolve(path);
    const raw = await readFile(resolved, 'utf8');
    const parsed = raw
      .split('\n')
      .map((line) => parseLine(line))
      .filter((line): line is SessionHeader | SessionEntry => line != null);
    const header = parsed.find(
      (line): line is SessionHeader => line.type === 'session'
    );
    if (!header) {
      throw new Error(`Invalid session file: ${resolved}`);
    }
    const entries = parsed.filter(
      (line): line is SessionEntry => line.type !== 'session'
    );
    return new JsonlSessionStore({ path: resolved, header, entries });
  }

  static async resolvePath(pathOrId: string): Promise<string> {
    const candidate = isAbsolute(pathOrId) ? pathOrId : resolve(pathOrId);
    if (await pathExists(candidate)) {
      return candidate;
    }
    const sessions = await JsonlSessionStore.listAll();
    const matches = sessions.filter(
      (item) =>
        item.id.startsWith(pathOrId) || basename(item.path).includes(pathOrId)
    );
    if (matches.length === 1) {
      return matches[0].path;
    }
    if (matches.length > 1) {
      throw new Error(`Session id "${pathOrId}" is ambiguous`);
    }
    throw new Error(`Session not found: ${pathOrId}`);
  }

  static async list(cwd: string): Promise<SessionListItem[]> {
    const dir = join(DEFAULT_SESSION_ROOT, sanitizeCwd(cwd));
    return JsonlSessionStore.listDirectory(dir);
  }

  static async listAll(
    root: string = DEFAULT_SESSION_ROOT
  ): Promise<SessionListItem[]> {
    if (!(await pathExists(root))) {
      return [];
    }
    const dirs = await readdir(root, { withFileTypes: true });
    const lists = await Promise.all(
      dirs
        .filter((entry) => entry.isDirectory())
        .map((entry) => JsonlSessionStore.listDirectory(join(root, entry.name)))
    );
    return lists.flat().sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  }

  private static async listDirectory(dir: string): Promise<SessionListItem[]> {
    if (!(await pathExists(dir))) {
      return [];
    }
    const files = await readdir(dir, { withFileTypes: true });
    const candidates = files
      .filter((entry) => entry.isFile() && entry.name.endsWith('.jsonl'))
      .map((entry) => join(dir, entry.name));
    const items = await Promise.all(
      candidates.map(async (path): Promise<SessionListItem | undefined> => {
        try {
          const store = await JsonlSessionStore.open(path);
          return {
            id: store.header.id,
            path,
            cwd: store.header.cwd,
            timestamp: store.header.timestamp,
            name: store.header.name,
            leafId: store.getLeafEntry()?.id ?? null,
          };
        } catch {
          return undefined;
        }
      })
    );
    return items
      .filter((item): item is SessionListItem => item != null)
      .sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  }

  getEntries(): SessionEntry[] {
    return [...this.entries];
  }

  getEntry(id: string): SessionEntry | undefined {
    return this.entries.find((entry) => entry.id === id);
  }

  getChildren(id: string): SessionEntry[] {
    return this.entries.filter((entry) => entry.parentId === id);
  }

  getTree(): SessionTreeNode[] {
    const byParent = new Map<string | null, SessionEntry[]>();
    for (const entry of this.entries) {
      const siblings = byParent.get(entry.parentId) ?? [];
      siblings.push(entry);
      byParent.set(entry.parentId, siblings);
    }
    const build = (parentId: string | null): SessionTreeNode[] =>
      (byParent.get(parentId) ?? []).map((entry) => ({
        entry,
        children: build(entry.id),
      }));
    return build(null);
  }

  getLeafEntry(): SessionEntry | undefined {
    const state = [...this.entries]
      .reverse()
      .find(
        (entry): entry is SessionStateEntry => entry.type === 'session_state'
      );
    if (state) {
      return state.data.leafId == null
        ? undefined
        : this.getEntry(state.data.leafId);
    }
    return [...this.entries]
      .reverse()
      .find((entry) => entry.type === 'message' || entry.type === 'summary');
  }

  getPath(
    entryId: string | undefined = this.getLeafEntry()?.id
  ): SessionEntry[] {
    if (entryId == null || entryId === '') {
      return [];
    }
    const byId = new Map(this.entries.map((entry) => [entry.id, entry]));
    const path: SessionEntry[] = [];
    let current: SessionEntry | undefined = byId.get(entryId);
    while (current) {
      path.push(current);
      current =
        current.parentId == null ? undefined : byId.get(current.parentId);
    }
    return path.reverse();
  }

  getMessages(entryId?: string): BaseMessage[] {
    const messages: BaseMessage[] = [];
    for (const entry of this.getPath(entryId)) {
      if (entry.type === 'message') {
        messages.push(deserializeMessage(entry.data.message));
      } else if (entry.type === 'summary') {
        messages.push(new SystemMessage(entry.data.text));
      }
    }
    return messages;
  }

  getForkPoints(): SessionMessageEntry[] {
    return this.entries.filter(
      (entry): entry is SessionMessageEntry =>
        entry.type === 'message' && entry.data.role === 'user'
    );
  }

  getLabel(entryId: string): string | undefined {
    return [...this.entries]
      .reverse()
      .find(
        (entry): entry is SessionLabelEntry =>
          entry.type === 'label' && entry.data.targetEntryId === entryId
      )?.data.label;
  }

  async setLabel(entryId: string, label: string): Promise<SessionLabelEntry> {
    return this.appendEntry<SessionLabelEntry>({
      type: 'label',
      parentId: this.getLeafEntry()?.id ?? null,
      data: { targetEntryId: entryId, label },
    });
  }

  async appendMessage(
    message: BaseMessage,
    parentId: string | null = this.getLeafEntry()?.id ?? null
  ): Promise<SessionMessageEntry> {
    const entry = await this.appendEntry<SessionMessageEntry>({
      type: 'message',
      parentId,
      data: {
        role: getMessageRole(message),
        message: serializeMessage(message),
      },
    });
    await this.setLeaf(entry.id);
    return entry;
  }

  async appendRunEvent(
    event: string,
    payload?: unknown,
    params?: { runId?: string; threadId?: string }
  ): Promise<SessionRunEventEntry> {
    return this.appendEntry<SessionRunEventEntry>({
      type: 'run_event',
      parentId: this.getLeafEntry()?.id ?? null,
      data: {
        event,
        ...(params?.runId != null && params.runId !== ''
          ? { runId: params.runId }
          : {}),
        ...(params?.threadId != null && params.threadId !== ''
          ? { threadId: params.threadId }
          : {}),
        ...(typeof payload !== 'undefined'
          ? { payload: toJsonValue(payload) }
          : {}),
      },
    });
  }

  async setLeaf(leafId: string | null): Promise<SessionStateEntry> {
    return this.appendEntry<SessionStateEntry>({
      type: 'session_state',
      parentId: leafId,
      data: { leafId },
    });
  }

  async appendEntryForCompaction(params: {
    text: string;
    tokenCount?: number;
    retainedEntryIds: string[];
    summarizedEntryIds: string[];
    instructions?: string;
    parentId?: string | null;
  }): Promise<SessionSummaryEntry> {
    const tokenCount =
      typeof params.tokenCount === 'number' &&
      Number.isFinite(params.tokenCount) &&
      params.tokenCount >= 0
        ? params.tokenCount
        : undefined;
    const entry = await this.appendEntry<SessionSummaryEntry>({
      type: 'summary',
      parentId:
        'parentId' in params
          ? (params.parentId ?? null)
          : (this.getLeafEntry()?.id ?? null),
      data: {
        text: params.text,
        ...(tokenCount != null ? { tokenCount } : {}),
        retainedEntryIds: params.retainedEntryIds,
        summarizedEntryIds: params.summarizedEntryIds,
        ...(params.instructions != null && params.instructions !== ''
          ? { instructions: params.instructions }
          : {}),
      },
    });
    await this.setLeaf(entry.id);
    return entry;
  }

  async appendCompactionEntry(params: {
    summaryEntryId: string;
    retainedEntryIds: string[];
    summarizedEntryIds: string[];
  }): Promise<SessionCompactionEntry> {
    return this.appendEntry<SessionCompactionEntry>({
      type: 'compaction',
      parentId: this.getLeafEntry()?.id ?? null,
      data: {
        summaryEntryId: params.summaryEntryId,
        retainedEntryIds: params.retainedEntryIds,
        summarizedEntryIds: params.summarizedEntryIds,
      },
    });
  }

  async appendCheckpoint(params: {
    source: SessionCheckpointEntry['data']['source'];
    threadId: string;
    runId?: string;
    checkpointId?: string;
    checkpointNs?: string;
    parentCheckpointId?: string;
    reason?: string;
  }): Promise<SessionCheckpointEntry> {
    return this.appendEntry<SessionCheckpointEntry>({
      type: 'checkpoint',
      parentId: this.getLeafEntry()?.id ?? null,
      data: {
        provider: 'langgraph',
        source: params.source,
        threadId: params.threadId,
        ...(params.runId != null && params.runId !== ''
          ? { runId: params.runId }
          : {}),
        ...(params.checkpointId != null && params.checkpointId !== ''
          ? { checkpointId: params.checkpointId }
          : {}),
        ...(params.checkpointNs != null
          ? { checkpointNs: params.checkpointNs }
          : {}),
        ...(params.parentCheckpointId != null &&
        params.parentCheckpointId !== ''
          ? { parentCheckpointId: params.parentCheckpointId }
          : {}),
        ...(params.reason != null && params.reason !== ''
          ? { reason: params.reason }
          : {}),
      },
    });
  }

  getCheckpoints(threadId?: string): SessionCheckpointEntry[] {
    return this.entries.filter(
      (entry): entry is SessionCheckpointEntry =>
        entry.type === 'checkpoint' &&
        (threadId == null || entry.data.threadId === threadId)
    );
  }

  getLatestCheckpoint(threadId?: string): SessionCheckpointEntry | undefined {
    const checkpoints = this.getCheckpoints(threadId);
    for (let i = checkpoints.length - 1; i >= 0; i--) {
      const checkpoint = checkpoints[i];
      if (checkpoint.data.source === 'reset') {
        return undefined;
      }
      return checkpoint;
    }
    return undefined;
  }

  async branch(
    entryId: string,
    options: SessionBranchOptions = {}
  ): Promise<SessionEntry | undefined> {
    const target = this.getBranchTarget(entryId, options.position ?? 'at');
    await this.setLeaf(target?.id ?? null);
    return target;
  }

  async createBranchedSession(
    entryId: string | undefined = this.getLeafEntry()?.id,
    options: SessionForkOptions = {}
  ): Promise<JsonlSessionStore> {
    const target =
      entryId != null && entryId !== ''
        ? this.getBranchTarget(entryId, options.position ?? 'at')
        : this.getLeafEntry();
    const newStore = await JsonlSessionStore.create({
      cwd: options.cwd ?? this.header.cwd,
      name: options.name ?? this.header.name,
      parentSession: this.path,
    });
    const pathEntries = target ? this.getPath(target.id) : [];
    for (const entry of pathEntries) {
      await newStore.appendExistingEntry(entry);
    }
    await newStore.setLeaf(target?.id ?? null);
    return newStore;
  }

  async clone(options: SessionForkOptions = {}): Promise<JsonlSessionStore> {
    return this.createBranchedSession(this.getLeafEntry()?.id, {
      ...options,
      position: 'at',
    });
  }

  async fork(
    entryId: string,
    options: SessionForkOptions = {}
  ): Promise<JsonlSessionStore> {
    return this.createBranchedSession(entryId, {
      ...options,
      position: options.position ?? 'before',
    });
  }

  private getBranchTarget(
    entryId: string,
    position: 'before' | 'at'
  ): SessionEntry | undefined {
    const entry = this.getEntry(entryId);
    if (!entry) {
      throw new Error(`Entry not found: ${entryId}`);
    }
    if (position === 'at') {
      return entry;
    }
    return entry.parentId == null ? undefined : this.getEntry(entry.parentId);
  }

  private async appendExistingEntry(entry: SessionEntry): Promise<void> {
    this.entries.push(entry);
    await appendFile(this.path, `${JSON.stringify(entry)}\n`, 'utf8');
  }

  private async appendEntry<TEntry extends SessionEntry>(params: {
    type: TEntry['type'];
    parentId: string | null;
    data: TEntry['data'];
  }): Promise<TEntry> {
    const entry = {
      type: params.type,
      id: createEntryId(),
      parentId: params.parentId,
      timestamp: createTimestamp(),
      data: params.data,
    } as TEntry;
    this.entries.push(entry);
    await appendFile(this.path, `${JSON.stringify(entry)}\n`, 'utf8');
    return entry;
  }

  async existsOnDisk(): Promise<boolean> {
    const fileStat = await stat(this.path).catch(() => undefined);
    return fileStat?.isFile() === true;
  }
}

export const SessionManager = JsonlSessionStore;
