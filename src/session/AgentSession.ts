import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type {
  BaseCheckpointSaver,
  CheckpointTuple,
} from '@langchain/langgraph';
import type { HookRegistry } from '@/hooks';
import type {
  AgentSessionConfig,
  AgentSessionCheckpointing,
  AgentSessionCheckpointLookupOptions,
  AgentSessionCheckpointReference,
  AgentSessionInput,
  AgentSessionRunOptions,
  AgentSessionRunResult,
  AgentSessionStream,
  AgentSessionStreamEvent,
  SessionBranchOptions,
  SessionCompactOptions,
  SessionEntry,
  SessionForkOptions,
} from './types';
import type * as t from '@/types';
import { AgentContext } from '@/agents/AgentContext';
import { ContentTypes, GraphEvents } from '@/common';
import { Run } from '@/run';
import { createRunId, createSessionId } from './ids';
import { deserializeMessage } from './messageSerialization';
import { JsonlSessionStore } from './JsonlSessionStore';
import { createRunHandlers } from './handlers';
import { createSummarizeNode } from '@/summarization/node';

function isBaseMessage(value: unknown): value is BaseMessage {
  return (
    value instanceof BaseMessage ||
    (value != null &&
      typeof value === 'object' &&
      '_getType' in value &&
      typeof (value as { _getType?: unknown })._getType === 'function')
  );
}

interface NormalizedSessionInput {
  messages: BaseMessage[];
  state: t.IState;
}

function normalizeInput(input: AgentSessionInput): NormalizedSessionInput {
  if (typeof input === 'string') {
    const messages = [new HumanMessage(input)];
    return { messages, state: { messages } };
  }
  if (Array.isArray(input)) {
    return { messages: input, state: { messages: input } };
  }
  if (isBaseMessage(input)) {
    const messages = [input];
    return { messages, state: { messages } };
  }
  return { messages: input.messages, state: input };
}

function contentToText(
  content: Array<t.MessageContentComplex | undefined>
): string {
  const chunks: string[] = [];
  for (const part of content) {
    if (!part) {
      continue;
    }
    if (part.type === ContentTypes.TEXT && typeof part.text === 'string') {
      chunks.push(part.text);
    }
  }
  return chunks.join('');
}

function normalizeConfig(config: AgentSessionConfig): {
  runConfig: t.RunConfig;
  cwd: string;
  sessionPath?: string;
  name?: string;
  ephemeral?: boolean;
  checkpointing?: AgentSessionCheckpointing;
} {
  if ('runConfig' in config) {
    return {
      runConfig: config.runConfig,
      cwd: config.cwd ?? process.cwd(),
      sessionPath: config.sessionPath,
      name: config.name,
      ephemeral: config.ephemeral,
      checkpointing: config.checkpointing,
    };
  }
  const {
    cwd,
    sessionPath,
    name,
    ephemeral,
    checkpointing,
    sessionId: _sessionId,
    ...runConfig
  } = config;
  return {
    runConfig: {
      ...runConfig,
      runId: config.runId ?? createRunId(),
    },
    cwd: cwd ?? process.cwd(),
    sessionPath,
    name,
    ephemeral,
    checkpointing,
  };
}

function isMissingSessionError(error: unknown): boolean {
  if (error == null || typeof error !== 'object') {
    return false;
  }
  const candidate = error as { code?: string; message?: string };
  if (candidate.code === 'ENOENT') {
    return true;
  }
  return candidate.message?.startsWith('Session not found:') === true;
}

async function createStore(params: {
  cwd: string;
  sessionPath?: string;
  name?: string;
  sessionId?: string;
  ephemeral?: boolean;
}): Promise<JsonlSessionStore | undefined> {
  if (params.ephemeral === true) {
    return undefined;
  }
  if (params.sessionPath != null && params.sessionPath !== '') {
    try {
      return await JsonlSessionStore.openPath(params.sessionPath);
    } catch (error) {
      if (!isMissingSessionError(error)) {
        throw error;
      }
      return JsonlSessionStore.create({
        path: params.sessionPath,
        cwd: params.cwd,
        name: params.name,
        sessionId: params.sessionId,
      });
    }
  }
  return JsonlSessionStore.create({
    cwd: params.cwd,
    name: params.name,
    sessionId: params.sessionId,
  });
}

type InitialSummary = NonNullable<t.AgentInputs['initialSummary']>;

function mergeInitialSummary(
  existing: InitialSummary | undefined,
  sessionSummary: InitialSummary | undefined
): InitialSummary | undefined {
  if (!existing) {
    return sessionSummary;
  }
  if (!sessionSummary) {
    return existing;
  }
  if (existing.text === sessionSummary.text) {
    return existing;
  }
  return {
    text: `${existing.text}\n\n${sessionSummary.text}`,
    tokenCount: existing.tokenCount + sessionSummary.tokenCount,
  };
}

function applyInitialSummaryToAgent(
  agent: t.AgentInputs,
  initialSummary: InitialSummary | undefined
): t.AgentInputs {
  const merged = mergeInitialSummary(agent.initialSummary, initialSummary);
  return merged ? { ...agent, initialSummary: merged } : agent;
}

function applyInitialSummaryToGraphConfig(
  graphConfig: t.RunConfig['graphConfig'],
  initialSummary: InitialSummary | undefined
): t.RunConfig['graphConfig'] {
  if (!initialSummary) {
    return graphConfig;
  }
  if ('agents' in graphConfig) {
    return {
      ...graphConfig,
      agents: graphConfig.agents.map((agent) =>
        applyInitialSummaryToAgent(agent, initialSummary)
      ),
    };
  }
  return {
    ...graphConfig,
    initialSummary: mergeInitialSummary(
      graphConfig.initialSummary,
      initialSummary
    ),
  };
}

interface SessionCheckpointingState {
  enabled: boolean;
  checkpointer?: BaseCheckpointSaver;
  disableGraphCheckpointer?: boolean;
}

function isCheckpointSaver(value: unknown): value is BaseCheckpointSaver {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const candidate = value as Partial<BaseCheckpointSaver>;
  return (
    typeof candidate.getTuple === 'function' &&
    typeof candidate.list === 'function' &&
    typeof candidate.put === 'function' &&
    typeof candidate.putWrites === 'function' &&
    typeof candidate.deleteThread === 'function'
  );
}

function getGraphCheckpointer(
  graphConfig: t.RunConfig['graphConfig']
): BaseCheckpointSaver | undefined {
  const checkpointer = graphConfig.compileOptions?.checkpointer;
  return isCheckpointSaver(checkpointer) ? checkpointer : undefined;
}

function createCheckpointingState(
  runConfig: t.RunConfig,
  checkpointing: AgentSessionCheckpointing | undefined
): SessionCheckpointingState {
  const graphCheckpointer = getGraphCheckpointer(runConfig.graphConfig);
  if (checkpointing === false) {
    return {
      enabled: false,
      disableGraphCheckpointer: true,
    };
  }
  if (typeof checkpointing === 'object') {
    if (checkpointing.enabled === false) {
      return {
        enabled: false,
        disableGraphCheckpointer: true,
      };
    }
    return {
      enabled: true,
      checkpointer:
        checkpointing.checkpointer ?? graphCheckpointer ?? new MemorySaver(),
    };
  }
  if (checkpointing === true || runConfig.humanInTheLoop?.enabled === true) {
    return {
      enabled: true,
      checkpointer: graphCheckpointer ?? new MemorySaver(),
    };
  }
  return {
    enabled: graphCheckpointer != null,
    checkpointer: graphCheckpointer,
  };
}

function removeCheckpointerFromGraphConfig(
  graphConfig: t.RunConfig['graphConfig']
): t.RunConfig['graphConfig'] {
  if (graphConfig.compileOptions?.checkpointer == null) {
    return graphConfig;
  }
  const { checkpointer: _checkpointer, ...compileOptions } =
    graphConfig.compileOptions;
  return {
    ...graphConfig,
    compileOptions,
  };
}

function applyCheckpointerToGraphConfig(
  graphConfig: t.RunConfig['graphConfig'],
  checkpointer: BaseCheckpointSaver | undefined
): t.RunConfig['graphConfig'] {
  if (!checkpointer) {
    return graphConfig;
  }
  if (graphConfig.compileOptions?.checkpointer === checkpointer) {
    return graphConfig;
  }
  return {
    ...graphConfig,
    compileOptions: {
      ...(graphConfig.compileOptions ?? {}),
      checkpointer,
    },
  };
}

function applyCheckpointingToGraphConfig(
  graphConfig: t.RunConfig['graphConfig'],
  checkpointing: SessionCheckpointingState
): t.RunConfig['graphConfig'] {
  if (checkpointing.disableGraphCheckpointer === true) {
    return removeCheckpointerFromGraphConfig(graphConfig);
  }
  return applyCheckpointerToGraphConfig(
    graphConfig,
    checkpointing.checkpointer
  );
}

function getConfigString(
  config: RunnableConfig | undefined,
  key: string
): string | undefined {
  const configurable = config?.configurable as
    | Partial<Record<string, unknown>>
    | undefined;
  const value = configurable?.[key];
  return typeof value === 'string' && value !== '' ? value : undefined;
}

function createCallerConfig(
  threadId: string,
  options: AgentSessionRunOptions
): RunnableConfig & { version: 'v1' | 'v2' } {
  return {
    recursionLimit: 50,
    ...(options.config ?? {}),
    configurable: {
      ...(options.config?.configurable ?? {}),
      thread_id: threadId,
    },
    version: options.config?.version ?? 'v2',
  };
}

function createCheckpointLookupConfig(config: RunnableConfig): RunnableConfig {
  const threadId = getConfigString(config, 'thread_id');
  if (threadId == null) {
    return config;
  }
  const checkpointNs = getConfigString(config, 'checkpoint_ns') ?? '';
  const checkpointId = getConfigString(config, 'checkpoint_id');
  return {
    configurable: {
      ...config.configurable,
      thread_id: threadId,
      checkpoint_ns: checkpointNs,
      ...(checkpointId != null ? { checkpoint_id: checkpointId } : {}),
    },
  };
}

function createLatestCheckpointLookupConfig(
  config: RunnableConfig
): RunnableConfig {
  const lookup = createCheckpointLookupConfig(config);
  const { checkpoint_id: _checkpointId, ...configurable } =
    lookup.configurable ?? {};
  return { configurable };
}

async function getLatestCheckpointTuple(
  checkpointer: BaseCheckpointSaver | undefined,
  config: RunnableConfig
): Promise<CheckpointTuple | undefined> {
  if (!checkpointer) {
    return undefined;
  }
  const lookupConfig = createLatestCheckpointLookupConfig(config);
  const tuple = await checkpointer.getTuple(lookupConfig);
  if (tuple) {
    return tuple;
  }
  for await (const checkpoint of checkpointer.list(lookupConfig, {
    limit: 1,
  })) {
    return checkpoint;
  }
  return undefined;
}

async function getSelectedCheckpointTuple(
  checkpointer: BaseCheckpointSaver | undefined,
  config: RunnableConfig
): Promise<CheckpointTuple | undefined> {
  return checkpointer?.getTuple(createCheckpointLookupConfig(config));
}

function createCheckpointReference(params: {
  threadId: string;
  tuple: CheckpointTuple;
}): AgentSessionCheckpointReference {
  const checkpointNs =
    getConfigString(params.tuple.config, 'checkpoint_ns') ?? '';
  const parentCheckpointId = getConfigString(
    params.tuple.parentConfig,
    'checkpoint_id'
  );
  return {
    provider: 'langgraph',
    threadId: params.threadId,
    checkpointId: params.tuple.checkpoint.id,
    checkpointNs,
    ...(parentCheckpointId != null ? { parentCheckpointId } : {}),
  };
}

function createSessionRunState(entries: SessionEntry[]): {
  messages: BaseMessage[];
  initialSummary?: InitialSummary;
} {
  const messages: BaseMessage[] = [];
  let initialSummary: InitialSummary | undefined;
  for (const entry of entries) {
    if (entry.type === 'summary') {
      initialSummary = {
        text: entry.data.text,
        tokenCount:
          typeof entry.data.tokenCount === 'number' &&
          Number.isFinite(entry.data.tokenCount)
            ? entry.data.tokenCount
            : 0,
      };
      messages.length = 0;
      continue;
    }
    if (entry.type === 'message') {
      messages.push(deserializeMessage(entry.data.message));
    }
  }
  return { messages, initialSummary };
}

function isMessageEntry(
  entry: SessionEntry
): entry is Extract<SessionEntry, { type: 'message' }> {
  return entry.type === 'message';
}

function getSessionBranchTarget(
  store: JsonlSessionStore,
  entryId: string,
  position: 'before' | 'at'
): SessionEntry | undefined {
  const entry = store.getEntry(entryId);
  if (!entry) {
    throw new Error(`Entry not found: ${entryId}`);
  }
  if (position === 'at') {
    return entry;
  }
  return entry.parentId == null ? undefined : store.getEntry(entry.parentId);
}

function getAbandonedPathForBranch(
  store: JsonlSessionStore,
  previousLeafId: string | null,
  targetLeafId: string | null
): SessionEntry[] {
  const previousPath = store.getPath(previousLeafId ?? undefined);
  if (previousPath.length === 0) {
    return [];
  }
  const targetPath = targetLeafId == null ? [] : store.getPath(targetLeafId);
  const maxSharedLength = Math.min(previousPath.length, targetPath.length);
  let sharedLength = 0;
  while (
    sharedLength < maxSharedLength &&
    previousPath[sharedLength].id === targetPath[sharedLength].id
  ) {
    sharedLength++;
  }
  return previousPath.slice(sharedLength);
}

function createAgentInputFromGraphConfig(
  graphConfig: t.RunConfig['graphConfig'],
  initialSummary: InitialSummary | undefined,
  retainRecentTurns: number | undefined,
  instructions: string | undefined
): t.AgentInputs {
  let agent: t.AgentInputs;
  if ('agents' in graphConfig) {
    if (graphConfig.agents.length === 0) {
      throw new Error('Cannot compact a session with no agents');
    }
    agent = graphConfig.agents[0];
  } else {
    const {
      type: _type,
      llmConfig,
      signal: _signal,
      tools = [],
      ...agentInputs
    } = graphConfig;
    const { provider, ...clientOptions } = llmConfig;
    agent = {
      ...agentInputs,
      tools,
      provider,
      clientOptions,
      agentId: 'default',
    };
  }
  const summarizationConfig: t.SummarizationConfig = {
    ...(agent.summarizationConfig ?? {}),
    ...(instructions != null && instructions !== ''
      ? { prompt: instructions }
      : {}),
    retainRecent: {
      ...(agent.summarizationConfig?.retainRecent ?? {}),
      ...(retainRecentTurns != null ? { turns: retainRecentTurns } : {}),
    },
  };
  return {
    ...applyInitialSummaryToAgent(agent, initialSummary),
    summarizationEnabled: true,
    summarizationConfig,
  };
}

function createManualCompactGraph(params: {
  runId: string;
  customHandlers?: Record<string, t.EventHandler>;
  hooks?: HookRegistry;
}): {
  graph: Parameters<typeof createSummarizeNode>[0]['graph'];
  completedSummary?: t.SummaryContentBlock;
} {
  const contentData: t.RunStep[] = [];
  const contentIndexMap = new Map<string, number>();
  const result: {
    graph: Parameters<typeof createSummarizeNode>[0]['graph'];
    completedSummary?: t.SummaryContentBlock;
  } = {
    graph: {
      contentData,
      contentIndexMap,
      runId: params.runId,
      isMultiAgent: false,
      hookRegistry: params.hooks,
      dispatchRunStep: async (runStep): Promise<void> => {
        contentData.push(runStep);
        contentIndexMap.set(runStep.id, runStep.index);
        await params.customHandlers?.[GraphEvents.ON_RUN_STEP]?.handle(
          GraphEvents.ON_RUN_STEP,
          runStep
        );
      },
      dispatchRunStepCompleted: async (stepId, completed): Promise<void> => {
        const runStep = contentData.find((step) => step.id === stepId);
        const resultWithStep = {
          ...completed,
          id: stepId,
          index: runStep?.index ?? 0,
        };
        if (completed.type === 'summary') {
          result.completedSummary = completed.summary;
        }
        await params.customHandlers?.[
          GraphEvents.ON_RUN_STEP_COMPLETED
        ]?.handle(GraphEvents.ON_RUN_STEP_COMPLETED, {
          result: resultWithStep,
        } as unknown as Parameters<t.EventHandler['handle']>[1]);
      },
    },
  };
  return result;
}

function getSummaryText(summary: t.SummaryContentBlock | undefined): string {
  const firstBlock = summary?.content?.[0];
  return firstBlock != null &&
    typeof firstBlock === 'object' &&
    'text' in firstBlock &&
    typeof firstBlock.text === 'string'
    ? firstBlock.text
    : '';
}

function getSummaryTokenCount(
  summary: t.SummaryContentBlock | undefined
): number {
  return typeof summary?.tokenCount === 'number' &&
    Number.isFinite(summary.tokenCount)
    ? summary.tokenCount
    : 0;
}

function filterRemoveMessages(messages: BaseMessage[]): BaseMessage[] {
  return messages.filter((message) => message._getType() !== 'remove');
}

class LiveAgentSessionStream implements AgentSessionStream {
  private resultPromise: Promise<AgentSessionRunResult> | undefined;
  private readonly events: AgentSessionStreamEvent[] = [];
  private readonly waiters: Array<{
    resolve: (result: IteratorResult<AgentSessionStreamEvent>) => void;
    reject: (error: unknown) => void;
  }> = [];
  private closed = false;
  private failed = false;
  private error: unknown;

  setResultPromise(resultPromise: Promise<AgentSessionRunResult>): void {
    this.resultPromise = resultPromise;
  }

  push(event: AgentSessionStreamEvent): void {
    if (this.closed) {
      return;
    }
    const waiter = this.waiters.shift();
    if (waiter) {
      waiter.resolve({ value: event, done: false });
      return;
    }
    this.events.push(event);
  }

  complete(): void {
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.resolve({
        value: undefined as unknown as AgentSessionStreamEvent,
        done: true,
      });
    }
  }

  fail(error: unknown): void {
    this.error = error;
    this.failed = true;
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.reject(error);
    }
  }

  private nextEvent(): Promise<IteratorResult<AgentSessionStreamEvent>> {
    const event = this.events.shift();
    if (event !== undefined) {
      return Promise.resolve({ value: event, done: false });
    }
    if (this.failed) {
      return Promise.reject(this.error);
    }
    if (this.closed) {
      return Promise.resolve({
        value: undefined as unknown as AgentSessionStreamEvent,
        done: true,
      });
    }
    return new Promise((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }

  async *[Symbol.asyncIterator](): AsyncIterator<AgentSessionStreamEvent> {
    for (;;) {
      const next = await this.nextEvent();
      if (next.done === true) {
        return;
      }
      yield next.value;
    }
  }

  async *toTextStream(): AsyncIterable<string> {
    for await (const event of this) {
      if (event.type !== 'message.delta' || event.data == null) {
        continue;
      }
      const data = event.data;
      if (typeof data === 'object' && !Array.isArray(data)) {
        const delta = data.delta;
        if (
          delta != null &&
          typeof delta === 'object' &&
          !Array.isArray(delta)
        ) {
          const content = delta.content;
          if (Array.isArray(content)) {
            for (const part of content) {
              if (
                part != null &&
                typeof part === 'object' &&
                !Array.isArray(part) &&
                typeof part.text === 'string'
              ) {
                yield part.text;
              }
            }
          }
        }
      }
    }
  }

  finalResult(): Promise<AgentSessionRunResult> {
    if (!this.resultPromise) {
      return Promise.reject(new Error('Session stream has not started'));
    }
    return this.resultPromise;
  }
}

export class AgentSession {
  private runConfig: t.RunConfig;
  private store: JsonlSessionStore | undefined;
  private calibrationRatio: number | undefined;
  private checkpointing: SessionCheckpointingState;
  cwd: string;
  threadId: string;

  private constructor(params: {
    runConfig: t.RunConfig;
    cwd: string;
    threadId: string;
    checkpointing: SessionCheckpointingState;
    store?: JsonlSessionStore;
  }) {
    this.runConfig = params.runConfig;
    this.cwd = params.cwd;
    this.threadId = params.threadId;
    this.store = params.store;
    this.checkpointing = params.checkpointing;
  }

  static async create(config: AgentSessionConfig): Promise<AgentSession> {
    const normalized = normalizeConfig(config);
    const explicitSessionId =
      'sessionId' in config && typeof config.sessionId === 'string'
        ? config.sessionId
        : undefined;
    const store = await createStore({
      cwd: normalized.cwd,
      sessionPath: normalized.sessionPath,
      name: normalized.name,
      sessionId: explicitSessionId,
      ephemeral: normalized.ephemeral,
    });
    return new AgentSession({
      runConfig: normalized.runConfig,
      cwd: normalized.cwd,
      threadId: store?.header.id ?? explicitSessionId ?? createSessionId(),
      checkpointing: createCheckpointingState(
        normalized.runConfig,
        normalized.checkpointing
      ),
      store,
    });
  }

  get sessionPath(): string | undefined {
    return this.store?.path;
  }

  getSessionStore(): JsonlSessionStore | undefined {
    return this.store;
  }

  getCheckpointer(): BaseCheckpointSaver | undefined {
    return this.checkpointing.checkpointer;
  }

  async getLatestCheckpoint(
    options: AgentSessionCheckpointLookupOptions = {}
  ): Promise<AgentSessionCheckpointReference | undefined> {
    const threadId = options.threadId ?? this.threadId;
    const baseConfig = options.config ?? {};
    const config = createCheckpointLookupConfig({
      ...baseConfig,
      configurable: {
        ...(baseConfig.configurable ?? {}),
        thread_id: threadId,
        ...(options.checkpointNs != null
          ? { checkpoint_ns: options.checkpointNs }
          : {}),
      },
    });
    const tuple = await getLatestCheckpointTuple(
      this.checkpointing.checkpointer,
      config
    );
    return tuple ? createCheckpointReference({ threadId, tuple }) : undefined;
  }

  private async hasCheckpointState(config: RunnableConfig): Promise<boolean> {
    if (!this.checkpointing.enabled) {
      return false;
    }
    const tuple = await getSelectedCheckpointTuple(
      this.checkpointing.checkpointer,
      config
    );
    return tuple != null;
  }

  private async recordCheckpoint(params: {
    source: 'run' | 'resume';
    runId: string;
    threadId: string;
    config: RunnableConfig;
  }): Promise<void> {
    if (!this.checkpointing.enabled) {
      return;
    }
    const tuple = await getLatestCheckpointTuple(
      this.checkpointing.checkpointer,
      params.config
    );
    if (!tuple) {
      return;
    }
    const reference = createCheckpointReference({
      threadId: params.threadId,
      tuple,
    });
    await this.store?.appendCheckpoint({
      source: params.source,
      runId: params.runId,
      threadId: params.threadId,
      checkpointId: reference.checkpointId,
      checkpointNs: reference.checkpointNs,
      parentCheckpointId: reference.parentCheckpointId,
    });
  }

  private getCheckpointThreadIds(): string[] {
    const threadIds = new Set<string>([this.threadId]);
    for (const checkpoint of this.store?.getCheckpoints() ?? []) {
      threadIds.add(checkpoint.data.threadId);
    }
    return [...threadIds];
  }

  private async resetCheckpointThreads(reason: string): Promise<void> {
    const checkpointer = this.checkpointing.checkpointer;
    if (!this.checkpointing.enabled || checkpointer == null) {
      return;
    }
    for (const threadId of this.getCheckpointThreadIds()) {
      await checkpointer.deleteThread(threadId);
      await this.store?.appendCheckpoint({
        source: 'reset',
        threadId,
        reason,
      });
    }
  }

  private async runInternal(
    input: AgentSessionInput,
    options: AgentSessionRunOptions = {},
    onEvent?: (event: AgentSessionStreamEvent) => void
  ): Promise<AgentSessionRunResult> {
    const runId = options.runId ?? createRunId();
    const threadId = options.threadId ?? this.threadId;
    const isSessionThread = threadId === this.threadId;
    const normalizedInput = normalizeInput(input);
    const inputMessages = normalizedInput.messages;
    const callerConfig = createCallerConfig(threadId, options);
    const useCheckpointState = await this.hasCheckpointState(callerConfig);
    let parentId = this.store?.getLeafEntry()?.id ?? null;
    if (isSessionThread) {
      for (const message of inputMessages) {
        const entry = await this.store?.appendMessage(message, parentId);
        parentId = entry?.id ?? parentId;
      }
    }
    await this.store?.appendRunEvent('run.started', undefined, {
      runId,
      threadId,
    });

    const handlerResult = createRunHandlers({
      runId,
      threadId,
      userHandlers: this.runConfig.customHandlers,
      onEvent,
    });
    const emitTerminalEvent = (
      event: Omit<
        AgentSessionStreamEvent,
        'sequence' | 'runId' | 'threadId' | 'timestamp'
      >
    ): void => {
      const streamEvent: AgentSessionStreamEvent = {
        ...event,
        sequence: handlerResult.events.length,
        runId,
        threadId,
        timestamp: new Date().toISOString(),
      };
      handlerResult.events.push(streamEvent);
      onEvent?.(streamEvent);
    };
    const sessionState = createSessionRunState(
      isSessionThread ? (this.store?.getPath() ?? []) : []
    );
    try {
      const runConfig: t.RunConfig = {
        ...this.runConfig,
        runId,
        graphConfig: applyCheckpointingToGraphConfig(
          applyInitialSummaryToGraphConfig(
            this.runConfig.graphConfig,
            sessionState.initialSummary
          ),
          this.checkpointing
        ),
        returnContent: true,
        calibrationRatio: this.calibrationRatio,
        customHandlers: {
          ...(this.runConfig.customHandlers ?? {}),
          ...handlerResult.handlers,
        },
      };
      const run = await Run.create<t.IState>(runConfig);
      let messages = inputMessages;
      if (!useCheckpointState && sessionState.messages.length > 0) {
        messages = sessionState.messages;
      }
      const content = await run.processStream(
        { ...normalizedInput.state, messages },
        callerConfig,
        options.streamOptions
      );
      const runMessages = run.getRunMessages() ?? [];
      if (isSessionThread) {
        for (const message of runMessages) {
          await this.store?.appendMessage(message);
        }
      }
      this.calibrationRatio = run.getCalibrationRatio();
      const interrupt = run.getInterrupt();
      const haltedReason = run.getHaltReason();
      if (interrupt) {
        emitTerminalEvent({ type: 'run.interrupted' });
        await this.store?.appendRunEvent('run.interrupted', interrupt, {
          runId,
          threadId,
        });
      } else if (haltedReason != null && haltedReason !== '') {
        emitTerminalEvent({ type: 'run.halted', data: haltedReason });
        await this.store?.appendRunEvent('run.halted', haltedReason, {
          runId,
          threadId,
        });
      } else {
        emitTerminalEvent({ type: 'run.completed' });
        await this.store?.appendRunEvent('run.completed', undefined, {
          runId,
          threadId,
        });
      }
      await this.recordCheckpoint({
        source: 'run',
        runId,
        threadId,
        config: callerConfig,
      });
      const contentParts = (content ?? handlerResult.contentParts).filter(
        (part): part is t.MessageContentComplex => part != null
      );
      return {
        text: contentToText(contentParts),
        content: contentParts,
        messages: runMessages,
        usage: handlerResult.usage,
        steps: handlerResult.steps,
        interrupt,
        haltedReason,
        runId,
        threadId,
      };
    } catch (error) {
      emitTerminalEvent({
        type: 'run.failed',
        data: error instanceof Error ? error.message : String(error),
      });
      await this.store?.appendRunEvent('run.failed', error, {
        runId,
        threadId,
      });
      await this.recordCheckpoint({
        source: 'run',
        runId,
        threadId,
        config: callerConfig,
      });
      throw error;
    }
  }

  async run(
    input: AgentSessionInput,
    options: AgentSessionRunOptions = {}
  ): Promise<AgentSessionRunResult> {
    return this.runInternal(input, options);
  }

  stream(
    input: AgentSessionInput,
    options: AgentSessionRunOptions = {}
  ): AgentSessionStream {
    const stream = new LiveAgentSessionStream();
    const resultPromise = this.runInternal(input, options, (event) => {
      stream.push(event);
    }).then(
      (result) => {
        stream.complete();
        return result;
      },
      (error: unknown) => {
        stream.fail(error);
        throw error;
      }
    );
    stream.setResultPromise(resultPromise);
    return stream;
  }

  async resumeSession(pathOrId?: string): Promise<AgentSession> {
    if (pathOrId == null || pathOrId === '') {
      const sessions = await JsonlSessionStore.list(this.cwd);
      if (sessions.length === 0) {
        throw new Error(`No sessions found for ${this.cwd}`);
      }
      this.store = await JsonlSessionStore.open(sessions[0].path);
      this.cwd = this.store.header.cwd;
      this.threadId = this.store.header.id;
      return this;
    }
    this.store = await JsonlSessionStore.open(pathOrId);
    this.cwd = this.store.header.cwd;
    this.threadId = this.store.header.id;
    return this;
  }

  async clone(options: SessionForkOptions = {}): Promise<AgentSession> {
    if (!this.store) {
      throw new Error('Cannot clone an ephemeral session');
    }
    const store = await this.store.clone(options);
    return new AgentSession({
      runConfig: this.runConfig,
      cwd: store.header.cwd,
      threadId: store.header.id,
      checkpointing: this.checkpointing,
      store,
    });
  }

  async fork(
    entryId: string,
    options: SessionForkOptions = {}
  ): Promise<AgentSession> {
    if (!this.store) {
      throw new Error('Cannot fork an ephemeral session');
    }
    const store = await this.store.fork(entryId, options);
    return new AgentSession({
      runConfig: this.runConfig,
      cwd: store.header.cwd,
      threadId: store.header.id,
      checkpointing: this.checkpointing,
      store,
    });
  }

  async branch(
    entryId: string,
    options: SessionBranchOptions = {}
  ): Promise<void> {
    const store = this.store;
    if (!store) {
      throw new Error('Cannot branch an ephemeral session');
    }
    const target = getSessionBranchTarget(
      store,
      entryId,
      options.position ?? 'at'
    );
    const previousLeafId = store.getLeafEntry()?.id ?? null;
    let leafId = target?.id ?? null;
    const summarizeAbandoned = options.summarizeAbandoned;
    if (summarizeAbandoned !== undefined && summarizeAbandoned !== false) {
      const instructions =
        typeof summarizeAbandoned === 'object'
          ? summarizeAbandoned.instructions
          : undefined;
      const abandonedPath = getAbandonedPathForBranch(
        store,
        previousLeafId,
        leafId
      );
      const summary = await this.compactActivePath(
        { instructions, retainRecentTurns: 0 },
        leafId,
        abandonedPath
      );
      leafId = summary?.id ?? leafId;
    }
    if (leafId === previousLeafId) {
      return;
    }
    await store.setLeaf(leafId);
    await this.resetCheckpointThreads('branch');
  }

  async compact(options: SessionCompactOptions = {}): Promise<void> {
    const summary = await this.compactActivePath(options, null);
    if (summary) {
      await this.resetCheckpointThreads('compact');
    }
  }

  private async compactActivePath(
    options: SessionCompactOptions = {},
    parentId: string | null,
    path?: SessionEntry[]
  ): Promise<Extract<SessionEntry, { type: 'summary' }> | undefined> {
    const store = this.store;
    if (!store) {
      throw new Error('Cannot compact an ephemeral session');
    }
    const activePath = path ?? store.getPath();
    const sessionState = createSessionRunState(activePath);
    const messageEntries = activePath.filter(isMessageEntry);
    if (sessionState.messages.length === 0) {
      return undefined;
    }
    const compactRunId = createRunId();
    const agentContext = AgentContext.fromConfig(
      createAgentInputFromGraphConfig(
        this.runConfig.graphConfig,
        sessionState.initialSummary,
        options.retainRecentTurns,
        options.instructions
      ),
      this.runConfig.tokenCounter
    );
    const graph = createManualCompactGraph({
      runId: compactRunId,
      customHandlers: this.runConfig.customHandlers,
      hooks: this.runConfig.hooks,
    });
    const summarizeNode = createSummarizeNode({
      agentContext,
      graph: graph.graph,
      generateStepId: (stepKey): [string, number] => [
        `${stepKey}-${compactRunId}`,
        graph.graph.contentData.length,
      ],
    });
    const summarizedState = await summarizeNode(
      {
        messages: sessionState.messages,
        summarizationRequest: {
          remainingContextTokens: agentContext.maxContextTokens ?? 0,
          agentId: agentContext.agentId,
        },
      },
      {
        configurable: { thread_id: this.threadId },
        metadata: { run_id: compactRunId },
      }
    );
    const completedSummaryText = getSummaryText(graph.completedSummary);
    const contextSummaryText = agentContext.getSummaryText();
    let summaryText = completedSummaryText;
    if (summaryText === '' && contextSummaryText != null) {
      summaryText = contextSummaryText;
    }
    if (summaryText === '') {
      return undefined;
    }
    const retainedMessages = filterRemoveMessages(
      summarizedState.messages ?? []
    );
    let retainedEntryIds: string[] = [];
    if (retainedMessages.length > 0) {
      retainedEntryIds = messageEntries
        .slice(-retainedMessages.length)
        .map((entry) => entry.id);
    }
    const retainedEntryIdSet = new Set(retainedEntryIds);
    const summarized = messageEntries.filter(
      (entry) => !retainedEntryIdSet.has(entry.id)
    );
    const summary = await store.appendEntryForCompaction({
      text: summaryText,
      tokenCount: getSummaryTokenCount(graph.completedSummary),
      retainedEntryIds,
      summarizedEntryIds: summarized.map((entry) => entry.id),
      instructions: options.instructions,
      parentId,
    });
    let retainedParentId: string | null = summary.id;
    for (const message of retainedMessages) {
      const retainedMessage = await store.appendMessage(
        message,
        retainedParentId
      );
      retainedParentId = retainedMessage.id;
    }
    await store.appendCompactionEntry({
      summaryEntryId: summary.id,
      retainedEntryIds,
      summarizedEntryIds: summarized.map((entry) => entry.id),
    });
    return summary;
  }

  async resumeInterrupt<TResume>(
    resumeValue: TResume,
    options: AgentSessionRunOptions = {}
  ): Promise<AgentSessionRunResult> {
    const runId = options.runId ?? createRunId();
    const threadId = options.threadId ?? this.threadId;
    const isSessionThread = threadId === this.threadId;
    const callerConfig = createCallerConfig(threadId, options);
    await this.store?.appendRunEvent('run.started', undefined, {
      runId,
      threadId,
    });
    const handlerResult = createRunHandlers({
      runId,
      threadId,
      userHandlers: this.runConfig.customHandlers,
    });
    const sessionState = createSessionRunState(
      isSessionThread ? (this.store?.getPath() ?? []) : []
    );
    try {
      const run = await Run.create<t.IState>({
        ...this.runConfig,
        runId,
        graphConfig: applyCheckpointingToGraphConfig(
          applyInitialSummaryToGraphConfig(
            this.runConfig.graphConfig,
            sessionState.initialSummary
          ),
          this.checkpointing
        ),
        returnContent: true,
        calibrationRatio: this.calibrationRatio,
        customHandlers: {
          ...(this.runConfig.customHandlers ?? {}),
          ...handlerResult.handlers,
        },
      });
      const content = await run.resume(resumeValue, callerConfig);
      const runMessages = run.getRunMessages() ?? [];
      if (isSessionThread) {
        for (const message of runMessages) {
          await this.store?.appendMessage(message);
        }
      }
      this.calibrationRatio = run.getCalibrationRatio();
      const interrupt = run.getInterrupt();
      const haltedReason = run.getHaltReason();
      if (interrupt) {
        await this.store?.appendRunEvent('run.interrupted', interrupt, {
          runId,
          threadId,
        });
      } else if (haltedReason != null && haltedReason !== '') {
        await this.store?.appendRunEvent('run.halted', haltedReason, {
          runId,
          threadId,
        });
      } else {
        await this.store?.appendRunEvent('run.completed', undefined, {
          runId,
          threadId,
        });
      }
      await this.recordCheckpoint({
        source: 'resume',
        runId,
        threadId,
        config: callerConfig,
      });
      const contentParts = (content ?? handlerResult.contentParts).filter(
        (part): part is t.MessageContentComplex => part != null
      );
      return {
        text: contentToText(contentParts),
        content: contentParts,
        messages: runMessages,
        usage: handlerResult.usage,
        steps: handlerResult.steps,
        interrupt,
        haltedReason,
        runId,
        threadId,
      };
    } catch (error) {
      await this.store?.appendRunEvent('run.failed', error, {
        runId,
        threadId,
      });
      await this.recordCheckpoint({
        source: 'resume',
        runId,
        threadId,
        config: callerConfig,
      });
      throw error;
    }
  }
}

export function createAgentSession(
  config: AgentSessionConfig
): Promise<AgentSession> {
  return AgentSession.create(config);
}
