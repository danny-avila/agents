import { Command, MemorySaver, isCommand } from '@langchain/langgraph';
import { isBaseMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import { isToolApprovalInterrupt } from '@/types/hitl';
import { TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY } from '@/hooks/types';
import {
  TOOL_APPROVAL_REVIEW_CONFIG_KEY,
  getToolApprovalReviewEvidence,
  createToolApprovalReviewEvidence,
} from '@/hitl/approvalReview';
import {
  attachRunStepResumeState,
  getRunStepResumeState,
  stripRunStepResumeState,
} from '@/tools/runStepResume';

export const TOOL_BATCH_REPLAY_KEY = '__librechat_tool_batch_replay';
const TOOL_BATCH_PAYLOAD_KEY = '__librechat_tool_batch_payload';
const TOOL_BATCH_WRAPPER_KEY = '__librechat_tool_batch_wrapper';

export interface SettledToolBatchResult {
  proposal?: { name: string; args: Record<string, unknown> };
  output: BaseMessage | Command;
  additionalContexts: string[];
  resolvedArgs?: Record<string, unknown>;
  completionHandled?: boolean;
  referenceContent?: string;
}

export interface ToolBatchReplayRecord {
  owner: string;
  batch: string;
  encoding: string;
  data: string;
}

interface ToolBatchReplayState {
  version: 1;
  approvalOwner?: string;
  records: ToolBatchReplayRecord[];
}

/** Use the same message/Command codec as LangGraph checkpoints. */
const serializer = new MemorySaver().serde;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

/** The only host-to-runtime entry point for checkpoint-owned tool replay state. */
export function restoreToolReplayConfig(
  configurable: Record<string, unknown>,
  interruptId: string | undefined,
  payload: unknown
): void {
  delete configurable[TOOL_BATCH_REPLAY_KEY];
  delete configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY];
  const state = getToolBatchReplayState(payload);
  const publicPayload = stripToolBatchReplayState(payload);
  const review = createToolApprovalReviewEvidence(
    interruptId,
    publicPayload,
    state?.approvalOwner
  );
  if (isToolApprovalInterrupt(publicPayload) && review == null) {
    throw new Error('Invalid tool approval checkpoint');
  }
  if (state != null) {
    configurable[TOOL_BATCH_REPLAY_KEY] = state;
  }
  if (review != null) {
    configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY] = review;
  }
}

export function getToolBatchReplayOwner(
  config: RunnableConfig,
  agentId: string
): string {
  return JSON.stringify([
    getToolBatchReplayScope(config),
    config.configurable?.[TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY] == null
      ? (config.configurable?.checkpoint_ns ?? '')
      : '',
    agentId,
  ]);
}

export function getToolBatchReplayScope(
  config: RunnableConfig
): string | undefined {
  const scope =
    config.configurable?.[TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY] ??
    config.configurable?.thread_id;
  return typeof scope === 'string' ? scope : undefined;
}

/** Rebind only the checkpoint-proven source execution when a child fork is created. */
export function rebindToolBatchReplayScope(
  configurable: Record<string, unknown>,
  sourceScope: string,
  destinationScope: string
): void {
  const rebind = (key: string): string => {
    const parts: unknown = JSON.parse(key);
    if (!Array.isArray(parts) || parts[0] !== sourceScope) {
      return key;
    }
    return JSON.stringify([destinationScope, ...parts.slice(1)]);
  };
  const config = { configurable };
  const review = getToolApprovalReviewEvidence(config);
  if (review?.owner != null) {
    configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY] = {
      ...review,
      owner: rebind(review.owner),
    };
  }
  const state = getToolBatchReplayState(configurable);
  if (state != null) {
    configurable[TOOL_BATCH_REPLAY_KEY] = {
      ...state,
      approvalOwner:
        state.approvalOwner == null ? undefined : rebind(state.approvalOwner),
      records: state.records.map((record) => ({
        ...record,
        owner: rebind(record.owner),
        batch: rebind(record.batch),
      })),
    };
  }
}

export function getToolBatchReplayState(
  payload: unknown
): ToolBatchReplayState | undefined {
  const value = stripRunStepResumeState(payload);
  if (
    value == null ||
    typeof value !== 'object' ||
    !(TOOL_BATCH_REPLAY_KEY in value)
  ) {
    return undefined;
  }
  const candidate: unknown = value[TOOL_BATCH_REPLAY_KEY];
  if (candidate === undefined) {
    return undefined;
  }
  const state = candidate as Partial<ToolBatchReplayState> | null;
  if (
    state?.version !== 1 ||
    (state.approvalOwner != null && typeof state.approvalOwner !== 'string') ||
    !Array.isArray(state.records) ||
    state.records.some((value: unknown) => {
      if (value == null || typeof value !== 'object') {
        return true;
      }
      const record = value as Partial<ToolBatchReplayRecord>;
      return (
        typeof record.owner !== 'string' ||
        typeof record.batch !== 'string' ||
        typeof record.encoding !== 'string' ||
        typeof record.data !== 'string'
      );
    })
  ) {
    throw new Error('Invalid tool batch replay checkpoint');
  }
  return structuredClone(state as ToolBatchReplayState);
}

export function stripToolBatchReplayState(payload: unknown): unknown {
  if (
    payload == null ||
    typeof payload !== 'object' ||
    !(TOOL_BATCH_REPLAY_KEY in payload)
  ) {
    return payload;
  }
  const { [TOOL_BATCH_REPLAY_KEY]: _state, ...publicPayload } = payload;
  if (TOOL_BATCH_WRAPPER_KEY in publicPayload && publicPayload[TOOL_BATCH_WRAPPER_KEY] === 1 && TOOL_BATCH_PAYLOAD_KEY in publicPayload) {
    return publicPayload[TOOL_BATCH_PAYLOAD_KEY];
  }
  return publicPayload;
}

export async function attachToolBatchReplayState(
  payload: unknown,
  owner: string,
  batches: ReadonlyMap<string, ReadonlyMap<string, object>>
): Promise<unknown> {
  const publicPayload = stripRunStepResumeState(payload);
  const previous = getToolBatchReplayState(publicPayload);
  const records =
    previous?.records.filter((record) => record.owner !== owner) ?? [];
  for (const [batch, results] of batches) {
    const [encoding, bytes] = await serializer.dumpsTyped([...results]);
    records.push({
      owner,
      batch,
      encoding,
      data: Buffer.from(bytes).toString('base64'),
    });
  }
  const approvalOwner =
    previous?.approvalOwner ??
    (isToolApprovalInterrupt(publicPayload) ? owner : undefined);
  if (records.length === 0 && approvalOwner == null) {
    return payload;
  }
  const result = {
    ...(isRecord(publicPayload) ? publicPayload : {
      [TOOL_BATCH_WRAPPER_KEY]: 1,
      [TOOL_BATCH_PAYLOAD_KEY]: publicPayload,
    }),
    [TOOL_BATCH_REPLAY_KEY]: {
      version: 1,
      approvalOwner,
      records,
    } satisfies ToolBatchReplayState,
  };
  const runStepState = getRunStepResumeState(payload);
  return runStepState == null
    ? result
    : attachRunStepResumeState(result, runStepState);
}

export async function restoreToolBatchReplayState(
  config: RunnableConfig,
  owner: string
): Promise<
  Array<{ batch: string; results: Array<[string, SettledToolBatchResult]> }>
> {
  const state = getToolBatchReplayState(config.configurable);
  if (state == null) {
    return [];
  }
  return Promise.all(
    state.records
      .filter((record) => record.owner === owner)
      .map(async (record) => {
        const values: unknown = await serializer.loadsTyped(
          record.encoding,
          Buffer.from(record.data, 'base64')
        );
        if (!Array.isArray(values)) {
          throw new Error('Invalid settled tool batch results');
        }
        const seen = new Set<string>();
        const results = values.map(
          (entry: unknown): [string, SettledToolBatchResult] => {
            if (
              !Array.isArray(entry) ||
              entry.length !== 2 ||
              typeof entry[0] !== 'string' ||
              seen.has(entry[0])
            ) {
              throw new Error('Invalid settled tool batch results');
            }
            const result = entry[1] as Partial<SettledToolBatchResult> | null;
            if (
              result == null ||
              (!isBaseMessage(result.output) && !isCommand(result.output)) ||
              !Array.isArray(result.additionalContexts) ||
              result.additionalContexts.some(
                (context) => typeof context !== 'string'
              ) ||
              (result.completionHandled != null &&
                typeof result.completionHandled !== 'boolean') ||
              (result.referenceContent != null &&
                typeof result.referenceContent !== 'string') ||
              (result.proposal != null &&
                (typeof result.proposal.name !== 'string' ||
                  !isRecord(result.proposal.args))) ||
              (result.resolvedArgs != null && !isRecord(result.resolvedArgs))
            ) {
              throw new Error('Invalid settled tool batch results');
            }
            seen.add(entry[0]);
            const output = isCommand(result.output)
              ? new Command({
                update: result.output.update,
                goto: result.output.goto,
                graph: result.output.graph,
                resume: result.output.resume,
              })
              : result.output;
            return [
              entry[0],
              {
                ...result,
                output,
                additionalContexts: result.additionalContexts,
              },
            ];
          }
        );
        return { batch: record.batch, results };
      })
  );
}
