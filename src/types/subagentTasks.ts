import type { BaseMessage } from '@langchain/core/messages';
import type { SubagentUpdateEvent } from './graph';
import type { InjectedMessage } from './tools';

/** Terminal and in-flight states for a detached subagent task. */
export type SubagentTaskStatus =
  | 'running'
  | 'completed'
  | 'error'
  | 'cancelled';

/** Where a pending parent message may enter the child run. */
export type SubagentTaskBoundary = 'preempt' | 'tool' | 'turn';

/** Parent-to-child control operations accepted while a task is running. */
export type SubagentTaskControlCommand =
  | { action: 'steer' | 'queue' | 'interrupt'; message: string }
  | { action: 'cancel' }
  | { action: 'cancel_message'; controlId: string };

/** Small, payload-free progress view safe to retain between parent turns. */
export interface SubagentTaskProgress {
  phase: SubagentUpdateEvent['phase'];
  at: number;
  eventCount: number;
  label?: string;
}
/** Read-only task metadata. Results are exposed only through `claim`. */
export interface SubagentTaskSnapshot {
  /** Handle for this child-conversation execution within its trusted scope. */
  taskId: string;
  subagentType: string;
  status: SubagentTaskStatus;
  createdAt: number;
  updatedAt: number;
  resultAvailable: boolean;
  resultClaimed: boolean;
  pendingControls: number;
  progress?: SubagentTaskProgress;
  error?: string;
}

export type SubagentTaskClaim =
  | { status: 'running'; task: SubagentTaskSnapshot }
  | { status: 'completed'; task: SubagentTaskSnapshot; result: string }
  | { status: 'error'; task: SubagentTaskSnapshot; error: string }
  | { status: 'cancelled'; task: SubagentTaskSnapshot; error: string }
  | { status: 'claimed'; task: SubagentTaskSnapshot }
  | { status: 'not_found' };

export type SubagentTaskControlResult =
  | {
      status: 'accepted';
      task: SubagentTaskSnapshot;
      controlId?: string;
    }
  | { status: 'cancelled'; task: SubagentTaskSnapshot }
  | { status: 'not_running'; task: SubagentTaskSnapshot }
  | { status: 'not_found' }
  | { status: 'control_not_found'; task: SubagentTaskSnapshot }
  | { status: 'invalid'; message: string };

/**
 * Child-side view supplied to one detached execution. It intentionally owns
 * only cancellation, bounded message drains, and payload-free progress — no
 * request/response object or host transport can leak into retained task state.
 */
export interface SubagentTaskRuntime {
  readonly taskId: string;
  readonly signal: AbortSignal;
  shouldPreempt(): boolean;
  drain(boundary: SubagentTaskBoundary): InjectedMessage[];
  closeTurn(): { closed: boolean; messages: InjectedMessage[] };
  reportProgress(event: SubagentUpdateEvent): void;
}

export interface SubagentTaskStartRequest {
  scopeId: string;
  idempotencyKey: string;
  /** Stable hash of model-writable inputs used to reject conflicting replays. */
  requestFingerprint?: string;
  subagentType: string;
  /**
   * Starts one ephemeral execution lease. The canonical child transcript is
   * returned so a host-owned store may persist it for a later fresh run;
   * retaining a graph/checkpoint after terminal completion is unnecessary.
   */
  run(runtime: SubagentTaskRuntime): Promise<{
    content: string;
    messages?: BaseMessage[];
  }>;
}

export type SubagentTaskStartResult =
  | {
      accepted: true;
      isNew: boolean;
      task: SubagentTaskSnapshot;
    }
  | { accepted: false; reason: 'capacity' }
  | {
      accepted: false;
      reason: 'conflict';
      task: SubagentTaskSnapshot;
    };

/**
 * Host-replaceable store contract used by the SDK's subagent tool. The store
 * should normally outlive individual `Run` instances. Durable hosts may
 * persist the transcript returned by `run` under the task/conversation
 * lineage and start a fresh execution for a later turn.
 */
export interface SubagentTaskStore {
  start(request: SubagentTaskStartRequest): SubagentTaskStartResult;
  get(scopeId: string, taskId: string): SubagentTaskSnapshot | undefined;
  list(scopeId: string): SubagentTaskSnapshot[];
  claim(scopeId: string, taskId: string): SubagentTaskClaim;
  control(
    scopeId: string,
    taskId: string,
    command: SubagentTaskControlCommand
  ): SubagentTaskControlResult;
}

/** Trusted, host-selected task namespace. It is never model-writable. */
export interface SubagentTaskConfig {
  store: SubagentTaskStore;
  scopeId: string;
}
