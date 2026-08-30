import type { RunStepResumeEntry, RunStepResumeState } from '@/types';

const RUN_STEP_RESUME_STATE_KEY = '__librechat_run_step_resume_state';
const RUN_STEP_RESUME_PAYLOAD_KEY = '__librechat_run_step_resume_payload';
const RUN_STEP_RESUME_WRAPPER_KEY = '__librechat_run_step_resume_wrapper';

type RunStepResumePayload = {
  [RUN_STEP_RESUME_STATE_KEY]: RunStepResumeState;
  [RUN_STEP_RESUME_PAYLOAD_KEY]: unknown;
  [RUN_STEP_RESUME_WRAPPER_KEY]: 1;
};

export function isRunStepResumeState(
  value: unknown
): value is RunStepResumeState {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const state = value as Partial<RunStepResumeState>;
  if (
    state.version !== 1 ||
    !Number.isSafeInteger(state.revision) ||
    (state.revision ?? -1) < 0 ||
    !Number.isSafeInteger(state.nextIndex) ||
    (state.nextIndex ?? -1) < 0 ||
    (state.stopContinuationCount != null &&
      (!Number.isSafeInteger(state.stopContinuationCount) ||
        state.stopContinuationCount < 0)) ||
    (state.stopContinuationExecutionId != null &&
      (typeof state.stopContinuationExecutionId !== 'string' ||
        state.stopContinuationExecutionId.length === 0)) ||
    (state.streamSegment != null &&
      (!Number.isSafeInteger(state.streamSegment) ||
        state.streamSegment < 0)) ||
    !Array.isArray(state.toolCallSteps) ||
    !Array.isArray(state.steps)
  ) {
    return false;
  }
  const toolCallSteps = new Map<string, string>();
  for (const reference of state.toolCallSteps as unknown[]) {
    if (reference == null || typeof reference !== 'object') {
      return false;
    }
    const { toolCallId, stepId } = reference as {
      toolCallId?: unknown;
      stepId?: unknown;
    };
    if (
      typeof toolCallId !== 'string' ||
      toolCallId === '' ||
      typeof stepId !== 'string' ||
      stepId === '' ||
      toolCallSteps.has(toolCallId)
    ) {
      return false;
    }
    toolCallSteps.set(toolCallId, stepId);
  }
  const stepIds = new Set<string>();
  for (const value of state.steps as unknown[]) {
    if (value == null || typeof value !== 'object') {
      return false;
    }
    const entry = value as Partial<RunStepResumeEntry>;
    const step = entry.step;
    if (
      step == null ||
      typeof step.id !== 'string' ||
      step.id === '' ||
      !Number.isSafeInteger(step.index) ||
      step.index < 0 ||
      !Array.isArray(entry.pendingToolCallIds) ||
      entry.pendingToolCallIds.some(
        (toolCallId) => typeof toolCallId !== 'string' || toolCallId === ''
      ) ||
      (entry.latestCompletionAt != null &&
        !Number.isFinite(entry.latestCompletionAt)) ||
      typeof entry.openMessageStep !== 'boolean' ||
      entry.pendingToolCallIds.some(
        (toolCallId) => toolCallSteps.get(toolCallId) !== step.id
      ) ||
      stepIds.has(step.id)
    ) {
      return false;
    }
    stepIds.add(step.id);
  }
  return true;
}

function isRunStepResumePayload(value: unknown): value is RunStepResumePayload {
  if (value == null || typeof value !== 'object') {
    return false;
  }
  const payload = value as Partial<RunStepResumePayload>;
  return (
    payload[RUN_STEP_RESUME_WRAPPER_KEY] === 1 &&
    Object.prototype.hasOwnProperty.call(value, RUN_STEP_RESUME_PAYLOAD_KEY) &&
    isRunStepResumeState(payload[RUN_STEP_RESUME_STATE_KEY])
  );
}

export function attachRunStepResumeState(
  payload: unknown,
  state: RunStepResumeState
): object {
  const publicPayload = isRunStepResumePayload(payload)
    ? payload[RUN_STEP_RESUME_PAYLOAD_KEY]
    : payload;
  return {
    [RUN_STEP_RESUME_WRAPPER_KEY]: 1,
    [RUN_STEP_RESUME_PAYLOAD_KEY]: publicPayload,
    [RUN_STEP_RESUME_STATE_KEY]: state,
  } satisfies RunStepResumePayload;
}

export function getRunStepResumeState(
  payload: unknown
): RunStepResumeState | undefined {
  return isRunStepResumePayload(payload)
    ? payload[RUN_STEP_RESUME_STATE_KEY]
    : undefined;
}

export function stripRunStepResumeState(payload: unknown): unknown {
  return isRunStepResumePayload(payload)
    ? payload[RUN_STEP_RESUME_PAYLOAD_KEY]
    : payload;
}
