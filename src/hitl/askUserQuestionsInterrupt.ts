import type {
  AskUserQuestionBatchItem,
  AskUserQuestionOption,
  AskUserQuestionRequest,
  AskUserQuestionsInterruptPayload,
} from '@/types/hitl';
import { isAskUserQuestionInterrupt } from '@/types/hitl';

/** Maximum questions supported by one batched clarification interaction. */
export const MAX_ASK_USER_QUESTIONS = 4;

/** Safe identifier format for answer-map keys in a batched question. */
export const ASK_USER_QUESTION_ID_PATTERN = /^[A-Za-z][A-Za-z0-9_-]{0,63}$/;

interface AskUserQuestionOptionCandidate {
  label?: unknown;
  value?: unknown;
}

interface AskUserQuestionCandidate {
  question?: unknown;
  description?: unknown;
  options?: unknown;
  multiSelect?: unknown;
}

interface AskUserQuestionBatchItemCandidate extends AskUserQuestionCandidate {
  id?: unknown;
  header?: unknown;
}

function isAskUserQuestionOption(
  value: unknown
): value is AskUserQuestionOption {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const option = value as AskUserQuestionOptionCandidate;
  return typeof option.label === 'string' && typeof option.value === 'string';
}

function isAskUserQuestionOptions(
  value: unknown
): value is AskUserQuestionOption[] {
  if (!Array.isArray(value)) {
    return false;
  }
  for (let index = 0; index < value.length; index++) {
    if (!Object.hasOwn(value, index) || !isAskUserQuestionOption(value[index])) {
      return false;
    }
  }
  return true;
}

function isAskUserQuestionRequest(
  value: unknown
): value is AskUserQuestionRequest {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const question = value as AskUserQuestionCandidate;
  return (
    typeof question.question === 'string' &&
    (question.description === undefined ||
      typeof question.description === 'string') &&
    (question.options === undefined ||
      isAskUserQuestionOptions(question.options)) &&
    (question.multiSelect === undefined ||
      typeof question.multiSelect === 'boolean')
  );
}

function isAskUserQuestionBatchItem(
  value: unknown
): value is AskUserQuestionBatchItem {
  if (!isAskUserQuestionRequest(value)) {
    return false;
  }
  const question = value as AskUserQuestionBatchItemCandidate;
  return (
    typeof question.id === 'string' &&
    ASK_USER_QUESTION_ID_PATTERN.test(question.id) &&
    (question.header === undefined || typeof question.header === 'string')
  );
}

/**
 * Type guard for the batched form of an `ask_user_question` interrupt. Hosts
 * use this to select the multi-question UI and `AskUserQuestionsResolution`.
 */
export function isAskUserQuestionsInterrupt(
  payload: unknown
): payload is AskUserQuestionsInterruptPayload {
  if (
    !isAskUserQuestionInterrupt(payload) ||
    !isAskUserQuestionRequest(payload.question) ||
    (payload.tool_call_id !== undefined &&
      typeof payload.tool_call_id !== 'string') ||
    !Array.isArray(payload.questions) ||
    payload.questions.length === 0 ||
    payload.questions.length > MAX_ASK_USER_QUESTIONS
  ) {
    return false;
  }

  const ids = new Set<string>();
  for (const question of payload.questions) {
    if (!isAskUserQuestionBatchItem(question) || ids.has(question.id)) {
      return false;
    }
    ids.add(question.id);
  }
  return true;
}
