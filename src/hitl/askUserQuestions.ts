import { interrupt } from '@langchain/langgraph';
import type {
  AskUserQuestionBatchItem,
  AskUserQuestionRequest,
  AskUserQuestionsInterruptPayload,
  AskUserQuestionsRequest,
  AskUserQuestionsResolution,
} from '@/types/hitl';
import {
  ASK_USER_QUESTION_ID_PATTERN,
  MAX_ASK_USER_QUESTIONS,
} from './askUserQuestionsInterrupt';

function validateQuestions(
  questions: readonly AskUserQuestionBatchItem[]
): AskUserQuestionBatchItem {
  if (questions.length === 0) {
    throw new RangeError('askUserQuestions requires at least one question.');
  }
  if (questions.length > MAX_ASK_USER_QUESTIONS) {
    throw new RangeError(
      `askUserQuestions accepts at most ${MAX_ASK_USER_QUESTIONS} questions.`
    );
  }

  const ids = new Set<string>();
  for (const question of questions) {
    if (!ASK_USER_QUESTION_ID_PATTERN.test(question.id)) {
      throw new Error(
        'askUserQuestions requires each question id to match [A-Za-z][A-Za-z0-9_-]{0,63}.'
      );
    }
    if (ids.has(question.id)) {
      throw new Error(
        `askUserQuestions requires unique question ids; received "${question.id}" more than once.`
      );
    }
    ids.add(question.id);
  }
  return questions[0];
}

interface AskUserQuestionsResolutionCandidate {
  answers?: unknown;
}

function validateResolution(
  value: unknown,
  questions: readonly AskUserQuestionBatchItem[]
): AskUserQuestionsResolution {
  if (typeof value !== 'object' || value === null) {
    throw new TypeError('askUserQuestions requires an answers object.');
  }
  const answers = (value as AskUserQuestionsResolutionCandidate).answers;
  if (
    typeof answers !== 'object' ||
    answers === null ||
    Array.isArray(answers)
  ) {
    throw new TypeError('askUserQuestions requires an answers object.');
  }

  const validated: Record<string, string> = {};
  for (const question of questions) {
    const descriptor = Object.getOwnPropertyDescriptor(answers, question.id);
    const answer: unknown = descriptor?.value;
    if (descriptor == null || typeof answer !== 'string') {
      throw new TypeError(
        `askUserQuestions requires a string answer for question id "${question.id}".`
      );
    }
    validated[question.id] = answer;
  }
  return { answers: validated };
}

/**
 * Suspend once to collect answers to several related questions. The first
 * question is also included in the legacy `question` field so existing hosts
 * can render a useful fallback during a staged rollout.
 *
 * Question ids must be non-empty and unique within the batch. The helper
 * accepts at most four questions so hosts can render the interaction as one
 * focused decision surface rather than an unbounded form.
 *
 * @example
 * ```ts
 * const { answers } = askUserQuestions({
 *   questions: [
 *     { id: 'environment', question: 'Which environment?' },
 *     { id: 'region', question: 'Which region?' },
 *   ],
 * });
 * return `Deploy to ${answers.environment} in ${answers.region}`;
 * ```
 */
export function askUserQuestions(
  request: AskUserQuestionsRequest,
  options?: { toolCallId?: string }
): AskUserQuestionsResolution {
  const first = validateQuestions(request.questions);
  const fallback: AskUserQuestionRequest = {
    question: first.question,
    ...(first.description != null && { description: first.description }),
    ...(first.options != null && { options: first.options }),
    ...(first.multiSelect != null && { multiSelect: first.multiSelect }),
  };
  const payload: AskUserQuestionsInterruptPayload = {
    type: 'ask_user_question',
    question: fallback,
    questions: request.questions,
    ...(options?.toolCallId != null &&
      options.toolCallId !== '' && { tool_call_id: options.toolCallId }),
  };

  const resolution = interrupt<AskUserQuestionsInterruptPayload, unknown>(
    payload
  );
  return validateResolution(resolution, request.questions);
}
