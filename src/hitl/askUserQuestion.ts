/**
 * Typed convenience wrapper around LangGraph's `interrupt()` for the
 * `ask_user_question` interrupt category. Lets a custom graph node
 * (or a tool implementation) suspend execution to collect a free-form
 * answer from the human, without the host having to assemble the
 * interrupt payload by hand. The companion to `Run.resume(answer)` on
 * the host side.
 *
 * AsyncLocalStorage anchoring: this helper does NOT call
 * `runWithConfig` itself — it expects to be invoked from inside a
 * LangGraph node where the framework has already established the
 * runnable config. ToolNode is the one place in this codebase that
 * needs the manual `runWithConfig` shim, because its
 * `RunnableCallable.trace = false` skips the upstream tracing path
 * that normally sets up the AsyncLocalStorage frame; ordinary user
 * nodes (RunnableLambda, addNode callbacks) do not have that
 * constraint.
 */

import { interrupt } from '@langchain/langgraph';
import type {
  AskUserQuestionRequest,
  AskUserQuestionResolution,
  AskUserQuestionInterruptPayload,
} from '@/types/hitl';

/**
 * Suspend the current graph node to ask the human a question. Returns
 * the host-supplied resolution after `Run.resume(resolution)` is
 * called against a Run rebuilt with the same `thread_id` and
 * checkpointer.
 *
 * On the FIRST call (no resume value available), `interrupt()` throws
 * a `GraphInterrupt` that LangGraph catches; this function does not
 * return — execution unwinds, the SDK persists the checkpoint, and
 * the run completes with `run.getInterrupt()` returning a
 * `RunInterruptResult` whose `payload` is an
 * `AskUserQuestionInterruptPayload`.
 *
 * On RESUME, LangGraph re-runs the node from the start and this call
 * returns the host's `AskUserQuestionResolution` directly.
 *
 * Hosts that prefer the raw `interrupt()` (e.g., to attach extra
 * metadata) can construct an `AskUserQuestionInterruptPayload` and
 * call `interrupt()` themselves — this helper is purely convenience.
 *
 * @example
 * ```ts
 * const builder = new StateGraph(MessagesAnnotation)
 *   .addNode('clarifier', () => {
 *     const { answer } = askUserQuestion({
 *       question: 'Which environment should I deploy to?',
 *       options: [
 *         { label: 'Staging', value: 'staging' },
 *         { label: 'Production', value: 'production' },
 *       ],
 *     });
 *     return { messages: [new HumanMessage(`Use ${answer}`)] };
 *   });
 * ```
 */
export function askUserQuestion(
  question: AskUserQuestionRequest
): AskUserQuestionResolution {
  const payload: AskUserQuestionInterruptPayload = {
    type: 'ask_user_question',
    question,
  };
  return interrupt<AskUserQuestionInterruptPayload, AskUserQuestionResolution>(
    payload
  );
}
