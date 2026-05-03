// src/types/hitl.ts
//
// First-class human-in-the-loop (HITL) types for `@librechat/agents`.
// Surfaces the interrupt payload that `ToolNode` raises when a `PreToolUse`
// hook returns `decision: 'ask'` and HITL is enabled on the run, plus the
// resume-decision shape the host returns to continue or reject the tool.
//
// Mirrors the LangChain HITL middleware shape (action_requests /
// review_configs) so hosts and clients can share rendering/UI semantics
// across the langchain ecosystem.

/** Per-tool approval request emitted inside an interrupt payload. */
export interface ToolApprovalRequest {
  /** Stable id of the tool call (matches LangGraph `ToolCall.id`). */
  tool_call_id: string;
  /** Tool name being invoked. */
  name: string;
  /**
   * Arguments the tool is about to be invoked with — already resolved by
   * any `{{tool<i>turn<n>}}` references and any `updatedInput` returned
   * by the firing PreToolUse hook.
   */
  arguments: Record<string, unknown>;
  /**
   * Optional reason the hook supplied for asking (e.g., "destructive
   * filesystem write"). Hosts can render this verbatim.
   */
  description?: string;
}

/** Allowed host-side decisions for a `tool_approval` interrupt. */
export type ToolApprovalDecisionType =
  | 'approve'
  | 'reject'
  | 'edit'
  | 'respond';

/** Per-action review configuration paired with each action_request. */
export interface ToolApprovalReviewConfig {
  /** Tool name (matches the `name` field on the corresponding action_request). */
  action_name: string;
  /** Decisions the host UI is allowed to surface for this action. */
  allowed_decisions: ToolApprovalDecisionType[];
}

/**
 * Resume value the host returns through `Run.resume(decisions)` after a
 * `tool_approval` interrupt. One entry per action_request, in the same
 * order. Hosts may also return a record keyed by `tool_call_id`; the SDK
 * handles either shape.
 *
 * Variants:
 *   - `approve`: run the tool with its original (or hook-rewritten) args.
 *   - `reject`: skip the tool, emit a blocked error `ToolMessage` with
 *     `reason` surfaced to the model.
 *   - `edit`: replace the tool's args with `updatedInput` (re-resolves
 *     any `{{tool<i>turn<n>}}` placeholders) and run the tool.
 *   - `respond`: skip the tool entirely and emit `responseText` as a
 *     successful `ToolMessage`. Mirrors LangChain HITL middleware's
 *     `respond` semantic — the human supplies the result the model sees,
 *     bypassing tool execution. Useful when the user wants to short-circuit
 *     a tool call with a hand-written answer (e.g., "don't actually run
 *     the search, just tell the model 'no relevant results'").
 */
export type ToolApprovalDecision =
  | { type: 'approve' }
  | { type: 'reject'; reason?: string }
  | { type: 'edit'; updatedInput: Record<string, unknown> }
  | { type: 'respond'; responseText: string };

/** Map form of resume decisions, keyed by tool call id. */
export type ToolApprovalDecisionMap = Record<string, ToolApprovalDecision>;

/**
 * Categories of human-in-the-loop interrupts the SDK can raise. Hosts
 * narrow on `HumanInterruptPayload.type` to determine which payload
 * shape they're handling and which resume value to send back through
 * `Run.resume()`.
 */
export type HumanInterruptType = 'tool_approval' | 'ask_user_question';

/**
 * Structured payload the SDK passes to `interrupt()` when one or more
 * pending tool calls require host approval. All `ask`-decision tool calls
 * from a single ToolNode batch are bundled into one interrupt so the host
 * can render and resolve them together.
 *
 * Resume value: `ToolApprovalDecision[]` (in `action_requests` order) or
 * `ToolApprovalDecisionMap` (keyed by `tool_call_id`).
 */
export interface ToolApprovalInterruptPayload {
  type: 'tool_approval';
  action_requests: ToolApprovalRequest[];
  review_configs: ToolApprovalReviewConfig[];
}

/**
 * Pre-defined option the user can pick when answering an
 * `ask_user_question` interrupt. The selected option's `value` becomes
 * the resume value's `answer` field.
 */
export interface AskUserQuestionOption {
  /** Human-readable label rendered in the host UI. */
  label: string;
  /** Value returned via `AskUserQuestionResolution.answer` if picked. */
  value: string;
}

/** Question request emitted inside an `ask_user_question` interrupt. */
export interface AskUserQuestionRequest {
  /** The question to ask the human. */
  question: string;
  /** Optional context / description rendered alongside the question. */
  description?: string;
  /**
   * Optional pre-defined response options. When present, hosts can render
   * a picker; the user may still type a free-form answer when the host
   * UI allows it. Omit to require a free-form answer.
   */
  options?: AskUserQuestionOption[];
}

/**
 * Structured payload the SDK passes to `interrupt()` when an agent (or
 * a custom node) needs to ask the user a clarifying question. Mirrors
 * Claude Code's `AskUserQuestion` semantic. Resume value:
 * `AskUserQuestionResolution`.
 */
export interface AskUserQuestionInterruptPayload {
  type: 'ask_user_question';
  question: AskUserQuestionRequest;
}

/**
 * Discriminated union of every interrupt payload the SDK raises. New
 * variants can be added without breaking existing handlers as long as
 * those handlers check `payload.type` before reading variant-specific
 * fields. Use the `isToolApprovalInterrupt` / `isAskUserQuestionInterrupt`
 * type guards for ergonomic narrowing.
 */
export type HumanInterruptPayload =
  | ToolApprovalInterruptPayload
  | AskUserQuestionInterruptPayload;

/** Resume value the host returns for an `ask_user_question` interrupt. */
export interface AskUserQuestionResolution {
  /**
   * The human's answer. Free-form text, or — when `options` were
   * provided — one of the option `value`s. Hosts may also send any
   * structured object their custom UI defines; see the host docs for
   * what your downstream consumer expects.
   */
  answer: string;
}

/** Type guard narrowing a `HumanInterruptPayload` to the tool-approval variant. */
export function isToolApprovalInterrupt(
  payload: HumanInterruptPayload
): payload is ToolApprovalInterruptPayload {
  return payload.type === 'tool_approval';
}

/** Type guard narrowing a `HumanInterruptPayload` to the ask-user-question variant. */
export function isAskUserQuestionInterrupt(
  payload: HumanInterruptPayload
): payload is AskUserQuestionInterruptPayload {
  return payload.type === 'ask_user_question';
}

/**
 * Run-level configuration controlling HITL semantics. **HITL is enabled
 * by default** — the SDK assumes hosts want first-class human review
 * support unless they explicitly opt out. When enabled:
 *
 *   - `PreToolUse` hooks returning `decision: 'ask'` raise a real
 *     LangGraph `interrupt()` instead of being treated as a synchronous
 *     deny.
 *   - `Run.create` installs a `MemorySaver` checkpointer fallback on the
 *     run's compile options if the host did not provide one, since
 *     LangGraph requires a checkpointer to suspend and resume.
 *
 * Pass `{ enabled: false }` to fall back to the pre-HITL behavior: `ask`
 * decisions are fail-closed (blocked with an error `ToolMessage`) and no
 * checkpointer is implicitly attached.
 *
 * Note on idempotency: when an interrupt fires, LangGraph re-runs the
 * interrupted node from the start on resume, which fires `PreToolUse`
 * hooks again. Hooks that produce side effects (logging, external calls)
 * will see two invocations per paused turn.
 */
export interface HumanInTheLoopConfig {
  /**
   * Master switch. Defaults to `true` — omit the field to keep HITL on,
   * or set `false` to opt out.
   */
  enabled?: boolean;
}

/**
 * Snapshot of an in-flight interrupt surfaced from `Run.processStream`
 * via `run.getInterrupt()`. Hosts persist this alongside their job
 * record so they can later call `Run.resume(decisions)` against a Run
 * compiled with the same `thread_id` / checkpointer.
 */
export interface RunInterruptResult {
  /** Stable id of the LangGraph interrupt (from `Interrupt.id`). */
  interruptId: string;
  /** `thread_id` the run was bound to — required to resume. */
  threadId?: string;
  /** Structured payload describing what needs human input. */
  payload: HumanInterruptPayload;
}
