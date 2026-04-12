// src/hooks/types.ts
import type { BaseMessage } from '@langchain/core/messages';

/**
 * Closed set of hook lifecycle events supported by the hooks system.
 *
 * These mirror the subset of Claude Code's event surface that makes sense
 * for a library context (no filesystem/CLI-specific events). See
 * `docs/hooks-design-report.md` §3.2 for the mapping to existing
 * `@librechat/agents` emission points.
 */
export const HOOK_EVENTS = [
  'RunStart',
  'UserPromptSubmit',
  'PreToolUse',
  'PostToolUse',
  'PostToolUseFailure',
  'PermissionDenied',
  'SubagentStart',
  'SubagentStop',
  'Stop',
  'StopFailure',
  'PreCompact',
  'PostCompact',
] as const;

export type HookEvent = (typeof HOOK_EVENTS)[number];

/** Tool-gating decision; executeHooks folds with `deny > ask > allow` precedence. */
export type ToolDecision = 'allow' | 'deny' | 'ask';

/** Stop-loop decision; `block` means "do not stop, run another turn". Any `block` wins. */
export type StopDecision = 'continue' | 'block';

/**
 * Fields shared by every `HookInput`. Discriminated by `hook_event_name`.
 *
 * - `runId` identifies the current agent run and is always present.
 * - `threadId` identifies the conversation thread when the host has one.
 * - `agentId` is only set when the hook fires inside a subagent scope.
 */
export interface BaseHookInput {
  runId: string;
  threadId?: string;
  agentId?: string;
}

export interface RunStartHookInput extends BaseHookInput {
  hook_event_name: 'RunStart';
  messages: BaseMessage[];
}

export interface UserPromptSubmitHookInput extends BaseHookInput {
  hook_event_name: 'UserPromptSubmit';
  prompt: string;
  attachments?: BaseMessage[];
}

/**
 * Fires before a tool is invoked. Hook may return `deny`/`ask`/`allow` and/or
 * an `updatedInput` that replaces the tool arguments before invocation.
 *
 * `toolInput` is intentionally typed as `Record<string, unknown>` because the
 * SDK is tool-agnostic — concrete tool argument shapes are only known at the
 * call site and are narrowed by the host. This is the one escape hatch in
 * the hook type system.
 */
export interface PreToolUseHookInput extends BaseHookInput {
  hook_event_name: 'PreToolUse';
  toolName: string;
  toolInput: Record<string, unknown>;
  toolUseId: string;
  stepId?: string;
  turn?: number;
}

export interface PostToolUseHookInput extends BaseHookInput {
  hook_event_name: 'PostToolUse';
  toolName: string;
  toolInput: Record<string, unknown>;
  toolOutput: unknown;
  toolUseId: string;
  stepId?: string;
  turn?: number;
}

export interface PostToolUseFailureHookInput extends BaseHookInput {
  hook_event_name: 'PostToolUseFailure';
  toolName: string;
  toolInput: Record<string, unknown>;
  toolUseId: string;
  error: string;
  stepId?: string;
  turn?: number;
}

export interface PermissionDeniedHookInput extends BaseHookInput {
  hook_event_name: 'PermissionDenied';
  toolName: string;
  toolInput: Record<string, unknown>;
  toolUseId: string;
  reason: string;
}

export interface SubagentStartHookInput extends BaseHookInput {
  hook_event_name: 'SubagentStart';
  parentAgentId?: string;
  agentId: string;
  agentType: string;
  inputs: BaseMessage[];
}

export interface SubagentStopHookInput extends BaseHookInput {
  hook_event_name: 'SubagentStop';
  agentId: string;
  agentType: string;
  messages: BaseMessage[];
}

export interface StopHookInput extends BaseHookInput {
  hook_event_name: 'Stop';
  messages: BaseMessage[];
  stopReason?: string;
  stopHookActive: boolean;
}

export interface StopFailureHookInput extends BaseHookInput {
  hook_event_name: 'StopFailure';
  error: string;
  lastAssistantMessage?: BaseMessage;
}

export interface PreCompactHookInput extends BaseHookInput {
  hook_event_name: 'PreCompact';
  messagesBeforeCount: number;
  trigger: 'threshold' | 'manual' | 'error';
}

export interface PostCompactHookInput extends BaseHookInput {
  hook_event_name: 'PostCompact';
  summary: string;
  messagesAfterCount: number;
}

/** Discriminated union of every hook input shape. */
export type HookInput =
  | RunStartHookInput
  | UserPromptSubmitHookInput
  | PreToolUseHookInput
  | PostToolUseHookInput
  | PostToolUseFailureHookInput
  | PermissionDeniedHookInput
  | SubagentStartHookInput
  | SubagentStopHookInput
  | StopHookInput
  | StopFailureHookInput
  | PreCompactHookInput
  | PostCompactHookInput;

/** Compile-time map from event name to its input shape. */
export type HookInputByEvent = {
  RunStart: RunStartHookInput;
  UserPromptSubmit: UserPromptSubmitHookInput;
  PreToolUse: PreToolUseHookInput;
  PostToolUse: PostToolUseHookInput;
  PostToolUseFailure: PostToolUseFailureHookInput;
  PermissionDenied: PermissionDeniedHookInput;
  SubagentStart: SubagentStartHookInput;
  SubagentStop: SubagentStopHookInput;
  Stop: StopHookInput;
  StopFailure: StopFailureHookInput;
  PreCompact: PreCompactHookInput;
  PostCompact: PostCompactHookInput;
};

/**
 * Fields common to every hook output. Hooks that have nothing to say simply
 * return `{}` (or omit the fields below).
 */
export interface BaseHookOutput {
  /** Context string to inject into the conversation. Accumulated across hooks. */
  additionalContext?: string;
  /** True to prevent the next model turn. Any hook can set this. */
  preventContinuation?: boolean;
  /** Reason reported alongside `preventContinuation`. */
  stopReason?: string;
}

export type RunStartHookOutput = BaseHookOutput;

export interface UserPromptSubmitHookOutput extends BaseHookOutput {
  decision?: ToolDecision;
  reason?: string;
}

export interface PreToolUseHookOutput extends BaseHookOutput {
  decision?: ToolDecision;
  reason?: string;
  /**
   * Replacement tool input. Merged into the pending tool call by the host.
   *
   * When multiple hooks set `updatedInput` within a single `executeHooks`
   * call, the last writer in registration order wins (outer loop: matcher
   * registration order; inner loop: hook position within the matcher). The
   * winner is deterministic — `Promise.all` preserves input-array order.
   * Consumers that need a single authoritative rewrite should still scope
   * `updatedInput` to one hook per matcher to avoid confusing precedence.
   */
  updatedInput?: Record<string, unknown>;
}

export interface PostToolUseHookOutput extends BaseHookOutput {
  /**
   * Replacement tool output. Flows through the aggregated result so the
   * host can substitute it before appending the tool result message.
   * Ordering semantics match `PreToolUseHookOutput.updatedInput`:
   * last-writer-wins in registration order.
   */
  updatedOutput?: unknown;
}

export type PostToolUseFailureHookOutput = BaseHookOutput;

export type PermissionDeniedHookOutput = BaseHookOutput;

export interface SubagentStartHookOutput extends BaseHookOutput {
  decision?: ToolDecision;
  reason?: string;
}

export type SubagentStopHookOutput = BaseHookOutput;

export interface StopHookOutput extends BaseHookOutput {
  decision?: StopDecision;
  reason?: string;
}

export type StopFailureHookOutput = BaseHookOutput;

export type PreCompactHookOutput = BaseHookOutput;

export type PostCompactHookOutput = BaseHookOutput;

/** Compile-time map from event name to its output shape. */
export type HookOutputByEvent = {
  RunStart: RunStartHookOutput;
  UserPromptSubmit: UserPromptSubmitHookOutput;
  PreToolUse: PreToolUseHookOutput;
  PostToolUse: PostToolUseHookOutput;
  PostToolUseFailure: PostToolUseFailureHookOutput;
  PermissionDenied: PermissionDeniedHookOutput;
  SubagentStart: SubagentStartHookOutput;
  SubagentStop: SubagentStopHookOutput;
  Stop: StopHookOutput;
  StopFailure: StopFailureHookOutput;
  PreCompact: PreCompactHookOutput;
  PostCompact: PostCompactHookOutput;
};

/** Superset output shape used by the executor's fold loop. */
export type HookOutput =
  | RunStartHookOutput
  | UserPromptSubmitHookOutput
  | PreToolUseHookOutput
  | PostToolUseHookOutput
  | PostToolUseFailureHookOutput
  | PermissionDeniedHookOutput
  | SubagentStartHookOutput
  | SubagentStopHookOutput
  | StopHookOutput
  | StopFailureHookOutput
  | PreCompactHookOutput
  | PostCompactHookOutput;

/**
 * A hook callback is a plain async function registered against a specific
 * event. The `signal` is always supplied by `executeHooks` and combines the
 * batch's parent signal with the per-hook timeout — callbacks that perform
 * long-running work should observe it.
 */
export type HookCallback<E extends HookEvent = HookEvent> = (
  input: HookInputByEvent[E],
  signal: AbortSignal
) => HookOutputByEvent[E] | Promise<HookOutputByEvent[E]>;

/**
 * A matcher groups one or more callbacks under a shared regex filter and
 * shared timeout/once/internal flags. The generic `E` ties the callback
 * types to the event the matcher is registered against.
 */
export interface HookMatcher<E extends HookEvent = HookEvent> {
  /**
   * Regex pattern matched against the event's primary query string (e.g.
   * the tool name for `PreToolUse`, the agent type for `SubagentStart`).
   *
   * Omitted or empty means "always match". For events that do not supply a
   * query string (`RunStart`, `Stop`, etc.), only wildcard matchers fire —
   * a non-empty pattern on such events will never match.
   *
   * Patterns are treated as trusted input: `executeHooks` compiles them
   * with `new RegExp(pattern)` without any sandbox, and a pathological
   * pattern can block the event loop. Host registration code is expected
   * to validate or length-bound patterns that originate from user input.
   */
  pattern?: string;
  /** Callbacks that fire when the matcher hits. Executed in parallel. */
  hooks: HookCallback<E>[];
  /** Per-matcher timeout in ms. Defaults to the executor's batch timeout. */
  timeout?: number;
  /**
   * Atomically remove the matcher before its first dispatch.
   *
   * `executeHooks` claims `once: true` matchers synchronously — between
   * `getMatchers` and its first `await` — so two concurrent calls cannot
   * both dispatch the same matcher. Whichever call runs its sync prefix
   * first wins the matcher; the other sees an empty bucket.
   *
   * Semantics are "at most one dispatch, ever" — if every hook in the
   * matcher throws, the matcher is still gone. Use `once` for
   * fire-and-forget bootstrapping (registration, telemetry, setup). Hosts
   * that need retry semantics should register a normal matcher and
   * self-unregister via the callback returned from `registry.register`.
   */
  once?: boolean;
  /** Internal hooks are excluded from telemetry and non-fatal error logging. */
  internal?: boolean;
}

/**
 * Storage shape for matchers keyed by event. Each event's matcher list is
 * a generic array parameterized by that event type, so lookup via
 * `HooksByEvent[E]` preserves type-safe callback signatures.
 */
export type HooksByEvent = {
  [E in HookEvent]?: HookMatcher<E>[];
};

/**
 * Aggregated result of a single `executeHooks` call. Fields are populated
 * according to the fold rules in `executeHooks.ts`.
 */
export interface AggregatedHookResult {
  /** Folded tool-gating decision; `deny > ask > allow`. */
  decision?: ToolDecision;
  /** Folded stop decision; any `block` wins. */
  stopDecision?: StopDecision;
  /** Reason from the hook that set the winning decision. */
  reason?: string;
  /**
   * Replacement tool input from a `PreToolUse` hook.
   *
   * Last-writer-wins in **registration order**: `executeHooks` uses
   * `Promise.all`, which preserves input-array order, so the fold iterates
   * outcomes in the same order they were pushed — outer loop over matchers
   * as they sit in the registry, inner loop over each matcher's `hooks`
   * array. The winner is therefore deterministic but may not match the
   * order in which hooks actually completed. Consumers that want a single
   * authoritative rewrite should still register one `updatedInput`-setting
   * hook per matcher to avoid subtle precedence bugs.
   */
  updatedInput?: Record<string, unknown>;
  /**
   * Replacement tool output from a `PostToolUse` hook.
   *
   * Same last-writer-wins-in-registration-order semantics as
   * `updatedInput`. Present only when at least one hook set it; `undefined`
   * means "use the original tool output".
   */
  updatedOutput?: unknown;
  /** Accumulated `additionalContext` strings from every hook, in order. */
  additionalContexts: string[];
  /** True if any hook returned `preventContinuation`. */
  preventContinuation?: boolean;
  /**
   * Reason recorded alongside `preventContinuation`. First winner wins:
   * once a hook sets both flags, later hooks that also set
   * `preventContinuation` do not overwrite the reason.
   */
  stopReason?: string;
  /** Error messages from hooks that threw; always present (possibly empty). */
  errors: string[];
}
