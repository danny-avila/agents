// src/hooks/index.ts
//
// Hook lifecycle system for `@librechat/agents`. Re-exported from
// `src/index.ts` and consumed by `Run.processStream` (RunStart,
// UserPromptSubmit, Stop, StopFailure), `ToolNode.dispatchToolEvents`
// (PreToolUse, PostToolUse, PostToolUseFailure, PermissionDenied),
// `createSummarizeNode` (PreCompact, PostCompact), and
// `SubagentExecutor.execute` (SubagentStart, SubagentStop).
export { HookRegistry } from './HookRegistry';
export type { HookHaltSignal } from './HookRegistry';
export { executeHooks, DEFAULT_HOOK_TIMEOUT_MS } from './executeHooks';
/**
 * Feature probe for hosts: hook outputs support `injectedMessages`
 * (per-message graph-state injection at the `PostToolBatch` boundary).
 * Hosts must gate drain-style hooks on this so a queued message can never
 * be consumed by an SDK version that would silently drop it.
 */
export const HOOK_INJECTED_MESSAGES_CAPABLE = true;
export {
  matchesQuery,
  hasNestedQuantifier,
  MAX_PATTERN_LENGTH,
  MAX_CACHE_SIZE,
} from './matchers';
export { createToolPolicyHook } from './createToolPolicyHook';
export type { ToolPolicyMode, ToolPolicyConfig } from './createToolPolicyHook';
export { createWorkspacePolicyHook } from './createWorkspacePolicyHook';
export type {
  OutsideAccessPolicy,
  WorkspacePolicyConfig,
  PathExtractor,
} from './createWorkspacePolicyHook';
export { HOOK_EVENTS } from './types';
export type {
  HookEvent,
  HookInput,
  HookOutput,
  HookCallback,
  HookMatcher,
  HooksByEvent,
  HookInputByEvent,
  HookOutputByEvent,
  BaseHookInput,
  BaseHookOutput,
  ToolDecision,
  StopDecision,
  AggregatedHookResult,
  RunStartHookInput,
  UserPromptSubmitHookInput,
  PreToolUseHookInput,
  PostToolUseHookInput,
  PostToolUseFailureHookInput,
  PostToolBatchHookInput,
  PostToolBatchEntry,
  PermissionDeniedHookInput,
  SubagentStartHookInput,
  SubagentStopHookInput,
  StopHookInput,
  StopFailureHookInput,
  PreCompactHookInput,
  PostCompactHookInput,
  RunStartHookOutput,
  UserPromptSubmitHookOutput,
  PreToolUseHookOutput,
  PostToolUseHookOutput,
  PostToolUseFailureHookOutput,
  PostToolBatchHookOutput,
  PermissionDeniedHookOutput,
  SubagentStartHookOutput,
  SubagentStopHookOutput,
  StopHookOutput,
  StopFailureHookOutput,
  PreCompactHookOutput,
  PostCompactHookOutput,
} from './types';
export type { ExecuteHooksOptions } from './executeHooks';
