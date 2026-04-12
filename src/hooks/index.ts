// src/hooks/index.ts
//
// Hook lifecycle system for `@librechat/agents`. Re-exported from
// `src/index.ts` and consumed by `Run.processStream` (RunStart,
// UserPromptSubmit, Stop, StopFailure) and — once the tool-hook PR
// lands — by `ToolNode` (PreToolUse, PostToolUse, etc.).
export { HookRegistry } from './HookRegistry';
export { executeHooks, DEFAULT_HOOK_TIMEOUT_MS } from './executeHooks';
export {
  matchesQuery,
  hasNestedQuantifier,
  MAX_PATTERN_LENGTH,
  MAX_CACHE_SIZE,
} from './matchers';
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
  PermissionDeniedHookOutput,
  SubagentStartHookOutput,
  SubagentStopHookOutput,
  StopHookOutput,
  StopFailureHookOutput,
  PreCompactHookOutput,
  PostCompactHookOutput,
} from './types';
export type { ExecuteHooksOptions } from './executeHooks';
