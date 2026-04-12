// src/hooks/index.ts
//
// Phase 1 PR 1: this directory is purely additive and is NOT re-exported
// from `src/index.ts`. The types and classes below are consumed internally
// in subsequent PRs that wire the registry into `Run`, `Graph`, and
// `ToolNode`. Leaving it unexported keeps the public API surface frozen
// until the integration layer is in place.
export { HookRegistry } from './HookRegistry';
export { executeHooks, DEFAULT_HOOK_TIMEOUT_MS } from './executeHooks';
export {
  matchesQuery,
  clearMatcherCache,
  getMatcherCacheSize,
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
