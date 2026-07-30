/**
 * Declarative `PreToolUse` hook factory. Lets hosts express common
 * permission policies (allow / deny / ask lists + a global mode) without
 * hand-rolling matching, precedence, and decision logic per-host.
 *
 * Uses the Claude Code Agent SDK permission vocabulary (`allowed_tools` /
 * `disallowed_tools` / `permissionMode`) while treating modes as fallbacks
 * for calls that match no explicit rule. See the README's HITL section for
 * the cross-walk and `docs/hooks-design-report.md` for the broader hook
 * system context.
 */

import type { HookCallback, PreToolUseHookOutput, ToolDecision } from './types';

/**
 * Permission mode controlling how tool calls that match no rule are
 * resolved. Mirrors Claude Code's `permissionMode`.
 *
 *   - `default`  — unmatched tools fall through to `'ask'` (interrupt).
 *   - `dontAsk`  — unmatched tools are denied; the human is never
 *                  prompted. Useful for headless / API agents where a
 *                  silent denial is preferable to a hung interrupt.
 *   - `bypass`   — unmatched tools are approved. Explicit `deny` and `ask`
 *                  rules still apply.
 */
export type ToolPolicyMode = 'default' | 'dontAsk' | 'bypass';

export interface ToolPolicyConfig {
  /**
   * Global mode applied to tools that don't match any rule.
   * Defaults to `'default'` (ask the human).
   */
  mode?: ToolPolicyMode;
  /**
   * Tool name patterns that are auto-approved without a prompt.
   * Patterns support glob `*` wildcards: `read_file`, `mcp:github:*`,
   * `*search*`. Match is anchored (`^pattern$`).
   */
  allow?: readonly string[];
  /**
   * Tool name patterns that are blocked outright. Wins over `allow`
   * and `ask`, and overrides `mode: 'bypass'` — a deny rule always
   * holds, matching Claude Code's "deny rules are checked first" guarantee.
   */
  deny?: readonly string[];
  /**
   * Tool name patterns that always trigger human approval. Wins over
   * `allow` and every mode, but not `deny`.
   */
  ask?: readonly string[];
  /**
   * Optional reason attached to the resulting `ask` / `deny` hook
   * decision so the host UI can render why approval is required.
   * The literal token `{tool}` is replaced with the tool name.
   */
  reason?: string;
}

/**
 * Compile a glob string with `*` wildcards into a single anchored
 * `RegExp`. Other regex metacharacters are escaped, so `read_file.md`
 * matches the literal dot. Patterns are short (tool names), so we do
 * not cache here — the registry's `matchesQuery` already caches its own
 * regex compilations and our patterns are evaluated once per ToolNode
 * batch, not once per stream chunk.
 */
function globToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp('^' + escaped.replace(/\*/g, '.*') + '$');
}

/** Pre-compile a list of glob patterns into a single match function. */
function compileMatchers(
  patterns: readonly string[] | undefined
): (toolName: string) => boolean {
  if (patterns == null || patterns.length === 0) {
    return () => false;
  }
  const regexes = patterns.map(globToRegex);
  return (toolName: string): boolean => {
    for (const regex of regexes) {
      if (regex.test(toolName)) {
        return true;
      }
    }
    return false;
  };
}

function formatReason(
  template: string | undefined,
  toolName: string
): string | undefined {
  if (template == null) {
    return undefined;
  }
  return template.replace(/\{tool\}/g, toolName);
}

/**
 * Build a `PreToolUse` hook callback that applies a declarative tool
 * permission policy. Register it with a `HookRegistry` and the SDK's
 * `humanInTheLoop` machinery handles the rest:
 *
 * ```ts
 * const policyHook = createToolPolicyHook({
 *   mode: 'default',
 *   allow: ['read_*', 'grep', 'glob'],
 *   deny:  ['delete_*'],
 *   ask:   ['execute_*', 'mcp:*'],
 * });
 * registry.register('PreToolUse', { hooks: [policyHook] });
 * ```
 *
 * Explicit rules take precedence over fallback modes:
 *
 *   1. `deny` rule match → `'deny'` (always wins, even in `bypass`).
 *   2. `ask` rule match → `'ask'`.
 *   3. `allow` rule match → `'allow'`.
 *   4. `mode === 'bypass'` → `'allow'`.
 *   5. `mode === 'dontAsk'` → `'deny'`.
 *   6. fallthrough → `'ask'`.
 *
 * The returned callback is a single `HookCallback`, not a `HookMatcher` —
 * register it under the matcher with the pattern you want (omit the
 * pattern to fire on every tool call, which is the typical case since
 * the policy itself does the filtering).
 */
export function createToolPolicyHook(
  config: ToolPolicyConfig
): HookCallback<'PreToolUse'> {
  const denyMatcher = compileMatchers(config.deny);
  const allowMatcher = compileMatchers(config.allow);
  const askMatcher = compileMatchers(config.ask);
  const mode: ToolPolicyMode = config.mode ?? 'default';
  const reasonTemplate = config.reason;

  return async (input): Promise<PreToolUseHookOutput> => {
    const toolName = input.toolName;
    const decision = decide(
      toolName,
      mode,
      denyMatcher,
      allowMatcher,
      askMatcher
    );
    if (decision === 'allow') {
      return { decision };
    }
    const reason = formatReason(reasonTemplate, toolName);
    if (reason != null) {
      return { decision, reason };
    }
    return { decision };
  };
}

function decide(
  toolName: string,
  mode: ToolPolicyMode,
  denyMatch: (n: string) => boolean,
  allowMatch: (n: string) => boolean,
  askMatch: (n: string) => boolean
): ToolDecision {
  if (denyMatch(toolName)) {
    return 'deny';
  }
  if (askMatch(toolName)) {
    return 'ask';
  }
  if (allowMatch(toolName)) {
    return 'allow';
  }
  if (mode === 'bypass') {
    return 'allow';
  }
  if (mode === 'dontAsk') {
    return 'deny';
  }
  return 'ask';
}
