/**
 * Declarative `PreToolUse` hook factory for the bash execution tools
 * (`bash_tool` by default; `toolNames` to widen). Mirrors the shape
 * of {@link createWorkspacePolicyHook} for file paths and
 * {@link createToolPolicyHook} for tool names — same `allow` / `deny`
 * / `ask` vocabulary, same HITL integration via the existing
 * `PreToolUse` `'ask'` flow.
 *
 * Pattern DSL (matched against the bash `command` argument, after a
 * leading-whitespace trim):
 *
 *   - `"<cmd>"`            — exact match on the trimmed command,
 *                            e.g. `"npm test"` matches only `npm test`.
 *   - `"<prefix>:*"`       — prefix match on the command, e.g.
 *                            `"git:*"` matches `git status`, `git push`,
 *                            `git log -1`. `"git push:*"` matches
 *                            `git push`, `git push origin main`, …
 *   - `"*"`                — match anything. Useful only inside
 *                            `default` (`default: 'allow'` is similar to
 *                            an empty policy; `default: 'deny'` plus an
 *                            explicit `allow` list gives a pure allowlist).
 *
 * Patterns are case-sensitive. Quoting and shell-substitution are NOT
 * normalized — `"git:*"` matches `git push`, but not `"git" push` or
 * `git\push`. Hosts that want bulletproof matching should pair this
 * hook with the always-on hard floor in the executor itself.
 *
 * `:*` boundary is whitespace-only. Shell separators (`;`, `&`, `|`,
 * `<`, `>`) do NOT count as a boundary, so `"git:*"` does NOT match
 * `git status; curl evil.com` — chained commands need their own rule.
 * This is the safe default for allowlist postures.
 *
 * Evaluation order (mirrors Claude Code's permission flow):
 *
 *   1. `deny` rule match → `'deny'`.
 *   2. `ask` rule match → `'ask'`.
 *   3. `allow` rule match → `'allow'`.
 *   4. `default` decision (defaults to `'allow'` so an empty policy
 *      is a no-op).
 *
 * The hook only fires for tool calls whose name is in `toolNames`
 * (defaults to `[BASH_TOOL]`). Calls to any other tool short-circuit
 * to `'allow'` so the host can combine this hook with
 * `createWorkspacePolicyHook` / `createToolPolicyHook` on the same
 * registry without cross-talk.
 */

import { Constants } from '@/common';
import type {
  HookCallback,
  PreToolUseHookInput,
  PreToolUseHookOutput,
  ToolDecision,
} from './types';

export type BashPolicyDecision = 'allow' | 'ask' | 'deny';

export interface BashPolicyConfig {
  /** Patterns that auto-allow without prompting. */
  allow?: readonly string[];
  /** Patterns that block outright. Wins over `ask` and `allow`. */
  deny?: readonly string[];
  /**
   * Patterns that trigger human approval. Collapses to `'deny'`
   * when the host has HITL disabled (matching the rest of the SDK).
   */
  ask?: readonly string[];
  /**
   * Decision for commands that match none of the rules. Defaults to
   * `'allow'` so an empty policy is a no-op. Set to `'ask'` or
   * `'deny'` for stricter postures (allowlist-only).
   */
  default?: BashPolicyDecision;
  /**
   * Tool names this hook gates. Defaults to `[BASH_TOOL]`. Set to
   * `[BASH_TOOL, BASH_PROGRAMMATIC_TOOL_CALLING]` to also gate the
   * PTC-bash flow. Calls to tools not in this set short-circuit to
   * `'allow'`.
   */
  toolNames?: readonly string[];
  /**
   * Optional reason template attached to `ask`/`deny` decisions.
   * Supports `{tool}`, `{command}`, `{pattern}` substitution.
   */
  reason?: string;
}

const DEFAULT_TOOL_NAMES: readonly string[] = [Constants.BASH_TOOL];

/**
 * Extract the bash command string from a tool-call input shape. The
 * `bash_tool` schema has `{ command: string, args?: string[] }`; the
 * legacy `execute_code` schema with `lang: 'bash'` uses
 * `{ lang, code, args? }`. Both forms are handled so a host that
 * widens `toolNames` to include `EXECUTE_CODE` still gets matching.
 */
function extractCommand(
  toolInput: Record<string, unknown>
): string | undefined {
  if (typeof toolInput.command === 'string') return toolInput.command;
  if (typeof toolInput.code === 'string' && toolInput.lang === 'bash') {
    return toolInput.code;
  }
  return undefined;
}

type CompiledMatcher = {
  source: string;
  test: (command: string) => boolean;
};

/**
 * Quote-aware check for shell command-chaining and command-substitution
 * metacharacters. Returns true if any of these appears in a position
 * bash would interpret:
 *
 *   - Command list / pipeline / redirection: `;`, `&`, `|`, `<`, `>`,
 *     plus `\n` / `\r` (treated as `;` by bash). Detected ONLY when
 *     not inside any quoted span (`"…"` and `'…'` both make them
 *     literal text).
 *
 *   - Command substitution: `$(…)` and backticks. Detected when not
 *     inside SINGLE quotes — single quotes suppress all expansion,
 *     but double quotes still interpolate `$(…)` and backticks, so
 *     `"$(curl evil)"` runs `curl evil` and the policy must reject it
 *     (Codex P1 round-4 bypass: `git status $(curl evil)` slipped
 *     through because `$(` wasn't in the separator set, and even after
 *     adding it the double-quote case still slipped because the prior
 *     scan treated `"…"` contents as literal across the board).
 *
 * Used by prefix `:*` patterns to refuse any match on commands that
 * contain chaining or substitution, so an allow rule like `git:*`
 * doesn't accidentally authorize a wrapped subcommand bash will
 * actually execute.
 */
function containsShellSeparator(command: string): boolean {
  let quote: '"' | '\'' | undefined;
  let escaped = false;
  for (let i = 0; i < command.length; i++) {
    const char = command[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    // Inside single quotes: NOTHING escapes. Backslash is literal.
    // Pre-fix the `\\` case ran before this check, so `'abc\\'`
    // never closed the quote in our scanner — and bash's actual
    // semantics (which DO close the quote) left trailing `; curl
    // evil.com` looking like it was still inside the single-quoted
    // span. (Codex P1 round-6 bypass.)
    if (quote === '\'') {
      if (char === '\'') quote = undefined;
      continue;
    }
    if (char === '\\') {
      escaped = true;
      continue;
    }
    // Open a quoted span when we're not inside one.
    if (quote == null && (char === '"' || char === '\'')) {
      quote = char;
      continue;
    }
    // Close a double-quoted span.
    if (quote === '"' && char === '"') {
      quote = undefined;
      continue;
    }
    // Command substitution. Active both outside quotes AND inside
    // double quotes — only single quotes suppress it.
    if (char === '`') return true;
    if (char === '$' && i + 1 < command.length && command[i + 1] === '(') {
      return true;
    }
    // Other separators are literal text inside double quotes; only
    // active when we're outside any quote.
    if (quote === '"') continue;
    if (
      char === ';' ||
      char === '&' ||
      char === '|' ||
      char === '<' ||
      char === '>' ||
      char === '\n' ||
      char === '\r'
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Compile a single DSL pattern. Three forms:
 *   - `*`         → matches anything.
 *   - `<x>:*`     → prefix match; trims trailing `:*` and matches
 *                   when the trimmed command starts with the prefix
 *                   followed by whitespace or end-of-string, AND the
 *                   command contains no shell separators (so the rule
 *                   can't accidentally authorize chained commands).
 *   - `<x>`       — exact match on the trimmed command (after collapsing
 *                   inner whitespace runs).
 */
function compilePattern(pattern: string): CompiledMatcher {
  const source = pattern;
  if (pattern === '*') {
    return { source, test: () => true };
  }
  if (pattern.endsWith(':*')) {
    const prefix = pattern.slice(0, -2);
    return {
      source,
      test: (command: string): boolean => {
        // Refuse to match any command containing a shell separator
        // (outside quotes). `git:*` matching `git status; curl evil`
        // would let bash run the trailing command unauthorized
        // (Codex P1 round-2). Hosts that want to allow chains must
        // write rules for each segment, or use exact-match patterns.
        if (containsShellSeparator(command)) return false;
        if (!command.startsWith(prefix)) return false;
        if (command.length === prefix.length) return true;
        const next = command.charAt(prefix.length);
        // Boundary char must be whitespace; `gitlab` doesn't match `git:*`.
        return /\s/.test(next);
      },
    };
  }
  // Exact match: normalize space + tab only (NOT newlines/CR — those
  // are shell separators bash treats as `;`). Pre-fix the `\s+`
  // collapse turned `npm\ntest` into `npm test`, so an exact rule
  // `"npm test"` matched multi-line input that bash would split into
  // two commands (Codex P2 round-6). Also gate on
  // `containsShellSeparator` so any chained / substituted input is
  // refused even when the first command happens to collapse to the
  // exact pattern.
  const exact = pattern.replace(/[ \t]+/g, ' ').trim();
  return {
    source,
    test: (command: string): boolean => {
      if (containsShellSeparator(command)) return false;
      return command.replace(/[ \t]+/g, ' ').trim() === exact;
    },
  };
}

function compileMatchers(
  patterns: readonly string[] | undefined
): CompiledMatcher[] {
  if (patterns == null || patterns.length === 0) return [];
  return patterns.map(compilePattern);
}

function firstMatch(
  command: string,
  matchers: readonly CompiledMatcher[]
): CompiledMatcher | undefined {
  for (const m of matchers) {
    if (m.test(command)) return m;
  }
  return undefined;
}

function formatReason(
  template: string | undefined,
  toolName: string,
  command: string,
  pattern: string | undefined
): string | undefined {
  if (template == null) return undefined;
  return template
    .replace(/\{tool\}/g, toolName)
    .replace(/\{command\}/g, command)
    .replace(/\{pattern\}/g, pattern ?? '');
}

/**
 * Build a `PreToolUse` hook callback that gates bash command
 * invocations against the configured allow / deny / ask DSL.
 *
 * Example — pure allowlist (everything else is denied):
 *
 * ```ts
 * registry.register('PreToolUse', {
 *   hooks: [
 *     createBashPolicyHook({
 *       allow: ['git status', 'git diff:*', 'npm test', 'ls:*'],
 *       default: 'deny',
 *     }),
 *   ],
 * });
 * ```
 *
 * Example — ask-on-mutation, allow read-only by default:
 *
 * ```ts
 * createBashPolicyHook({
 *   ask: ['git push:*', 'npm publish:*', 'rm:*'],
 *   default: 'allow',
 * });
 * ```
 *
 * Composes with `createWorkspacePolicyHook` and `createToolPolicyHook`
 * on the same registry — `executeHooks` precedence (`deny > ask >
 * allow`) sorts out which decision wins per call.
 */
export function createBashPolicyHook(
  config: BashPolicyConfig
): HookCallback<'PreToolUse'> {
  const denyMatchers = compileMatchers(config.deny);
  const askMatchers = compileMatchers(config.ask);
  const allowMatchers = compileMatchers(config.allow);
  const defaultDecision: BashPolicyDecision = config.default ?? 'allow';
  const toolNames = new Set<string>(config.toolNames ?? DEFAULT_TOOL_NAMES);
  const reasonTemplate = config.reason;

  return async (input: PreToolUseHookInput): Promise<PreToolUseHookOutput> => {
    if (!toolNames.has(input.toolName)) return { decision: 'allow' };

    const command = extractCommand(input.toolInput);
    if (command == null || command === '') return { decision: 'allow' };

    const trimmed = command.trim();

    const denyHit = firstMatch(trimmed, denyMatchers);
    if (denyHit != null) {
      return buildDecision(
        'deny',
        input.toolName,
        command,
        denyHit,
        reasonTemplate
      );
    }
    const askHit = firstMatch(trimmed, askMatchers);
    if (askHit != null) {
      return buildDecision(
        'ask',
        input.toolName,
        command,
        askHit,
        reasonTemplate
      );
    }
    const allowHit = firstMatch(trimmed, allowMatchers);
    if (allowHit != null) {
      return { decision: 'allow' };
    }
    if (defaultDecision === 'allow') {
      return { decision: 'allow' };
    }
    return buildDecision(
      defaultDecision,
      input.toolName,
      command,
      undefined,
      reasonTemplate
    );
  };
}

function buildDecision(
  decision: ToolDecision,
  toolName: string,
  command: string,
  match: CompiledMatcher | undefined,
  reasonTemplate: string | undefined
): PreToolUseHookOutput {
  const reason = formatReason(reasonTemplate, toolName, command, match?.source);
  const baseAllowed =
    decision === 'ask'
      ? { allowedDecisions: ['approve', 'reject'] as const }
      : {};
  if (reason != null) {
    return { decision, reason, ...baseAllowed };
  }
  return { decision, ...baseAllowed };
}
