import type * as t from '@/types';
import { runBashAstChecks, bashAstFindingsToErrors } from './bashAst';

/**
 * Shared synchronous bash-command validation primitives consumed by
 * BOTH the local engine (`LocalExecutionEngine.validateBashCommand`)
 * and the remote `BashExecutor`.
 *
 * Two entry points:
 *
 *   - `validateBashCommandHardFloor` — the "never under any
 *     circumstance" set. Always-on for the remote `/exec` path; not
 *     bypassable by `allowDangerousCommands` because the host doesn't
 *     own the remote sandbox. Narrow by design: zero false positives
 *     on real workloads.
 *
 *   - `validateBashCommandStatic` — the full regex + heuristic pass
 *     (destructive patterns, positional-arg target check, optional
 *     `bashAst` shape findings, optional `readOnly` mutating gate).
 *     Synchronous; no spawn. Suitable for the remote path when the
 *     host opts in via `validation: 'auto' | 'strict'`.
 *
 * The local-engine path layers a `bash -n` syntax preflight on top of
 * `validateBashCommandStatic`; the remote path never spawns anything
 * (there is no local bash to spawn against), so it stops here.
 */

const DESTRUCTIVE_TARGET = '(?:\\/|~|\\$\\{?HOME\\}?|\\.)(?:\\/?\\.?\\*|\\/)?';

const dangerousCommandPatterns: ReadonlyArray<RegExp> = [
  new RegExp(
    `\\brm\\s+(?:-[^\\s]*[rf][^\\s]*\\s+|-[^\\s]*[r][^\\s]*\\s+-[^\\s]*[f][^\\s]*\\s+)(?:--\\s+)?${DESTRUCTIVE_TARGET}\\s*(?:$|[;&|])`
  ),
  /\b(?:mkfs|mkswap|fdisk|parted|diskutil)\b/,
  /\bdd\s+[^;&|]*\bof=\/dev\//,
  new RegExp(
    `\\bchmod\\s+-R\\s+(?:777|a\\+w)\\s+(?:--\\s+)?${DESTRUCTIVE_TARGET}(?:$|\\s|[;&|])`
  ),
  new RegExp(
    `\\bchown\\s+-R\\s+[^;&|]+\\s+(?:--\\s+)?${DESTRUCTIVE_TARGET}(?:$|\\s|[;&|])`
  ),
  /:\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:/,
];

const quotedDestructivePatterns: ReadonlyArray<RegExp> = [
  new RegExp(
    `\\brm\\s+(?:-[^\\s]*[rf][^\\s]*\\s+){1,3}(?:--\\s+)?["']${DESTRUCTIVE_TARGET}["']`
  ),
  new RegExp(
    `\\bchmod\\s+-R\\s+(?:777|a\\+w)\\s+(?:--\\s+)?["']${DESTRUCTIVE_TARGET}["']`
  ),
  new RegExp(
    `\\bchown\\s+-R\\s+[^;&|]+\\s+(?:--\\s+)?["']${DESTRUCTIVE_TARGET}["']`
  ),
];

const NESTED_SHELL_PREFIX = '(?:(?:ba|z|da|k)?sh|eval)\\s+(?:-l?c\\s+)?';
const nestedShellDestructivePatterns: ReadonlyArray<RegExp> = [
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\brm\\s+-[^\\s"\']*[rf][^\\s"\']*\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\bchmod\\s+-R\\s+(?:777|a\\+w)\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
  new RegExp(
    NESTED_SHELL_PREFIX +
      '["\'][^"\']*\\bchown\\s+-R\\s+[^;&|]+\\s+(?:--\\s+)?(?:\\/|~|\\$\\{?HOME\\}?|\\.)'
  ),
];

const PROTECTED_TARGET_ARG_RE = /^(?:\/|~|\$\{?HOME\}?|\.)(?:\/?\.?\*|\/)?$/;
const DESTRUCTIVE_OP_IN_COMMAND_RE =
  /\b(?:rm\s+-[^\s]*[rf]|chmod\s+-R|chown\s+-R)\b/;

const mutatingCommandPattern =
  /\b(?:rm|mv|cp|touch|mkdir|rmdir|ln|truncate|tee|sed\s+-i|perl\s+-pi|python(?:3)?\s+-c|node\s+-e|npm\s+(?:install|ci|update|publish)|pnpm\s+(?:install|update|publish)|yarn\s+(?:install|add|publish)|git\s+(?:add|commit|checkout|switch|reset|clean|rebase|merge|push|pull|stash|tag|branch)|chmod|chown)\b|(?:^|[^<])>\s*[^&]|\bcat\s+[^|;&]*>\s*/;

/**
 * Strips `# …` shell comments (outside quoted spans) to a `\n` while
 * preserving quoted content as-is. Used by `validateBashCommandHardFloor`
 * so the deny-only AST scan doesn't false-positive on text that bash
 * would skip — e.g. `echo ok # cat /proc/self/environ`. Differs from
 * `stripQuotedContent` (which BLANKS quotes too); we want to preserve
 * nested-shell payloads so the deny patterns still see their contents.
 */
export function stripComments(command: string): string {
  let output = '';
  let quote: '"' | '\'' | '`' | undefined;
  let escaped = false;
  for (let i = 0; i < command.length; i++) {
    const char = command[i];
    if (escaped) {
      escaped = false;
      output += char;
      continue;
    }
    if (char === '\\') {
      escaped = true;
      output += char;
      continue;
    }
    if (quote != null) {
      if (char === quote) quote = undefined;
      output += char;
      continue;
    }
    if (char === '"' || char === '\'' || char === '`') {
      quote = char;
      output += char;
      continue;
    }
    if (char === '#') {
      while (i < command.length && command[i] !== '\n') {
        i++;
      }
      output += '\n';
      continue;
    }
    output += char;
  }
  return output;
}

/**
 * Strips the contents of quoted spans (`"…"`, `'…'`, `` `…` ``) and
 * comments to `\n`-normalized whitespace so the destructive regex set
 * can be applied against the "structural" form of the command without
 * being fooled by quoted literals. Exported for the higher-level
 * validator wrapper.
 */
export function stripQuotedContent(command: string): string {
  let output = '';
  let quote: '"' | '\'' | '`' | undefined;
  let escaped = false;

  for (let i = 0; i < command.length; i++) {
    const char = command[i];

    if (escaped) {
      escaped = false;
      output += ' ';
      continue;
    }

    if (char === '\\') {
      escaped = true;
      output += ' ';
      continue;
    }

    if (quote != null) {
      if (char === quote) {
        quote = undefined;
      }
      output += ' ';
      continue;
    }

    if (char === '"' || char === '\'' || char === '`') {
      quote = char;
      output += ' ';
      continue;
    }

    if (char === '#') {
      while (i < command.length && command[i] !== '\n') {
        output += ' ';
        i++;
      }
      output += '\n';
      continue;
    }

    output += char;
  }

  return output;
}

export type BashStaticValidationConfig = {
  readOnly?: boolean;
  bashAst?: t.LocalBashAstMode;
  allowDangerousCommands?: boolean;
};

export type BashValidationResult = {
  valid: boolean;
  errors: string[];
  warnings: string[];
};

function matchAnyDestructive(
  command: string,
  normalized: string
): string | undefined {
  for (const pattern of dangerousCommandPatterns) {
    if (pattern.test(normalized)) {
      return 'Command matches a destructive command pattern.';
    }
  }
  for (const pattern of quotedDestructivePatterns) {
    if (pattern.test(command)) {
      return 'Command matches a destructive command pattern (quoted target).';
    }
  }
  for (const pattern of nestedShellDestructivePatterns) {
    if (pattern.test(command)) {
      return 'Command matches a destructive command pattern (nested shell payload).';
    }
  }
  return undefined;
}

function findOffendingArg(
  command: string,
  args: readonly string[] | undefined
): string | undefined {
  if (args == null || args.length === 0) return undefined;
  if (!DESTRUCTIVE_OP_IN_COMMAND_RE.test(command)) return undefined;
  return args.find((a) => PROTECTED_TARGET_ARG_RE.test(a));
}

/**
 * Pure synchronous validation pass — no spawn. Returns the same
 * `{ valid, errors, warnings }` shape `validateBashCommand` has used
 * historically, so existing call sites and tests keep working.
 */
export function validateBashCommandStatic(
  command: string,
  args: readonly string[] | undefined,
  config: BashStaticValidationConfig = {}
): BashValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const normalized = stripQuotedContent(command);

  if (command.trim() === '') {
    errors.push('Command is empty.');
  }
  if (command.includes('\0')) {
    errors.push('Command contains a NUL byte.');
  }

  if (config.allowDangerousCommands !== true) {
    const reason = matchAnyDestructive(command, normalized);
    if (reason != null) {
      errors.push(reason);
    } else {
      const offending = findOffendingArg(command, args);
      if (offending !== undefined) {
        errors.push(
          `Command matches a destructive command pattern (protected target "${offending}" passed via positional arg).`
        );
      }
    }
  }

  const bashAstMode = config.bashAst ?? 'off';
  if (bashAstMode !== 'off' && config.allowDangerousCommands !== true) {
    const findings = runBashAstChecks(normalized, bashAstMode);
    const split = bashAstFindingsToErrors(findings);
    errors.push(...split.errors);
    warnings.push(...split.warnings);
  }

  if (config.readOnly === true && mutatingCommandPattern.test(normalized)) {
    errors.push(
      'Command appears to mutate files or repository state in read-only local mode.'
    );
  }

  if (/\bsudo\b/.test(normalized)) {
    warnings.push('Command requests elevated privileges with sudo.');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Always-on "never under any circumstance" check for the remote
 * `BashExecutor`. Covers exactly the patterns whose presence in a
 * legitimate workload would be a red flag regardless of the
 * sandbox's ephemerality:
 *
 *   - Destructive shapes against protected roots (`rm -rf /`,
 *     `chmod -R 777 ~`, etc.) — including quoted, nested-shell, and
 *     positional-arg variants.
 *   - Disk-tampering utilities (`mkfs`, `dd of=/dev/…`, `fdisk`, …).
 *   - Fork bombs.
 *   - `/proc/<pid>/environ` reads — the most important one: leaks
 *     sandbox env vars that may carry credentials.
 *   - `source $UNBOUND_VAR` / `. $UNBOUND_VAR` — obfuscated source.
 *   - Zsh privileged builtins (`zmodload`, `sysopen`, …) — privileged
 *     shell extensions with no legitimate use inside `/exec`.
 *
 * Explicitly NOT in the floor (kept opt-in behind the host's
 * `validation: 'auto' | 'strict'` config so common scripts don't
 * trip on the remote path):
 *
 *   - Plain command substitution (`$(…)`) — too common in legit code.
 *   - `IFS=` reassignment.
 *   - Hex escapes (`\xNN`).
 *   - `eval` / `exec`.
 *   - `readOnly` mutating-command set.
 *   - `chmod -R 777` against non-protected paths.
 *
 * The floor is NOT bypassable by `allowDangerousCommands`. That
 * config is a local-engine escape hatch (the host owns their own
 * machine); on the remote sandbox the host doesn't own the runtime,
 * so the floor stands.
 */
export function validateBashCommandHardFloor(
  command: string,
  args?: readonly string[]
): BashValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const normalized = stripQuotedContent(command);

  if (command.trim() === '') {
    errors.push('Command is empty.');
  }
  if (command.includes('\0')) {
    errors.push('Command contains a NUL byte.');
  }

  const destructive = matchAnyDestructive(command, normalized);
  if (destructive != null) {
    errors.push(destructive);
  } else {
    const offending = findOffendingArg(command, args);
    if (offending !== undefined) {
      errors.push(
        `Command matches a destructive command pattern (protected target "${offending}" passed via positional arg).`
      );
    }
  }

  // Categorical deny-only AST findings: /proc/environ, source-from-var,
  // zsh privileged builtins. Skip the warn-severity ones (cmd subst,
  // IFS, hex escape, eval/exec) — they're too noisy for an always-on
  // floor.
  //
  // Scan an "effective command" assembled from the comment-stripped
  // command + every positional arg (rather than the quote-stripped
  // `normalized` form). Three bypass classes the original
  // `runBashAstChecks(normalized, …)` call missed:
  //
  //   1. Nested-shell payloads (Codex P1): `bash -lc 'cat
  //      /proc/self/environ'` — `stripQuotedContent` blanks the inner
  //      payload to whitespace, so `/proc/self/environ` disappeared
  //      before the AST scan ran. Scanning the original (quotes
  //      intact) command text catches it.
  //
  //   2. Positional-arg exfil (Codex P1): `command: 'cat "$1"',
  //      args: ['/proc/self/environ']` — the command body never
  //      references the path. Appending args to the scanned string
  //      catches it.
  //
  //   3. Comment false-positive (Codex P2): `echo ok # cat
  //      /proc/self/environ` would trip the deny pattern even though
  //      bash never executes commented text. `stripComments` blanks
  //      `#…\n` runs (outside quoted spans) so commented mentions
  //      pass through.
  //
  // False-positive tradeoff: `echo "discussing /proc/self/environ"`
  // (the path mentioned inside a string literal that bash WILL pass
  // to echo) still trips. Acceptable — the deny patterns are narrow
  // enough that legitimate workloads basically never include them
  // even as literal strings, and the cost of refusing such a print is
  // negligible.
  const stripped = stripComments(command);
  const effectiveCommand =
    args != null && args.length > 0
      ? `${stripped} ${args.join(' ')}`
      : stripped;
  const findings = runBashAstChecks(effectiveCommand, 'auto');
  for (const finding of findings) {
    if (finding.severity === 'deny') {
      errors.push(`[bashAst:${finding.code}] ${finding.message}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}
