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
  // Terminator `(?:$|\s|[;&|])` (not `\s*(?:$|[;&|])`): the previous
  // form let `rm -rf /\ncurl evil` slip past because `\s*` greedily
  // consumed the `\n` and then required `$` or `;&|` afterward (and
  // `c` of `curl` matched neither). Newline is a bash command
  // separator equivalent to `;`, so it must be an accepted terminator.
  // Mirrors the chmod/chown terminator shape. (Codex P1 round-12).
  new RegExp(
    `\\brm\\s+(?:-[^\\s]*[rf][^\\s]*\\s+|-[^\\s]*[r][^\\s]*\\s+-[^\\s]*[f][^\\s]*\\s+)(?:--\\s+)?${DESTRUCTIVE_TARGET}(?:$|\\s|[;&|])`
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
  // `dd` with a quoted device target — companion to the unquoted
  // `dd … of=/dev/…` pattern in `dangerousCommandPatterns`. Pre-fix,
  // `dd if=/dev/zero of='/dev/sda'` slipped past the dd guard because
  // `stripQuotedContent` blanked the quoted path before the unquoted
  // regex could match (Codex P1 round-7).
  /\bdd\s+[^;&|]*\bof=["']\/dev\//,
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
 * Bash treats `#` as a comment marker only when it begins a word —
 * at start-of-input or after whitespace / a shell separator. `cat#foo`
 * is one word and the `#` is literal text; `cat #foo` is `cat` plus
 * a comment. Both `stripComments` and `stripQuotedContent` use this
 * predicate so a mid-word `#` doesn't accidentally blank everything
 * after it (which previously hid destructive commands like
 * `echo foo#bar; rm -rf /` — Codex P1 rounds 5 and 10).
 */
function isCommentBoundary(prevChar: string | undefined): boolean {
  return (
    prevChar === undefined ||
    prevChar === ' ' ||
    prevChar === '\t' ||
    prevChar === '\n' ||
    prevChar === '\r' ||
    prevChar === ';' ||
    prevChar === '&' ||
    prevChar === '|' ||
    prevChar === '<' ||
    prevChar === '>' ||
    prevChar === '(' ||
    prevChar === '{'
  );
}

/**
 * Strips `# …` shell comments (outside quoted spans) to a `\n` while
 * preserving quoted content as-is. Used by `validateBashCommandHardFloor`
 * so the deny-only AST scan doesn't false-positive on text that bash
 * would skip — e.g. `echo ok # cat /proc/self/environ`.
 *
 * Bash's `#` rule is narrower than "anywhere outside quotes":
 *
 *   - `#` is a comment marker only when it begins a word (i.e., at
 *     start-of-input or after whitespace / a shell separator).
 *     `cat#foo` is one word and the `#` is literal text bash passes
 *     to `cat`.
 *
 *   - Inside `${…}` parameter expansion, `#` is an OPERATOR
 *     (`${var#prefix}`, `${var##prefix}`), not a comment. Pre-fix
 *     this case bypassed the hard-floor exfil guard: a command like
 *     `x=1; echo ${x#1}; cat /proc/self/environ` got truncated at
 *     `${x#1}`, so the `/proc/self/environ` read never reached the
 *     deny scan (Codex P1 round-5).
 *
 * Other parenthesized contexts (`$(…)`, plain `(…)` subshells) are
 * not tracked — `#` rules inside them follow the same word-boundary
 * gate at the top level. A pattern in a comment INSIDE `$(…)` would
 * false-positive against the deny scan; acceptable for the hard
 * floor's defense-in-depth posture.
 */
export function stripComments(command: string): string {
  let output = '';
  let quote: '"' | '\'' | '`' | undefined;
  let escaped = false;
  let braceDepth = 0;
  let prevChar: string | undefined;
  for (let i = 0; i < command.length; i++) {
    const char = command[i];
    if (escaped) {
      escaped = false;
      output += char;
      prevChar = char;
      continue;
    }
    // Inside single quotes: NOTHING escapes — backslash is literal,
    // only the closing `'` matters. Pre-fix the `\\` branch ran
    // before this check, so `'abc\'` never closed the quote in our
    // scanner — and `; rm -rf /` after looked "still inside the
    // quote", suppressing subsequent comment-strip / quote-aware
    // handling (same bug class as Codex P1 round-6 in
    // `containsShellSeparator`; round-11 found it here too).
    if (quote === '\'') {
      if (char === '\'') quote = undefined;
      output += char;
      prevChar = char;
      continue;
    }
    if (char === '\\') {
      escaped = true;
      output += char;
      prevChar = char;
      continue;
    }
    if (quote != null) {
      if (char === quote) quote = undefined;
      output += char;
      prevChar = char;
      continue;
    }
    if (char === '"' || char === '\'' || char === '`') {
      quote = char;
      output += char;
      prevChar = char;
      continue;
    }
    // Track `${…}` parameter-expansion depth. Bash uses `#` inside as
    // an expansion operator, not a comment start.
    if (char === '$' && i + 1 < command.length && command[i + 1] === '{') {
      output += '$';
      output += '{';
      i++;
      braceDepth++;
      prevChar = '{';
      continue;
    }
    if (braceDepth > 0) {
      if (char === '{') braceDepth++;
      else if (char === '}') braceDepth--;
      output += char;
      prevChar = char;
      continue;
    }
    if (char === '#' && isCommentBoundary(prevChar)) {
      while (i < command.length && command[i] !== '\n') {
        i++;
      }
      output += '\n';
      prevChar = '\n';
      continue;
    }
    output += char;
    prevChar = char;
  }
  return output;
}

/**
 * Strips the contents of quoted spans (`"…"`, `'…'`, `` `…` ``) and
 * shell comments to `\n`-normalized whitespace so the destructive
 * regex set can be applied against the "structural" form of the
 * command without being fooled by quoted literals.
 *
 * Comment handling: same word-boundary rule as `stripComments` —
 * mid-word `#` (e.g. `echo foo#bar; rm -rf /`) is literal text in
 * bash and must NOT cause a blanket strip of the trailing
 * destructive command (Codex P1 round-10).
 *
 * Parameter-expansion tracking: `${var#prefix}` uses `#` as an
 * expansion operator, not a comment start. Tracking `${…}` depth
 * keeps that text intact for the destructive-pattern regex.
 */
export function stripQuotedContent(command: string): string {
  let output = '';
  let quote: '"' | '\'' | '`' | undefined;
  let escaped = false;
  let braceDepth = 0;
  let prevChar: string | undefined;

  const append = (out: string): void => {
    output += out;
    // `prevChar` tracks what bash's tokenizer would see, not the
    // possibly-blanked output char. We update it via a separate
    // `setPrev` call so the word-boundary check on `#` reflects the
    // original char (e.g. `o` in `foo#bar` stays `o`, not ` `).
  };
  const setPrev = (originalChar: string): void => {
    prevChar = originalChar;
  };

  for (let i = 0; i < command.length; i++) {
    const char = command[i];

    if (escaped) {
      escaped = false;
      append(' ');
      setPrev(char);
      continue;
    }

    // Inside single quotes: NOTHING escapes — backslash is literal,
    // only the closing `'` matters. Pre-fix the `\\` branch ran
    // before this check, so `'abc\'` never closed the quote in our
    // scanner — and the trailing `; rm -rf /` looked "still inside
    // the quote", getting blanked. `matchAnyDestructive` then
    // missed the destructive tail and the hard-floor was bypassable.
    // Same bug class Codex flagged for `containsShellSeparator` in
    // round 6 (Codex P1 round-11).
    if (quote === '\'') {
      if (char === '\'') {
        quote = undefined;
      }
      append(' ');
      setPrev(char);
      continue;
    }

    if (char === '\\') {
      escaped = true;
      append(' ');
      setPrev(char);
      continue;
    }

    if (quote != null) {
      if (char === quote) {
        quote = undefined;
      }
      append(' ');
      setPrev(char);
      continue;
    }

    if (char === '"' || char === '\'' || char === '`') {
      quote = char;
      append(' ');
      setPrev(char);
      continue;
    }

    // Track `${…}` parameter-expansion depth. Inside `${…}`, `#` is
    // an expansion operator (e.g. `${var#prefix}`), not a comment
    // marker; we must NOT blank from it. Pre-fix
    // `echo ${x#1}; rm -rf /` had the rest of the line stripped
    // because the first unquoted `#` was treated as comment start
    // (mirrors the same issue we fixed for `stripComments` in
    // Codex P1 round-5; round-10 found it again here).
    if (char === '$' && i + 1 < command.length && command[i + 1] === '{') {
      append('$');
      append('{');
      i++;
      braceDepth++;
      setPrev('{');
      continue;
    }
    if (braceDepth > 0) {
      if (char === '{') braceDepth++;
      else if (char === '}') braceDepth--;
      append(char);
      setPrev(char);
      continue;
    }

    // `#` only starts a comment at a word boundary (start-of-input
    // or after whitespace / shell separator). `echo foo#bar; rm -rf
    // /` keeps the `; rm -rf /` visible to the destructive regex
    // because the `#` is mid-word here (Codex P1 round-10).
    if (char === '#' && isCommentBoundary(prevChar)) {
      while (i < command.length && command[i] !== '\n') {
        append(' ');
        i++;
      }
      append('\n');
      setPrev('\n');
      continue;
    }

    append(char);
    setPrev(char);
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
  // We scan MULTIPLE candidate forms of the command+args so neither
  // the way args are quoted nor the way they're concatenated at
  // runtime can hide a deny pattern:
  //
  //   1. `stripped` alone — catches the bare case (no args, or
  //      command contains the full pattern).
  //
  //   2. `${stripped} ${args.join(' ')}` — catches the
  //      "one arg holds the full pattern" case (`args:
  //      ['/proc/self/environ']`). Each arg is a separate word in
  //      the scan, separated by spaces.
  //
  //   3. `${stripped} ${args.join('')}` — catches the
  //      "pattern split across args" case (Codex P1 round-9):
  //      `command: 'cat "$1$2"', args: ['/proc/self', '/environ']`
  //      executes `cat /proc/self/environ` at runtime because bash
  //      concatenates `$1$2`. The space-joined form `... /proc/self
  //      /environ` misses the regex; the raw-joined form
  //      `.../proc/self/environ` matches.
  //
  // Original-command bypasses also caught here via the stripped form:
  //
  //   - Nested-shell payloads (Codex P1 round-1):
  //     `bash -lc 'cat /proc/self/environ'` — the original (quotes
  //     intact) is scanned, so the payload's contents are visible.
  //
  //   - Comment false-positive (Codex P2 round-2): `echo ok #
  //     cat /proc/self/environ` is stripped before the scan, so
  //     commented mentions pass through.
  //
  // False-positive tradeoff: legitimate workloads with two adjacent
  // path-shaped args like `cat "$1" "$2"` + `args: ['/proc/self',
  // '/environ']` would now trip the floor. Acceptable — the deny
  // patterns are narrow, the workload would be unusual, and refusing
  // is cheaper than missing the real exfil shape.
  const stripped = stripComments(command);
  const candidates: string[] = [stripped];
  if (args != null && args.length > 0) {
    const joinedSpaces = args.join(' ');
    const joinedRaw = args.join('');
    candidates.push(`${stripped} ${joinedSpaces}`);
    if (joinedRaw !== joinedSpaces) {
      candidates.push(`${stripped} ${joinedRaw}`);
    }
  }
  const seenDenyCodes = new Set<string>();
  for (const candidate of candidates) {
    const findings = runBashAstChecks(candidate, 'auto');
    for (const finding of findings) {
      if (finding.severity !== 'deny') continue;
      if (seenDenyCodes.has(finding.code)) continue;
      seenDenyCodes.add(finding.code);
      errors.push(`[bashAst:${finding.code}] ${finding.message}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}
