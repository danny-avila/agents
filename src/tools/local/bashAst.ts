import type * as t from '@/types';

export type BashAstFinding = {
  code: string;
  message: string;
  severity: 'warn' | 'deny';
};

/**
 * Categorical-hazard checks layered on top of the existing dangerous-
 * command regex set. These match command-shape signatures that
 * claude-code's tree-sitter AST validator catches via categorical
 * deny-lists (command substitution, zsh-only privileged commands,
 * /proc/<pid>/environ access, IFS injection, etc.).
 *
 * This is *not* a real AST parser. It is a deliberately conservative
 * heuristic pass intended for the local engine's `bashAst: 'auto' |
 * 'strict'` modes; a future PR can swap in a true tree-sitter-bash
 * pass behind the same config field without changing the public API.
 *
 * `runBashAstChecks` runs on the *quote-stripped* command (so quoted
 * strings inside the script don't generate false positives) and
 * returns one finding per matched category.
 */

const COMMAND_SUBSTITUTION_PATTERNS: { code: string; rx: RegExp }[] = [
  { code: 'cmd-subst-dollar-paren', rx: /\$\(/ },
  { code: 'cmd-subst-backtick', rx: /`[^`]*`/ },
  { code: 'cmd-subst-process-sub', rx: /[<>]\(/ },
  { code: 'cmd-subst-zsh-eq', rx: /(?:^|\s)=[A-Za-z_]/ },
];

const ZSH_DANGEROUS_BUILTINS = [
  'zmodload',
  'emulate',
  'sysopen',
  'sysread',
  'syswrite',
  'ztcp',
  'zsocket',
  'zf_rm',
  'zselect',
];

const STRICT_DENIED_BUILTINS = ['eval', 'exec'];

function rxForBuiltin(name: string): RegExp {
  return new RegExp(`\\b${name}\\b`);
}

// `/proc/<pid>/environ` exfiltrates process env vars. Match any
// pid-shaped position between `/proc/` and `/environ` — `\d+`, `self`,
// `$VAR`, `${VAR}`, `$$` (current PID), `$!` (last bg PID), `$0`–`$9`
// (positional args), AND quoted variants like `"self"` (bash resolves
// the quoted token to the same path at runtime — Codex P1 round-13).
// The negated class only excludes slash and whitespace; quote chars
// are part of the matched segment so `/proc/"self"/environ` and
// `/proc/'1'/environ` are caught.
//
// Pre-fixes (round-12): only `\d+|self|\$[A-Za-z_]` matched; common
// `/proc/$$/environ` and `/proc/${pid}/environ` slipped through.
// Pre-fixes (round-13): the `[^/\s'"]` negated class excluded quote
// chars, so quoted-segment variants weren't caught.
const PROC_ENVIRON_RX = /\/proc\/[^/\s]+\/environ\b/;
const IFS_INJECTION_RX = /\bIFS\s*=/;
const HEX_ESCAPE_OBFUSCATION_RX = /\\x[0-9a-fA-F]{2}/;
// Source-from-expansion: any `source $...` / `. $...` form where the
// next token starts with `$`. Catches `$VAR`, `${VAR}`, `$1`, etc.
// (Codex P1 round-12 — pre-fix only `\$[A-Za-z_]` matched, missing
// `${VAR}` and other expansions.)
const SOURCE_FROM_VAR_RX = /(?:^|\s)(?:source|\.)\s+["']?\$/;

export function runBashAstChecks(
  command: string,
  mode: t.LocalBashAstMode = 'off'
): BashAstFinding[] {
  if (mode === 'off') {
    return [];
  }
  const findings: BashAstFinding[] = [];
  const strict = mode === 'strict';

  for (const { code, rx } of COMMAND_SUBSTITUTION_PATTERNS) {
    if (rx.test(command)) {
      findings.push({
        code,
        message:
          'Command substitution can mask intent and exfiltrate variables; not allowed under bashAst.',
        severity: strict ? 'deny' : 'warn',
      });
    }
  }

  for (const builtin of ZSH_DANGEROUS_BUILTINS) {
    if (rxForBuiltin(builtin).test(command)) {
      findings.push({
        code: `zsh-builtin-${builtin}`,
        message: `Zsh privileged builtin "${builtin}" is denied.`,
        severity: 'deny',
      });
    }
  }

  if (PROC_ENVIRON_RX.test(command)) {
    findings.push({
      code: 'proc-environ-read',
      message:
        'Reads from /proc/<pid>/environ are denied — leaks host secrets.',
      severity: 'deny',
    });
  }

  if (IFS_INJECTION_RX.test(command)) {
    findings.push({
      code: 'ifs-injection',
      message: 'Inline IFS reassignment is suspicious; review the command.',
      severity: strict ? 'deny' : 'warn',
    });
  }

  if (HEX_ESCAPE_OBFUSCATION_RX.test(command)) {
    findings.push({
      code: 'hex-escape',
      message:
        'Hex-escaped bytes (\\xNN) often hide intent; review the command.',
      severity: strict ? 'deny' : 'warn',
    });
  }

  if (SOURCE_FROM_VAR_RX.test(command)) {
    findings.push({
      code: 'source-from-variable',
      message: 'Sourcing a script from an unbound variable is denied.',
      severity: 'deny',
    });
  }

  if (strict) {
    for (const builtin of STRICT_DENIED_BUILTINS) {
      if (rxForBuiltin(builtin).test(command)) {
        findings.push({
          code: `strict-${builtin}`,
          message: `In strict mode, "${builtin}" is denied.`,
          severity: 'deny',
        });
      }
    }
  }

  return findings;
}

export function bashAstFindingsToErrors(findings: BashAstFinding[]): {
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];
  for (const f of findings) {
    const formatted = `[bashAst:${f.code}] ${f.message}`;
    if (f.severity === 'deny') {
      errors.push(formatted);
    } else {
      warnings.push(formatted);
    }
  }
  return { errors, warnings };
}
