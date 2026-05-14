import fetch from 'node-fetch';
import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { createBashExecutionTool } from '../BashExecutor';
import {
  validateBashCommandHardFloor,
  validateBashCommandStatic,
} from '../local/bashValidation';

jest.mock('node-fetch', () => ({
  __esModule: true,
  default: jest.fn(),
}));

type FetchMock = jest.MockedFunction<
  (url: unknown, init?: unknown) => Promise<unknown>
>;

const fetchMock = fetch as unknown as FetchMock;

function jsonResponse(body: unknown): unknown {
  return {
    ok: true,
    json: jest.fn(async () => body),
    text: jest.fn(async () => JSON.stringify(body)),
  };
}

beforeEach(() => {
  fetchMock.mockReset();
  fetchMock.mockResolvedValue(
    jsonResponse({ session_id: 'session_abc', stdout: 'ok\n' })
  );
});

describe('validateBashCommandHardFloor', () => {
  it('passes obviously safe commands', () => {
    expect(validateBashCommandHardFloor('echo hi').valid).toBe(true);
    expect(validateBashCommandHardFloor('ls -la').valid).toBe(true);
    expect(validateBashCommandHardFloor('git status').valid).toBe(true);
    expect(validateBashCommandHardFloor('python3 -m pytest').valid).toBe(true);
  });

  it('blocks rm -rf against protected roots', () => {
    expect(validateBashCommandHardFloor('rm -rf /').valid).toBe(false);
    expect(validateBashCommandHardFloor('rm -rf ~').valid).toBe(false);
    expect(validateBashCommandHardFloor('rm -rf $HOME').valid).toBe(false);
    expect(validateBashCommandHardFloor('rm -rf ${HOME}').valid).toBe(false);
    expect(validateBashCommandHardFloor('rm -rf "/"').valid).toBe(false);
    expect(validateBashCommandHardFloor('rm -rf \'/\'').valid).toBe(false);
  });

  it('blocks rm -rf via the positional-arg bypass shape', () => {
    const result = validateBashCommandHardFloor('rm -rf "$1"', ['/']);
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('positional arg');
  });

  it('allows rm -rf against non-protected paths', () => {
    expect(validateBashCommandHardFloor('rm -rf /tmp/build').valid).toBe(true);
    expect(validateBashCommandHardFloor('rm -rf node_modules').valid).toBe(
      true
    );
  });

  it('blocks disk-tampering utilities', () => {
    expect(validateBashCommandHardFloor('mkfs.ext4 /dev/sda1').valid).toBe(
      false
    );
    expect(
      validateBashCommandHardFloor('dd if=/dev/zero of=/dev/sda').valid
    ).toBe(false);
    expect(validateBashCommandHardFloor('fdisk /dev/sda').valid).toBe(false);
    expect(validateBashCommandHardFloor('mkswap /dev/sdb').valid).toBe(false);
  });

  it('blocks dd with quoted device target (Codex P1 round-7)', () => {
    // Pre-fix `dd if=/dev/zero of='/dev/sda'` slipped past the dd
    // guard because the unquoted pattern ran on the quote-stripped
    // form, which blanked the device path before regex match.
    expect(
      validateBashCommandHardFloor('dd if=/dev/zero of=\'/dev/sda\'').valid
    ).toBe(false);
    expect(
      validateBashCommandHardFloor('dd if=/dev/zero of="/dev/sda"').valid
    ).toBe(false);
    expect(
      validateBashCommandHardFloor('dd bs=1M if=/tmp/in of="/dev/nvme0n1"')
        .valid
    ).toBe(false);
  });

  it('blocks fork bombs', () => {
    expect(validateBashCommandHardFloor(':(){ :|:& };:').valid).toBe(false);
  });

  it('blocks /proc/<pid>/environ reads (sandbox env exfil)', () => {
    expect(validateBashCommandHardFloor('cat /proc/1/environ').valid).toBe(
      false
    );
    expect(validateBashCommandHardFloor('cat /proc/self/environ').valid).toBe(
      false
    );
    const result = validateBashCommandHardFloor('cat /proc/self/environ');
    expect(result.errors.join('\n')).toContain('proc-environ-read');
  });

  it('blocks zsh privileged builtins', () => {
    const result = validateBashCommandHardFloor('zmodload zsh/net/tcp');
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('zsh-builtin-zmodload');
  });

  it('blocks /proc/<pid>/environ inside nested-shell payloads (Codex P1 #1)', () => {
    // Pre-fix `stripQuotedContent` blanked the inner payload to
    // whitespace before the AST scan, so the deny check never saw
    // `/proc/self/environ`. Verify both single- and double-quoted forms.
    const single = validateBashCommandHardFloor(
      'bash -lc \'cat /proc/self/environ\''
    );
    expect(single.valid).toBe(false);
    expect(single.errors.join('\n')).toContain('proc-environ-read');

    const double = validateBashCommandHardFloor(
      'sh -c "cat /proc/self/environ"'
    );
    expect(double.valid).toBe(false);
    expect(double.errors.join('\n')).toContain('proc-environ-read');

    const evalForm = validateBashCommandHardFloor('eval "cat /proc/1/environ"');
    expect(evalForm.valid).toBe(false);
    expect(evalForm.errors.join('\n')).toContain('proc-environ-read');
  });

  it('blocks /proc/<pid>/environ split across positional args (Codex P1 round-9)', () => {
    // Pre-fix the hard-floor AST scan only saw args joined with
    // spaces, so `command: 'cat "$1$2"', args: ['/proc/self', '/environ']`
    // looked like `... /proc/self /environ` and missed the regex.
    // Bash concatenates `$1$2` at runtime → reads /proc/self/environ.
    // The fix scans both space-joined AND raw-joined forms; the raw
    // join reassembles the path.
    const slashBoundary = validateBashCommandHardFloor('cat "$1$2"', [
      '/proc/self',
      '/environ',
    ]);
    expect(slashBoundary.valid).toBe(false);
    expect(slashBoundary.errors.join('\n')).toContain('proc-environ-read');

    const midWord = validateBashCommandHardFloor('cat "$1$2"', [
      '/proc/self/envi',
      'ron',
    ]);
    expect(midWord.valid).toBe(false);

    const threeWay = validateBashCommandHardFloor('cat "$1$2$3"', [
      '/proc/',
      'self/',
      'environ',
    ]);
    expect(threeWay.valid).toBe(false);
  });

  it('blocks /proc/<pid>/environ smuggled via positional arg (Codex P1 #2)', () => {
    // Pre-fix `findOffendingArg` only ran when the command matched
    // `rm/chmod/chown`, so a benign-looking `cat "$1"` with the
    // sensitive path in args silently slipped through.
    const result = validateBashCommandHardFloor('cat "$1"', [
      '/proc/self/environ',
    ]);
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('proc-environ-read');

    const numbered = validateBashCommandHardFloor('cat "$2"', [
      'ignored',
      '/proc/1/environ',
    ]);
    expect(numbered.valid).toBe(false);
  });

  it('blocks source-from-unbound-variable', () => {
    expect(validateBashCommandHardFloor('source $REMOTE_SCRIPT').valid).toBe(
      false
    );
    expect(validateBashCommandHardFloor('. $REMOTE_SCRIPT').valid).toBe(false);
  });

  it('blocks nested-shell destructive payloads', () => {
    expect(validateBashCommandHardFloor('bash -lc "rm -rf $HOME"').valid).toBe(
      false
    );
    expect(validateBashCommandHardFloor('sh -c \'chmod -R 777 /\'').valid).toBe(
      false
    );
    expect(validateBashCommandHardFloor('eval \'rm -rf /\'').valid).toBe(false);
  });

  it('blocks destructive commands hidden after a mid-word # (Codex P1 round-10)', () => {
    // Pre-fix `stripQuotedContent` treated every unquoted `#` as a
    // comment start, blanking the trailing `; rm -rf /` from the
    // destructive-pattern regex. Bash treats `#` as a comment only
    // at word boundaries, so `foo#bar` keeps `#` literal and `; rm
    // -rf /` is still an executed command.
    const result = validateBashCommandHardFloor('echo foo#bar; rm -rf /');
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('destructive');

    // Same shape with `${var#prefix}` parameter expansion in front.
    const withParamExpansion = validateBashCommandHardFloor(
      'x=foo; echo ${x#1}; rm -rf /'
    );
    expect(withParamExpansion.valid).toBe(false);
  });

  it('does NOT trip on commands that merely print destructive strings', () => {
    expect(validateBashCommandHardFloor('echo "rm -rf /"').valid).toBe(true);
  });

  it('does NOT trip on deny patterns inside shell comments (Codex P2 round-2)', () => {
    // Pre-fix the hard-floor AST scan ran on raw command text, so a
    // comment like `# cat /proc/self/environ` triggered the proc-environ
    // deny check even though bash never executes commented text.
    expect(
      validateBashCommandHardFloor('echo ok # cat /proc/self/environ').valid
    ).toBe(true);
    expect(
      validateBashCommandHardFloor(
        'ls -la\n# todo: investigate /proc/self/environ\necho done'
      ).valid
    ).toBe(true);
  });

  it('does NOT mis-strip deny patterns past `${var#prefix}` expansion (Codex P1 round-5)', () => {
    // Pre-fix `stripComments` treated every unquoted `#` as a comment
    // start, so `${x#1}` truncated the rest of the command — letting
    // a subsequent `/proc/self/environ` read slip past the hard floor.
    // Now `#` inside `${…}` is recognized as the parameter-expansion
    // operator and the scan continues past it.
    const result = validateBashCommandHardFloor(
      'x=1; echo ${x#1}; cat /proc/self/environ'
    );
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('proc-environ-read');

    // Also catches the `##` (greedy prefix removal) variant.
    const greedy = validateBashCommandHardFloor(
      'x=foo; echo ${x##*foo}; cat /proc/1/environ'
    );
    expect(greedy.valid).toBe(false);
  });

  it('does NOT treat mid-word `#` as a comment start', () => {
    // `cat#foo` is one word and the `#` is literal text bash passes
    // to `cat` (which would fail, but isn't a comment). Verify the
    // hard floor still sees text after such a `#`.
    const result = validateBashCommandHardFloor(
      'echo cat#hello; cat /proc/self/environ'
    );
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('proc-environ-read');
  });

  it('still trips on deny patterns inside quoted strings (not comments)', () => {
    // `echo "cat /proc/self/environ"` — the path lives inside a
    // double-quoted string that bash WILL evaluate. The hard floor
    // remains aggressive here (the deny patterns are narrow enough
    // that legitimate workloads don't reference them as literals).
    // This is the defense-in-depth posture, distinct from the
    // comment case above.
    expect(
      validateBashCommandHardFloor('echo "cat /proc/self/environ"').valid
    ).toBe(false);
  });

  it('does NOT trip on plain command substitution (only opt-in flags that)', () => {
    expect(validateBashCommandHardFloor('echo "$(date)"').valid).toBe(true);
    expect(validateBashCommandHardFloor('echo $(whoami)').valid).toBe(true);
  });

  it('does NOT trip on IFS or hex escapes (warn-only, not in floor)', () => {
    expect(
      validateBashCommandHardFloor('IFS=, read a b c <<< "x,y,z"').valid
    ).toBe(true);
    expect(validateBashCommandHardFloor('printf "\\x41"').valid).toBe(true);
  });
});

describe('createBashExecutionTool — hard floor enforcement', () => {
  it('rejects rm -rf / before posting to /exec', async () => {
    const tool = createBashExecutionTool();
    await expect(tool.invoke({ command: 'rm -rf /' })).rejects.toThrow(
      /rejected by remote validator.*destructive command pattern/i
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects /proc/<pid>/environ reads before posting to /exec', async () => {
    const tool = createBashExecutionTool();
    await expect(
      tool.invoke({ command: 'cat /proc/self/environ' })
    ).rejects.toThrow(/rejected by remote validator.*proc-environ-read/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects fork bombs before posting to /exec', async () => {
    const tool = createBashExecutionTool();
    await expect(tool.invoke({ command: ':(){ :|:& };:' })).rejects.toThrow(
      /rejected by remote validator/
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('rejects positional-arg bypass shape (rm -rf "$1" + args:[/])', async () => {
    const tool = createBashExecutionTool();
    await expect(
      tool.invoke({ command: 'rm -rf "$1"', args: ['/'] })
    ).rejects.toThrow(/rejected by remote validator.*positional arg/i);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('forwards safe commands through to the remote endpoint', async () => {
    const tool = createBashExecutionTool();
    await tool.invoke({ command: 'echo hi' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('allows command substitution by default (only opt-in validation flags it)', async () => {
    const tool = createBashExecutionTool();
    await tool.invoke({ command: 'echo "$(date)"' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('createBashExecutionTool — opt-in validation modes', () => {
  it('validation: "auto" lets command substitution through as a warning (no block)', async () => {
    const tool = createBashExecutionTool({ validation: 'auto' });
    await tool.invoke({ command: 'echo "$(date)"' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('validation: "strict" rejects command substitution', async () => {
    const tool = createBashExecutionTool({ validation: 'strict' });
    // Unquoted $(…) because stripQuotedContent intentionally blanks
    // quoted spans first (otherwise `echo "rm -rf /"` would false-
    // positive). The strict-mode check operates on the post-strip
    // form, so we test the form an actual exfil attempt would take.
    await expect(tool.invoke({ command: 'echo $(date)' })).rejects.toThrow(
      /rejected by remote validator.*cmd-subst/i
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('validation: "strict" rejects eval', async () => {
    const tool = createBashExecutionTool({ validation: 'strict' });
    await expect(tool.invoke({ command: 'eval "ls -la"' })).rejects.toThrow(
      /rejected by remote validator.*strict-eval/i
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('validation: "off" runs only the hard floor — substitution allowed, rm -rf / blocked', async () => {
    const tool = createBashExecutionTool({ validation: 'off' });
    await tool.invoke({ command: 'echo "$(date)"' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await expect(tool.invoke({ command: 'rm -rf /' })).rejects.toThrow(
      /rejected by remote validator/
    );
  });
});

describe('validateBashCommandStatic — shared regex + heuristic pass', () => {
  it('matches the historical { valid, errors, warnings } shape', () => {
    const result = validateBashCommandStatic('rm -rf /', undefined, {});
    expect(result).toEqual(
      expect.objectContaining({
        valid: false,
        errors: expect.arrayContaining([
          expect.stringContaining('destructive'),
        ]),
        warnings: expect.any(Array),
      })
    );
  });

  it('sudo emits a warning but does not block', () => {
    const result = validateBashCommandStatic('sudo ls', undefined, {});
    expect(result.valid).toBe(true);
    expect(result.warnings.join('\n')).toContain('sudo');
  });

  it('readOnly blocks mutating commands', () => {
    const result = validateBashCommandStatic('echo hi > out.txt', undefined, {
      readOnly: true,
    });
    expect(result.valid).toBe(false);
    expect(result.errors.join('\n')).toContain('read-only');
  });

  it('allowDangerousCommands bypasses destructive patterns', () => {
    const result = validateBashCommandStatic('rm -rf /', undefined, {
      allowDangerousCommands: true,
    });
    expect(result.valid).toBe(true);
  });

  it('blocks empty commands', () => {
    expect(validateBashCommandStatic('', undefined, {}).valid).toBe(false);
    expect(validateBashCommandStatic('   ', undefined, {}).valid).toBe(false);
  });

  it('blocks NUL bytes', () => {
    expect(validateBashCommandStatic('echo hi\0bye', undefined, {}).valid).toBe(
      false
    );
  });
});
