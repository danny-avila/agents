import { tmpdir } from 'os';
import { join, resolve } from 'path';
import { mkdtemp, rm } from 'fs/promises';
import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { createWorkspacePolicyHook } from '../createWorkspacePolicyHook';
import type {
  HookCallback,
  PreToolUseHookInput,
  PreToolUseHookOutput,
} from '../types';

function call(
  hook: HookCallback<'PreToolUse'>,
  input: PreToolUseHookInput
): Promise<PreToolUseHookOutput> {
  return Promise.resolve(hook(input, new AbortController().signal));
}

function makeInput(
  toolName: string,
  toolInput: Record<string, unknown>
): PreToolUseHookInput {
  return {
    hook_event_name: 'PreToolUse',
    runId: 'r',
    threadId: 't',
    agentId: 'a',
    toolName,
    toolInput,
    toolUseId: 'tu',
    stepId: 's',
    turn: 0,
  };
}

describe('createWorkspacePolicyHook', () => {
  let workspace: string;
  let extra: string;

  beforeEach(async () => {
    workspace = await mkdtemp(join(tmpdir(), 'lc-wp-'));
    extra = await mkdtemp(join(tmpdir(), 'lc-wp-extra-'));
  });

  afterEach(async () => {
    await rm(workspace, { recursive: true, force: true });
    await rm(extra, { recursive: true, force: true });
  });

  it('allows in-workspace relative paths', async () => {
    const hook = createWorkspacePolicyHook({ root: workspace });
    const out = await call(hook,makeInput('read_file', { file_path: 'src/x.ts' }));
    expect(out.decision).toBe('allow');
  });

  it('allows in-workspace absolute paths', async () => {
    const hook = createWorkspacePolicyHook({ root: workspace });
    const out = await call(hook,
      makeInput('read_file', { file_path: join(workspace, 'src/x.ts') })
    );
    expect(out.decision).toBe('allow');
  });

  it('allows additionalRoots paths', async () => {
    const hook = createWorkspacePolicyHook({
      root: workspace,
      additionalRoots: [extra],
    });
    const out = await call(hook,
      makeInput('read_file', { file_path: join(extra, 'lib/y.ts') })
    );
    expect(out.decision).toBe('allow');
  });

  it('asks by default when path is outside the workspace (read tool)', async () => {
    const hook = createWorkspacePolicyHook({ root: workspace });
    const out = await call(hook,
      makeInput('read_file', { file_path: '/etc/passwd' })
    );
    expect(out.decision).toBe('ask');
    expect(out.reason).toContain('/etc/passwd');
    expect(out.allowedDecisions).toEqual(['approve', 'reject']);
  });

  it('asks by default when path is outside the workspace (write tool)', async () => {
    const hook = createWorkspacePolicyHook({ root: workspace });
    const out = await call(hook,
      makeInput('write_file', {
        file_path: '/etc/foo',
        content: 'malicious',
      })
    );
    expect(out.decision).toBe('ask');
  });

  it('honours `outsideRead: deny` for read tools', async () => {
    const hook = createWorkspacePolicyHook({
      root: workspace,
      outsideRead: 'deny',
    });
    const out = await call(hook,
      makeInput('read_file', { file_path: '/etc/passwd' })
    );
    expect(out.decision).toBe('deny');
    expect(out.reason).toContain('/etc/passwd');
  });

  it('honours `outsideWrite: deny` for write tools', async () => {
    const hook = createWorkspacePolicyHook({
      root: workspace,
      outsideWrite: 'deny',
    });
    const out = await call(hook,
      makeInput('edit_file', {
        file_path: '/etc/foo',
        old_text: 'a',
        new_text: 'b',
      })
    );
    expect(out.decision).toBe('deny');
  });

  it('honours `outsideRead: allow`', async () => {
    const hook = createWorkspacePolicyHook({
      root: workspace,
      outsideRead: 'allow',
    });
    const out = await call(hook,
      makeInput('read_file', { file_path: '/etc/passwd' })
    );
    expect(out.decision).toBe('allow');
  });

  it('passes through when the tool has no extractor (e.g. bash)', async () => {
    const hook = createWorkspacePolicyHook({ root: workspace });
    const out = await call(hook,
      makeInput('bash', { command: 'cat /etc/passwd' })
    );
    expect(out.decision).toBe('allow');
  });

  it('passes through when the path arg is missing/empty', async () => {
    const hook = createWorkspacePolicyHook({ root: workspace });
    const out = await call(hook,makeInput('grep_search', { pattern: 'x' }));
    expect(out.decision).toBe('allow');
  });

  it('allows custom extractors to opt new tools in', async () => {
    const hook = createWorkspacePolicyHook({
      root: workspace,
      pathExtractors: {
        my_custom_tool: (i) =>
          typeof i.target === 'string' ? [i.target] : [],
      },
    });
    const inside = await call(hook,
      makeInput('my_custom_tool', { target: 'in/workspace.ts' })
    );
    expect(inside.decision).toBe('allow');

    const outside = await call(hook,
      makeInput('my_custom_tool', { target: '/elsewhere/x.ts' })
    );
    // Unknown tools default to write-policy (stricter): 'ask'.
    expect(outside.decision).toBe('ask');
  });

  it('formats the reason via {tool} and {paths}', async () => {
    const hook = createWorkspacePolicyHook({
      root: workspace,
      reason: '{tool} blocked from {paths}',
    });
    const out = await call(hook,
      makeInput('read_file', { file_path: '/somewhere/else.ts' })
    );
    expect(out.reason).toBe('read_file blocked from /somewhere/else.ts');
  });

  it('handles a tool input with an array of paths', async () => {
    const hook = createWorkspacePolicyHook({
      root: workspace,
      pathExtractors: {
        multi_file_tool: (i) =>
          Array.isArray(i.paths) ? (i.paths as string[]) : [],
      },
    });
    const out = await call(hook,
      makeInput('multi_file_tool', {
        paths: [join(workspace, 'a.ts'), '/etc/passwd'],
      })
    );
    expect(out.decision).toBe('ask');
    expect(out.reason).toContain('/etc/passwd');
    expect(out.reason).not.toContain(join(workspace, 'a.ts'));
  });

  it('respects resolved root paths (relative root config)', async () => {
    const hook = createWorkspacePolicyHook({ root: '.' });
    const out = await call(hook,
      makeInput('read_file', { file_path: resolve('.', 'src/x.ts') })
    );
    expect(out.decision).toBe('allow');
  });

  describe('symlink-escape (Codex P1 #10)', () => {
    it('rejects a symlink inside the workspace that points outside', async () => {
      const fs = await import('fs/promises');
      await fs.writeFile(join(extra, 'secret.txt'), 'top-secret\n');
      await fs.symlink(
        join(extra, 'secret.txt'),
        join(workspace, 'escape')
      );
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('read_file', { file_path: 'escape' })
      );
      expect(out.decision).toBe('deny');
      expect(out.reason).toContain('escape');
    });

    it('still allows a lexically-outside path that realpaths back inside (alternate mount via symlink)', async () => {
      const fs = await import('fs/promises');
      await fs.writeFile(join(workspace, 'file.ts'), 'export {};\n');
      const altMount = join(extra, 'alt-mount');
      await fs.symlink(workspace, altMount);
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('read_file', { file_path: join(altMount, 'file.ts') })
      );
      expect(out.decision).toBe('allow');
    });
  });

  describe('compile_check command path extraction (Codex P1 #26)', () => {
    it('extracts absolute paths from compile_check commands so the policy fires', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'cat /etc/passwd' })
      );
      expect(out.decision).toBe('deny');
      expect(out.reason).toContain('/etc/passwd');
    });

    it('extracts paths from --flag=/path-style tokens', async () => {
      // compile_check is in READ_TOOLS, so the policy uses outsideRead.
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'tsc --out=/etc/foo' })
      );
      expect(out.decision).toBe('deny');
      expect(out.reason).toContain('/etc/foo');
    });

    it('extracts ~/ and $HOME/ tokens', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const tilde = await call(
        hook,
        makeInput('compile_check', { command: 'cat ~/secret' })
      );
      expect(tilde.decision).toBe('deny');
      const homeVar = await call(
        hook,
        makeInput('compile_check', { command: 'cat $HOME/secret' })
      );
      expect(homeVar.decision).toBe('deny');
    });

    it('allows in-workspace absolute paths in the command', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', {
          command: `tsc --noEmit ${join(workspace, 'src/x.ts')}`,
        })
      );
      expect(out.decision).toBe('allow');
    });

    it('allows commands with no absolute paths (relative-only auto-detect form)', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'npx tsc --noEmit' })
      );
      expect(out.decision).toBe('allow');
    });

    it('extracts paths wrapped in double quotes (Codex P1 #31)', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'cat "/etc/passwd"' })
      );
      expect(out.decision).toBe('deny');
      expect(out.reason).toContain('/etc/passwd');
    });

    it('extracts paths wrapped in single quotes (Codex P1 #31)', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'cat \'/etc/passwd\'' })
      );
      expect(out.decision).toBe('deny');
      expect(out.reason).toContain('/etc/passwd');
    });

    it('extracts quoted --flag="/path" tokens (Codex P1 #31)', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'tsc --out="/tmp/x"' })
      );
      expect(out.decision).toBe('deny');
      expect(out.reason).toContain('/tmp/x');
    });

    it('extracts ../ parent-traversal tokens (Codex P2 #35)', async () => {
      // `../secrets.txt` resolves to `<parent-of-workspace>/secrets.txt`
      // which is outside the workspace boundary. Pre-fix the regex
      // ignored relative tokens entirely so this looked like an
      // empty-extract → allow.
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'cat ../secrets.txt' })
      );
      expect(out.decision).toBe('deny');
      expect(out.reason).toContain('../secrets.txt');
    });

    it('extracts deeper traversal like ../../etc/foo (Codex P2 #35)', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'cat ../../etc/foo' })
      );
      expect(out.decision).toBe('deny');
    });

    it('does NOT trip on `./` or single-dot tokens (no false positives)', async () => {
      const hook = createWorkspacePolicyHook({
        root: workspace,
        outsideRead: 'deny',
      });
      const out = await call(
        hook,
        makeInput('compile_check', { command: 'cd ./src && npx tsc' })
      );
      expect(out.decision).toBe('allow');
    });
  });
});
