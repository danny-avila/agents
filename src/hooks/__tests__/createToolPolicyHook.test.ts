import { describe, it, expect } from '@jest/globals';
import type {
  HookCallback,
  PreToolUseHookInput,
  PreToolUseHookOutput,
} from '../types';
import { createToolPolicyHook } from '../createToolPolicyHook';

const baseInput: Omit<PreToolUseHookInput, 'toolName'> = {
  hook_event_name: 'PreToolUse',
  runId: 'r-1',
  toolInput: {},
  toolUseId: 'call-1',
  stepId: 'step-1',
  turn: 0,
};

async function callHook(
  hook: HookCallback<'PreToolUse'>,
  toolName: string
): Promise<PreToolUseHookOutput> {
  const signal = new AbortController().signal;
  return await hook({ ...baseInput, toolName }, signal);
}

describe('createToolPolicyHook — default mode', () => {
  it('asks for tools that match no rule', async () => {
    const hook = createToolPolicyHook({ mode: 'default' });
    expect((await callHook(hook, 'unknown_tool')).decision).toBe('ask');
  });

  it('allows tools that match an allow pattern', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      allow: ['read_file', 'grep'],
    });
    expect((await callHook(hook, 'read_file')).decision).toBe('allow');
    expect((await callHook(hook, 'grep')).decision).toBe('allow');
    expect((await callHook(hook, 'write_file')).decision).toBe('ask');
  });

  it('denies tools that match a deny pattern', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      deny: ['delete_*'],
    });
    expect((await callHook(hook, 'delete_file')).decision).toBe('deny');
    expect((await callHook(hook, 'read_file')).decision).toBe('ask');
  });

  it('asks tools that match an ask pattern (redundant in default mode but explicit)', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      ask: ['execute_*'],
    });
    expect((await callHook(hook, 'execute_code')).decision).toBe('ask');
  });
});

describe('createToolPolicyHook — dontAsk mode', () => {
  it('denies tools that match no rule (no human prompt)', async () => {
    const hook = createToolPolicyHook({ mode: 'dontAsk' });
    expect((await callHook(hook, 'unknown_tool')).decision).toBe('deny');
  });

  it('still allows tools that match an allow pattern', async () => {
    const hook = createToolPolicyHook({
      mode: 'dontAsk',
      allow: ['read_*'],
    });
    expect((await callHook(hook, 'read_file')).decision).toBe('allow');
    expect((await callHook(hook, 'write_file')).decision).toBe('deny');
  });

  it('still asks tools that match an explicit ask pattern (overrides dontAsk default)', async () => {
    const hook = createToolPolicyHook({
      mode: 'dontAsk',
      ask: ['execute_*'],
    });
    expect((await callHook(hook, 'execute_code')).decision).toBe('ask');
    expect((await callHook(hook, 'unknown_tool')).decision).toBe('deny');
  });
});

describe('createToolPolicyHook — bypass mode', () => {
  it('allows everything by default', async () => {
    const hook = createToolPolicyHook({ mode: 'bypass' });
    expect((await callHook(hook, 'anything')).decision).toBe('allow');
    expect((await callHook(hook, 'execute_code')).decision).toBe('allow');
  });

  it('still denies tools that match a deny pattern (deny always wins)', async () => {
    const hook = createToolPolicyHook({
      mode: 'bypass',
      deny: ['delete_*'],
    });
    expect((await callHook(hook, 'delete_file')).decision).toBe('deny');
    expect((await callHook(hook, 'read_file')).decision).toBe('allow');
  });

  it('overrides explicit ask patterns (bypass means stop asking)', async () => {
    const hook = createToolPolicyHook({
      mode: 'bypass',
      ask: ['execute_*'],
    });
    expect((await callHook(hook, 'execute_code')).decision).toBe('allow');
  });
});

describe('createToolPolicyHook — pattern matching', () => {
  it('matches glob `*` wildcards', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      allow: ['mcp:github:*'],
    });
    expect((await callHook(hook, 'mcp:github:create_issue')).decision).toBe(
      'allow'
    );
    expect((await callHook(hook, 'mcp:github:list_repos')).decision).toBe(
      'allow'
    );
    expect((await callHook(hook, 'mcp:slack:post')).decision).toBe('ask');
  });

  it('matches exact tool names', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      allow: ['read_file'],
    });
    expect((await callHook(hook, 'read_file')).decision).toBe('allow');
    expect((await callHook(hook, 'read_file_lines')).decision).toBe('ask');
  });

  it('escapes regex metacharacters in literal portions', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      allow: ['tool.with.dots'],
    });
    expect((await callHook(hook, 'tool.with.dots')).decision).toBe('allow');
    /** A literal regex `.` would also match `tool_with_dots`; glob shouldn't. */
    expect((await callHook(hook, 'tool_with_dots')).decision).toBe('ask');
  });

  it('matches wildcards in the middle and end', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      ask: ['*search*'],
    });
    expect((await callHook(hook, 'web_search')).decision).toBe('ask');
    expect((await callHook(hook, 'searcher')).decision).toBe('ask');
    expect((await callHook(hook, 'read_file')).decision).toBe('ask'); // default mode
    /** Confirm the ask path tagged it (not the fallthrough): explicit ask hits before mode fallthrough. */
  });
});

describe('createToolPolicyHook — precedence', () => {
  it('deny wins over allow', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      allow: ['read_*'],
      deny: ['read_secret'],
    });
    expect((await callHook(hook, 'read_secret')).decision).toBe('deny');
    expect((await callHook(hook, 'read_file')).decision).toBe('allow');
  });

  it('deny wins over bypass mode', async () => {
    const hook = createToolPolicyHook({
      mode: 'bypass',
      deny: ['delete_*'],
    });
    expect((await callHook(hook, 'delete_file')).decision).toBe('deny');
    expect((await callHook(hook, 'anything_else')).decision).toBe('allow');
  });

  it('allow wins over ask in default mode', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      allow: ['execute_safe'],
      ask: ['execute_*'],
    });
    expect((await callHook(hook, 'execute_safe')).decision).toBe('allow');
    expect((await callHook(hook, 'execute_dangerous')).decision).toBe('ask');
  });
});

describe('createToolPolicyHook — reason', () => {
  it('attaches the configured reason to ask and deny decisions', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      deny: ['delete_*'],
      reason: 'Tool {tool} requires manual review',
    });
    const denied = await callHook(hook, 'delete_file');
    expect(denied.decision).toBe('deny');
    expect(denied.reason).toBe('Tool delete_file requires manual review');

    const asked = await callHook(hook, 'unknown_tool');
    expect(asked.decision).toBe('ask');
    expect(asked.reason).toBe('Tool unknown_tool requires manual review');
  });

  it('omits the reason field for allow decisions', async () => {
    const hook = createToolPolicyHook({
      mode: 'default',
      allow: ['read_*'],
      reason: 'never seen',
    });
    const result = await callHook(hook, 'read_file');
    expect(result.decision).toBe('allow');
    expect(result.reason).toBeUndefined();
  });

  it('does not add a reason field when no template is configured', async () => {
    const hook = createToolPolicyHook({ mode: 'dontAsk' });
    const result = await callHook(hook, 'unknown_tool');
    expect(result.decision).toBe('deny');
    expect(result.reason).toBeUndefined();
  });
});

describe('createToolPolicyHook — registry integration', () => {
  it('works when registered as a PreToolUse hook (round-trip via executeHooks)', async () => {
    const { HookRegistry, executeHooks } = await import('../index');
    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        createToolPolicyHook({
          mode: 'default',
          allow: ['read_file'],
          deny: ['delete_*'],
          reason: 'review {tool}',
        }),
      ],
    });

    const allow = await executeHooks({
      registry,
      input: { ...baseInput, toolName: 'read_file' },
      matchQuery: 'read_file',
    });
    expect(allow.decision).toBe('allow');

    const deny = await executeHooks({
      registry,
      input: { ...baseInput, toolName: 'delete_file' },
      matchQuery: 'delete_file',
    });
    expect(deny.decision).toBe('deny');
    expect(deny.reason).toBe('review delete_file');

    const ask = await executeHooks({
      registry,
      input: { ...baseInput, toolName: 'mystery_tool' },
      matchQuery: 'mystery_tool',
    });
    expect(ask.decision).toBe('ask');
  });
});
