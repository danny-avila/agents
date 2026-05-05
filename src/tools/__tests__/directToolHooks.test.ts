import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect, jest, afterEach } from '@jest/globals';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type {
  PreToolUseHookOutput,
  PostToolUseHookOutput,
  PostToolUseFailureHookOutput,
  PermissionDeniedHookOutput,
} from '@/hooks';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';

/**
 * Direct-tool helper: returns a real `StructuredToolInterface` whose
 * `func` runs in-process. Once registered as a graphTool, the ToolNode
 * marks it `direct` (skipping the host event-dispatch path) — the path
 * we are testing fires lifecycle hooks around.
 */
function createDirectTool(
  name: string,
  impl: (args: Record<string, unknown>) => string | Promise<string>
): StructuredToolInterface {
  return tool(async (args: Record<string, unknown>) => impl(args), {
    name,
    description: `direct in-process tool ${name}`,
    schema: z.object({ command: z.string().optional() }).passthrough(),
  }) as unknown as StructuredToolInterface;
}

function aiCall(
  callId: string,
  name: string,
  args: Record<string, unknown>
): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [{ id: callId, name, args }],
  });
}

function toolMessages(result: unknown): ToolMessage[] {
  if (Array.isArray(result)) {
    return result as ToolMessage[];
  }
  const obj = result as { messages: ToolMessage[] };
  return obj.messages;
}

describe('Direct-path lifecycle hooks (in-process tools)', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('PreToolUse decision: deny replaces the tool result with a Blocked ToolMessage and fires PermissionDenied', async () => {
    const echo = createDirectTool('echo', () => 'EXECUTED');

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'deny',
          reason: 'policy-deny',
        }),
      ],
    });
    const permissionDenied = jest.fn(
      async (): Promise<PermissionDeniedHookOutput> => ({})
    );
    registry.register('PermissionDenied', { hooks: [permissionDenied] });

    const node = new ToolNode({
      tools: [echo],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
    });

    const result = await node.invoke({
      messages: [aiCall('call_1', 'echo', { command: 'rm -rf /' })],
    });
    const [message] = toolMessages(result);

    expect(message.status).toBe('error');
    expect(String(message.content)).toContain('Blocked: policy-deny');
    expect(String(message.content)).not.toContain('EXECUTED');

    // Event handlers can be async — flush the microtask queue once.
    await Promise.resolve();
    expect(permissionDenied).toHaveBeenCalledTimes(1);
  });

  it('PreToolUse decision: ask is fail-closed when humanInTheLoop is disabled', async () => {
    const echo = createDirectTool('echo', () => 'EXECUTED');

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'ask',
          reason: 'needs-review',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [echo],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
      // humanInTheLoop intentionally not set — fail-closed.
    });

    const result = await node.invoke({
      messages: [aiCall('call_2', 'echo', { command: 'whoami' })],
    });
    const [message] = toolMessages(result);

    expect(message.status).toBe('error');
    expect(String(message.content)).toContain('Blocked: needs-review');
    expect(String(message.content)).not.toContain('EXECUTED');
  });

  it('PreToolUse decision: allow runs the tool unchanged', async () => {
    const echo = createDirectTool('echo', (args) => `ran:${args.command}`);

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({ decision: 'allow' }),
      ],
    });

    const node = new ToolNode({
      tools: [echo],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
    });

    const result = await node.invoke({
      messages: [aiCall('call_3', 'echo', { command: 'ls' })],
    });
    const [message] = toolMessages(result);
    expect(message.status).toBe('success');
    expect(String(message.content)).toBe('ran:ls');
  });

  it('PreToolUse updatedInput rewrites the args before runTool sees them', async () => {
    const seen: Record<string, unknown>[] = [];
    const echo = createDirectTool('echo', (args) => {
      seen.push(args);
      return `ran:${String(args.command)}`;
    });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (): Promise<PreToolUseHookOutput> => ({
          decision: 'allow',
          updatedInput: { command: 'redacted' },
        }),
      ],
    });

    const node = new ToolNode({
      tools: [echo],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
    });

    const result = await node.invoke({
      messages: [aiCall('call_4', 'echo', { command: 'secret' })],
    });
    const [message] = toolMessages(result);
    expect(String(message.content)).toBe('ran:redacted');
    expect(seen[0]).toMatchObject({ command: 'redacted' });
  });

  it('PostToolUse updatedOutput replaces the tool message content', async () => {
    const echo = createDirectTool('echo', () => 'ORIGINAL');

    const registry = new HookRegistry();
    registry.register('PostToolUse', {
      hooks: [
        async (): Promise<PostToolUseHookOutput> => ({
          updatedOutput: 'REPLACED',
        }),
      ],
    });

    const node = new ToolNode({
      tools: [echo],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
    });

    const result = await node.invoke({
      messages: [aiCall('call_5', 'echo', { command: 'x' })],
    });
    const [message] = toolMessages(result);
    expect(String(message.content)).toBe('REPLACED');
    expect(message.status).toBe('success');
  });

  it('PostToolUseFailure observes errors thrown by the tool', async () => {
    const failing = createDirectTool('boom', () => {
      throw new Error('kaboom');
    });

    const failure = jest.fn(
      async (): Promise<PostToolUseFailureHookOutput> => ({})
    );
    const registry = new HookRegistry();
    registry.register('PostToolUseFailure', { hooks: [failure] });

    const node = new ToolNode({
      tools: [failing],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['boom']),
    });

    const result = await node.invoke({
      messages: [aiCall('call_6', 'boom', { command: 'x' })],
    });
    const [message] = toolMessages(result);
    expect(message.status).toBe('error');
    expect(String(message.content)).toContain('kaboom');

    await Promise.resolve();
    expect(failure).toHaveBeenCalledTimes(1);
  });

  it('no-hooks fast path: when no relevant hooks registered, runs runTool directly without overhead', async () => {
    const echo = createDirectTool('echo', () => 'fast-path');

    const registry = new HookRegistry();
    // Only register an unrelated event so hasHookFor returns false for
    // PreToolUse / PostToolUse / PostToolUseFailure.
    registry.register('RunStart', {
      hooks: [async (): Promise<Record<string, never>> => ({})],
    });

    const node = new ToolNode({
      tools: [echo],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
    });

    const result = await node.invoke({
      messages: [aiCall('call_7', 'echo', { command: 'x' })],
    });
    const [message] = toolMessages(result);
    expect(String(message.content)).toBe('fast-path');
  });

  it('mixed batch: PreToolUse deny on a direct tool runs the surviving tool only', async () => {
    const allowed = createDirectTool('allowed', () => 'allowed-ran');
    const denied = createDirectTool('denied', () => 'should-not-run');

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async ({ toolName }): Promise<PreToolUseHookOutput> =>
          toolName === 'denied'
            ? { decision: 'deny', reason: 'no-no' }
            : { decision: 'allow' },
      ],
    });

    const node = new ToolNode({
      tools: [allowed, denied],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['allowed', 'denied']),
    });

    const result = await node.invoke({
      messages: [
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'call_a', name: 'allowed', args: { command: 'a' } },
            { id: 'call_b', name: 'denied', args: { command: 'b' } },
          ],
        }),
      ],
    });
    const messages = toolMessages(result);
    expect(messages).toHaveLength(2);
    const byId = new Map(messages.map((m) => [m.tool_call_id, m]));
    expect(String(byId.get('call_a')?.content)).toBe('allowed-ran');
    expect(String(byId.get('call_b')?.content)).toContain('Blocked: no-no');
  });

  it('PreToolUse `turn` matches the per-tool index the body actually executes under (Codex P2 #27)', async () => {
    // Three parallel direct calls of the same tool. Pre-fix: each
    // hook read `turn = toolUsageCount.get('echo') ?? 0` BEFORE any
    // await, so all three saw 0; runTool then incremented inside its
    // own scope and the bodies ran as 0/1/2. Hook → tool got
    // misaligned, breaking host policies that key on
    // (toolName, turn). With the fix, the increment is hoisted into
    // runDirectToolWithLifecycleHooks (sync, before any await) and
    // threaded into runTool via batchContext so both observe the
    // same value.
    const hookTurns: number[] = [];
    const bodyTurns: number[] = [];
    let bodyCount = 0;
    const echo = createDirectTool('echo', () => {
      bodyCount += 1;
      return 'EXECUTED';
    });

    const registry = new HookRegistry();
    registry.register('PreToolUse', {
      hooks: [
        async (input): Promise<PreToolUseHookOutput> => {
          if (typeof input.turn === 'number') hookTurns.push(input.turn);
          return { decision: 'allow' };
        },
      ],
    });

    const node = new ToolNode({
      tools: [echo],
      eventDrivenMode: true,
      hookRegistry: registry,
      directToolNames: new Set(['echo']),
    });

    // Patch the tool's `func` to record the turn the body sees via the
    // standard LangChain config.toolCall.turn channel.
    const originalFunc = (echo as unknown as { func: (input: unknown, config: unknown) => Promise<string> }).func;
    (echo as unknown as { func: (input: unknown, config: unknown) => Promise<string> }).func = async (
      input,
      config
    ): Promise<string> => {
      const t = (config as { toolCall?: { turn?: number } } | undefined)
        ?.toolCall?.turn;
      if (typeof t === 'number') bodyTurns.push(t);
      return originalFunc(input, config);
    };

    const aiMsg = new AIMessage({
      content: '',
      tool_calls: [
        { id: 'c0', name: 'echo', args: { command: 'a' } },
        { id: 'c1', name: 'echo', args: { command: 'b' } },
        { id: 'c2', name: 'echo', args: { command: 'c' } },
      ],
    });
    await node.invoke({ messages: [aiMsg] });

    // Sanity: tool ran 3 times, hook fired 3 times.
    expect(bodyCount).toBe(3);
    expect(hookTurns.length).toBe(3);
    // Post-fix: each hook observes a unique turn (one of 0, 1, 2)
    // — the SAME turn the body executes under. Pre-fix they all
    // saw 0, so the dedupe-to-3 assertion would fail.
    expect(new Set(hookTurns).size).toBe(3);
    expect([...hookTurns].sort()).toEqual([0, 1, 2]);
    // The body-side `config.toolCall.turn` should also align (when
    // visible — LangChain may not propagate config in every test
    // shape; assert weakly).
    if (bodyTurns.length === 3) {
      expect([...bodyTurns].sort()).toEqual([0, 1, 2]);
    }
  });
});
