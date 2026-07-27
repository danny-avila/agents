// src/hooks/__tests__/preemptBoundary.test.ts
import type { HookMatcher, HookCallback, HookOutput } from '../types';
import { HOOK_EVENTS } from '../types';
import { HookRegistry } from '../HookRegistry';
import { executeHooks } from '../executeHooks';
import {
  HOOK_PREEMPT_BOUNDARY_CAPABLE,
  HOOK_INJECTED_MESSAGES_CAPABLE,
} from '../index';

const noop: HookCallback<
  'PreemptBoundary'
> = async (): Promise<HookOutput> => ({});

function makeMatcher(
  hook: HookCallback<'PreemptBoundary'> = noop
): HookMatcher<'PreemptBoundary'> {
  return { hooks: [hook] };
}

describe('PreemptBoundary hook event', () => {
  it('is part of the closed event set', () => {
    expect(HOOK_EVENTS).toContain('PreemptBoundary');
  });

  it('advertises its own capability flag, separate from injectedMessages', () => {
    expect(HOOK_PREEMPT_BOUNDARY_CAPABLE).toBe(true);
    expect(HOOK_INJECTED_MESSAGES_CAPABLE).toBe(true);
  });

  /**
   * Load-bearing: `hasResultAlteringHooks` gates eager tool execution. If
   * `PreemptBoundary` ever counted as result-altering, registering a steering
   * drain would silently disable eager dispatch for every steering run.
   */
  it('is not result-altering, so eager tool execution stays enabled', () => {
    const registry = new HookRegistry();
    registry.register('PreemptBoundary', makeMatcher());
    expect(registry.hasResultAlteringHooks()).toBe(false);
  });

  it('still reports result-altering when a tool hook is also registered', () => {
    const registry = new HookRegistry();
    registry.register('PreemptBoundary', makeMatcher());
    registry.register('PreToolUse', { hooks: [async () => ({})] });
    expect(registry.hasResultAlteringHooks()).toBe(true);
  });

  /**
   * `hasHookFor` answers "is one registered"; the seal gate needs "will one
   * fire". A pattern-scoped matcher is inert for a query-less dispatch, so
   * treating registration as a proxy would seal into a boundary that injects
   * nothing and cut the answer short.
   */
  describe('hasDispatchableHookFor', () => {
    it('is true for a wildcard matcher with a callback', () => {
      const registry = new HookRegistry();
      registry.register('PreemptBoundary', makeMatcher());
      expect(registry.hasDispatchableHookFor('PreemptBoundary')).toBe(true);
    });

    it('is false for a pattern-scoped matcher that can never match', () => {
      const registry = new HookRegistry();
      registry.register('PreemptBoundary', { pattern: 'Bash', hooks: [noop] });
      expect(registry.hasHookFor('PreemptBoundary')).toBe(true);
      expect(registry.hasDispatchableHookFor('PreemptBoundary')).toBe(false);
    });

    it('is false for a matcher carrying no callbacks', () => {
      const registry = new HookRegistry();
      registry.register('PreemptBoundary', { hooks: [] });
      expect(registry.hasDispatchableHookFor('PreemptBoundary')).toBe(false);
    });

    it('is false when nothing is registered at all', () => {
      const registry = new HookRegistry();
      expect(registry.hasDispatchableHookFor('PreemptBoundary')).toBe(false);
    });

    it('finds a session-scoped dispatchable matcher', () => {
      const registry = new HookRegistry();
      registry.registerSession('run_1', 'PreemptBoundary', makeMatcher());
      expect(registry.hasDispatchableHookFor('PreemptBoundary')).toBe(false);
      expect(registry.hasDispatchableHookFor('PreemptBoundary', 'run_1')).toBe(
        true
      );
    });
  });

  it('keeps registration isolated from the tool boundary event', () => {
    const registry = new HookRegistry();
    const matcher = makeMatcher();
    registry.register('PreemptBoundary', matcher);
    expect(registry.getMatchers('PreemptBoundary')).toEqual([matcher]);
    expect(registry.getMatchers('PostToolBatch')).toEqual([]);
  });

  it('aggregates injectedMessages from the boundary dispatch', async () => {
    const registry = new HookRegistry();
    registry.register(
      'PreemptBoundary',
      makeMatcher(async () => ({
        injectedMessages: [
          { role: 'user', content: 'Skip phase two.', source: 'steer' },
        ],
      }))
    );

    const result = await executeHooks({
      registry,
      input: {
        hook_event_name: 'PreemptBoundary',
        runId: 'run_1',
        executingAgentId: 'agent_1',
        sealCount: 1,
      },
      sessionId: 'run_1',
    });

    expect(result.injectedMessages).toEqual([
      { role: 'user', content: 'Skip phase two.', source: 'steer' },
    ]);
    expect(result.errors).toEqual([]);
  });

  it('reports the 1-based seal index to the hook', async () => {
    const registry = new HookRegistry();
    const seen: number[] = [];
    registry.register(
      'PreemptBoundary',
      makeMatcher(async (input) => {
        seen.push(input.sealCount);
        return {};
      })
    );

    for (const sealCount of [1, 2]) {
      await executeHooks({
        registry,
        input: {
          hook_event_name: 'PreemptBoundary',
          runId: 'run_1',
          executingAgentId: 'agent_1',
          sealCount,
        },
        sessionId: 'run_1',
      });
    }

    expect(seen).toEqual([1, 2]);
  });
});
