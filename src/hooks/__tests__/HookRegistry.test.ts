// src/hooks/__tests__/HookRegistry.test.ts
import { HookRegistry } from '../HookRegistry';
import type {
  HookMatcher,
  HookCallback,
  PreToolUseHookOutput,
  PostToolUseHookOutput,
} from '../types';

const noop: HookCallback<
  'PreToolUse'
> = async (): Promise<PreToolUseHookOutput> => ({});
const noopPost: HookCallback<
  'PostToolUse'
> = async (): Promise<PostToolUseHookOutput> => ({});

function makePreToolUseMatcher(pattern?: string): HookMatcher<'PreToolUse'> {
  return {
    pattern,
    hooks: [noop],
  };
}

function makePostToolUseMatcher(): HookMatcher<'PostToolUse'> {
  return {
    hooks: [noopPost],
  };
}

describe('HookRegistry', () => {
  describe('global registration', () => {
    it('stores a matcher and returns it via getMatchers', () => {
      const registry = new HookRegistry();
      const matcher = makePreToolUseMatcher('Bash');
      registry.register('PreToolUse', matcher);
      const matchers = registry.getMatchers('PreToolUse');
      expect(matchers).toHaveLength(1);
      expect(matchers[0]).toBe(matcher);
    });

    it('keeps registrations isolated across events', () => {
      const registry = new HookRegistry();
      const pre = makePreToolUseMatcher('Bash');
      const post = makePostToolUseMatcher();
      registry.register('PreToolUse', pre);
      registry.register('PostToolUse', post);
      expect(registry.getMatchers('PreToolUse')).toEqual([pre]);
      expect(registry.getMatchers('PostToolUse')).toEqual([post]);
      expect(registry.getMatchers('Stop')).toEqual([]);
    });

    it('unregister removes the matcher from the registry', () => {
      const registry = new HookRegistry();
      const matcher = makePreToolUseMatcher();
      const unregister = registry.register('PreToolUse', matcher);
      expect(registry.getMatchers('PreToolUse')).toHaveLength(1);
      unregister();
      expect(registry.getMatchers('PreToolUse')).toHaveLength(0);
    });

    it('hasHookFor reflects registration state', () => {
      const registry = new HookRegistry();
      expect(registry.hasHookFor('PreToolUse')).toBe(false);
      registry.register('PreToolUse', makePreToolUseMatcher());
      expect(registry.hasHookFor('PreToolUse')).toBe(true);
      expect(registry.hasHookFor('PostToolUse')).toBe(false);
    });

    it('supports multiple matchers on the same event', () => {
      const registry = new HookRegistry();
      const a = makePreToolUseMatcher('Bash');
      const b = makePreToolUseMatcher('Edit');
      registry.register('PreToolUse', a);
      registry.register('PreToolUse', b);
      const matchers = registry.getMatchers('PreToolUse');
      expect(matchers).toHaveLength(2);
      expect(matchers).toContain(a);
      expect(matchers).toContain(b);
    });

    it('returns a fresh array on each getMatchers call', () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', makePreToolUseMatcher());
      const first = registry.getMatchers('PreToolUse');
      const second = registry.getMatchers('PreToolUse');
      expect(first).not.toBe(second);
      first.length = 0;
      expect(registry.getMatchers('PreToolUse')).toHaveLength(1);
    });
  });

  describe('session registration', () => {
    it('scopes matchers to a single session', () => {
      const registry = new HookRegistry();
      const sessionA = makePreToolUseMatcher('Bash');
      const sessionB = makePreToolUseMatcher('Edit');
      registry.registerSession('run-a', 'PreToolUse', sessionA);
      registry.registerSession('run-b', 'PreToolUse', sessionB);

      expect(registry.getMatchers('PreToolUse', 'run-a')).toEqual([sessionA]);
      expect(registry.getMatchers('PreToolUse', 'run-b')).toEqual([sessionB]);
      expect(registry.getMatchers('PreToolUse')).toEqual([]);
    });

    it('merges global matchers in front of session matchers', () => {
      const registry = new HookRegistry();
      const global = makePreToolUseMatcher('global');
      const session = makePreToolUseMatcher('session');
      registry.register('PreToolUse', global);
      registry.registerSession('run-a', 'PreToolUse', session);

      const matchers = registry.getMatchers('PreToolUse', 'run-a');
      expect(matchers).toEqual([global, session]);
    });

    it('clearSession drops only the given session', () => {
      const registry = new HookRegistry();
      const a = makePreToolUseMatcher('a');
      const b = makePreToolUseMatcher('b');
      registry.registerSession('run-a', 'PreToolUse', a);
      registry.registerSession('run-b', 'PreToolUse', b);

      registry.clearSession('run-a');
      expect(registry.getMatchers('PreToolUse', 'run-a')).toEqual([]);
      expect(registry.getMatchers('PreToolUse', 'run-b')).toEqual([b]);
    });

    it('removeMatcher removes from the correct scope', () => {
      const registry = new HookRegistry();
      const global = makePreToolUseMatcher('global');
      const session = makePreToolUseMatcher('session');
      registry.register('PreToolUse', global);
      registry.registerSession('run-a', 'PreToolUse', session);

      expect(registry.removeMatcher('PreToolUse', session, 'run-a')).toBe(true);
      expect(registry.getMatchers('PreToolUse', 'run-a')).toEqual([global]);

      expect(registry.removeMatcher('PreToolUse', global)).toBe(true);
      expect(registry.getMatchers('PreToolUse', 'run-a')).toEqual([]);
    });

    it('removeMatcher returns false when the matcher is not found', () => {
      const registry = new HookRegistry();
      const orphan = makePreToolUseMatcher('orphan');
      expect(registry.removeMatcher('PreToolUse', orphan)).toBe(false);
      expect(registry.removeMatcher('PreToolUse', orphan, 'run-a')).toBe(false);
    });

    it('session unregister function removes only that matcher', () => {
      const registry = new HookRegistry();
      const a = makePreToolUseMatcher('a');
      const b = makePreToolUseMatcher('b');
      const unregisterA = registry.registerSession('run-a', 'PreToolUse', a);
      registry.registerSession('run-a', 'PreToolUse', b);

      unregisterA();
      expect(registry.getMatchers('PreToolUse', 'run-a')).toEqual([b]);
    });

    it('hasHookFor honours the sessionId parameter', () => {
      const registry = new HookRegistry();
      registry.registerSession('run-a', 'PreToolUse', makePreToolUseMatcher());
      expect(registry.hasHookFor('PreToolUse')).toBe(false);
      expect(registry.hasHookFor('PreToolUse', 'run-a')).toBe(true);
      expect(registry.hasHookFor('PreToolUse', 'run-b')).toBe(false);
    });
  });

  describe('session isolation under parallel registration', () => {
    it('does not leak matchers between sessions registered in parallel', async () => {
      const registry = new HookRegistry();
      const sessions = Array.from({ length: 50 }, (_, i) => `run-${i}`);
      await Promise.all(
        sessions.map(async (sid): Promise<void> => {
          registry.registerSession(
            sid,
            'PreToolUse',
            makePreToolUseMatcher(sid)
          );
        })
      );

      for (const sid of sessions) {
        const matchers = registry.getMatchers('PreToolUse', sid);
        expect(matchers).toHaveLength(1);
        expect(matchers[0]?.pattern).toBe(sid);
      }
    });
  });
});
