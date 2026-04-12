// src/hooks/__tests__/executeHooks.test.ts
import { HookRegistry } from '../HookRegistry';
import { executeHooks } from '../executeHooks';
import type {
  HookCallback,
  HookMatcher,
  RunStartHookInput,
  RunStartHookOutput,
  StopHookInput,
  StopHookOutput,
  PreToolUseHookInput,
  PreToolUseHookOutput,
  PostToolUseHookInput,
  PostToolUseHookOutput,
} from '../types';

function preToolUseInput(
  toolName: string,
  overrides: Partial<PreToolUseHookInput> = {}
): PreToolUseHookInput {
  return {
    hook_event_name: 'PreToolUse',
    runId: 'run-1',
    threadId: 'thread-1',
    toolName,
    toolInput: { cmd: 'ls' },
    toolUseId: 'tool-call-1',
    ...overrides,
  };
}

function postToolUseInput(
  toolName: string,
  overrides: Partial<PostToolUseHookInput> = {}
): PostToolUseHookInput {
  return {
    hook_event_name: 'PostToolUse',
    runId: 'run-1',
    toolName,
    toolInput: {},
    toolOutput: null,
    toolUseId: 'tc-1',
    ...overrides,
  };
}

function stopInput(overrides: Partial<StopHookInput> = {}): StopHookInput {
  return {
    hook_event_name: 'Stop',
    runId: 'run-1',
    messages: [],
    stopHookActive: false,
    ...overrides,
  };
}

function runStartInput(): RunStartHookInput {
  return {
    hook_event_name: 'RunStart',
    runId: 'run-1',
    messages: [],
  };
}

function preToolHook(
  fn: HookCallback<'PreToolUse'>
): HookCallback<'PreToolUse'> {
  return fn;
}

function postToolHook(
  fn: HookCallback<'PostToolUse'>
): HookCallback<'PostToolUse'> {
  return fn;
}

function runStartHook(fn: HookCallback<'RunStart'>): HookCallback<'RunStart'> {
  return fn;
}

function stopHook(fn: HookCallback<'Stop'>): HookCallback<'Stop'> {
  return fn;
}

const emptyPreOutput: PreToolUseHookOutput = {};
const emptyRunStartOutput: RunStartHookOutput = {};

const noopPreHook = preToolHook(
  async (): Promise<PreToolUseHookOutput> => emptyPreOutput
);
const noopRunStartHook = runStartHook(
  async (): Promise<RunStartHookOutput> => emptyRunStartOutput
);

describe('executeHooks', () => {
  let consoleWarnSpy: jest.SpyInstance;

  beforeEach(() => {
    consoleWarnSpy = jest
      .spyOn(console, 'warn')
      .mockImplementation((): void => {
        /* silence expected warnings */
      });
  });

  afterEach(() => {
    consoleWarnSpy.mockRestore();
  });

  describe('empty matcher set', () => {
    it('returns an empty aggregated result when no matchers are registered', async () => {
      const registry = new HookRegistry();
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result).toEqual({ additionalContexts: [], errors: [] });
    });

    it('returns an empty result when no matcher pattern matches the query', async () => {
      const registry = new HookRegistry();
      let called = false;
      registry.register('PreToolUse', {
        pattern: '^Edit$',
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            called = true;
            return emptyPreOutput;
          }),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(called).toBe(false);
      expect(result).toEqual({ additionalContexts: [], errors: [] });
    });
  });

  describe('matcher regex filtering', () => {
    it('fires hooks whose matcher regex matches the query', async () => {
      const registry = new HookRegistry();
      const calls: string[] = [];
      registry.register('PreToolUse', {
        pattern: '^Bash$',
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            calls.push('bash-only');
            return emptyPreOutput;
          }),
        ],
      });
      registry.register('PreToolUse', {
        pattern: 'Bash|Edit',
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            calls.push('bash-or-edit');
            return emptyPreOutput;
          }),
        ],
      });
      registry.register('PreToolUse', {
        pattern: '^Edit$',
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            calls.push('edit-only');
            return emptyPreOutput;
          }),
        ],
      });

      await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(calls.sort()).toEqual(['bash-only', 'bash-or-edit']);
    });

    it('fires matchers with no pattern regardless of query', async () => {
      const registry = new HookRegistry();
      let fired = false;
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            fired = true;
            return emptyPreOutput;
          }),
        ],
      });
      await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(fired).toBe(true);
    });
  });

  describe('decision precedence (deny > ask > allow)', () => {
    it('deny beats allow', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              decision: 'allow',
              reason: 'all good',
            })
          ),
        ],
      });
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              decision: 'deny',
              reason: 'forbidden',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.decision).toBe('deny');
      expect(result.reason).toBe('forbidden');
    });

    it('deny beats ask', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              decision: 'ask',
              reason: 'needs prompt',
            })
          ),
        ],
      });
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              decision: 'deny',
              reason: 'forbidden',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.decision).toBe('deny');
      expect(result.reason).toBe('forbidden');
    });

    it('ask beats allow but not deny', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({ decision: 'allow' })
          ),
        ],
      });
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              decision: 'ask',
              reason: 'please confirm',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.decision).toBe('ask');
      expect(result.reason).toBe('please confirm');
    });

    it('allow is the default when any hook returned allow and none denied or asked', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              decision: 'allow',
              reason: 'ok',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.decision).toBe('allow');
      expect(result.reason).toBe('ok');
    });

    it('no decision is set when no hook returns one', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [noopPreHook],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.decision).toBeUndefined();
    });
  });

  describe('stop decision folding', () => {
    it('any block wins over continue', async () => {
      const registry = new HookRegistry();
      registry.register('Stop', {
        hooks: [
          stopHook(
            async (): Promise<StopHookOutput> => ({ decision: 'continue' })
          ),
        ],
      });
      registry.register('Stop', {
        hooks: [
          stopHook(
            async (): Promise<StopHookOutput> => ({
              decision: 'block',
              reason: 'more work to do',
            })
          ),
        ],
      });
      const result = await executeHooks({ registry, input: stopInput() });
      expect(result.stopDecision).toBe('block');
      expect(result.reason).toBe('more work to do');
    });

    it('continue is the aggregated result when no hook blocks', async () => {
      const registry = new HookRegistry();
      registry.register('Stop', {
        hooks: [
          stopHook(
            async (): Promise<StopHookOutput> => ({ decision: 'continue' })
          ),
        ],
      });
      const result = await executeHooks({ registry, input: stopInput() });
      expect(result.stopDecision).toBe('continue');
    });
  });

  describe('additionalContext accumulation', () => {
    it('accumulates non-empty additionalContext from every hook', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              additionalContext: 'context one',
            })
          ),
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              additionalContext: '',
            })
          ),
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              additionalContext: 'context two',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.additionalContexts.sort()).toEqual([
        'context one',
        'context two',
      ]);
    });
  });

  describe('updatedInput handling', () => {
    it('last-writer-wins on updatedInput follows registration order', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              updatedInput: { cmd: 'first' },
            })
          ),
        ],
      });
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              updatedInput: { cmd: 'second' },
            })
          ),
        ],
      });
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              updatedInput: { cmd: 'third' },
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.updatedInput).toEqual({ cmd: 'third' });
    });

    it('last-writer-wins within a single matcher follows hook array order', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              updatedInput: { cmd: 'inner-first' },
            })
          ),
          preToolHook(
            async (): Promise<PreToolUseHookOutput> => ({
              updatedInput: { cmd: 'inner-second' },
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.updatedInput).toEqual({ cmd: 'inner-second' });
    });
  });

  describe('updatedOutput handling', () => {
    it('flows updatedOutput through the aggregated result', async () => {
      const registry = new HookRegistry();
      registry.register('PostToolUse', {
        hooks: [
          postToolHook(
            async (): Promise<PostToolUseHookOutput> => ({
              updatedOutput: 'redacted',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: postToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.updatedOutput).toBe('redacted');
    });

    it('last-writer-wins on updatedOutput follows registration order', async () => {
      const registry = new HookRegistry();
      registry.register('PostToolUse', {
        hooks: [
          postToolHook(
            async (): Promise<PostToolUseHookOutput> => ({
              updatedOutput: { tag: 'first' },
            })
          ),
        ],
      });
      registry.register('PostToolUse', {
        hooks: [
          postToolHook(
            async (): Promise<PostToolUseHookOutput> => ({
              updatedOutput: { tag: 'second' },
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: postToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.updatedOutput).toEqual({ tag: 'second' });
    });

    it('leaves updatedOutput undefined when no hook sets it', async () => {
      const registry = new HookRegistry();
      registry.register('PostToolUse', {
        hooks: [postToolHook(async (): Promise<PostToolUseHookOutput> => ({}))],
      });
      const result = await executeHooks({
        registry,
        input: postToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.updatedOutput).toBeUndefined();
    });
  });

  describe('preventContinuation', () => {
    it('propagates preventContinuation and stopReason', async () => {
      const registry = new HookRegistry();
      registry.register('PostToolUse', {
        hooks: [
          postToolHook(
            async (): Promise<PostToolUseHookOutput> => ({
              preventContinuation: true,
              stopReason: 'budget exhausted',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: postToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.preventContinuation).toBe(true);
      expect(result.stopReason).toBe('budget exhausted');
    });

    it('keeps the first stopReason when multiple hooks set preventContinuation', async () => {
      const registry = new HookRegistry();
      registry.register('PostToolUse', {
        hooks: [
          postToolHook(
            async (): Promise<PostToolUseHookOutput> => ({
              preventContinuation: true,
              stopReason: 'first writer',
            })
          ),
        ],
      });
      registry.register('PostToolUse', {
        hooks: [
          postToolHook(
            async (): Promise<PostToolUseHookOutput> => ({
              preventContinuation: true,
              stopReason: 'second writer',
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: postToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.preventContinuation).toBe(true);
      expect(result.stopReason).toBe('first writer');
    });

    it('sets preventContinuation even if only the flag, no reason, is present', async () => {
      const registry = new HookRegistry();
      registry.register('PostToolUse', {
        hooks: [
          postToolHook(
            async (): Promise<PostToolUseHookOutput> => ({
              preventContinuation: true,
            })
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: postToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.preventContinuation).toBe(true);
      expect(result.stopReason).toBeUndefined();
    });
  });

  describe('session scoping', () => {
    it('runs session matchers only when sessionId is supplied', async () => {
      const registry = new HookRegistry();
      let globalFired = false;
      let sessionFired = false;
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            globalFired = true;
            return emptyPreOutput;
          }),
        ],
      });
      registry.registerSession('run-1', 'PreToolUse', {
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            sessionFired = true;
            return emptyPreOutput;
          }),
        ],
      });

      await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(globalFired).toBe(true);
      expect(sessionFired).toBe(false);

      globalFired = false;
      await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
        sessionId: 'run-1',
      });
      expect(globalFired).toBe(true);
      expect(sessionFired).toBe(true);
    });
  });

  describe('once: true self-removal', () => {
    it('removes the matcher after a successful fire', async () => {
      const registry = new HookRegistry();
      let calls = 0;
      const matcher: HookMatcher<'RunStart'> = {
        once: true,
        hooks: [
          runStartHook(async (): Promise<RunStartHookOutput> => {
            calls++;
            return emptyRunStartOutput;
          }),
        ],
      };
      registry.register('RunStart', matcher);

      await executeHooks({ registry, input: runStartInput() });
      expect(calls).toBe(1);
      expect(registry.getMatchers('RunStart')).toHaveLength(0);

      await executeHooks({ registry, input: runStartInput() });
      expect(calls).toBe(1);
    });

    it('does not remove the matcher when every hook in it throws', async () => {
      const registry = new HookRegistry();
      const matcher: HookMatcher<'RunStart'> = {
        once: true,
        hooks: [
          runStartHook(async (): Promise<RunStartHookOutput> => {
            throw new Error('hook failed');
          }),
          runStartHook(async (): Promise<RunStartHookOutput> => {
            throw new Error('hook also failed');
          }),
        ],
      };
      registry.register('RunStart', matcher);

      const result = await executeHooks({ registry, input: runStartInput() });
      expect(result.errors).toHaveLength(2);
      expect(registry.getMatchers('RunStart')).toHaveLength(1);
    });

    it('removes the matcher when at least one hook succeeds', async () => {
      const registry = new HookRegistry();
      const matcher: HookMatcher<'RunStart'> = {
        once: true,
        hooks: [
          runStartHook(async (): Promise<RunStartHookOutput> => {
            throw new Error('boom');
          }),
          noopRunStartHook,
        ],
      };
      registry.register('RunStart', matcher);

      const result = await executeHooks({ registry, input: runStartInput() });
      expect(result.errors).toHaveLength(1);
      expect(registry.getMatchers('RunStart')).toHaveLength(0);
    });

    it('removes once-matchers registered for a session from the session scope', async () => {
      const registry = new HookRegistry();
      const matcher: HookMatcher<'PreToolUse'> = {
        once: true,
        hooks: [noopPreHook],
      };
      registry.registerSession('run-1', 'PreToolUse', matcher);
      await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
        sessionId: 'run-1',
      });
      expect(registry.getMatchers('PreToolUse', 'run-1')).toHaveLength(0);
    });

    it('documents the concurrent-dispatch race: once fires per concurrent call, not globally', async () => {
      const registry = new HookRegistry();
      let calls = 0;
      const matcher: HookMatcher<'RunStart'> = {
        once: true,
        hooks: [
          runStartHook(async (): Promise<RunStartHookOutput> => {
            calls++;
            return emptyRunStartOutput;
          }),
        ],
      };
      registry.register('RunStart', matcher);

      await Promise.all([
        executeHooks({ registry, input: runStartInput() }),
        executeHooks({ registry, input: runStartInput() }),
      ]);

      expect(calls).toBe(2);
      expect(registry.getMatchers('RunStart')).toHaveLength(0);
    });
  });

  describe('timeout enforcement', () => {
    it('aborts hooks that exceed the matcher timeout', async () => {
      const registry = new HookRegistry();
      registry.register('RunStart', {
        timeout: 20,
        hooks: [
          runStartHook(
            (_input, signal): Promise<RunStartHookOutput> =>
              new Promise<RunStartHookOutput>((_resolve, reject) => {
                const id = setTimeout((): void => {
                  reject(new Error('hook should have been aborted'));
                }, 500);
                signal.addEventListener('abort', (): void => {
                  clearTimeout(id);
                  reject(new Error('aborted'));
                });
              })
          ),
        ],
      });

      const start = Date.now();
      const result = await executeHooks({ registry, input: runStartInput() });
      const elapsed = Date.now() - start;

      expect(result.errors).toHaveLength(1);
      expect(elapsed).toBeLessThan(400);
    });

    it('times out hooks that ignore the signal and surfaces an abort-shaped error', async () => {
      const registry = new HookRegistry();
      const pendingTimers: NodeJS.Timeout[] = [];
      registry.register('RunStart', {
        timeout: 15,
        hooks: [
          runStartHook(
            (): Promise<RunStartHookOutput> =>
              new Promise<RunStartHookOutput>((resolve): void => {
                const id = setTimeout(
                  (): void => resolve(emptyRunStartOutput),
                  500
                );
                pendingTimers.push(id);
              })
          ),
        ],
      });

      const start = Date.now();
      const result = await executeHooks({ registry, input: runStartInput() });
      const elapsed = Date.now() - start;

      for (const id of pendingTimers) {
        clearTimeout(id);
      }
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0]?.toLowerCase()).toMatch(
        /timeout|timed out|abort/
      );
      expect(elapsed).toBeLessThan(400);
    });

    it('honours the batch timeoutMs default when the matcher does not set its own', async () => {
      const registry = new HookRegistry();
      registry.register('RunStart', {
        hooks: [
          runStartHook(
            (_input, signal): Promise<RunStartHookOutput> =>
              new Promise<RunStartHookOutput>((_resolve, reject) => {
                signal.addEventListener('abort', (): void =>
                  reject(new Error('aborted'))
                );
              })
          ),
        ],
      });

      const start = Date.now();
      const result = await executeHooks({
        registry,
        input: runStartInput(),
        timeoutMs: 25,
      });
      const elapsed = Date.now() - start;

      expect(result.errors).toHaveLength(1);
      expect(elapsed).toBeLessThan(400);
    });
  });

  describe('error non-fatality', () => {
    it('swallows synchronous throws into the errors array and keeps going', async () => {
      const registry = new HookRegistry();
      let otherRan = false;
      registry.register('PreToolUse', {
        hooks: [
          preToolHook((): Promise<PreToolUseHookOutput> => {
            throw new Error('sync boom');
          }),
        ],
      });
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(async (): Promise<PreToolUseHookOutput> => {
            otherRan = true;
            return { additionalContext: 'still ran' };
          }),
        ],
      });

      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(otherRan).toBe(true);
      expect(result.additionalContexts).toEqual(['still ran']);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0]).toContain('sync boom');
    });

    it('swallows async rejections', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> =>
              Promise.reject(new Error('async boom'))
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0]).toContain('async boom');
    });

    it('excludes internal matcher errors from the errors array', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        internal: true,
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> =>
              Promise.reject(new Error('internal failure'))
          ),
        ],
      });
      const result = await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(result.errors).toHaveLength(0);
    });

    it('routes non-internal errors through an optional logger instead of console', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> =>
              Promise.reject(new Error('oops'))
          ),
        ],
      });
      const warnings: string[] = [];
      const fakeLogger = {
        warn: (msg: string): void => {
          warnings.push(msg);
        },
      } as unknown as import('winston').Logger;
      await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
        logger: fakeLogger,
      });
      expect(warnings).toHaveLength(1);
      expect(warnings[0]).toContain('oops');
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('falls back to console.warn when no logger is supplied', async () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', {
        hooks: [
          preToolHook(
            async (): Promise<PreToolUseHookOutput> =>
              Promise.reject(new Error('fallback'))
          ),
        ],
      });
      await executeHooks({
        registry,
        input: preToolUseInput('Bash'),
        matchQuery: 'Bash',
      });
      expect(consoleWarnSpy).toHaveBeenCalledTimes(1);
      const firstCall = consoleWarnSpy.mock.calls[0] as unknown[];
      expect(String(firstCall[0])).toContain('fallback');
    });
  });

  describe('parent AbortSignal combination', () => {
    it('aborts hooks when the caller signal fires', async () => {
      const registry = new HookRegistry();
      registry.register('RunStart', {
        hooks: [
          runStartHook(
            (_input, signal): Promise<RunStartHookOutput> =>
              new Promise<RunStartHookOutput>((_resolve, reject) => {
                signal.addEventListener('abort', (): void =>
                  reject(new Error('aborted'))
                );
              })
          ),
        ],
      });

      const controller = new AbortController();
      setTimeout((): void => controller.abort(), 20);

      const start = Date.now();
      const result = await executeHooks({
        registry,
        input: runStartInput(),
        signal: controller.signal,
        timeoutMs: 5_000,
      });
      const elapsed = Date.now() - start;

      expect(result.errors).toHaveLength(1);
      expect(elapsed).toBeLessThan(400);
    });
  });
});
