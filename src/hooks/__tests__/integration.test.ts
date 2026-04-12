// src/hooks/__tests__/integration.test.ts
import { HumanMessage } from '@langchain/core/messages';
import { HookRegistry } from '../HookRegistry';
import { Run } from '@/run';
import type * as t from '@/types';
import type {
  HookCallback,
  RunStartHookInput,
  RunStartHookOutput,
  UserPromptSubmitHookOutput,
  StopHookInput,
  StopHookOutput,
  StopFailureHookOutput,
} from '../types';
import { Providers } from '@/common';

const llmConfig: t.LLMConfig = {
  provider: Providers.OPENAI,
  streaming: true,
  streamUsage: false,
};

const callerConfig = {
  configurable: { thread_id: 'test-thread' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

function createRun(
  hooks: HookRegistry,
  runId = 'test-run'
): Promise<Run<t.IState>> {
  return Run.create<t.IState>({
    runId,
    graphConfig: { type: 'standard', llmConfig },
    returnContent: true,
    skipCleanup: true,
    hooks,
  });
}

describe('Run-level hook integration', () => {
  jest.setTimeout(15000);

  describe('RunStart', () => {
    it('fires with runId, threadId, and messages before the stream', async () => {
      const registry = new HookRegistry();
      let captured: RunStartHookInput | undefined;
      const hook: HookCallback<'RunStart'> = async (
        input
      ): Promise<RunStartHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('RunStart', { hooks: [hook] });

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['hello']);
      const inputs = { messages: [new HumanMessage('hi')] };
      await run.processStream(inputs, callerConfig);

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('RunStart');
      expect(captured!.runId).toBe('test-run');
      expect(captured!.threadId).toBe('test-thread');
      expect(captured!.messages).toHaveLength(1);
    });
  });

  describe('UserPromptSubmit', () => {
    it('extracts prompt text from the last human message', async () => {
      const registry = new HookRegistry();
      let capturedPrompt = '';
      const hook: HookCallback<'UserPromptSubmit'> = async (
        input
      ): Promise<UserPromptSubmitHookOutput> => {
        capturedPrompt = input.prompt;
        return {};
      };
      registry.register('UserPromptSubmit', { hooks: [hook] });

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['response']);
      const inputs = { messages: [new HumanMessage('hello world')] };
      await run.processStream(inputs, callerConfig);

      expect(capturedPrompt).toBe('hello world');
    });

    it('aborts the run when hook returns deny', async () => {
      const registry = new HookRegistry();
      let stopFired = false;
      const denyHook: HookCallback<
        'UserPromptSubmit'
      > = async (): Promise<UserPromptSubmitHookOutput> => ({
        decision: 'deny',
        reason: 'blocked by policy',
      });
      const stopHook: HookCallback<
        'Stop'
      > = async (): Promise<StopHookOutput> => {
        stopFired = true;
        return {};
      };
      registry.register('UserPromptSubmit', { hooks: [denyHook] });
      registry.register('Stop', { hooks: [stopHook] });

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['should not reach']);
      const inputs = { messages: [new HumanMessage('hi')] };
      const result = await run.processStream(inputs, callerConfig);

      expect(result).toBeUndefined();
      expect(stopFired).toBe(false);
    });

    it('aborts the run when hook returns ask (v1 — no interactive flow)', async () => {
      const registry = new HookRegistry();
      const askHook: HookCallback<
        'UserPromptSubmit'
      > = async (): Promise<UserPromptSubmitHookOutput> => ({
        decision: 'ask',
        reason: 'needs confirmation',
      });
      registry.register('UserPromptSubmit', { hooks: [askHook] });

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['should not reach']);
      const inputs = { messages: [new HumanMessage('hi')] };
      const result = await run.processStream(inputs, callerConfig);

      expect(result).toBeUndefined();
    });
  });

  describe('Stop', () => {
    it('fires after a successful stream with accumulated messages', async () => {
      const registry = new HookRegistry();
      let captured: StopHookInput | undefined;
      const hook: HookCallback<'Stop'> = async (
        input
      ): Promise<StopHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('Stop', { hooks: [hook] });

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['agent reply']);
      const inputs = { messages: [new HumanMessage('hi')] };
      await run.processStream(inputs, callerConfig);

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('Stop');
      expect(captured!.runId).toBe('test-run');
      expect(captured!.stopHookActive).toBe(false);
      expect(captured!.messages.length).toBeGreaterThanOrEqual(1);
    });

    it('does not fire when the stream throws an error', async () => {
      const registry = new HookRegistry();
      let stopFired = false;
      const hook: HookCallback<'Stop'> = async (): Promise<StopHookOutput> => {
        stopFired = true;
        return {};
      };
      registry.register('Stop', { hooks: [hook] });

      const run = await createRun(registry, 'error-run');
      run.Graph!.overrideTestModel([]);

      const inputs = { messages: [new HumanMessage('hi')] };
      try {
        await run.processStream(inputs, callerConfig);
      } catch {
        /* expected */
      }

      expect(stopFired).toBe(false);
    });
  });

  describe('StopFailure', () => {
    it('fires when the stream throws and preserves the original error', async () => {
      const registry = new HookRegistry();
      let capturedError = '';
      const hook: HookCallback<'StopFailure'> = async (
        input
      ): Promise<StopFailureHookOutput> => {
        capturedError = input.error;
        return {};
      };
      registry.register('StopFailure', { hooks: [hook] });

      const run = await createRun(registry, 'fail-run');
      run.Graph!.overrideTestModel([]);

      const inputs = { messages: [new HumanMessage('hi')] };
      let thrownError: Error | undefined;
      try {
        await run.processStream(inputs, callerConfig);
      } catch (err) {
        thrownError = err instanceof Error ? err : new Error(String(err));
      }

      expect(thrownError).toBeDefined();
      expect(capturedError).toBeTruthy();
    });
  });

  describe('session teardown', () => {
    it('clears session matchers after processStream completes', async () => {
      const registry = new HookRegistry();
      registry.registerSession('test-run', 'RunStart', {
        hooks: [async (): Promise<RunStartHookOutput> => ({})],
      });
      expect(registry.getMatchers('RunStart', 'test-run')).toHaveLength(1);

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['done']);
      const inputs = { messages: [new HumanMessage('hi')] };
      await run.processStream(inputs, callerConfig);

      expect(registry.getMatchers('RunStart', 'test-run')).toHaveLength(0);
    });

    it('clears session even when the stream errors', async () => {
      const registry = new HookRegistry();
      registry.registerSession('error-run', 'RunStart', {
        hooks: [async (): Promise<RunStartHookOutput> => ({})],
      });

      const run = await createRun(registry, 'error-run');
      run.Graph!.overrideTestModel([]);

      const inputs = { messages: [new HumanMessage('hi')] };
      try {
        await run.processStream(inputs, callerConfig);
      } catch {
        /* expected */
      }

      expect(registry.getMatchers('RunStart', 'error-run')).toHaveLength(0);
    });
  });

  describe('no-hooks baseline', () => {
    it('works identically when no hooks registry is provided', async () => {
      const run = await Run.create<t.IState>({
        runId: 'no-hooks-run',
        graphConfig: { type: 'standard', llmConfig },
        returnContent: true,
        skipCleanup: true,
      });
      run.Graph!.overrideTestModel(['response']);
      const inputs = { messages: [new HumanMessage('hi')] };
      const result = await run.processStream(inputs, callerConfig);

      expect(result).toBeDefined();
      expect(result!.length).toBeGreaterThan(0);
    });
  });
});
