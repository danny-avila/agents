// src/hooks/__tests__/integration.test.ts
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import type {
  HookCallback,
  RunStartHookInput,
  RunStartHookOutput,
  UserPromptSubmitHookOutput,
  StopHookInput,
  StopFinalizeHookInput,
  StopHookOutput,
  StopFailureHookOutput,
} from '../types';
import type * as t from '@/types';
import { Providers, StepTypes } from '@/common';
import { HookRegistry } from '../HookRegistry';
import { Run } from '@/run';

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

    it('extracts prompt from multi-part content (text + non-text blocks)', async () => {
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
      run.Graph!.overrideTestModel(['ok']);
      const msg = new HumanMessage({
        content: [
          { type: 'text', text: 'hello' },
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,...' },
          },
          { type: 'text', text: 'world' },
        ],
      });
      await run.processStream({ messages: [msg] }, callerConfig);

      expect(capturedPrompt).toBe('hello\nworld');
    });

    it('yields empty prompt for image-only content', async () => {
      const registry = new HookRegistry();
      let capturedPrompt: string | undefined;
      const hook: HookCallback<'UserPromptSubmit'> = async (
        input
      ): Promise<UserPromptSubmitHookOutput> => {
        capturedPrompt = input.prompt;
        return {};
      };
      registry.register('UserPromptSubmit', { hooks: [hook] });

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['ok']);
      const msg = new HumanMessage({
        content: [
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,...' },
          },
        ],
      });
      await run.processStream({ messages: [msg] }, callerConfig);

      expect(capturedPrompt).toBe('');
    });

    it('fires with empty prompt when human message has no text blocks', async () => {
      const registry = new HookRegistry();
      let capturedPrompt: string | undefined;
      const hook: HookCallback<'UserPromptSubmit'> = async (
        input
      ): Promise<UserPromptSubmitHookOutput> => {
        capturedPrompt = input.prompt;
        return {};
      };
      registry.register('UserPromptSubmit', { hooks: [hook] });

      const run = await createRun(registry);
      run.Graph!.overrideTestModel(['ok']);
      const msg = new HumanMessage({ content: [] });
      await run.processStream({ messages: [msg] }, callerConfig);

      expect(capturedPrompt).toBe('');
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
      expect(captured!.continuationCount).toBe(0);
      expect(captured!.continuationBudgetRemaining).toBe(8);
      expect(captured!.messages.length).toBeGreaterThanOrEqual(1);
    });

    it('keeps the same Run warm when Stop blocks with injected messages', async () => {
      const registry = new HookRegistry();
      const captured: StopHookInput[] = [];
      let runStarts = 0;
      let promptSubmissions = 0;
      registry.register('RunStart', {
        hooks: [
          async (): Promise<RunStartHookOutput> => {
            runStarts += 1;
            return {};
          },
        ],
      });
      registry.register('UserPromptSubmit', {
        hooks: [
          async (): Promise<UserPromptSubmitHookOutput> => {
            promptSubmissions += 1;
            return {};
          },
        ],
      });
      registry.register('Stop', {
        hooks: [
          async (input): Promise<StopHookOutput> => {
            captured.push(input);
            if (input.continuationCount > 0) {
              return { decision: 'continue' };
            }
            return {
              decision: 'block',
              injectedMessages: [
                { role: 'user', content: 'late steer', source: 'steer' },
              ],
            };
          },
        ],
      });

      const run = await createRun(registry, 'warm-run');
      run.Graph!.overrideTestModel(['first answer', 'continued answer']);
      await run.processStream(
        { messages: [new HumanMessage('initial prompt')] },
        callerConfig
      );

      expect(captured).toHaveLength(2);
      expect(captured.map((input) => input.stopHookActive)).toEqual([
        false,
        true,
      ]);
      expect(captured.map((input) => input.continuationCount)).toEqual([0, 1]);
      expect(
        captured.map((input) => input.continuationBudgetRemaining)
      ).toEqual([8, 7]);
      expect(runStarts).toBe(1);
      expect(promptSubmissions).toBe(1);
      expect(run.Graph!.messages.map((message) => message.content)).toEqual([
        'initial prompt',
        'first answer',
        'late steer',
        'continued answer',
      ]);
      expect(run.Graph!.messages[2].additional_kwargs).toMatchObject({
        role: 'user',
        source: 'steer',
      });
      const messageSteps = run.Graph!.contentData.filter(
        (step) => step.type === StepTypes.MESSAGE_CREATION
      );
      expect(messageSteps).toHaveLength(2);
      expect(new Set(messageSteps.map((step) => step.id)).size).toBe(2);
    });

    it('serializes StopFinalize after ordinary Stop decisions are folded', async () => {
      const registry = new HookRegistry();
      const finalized: StopFinalizeHookInput[] = [];
      registry.register('Stop', {
        hooks: [
          async (input): Promise<StopHookOutput> =>
            input.continuationCount === 0
              ? {
                decision: 'block',
                injectedMessages: [
                  { role: 'user', content: 'plugin continuation' },
                ],
              }
              : { decision: 'continue' },
        ],
      });
      registry.register('StopFinalize', {
        hooks: [
          async (input): Promise<StopHookOutput> => {
            finalized.push(input);
            return { decision: 'continue' };
          },
        ],
      });

      const run = await createRun(registry, 'finalized-warm-run');
      run.Graph!.overrideTestModel(['first answer', 'continued answer']);
      await run.processStream(
        { messages: [new HumanMessage('initial prompt')] },
        callerConfig
      );

      expect(finalized).toHaveLength(2);
      expect(finalized.map((input) => input.continuationPlanned)).toEqual([
        true,
        false,
      ]);
      expect(finalized.map((input) => input.continuationPrevented)).toEqual([
        false,
        false,
      ]);
      expect(run.Graph!.messages.map((message) => message.content)).toEqual([
        'initial prompt',
        'first answer',
        'plugin continuation',
        'continued answer',
      ]);
    });

    it('fails the run when final continuation admission is indeterminate', async () => {
      const registry = new HookRegistry();
      registry.register('StopFinalize', {
        hooks: [
          async (): Promise<StopHookOutput> => {
            throw new Error('claim response unavailable');
          },
        ],
      });

      const run = await createRun(registry, 'failed-final-admission');
      run.Graph!.overrideTestModel(['first answer']);

      await expect(
        run.processStream(
          { messages: [new HumanMessage('initial prompt')] },
          callerConfig
        )
      ).rejects.toThrow(
        'StopFinalize terminal admission failed: claim response unavailable'
      );
    });

    it('does not self-loop when Stop blocks without injectable content', async () => {
      const registry = new HookRegistry();
      let calls = 0;
      registry.register('Stop', {
        hooks: [
          async (): Promise<StopHookOutput> => {
            calls += 1;
            return { decision: 'block' };
          },
        ],
      });

      const run = await createRun(registry, 'empty-warm-run');
      run.Graph!.overrideTestModel(['only answer']);
      await run.processStream(
        { messages: [new HumanMessage('initial prompt')] },
        callerConfig
      );

      expect(calls).toBe(1);
      expect(run.Graph!.messages.map((message) => message.content)).toEqual([
        'initial prompt',
        'only answer',
      ]);
    });

    it('submits only the injected delta when a checkpointer owns prior state', async () => {
      const registry = new HookRegistry();
      registry.register('Stop', {
        hooks: [
          async (input): Promise<StopHookOutput> =>
            input.continuationCount === 0
              ? {
                decision: 'block',
                injectedMessages: [
                  {
                    role: 'user',
                    content: 'checkpoint steer',
                    source: 'steer',
                  },
                ],
              }
              : { decision: 'continue' },
        ],
      });
      const run = await Run.create<t.IState>({
        runId: 'checkpoint-warm-run',
        graphConfig: {
          type: 'standard',
          llmConfig,
          compileOptions: { checkpointer: new MemorySaver() },
        },
        returnContent: true,
        skipCleanup: true,
        hooks: registry,
      });
      run.Graph!.overrideTestModel(['first answer', 'continued answer']);

      await run.processStream(
        { messages: [new HumanMessage('initial prompt')] },
        callerConfig
      );

      expect(run.Graph!.messages.map((message) => message.content)).toEqual([
        'initial prompt',
        'first answer',
        'checkpoint steer',
        'continued answer',
      ]);
    });

    it('advances an explicitly pinned checkpoint before the warm segment', async () => {
      const checkpointer = new MemorySaver();
      const threadConfig = {
        ...callerConfig,
        configurable: { thread_id: 'pinned-warm-thread' },
      };
      const seedRun = await Run.create<t.IState>({
        runId: 'pinned-warm-seed',
        graphConfig: {
          type: 'standard',
          llmConfig,
          compileOptions: { checkpointer },
        },
        returnContent: true,
        skipCleanup: true,
      });
      seedRun.Graph!.overrideTestModel(['seed answer']);
      await seedRun.processStream(
        { messages: [new HumanMessage('seed prompt')] },
        threadConfig
      );
      const seedTuple = await checkpointer.getTuple(threadConfig);
      const checkpointId = seedTuple?.config.configurable?.checkpoint_id;
      expect(typeof checkpointId).toBe('string');

      const registry = new HookRegistry();
      let siblingCompleted = false;
      registry.register('Stop', {
        hooks: [
          async (input): Promise<StopHookOutput> => {
            if (input.continuationCount > 0) {
              return { decision: 'continue' };
            }
            const siblingRun = await Run.create<t.IState>({
              runId: 'pinned-warm-sibling',
              graphConfig: {
                type: 'standard',
                llmConfig,
                compileOptions: { checkpointer },
              },
              returnContent: true,
              skipCleanup: true,
            });
            siblingRun.Graph!.overrideTestModel(['sibling answer']);
            await siblingRun.processStream(
              { messages: [new HumanMessage('sibling prompt')] },
              threadConfig
            );
            siblingCompleted = true;
            return {
              decision: 'block',
              injectedMessages: [
                { role: 'user', content: 'branch steer', source: 'steer' },
              ],
            };
          },
        ],
      });
      const branchRun = await Run.create<t.IState>({
        runId: 'pinned-warm-branch',
        graphConfig: {
          type: 'standard',
          llmConfig,
          compileOptions: { checkpointer },
        },
        returnContent: true,
        skipCleanup: true,
        hooks: registry,
      });
      branchRun.Graph!.overrideTestModel([
        'branch answer',
        'continued branch answer',
      ]);
      await branchRun.processStream(
        { messages: [new HumanMessage('branch prompt')] },
        {
          ...threadConfig,
          configurable: {
            ...threadConfig.configurable,
            checkpoint_id: checkpointId,
          },
        }
      );

      expect(siblingCompleted).toBe(true);
      expect(
        branchRun.Graph!.messages.map((message) => message.content)
      ).toEqual([
        'seed prompt',
        'seed answer',
        'branch prompt',
        'branch answer',
        'branch steer',
        'continued branch answer',
      ]);
    });

    it('resets persisted continuation admission for the next fresh turn', async () => {
      const checkpointer = new MemorySaver();
      const counts: number[] = [];
      let admitted = false;
      const registry = new HookRegistry();
      registry.register('Stop', {
        hooks: [
          async (input): Promise<StopHookOutput> => {
            counts.push(input.continuationCount);
            if (admitted) {
              return { decision: 'continue' };
            }
            admitted = true;
            return {
              decision: 'block',
              injectedMessages: [
                { role: 'user', content: 'first-turn steer', source: 'steer' },
              ],
            };
          },
        ],
      });
      const run = await Run.create<t.IState>({
        runId: 'fresh-turn-continuation-reset',
        graphConfig: {
          type: 'standard',
          llmConfig,
          compileOptions: { checkpointer },
        },
        returnContent: true,
        skipCleanup: true,
        hooks: registry,
        maxStopContinuations: 1,
      });
      const config = {
        ...callerConfig,
        configurable: { thread_id: 'fresh-turn-continuation-reset-thread' },
      };
      run.Graph!.overrideTestModel(['first answer', 'continued answer']);
      await run.processStream(
        { messages: [new HumanMessage('first prompt')] },
        config
      );

      run.Graph!.overrideTestModel(['next answer']);
      await run.processStream(
        { messages: [new HumanMessage('next prompt')] },
        config
      );

      expect(counts).toEqual([0, 1, 0]);
    });

    it('reports a zero budget and refuses continuation past the configured cap', async () => {
      const registry = new HookRegistry();
      const remaining: number[] = [];
      registry.register('Stop', {
        hooks: [
          async (input): Promise<StopHookOutput> => {
            remaining.push(input.continuationBudgetRemaining);
            if (input.continuationBudgetRemaining === 0) {
              return { decision: 'continue' };
            }
            return {
              decision: 'block',
              injectedMessages: [
                {
                  role: 'user',
                  content: `steer ${input.continuationCount + 1}`,
                  source: 'steer',
                },
              ],
            };
          },
        ],
      });
      const run = await Run.create<t.IState>({
        runId: 'capped-warm-run',
        graphConfig: { type: 'standard', llmConfig },
        returnContent: true,
        skipCleanup: true,
        hooks: registry,
        maxStopContinuations: 1,
      });
      run.Graph!.overrideTestModel(['first answer', 'second answer']);

      await run.processStream(
        { messages: [new HumanMessage('initial prompt')] },
        callerConfig
      );

      expect(remaining).toEqual([1, 0]);
      expect(run.Graph!.messages.map((message) => message.content)).toEqual([
        'initial prompt',
        'first answer',
        'steer 1',
        'second answer',
      ]);
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
      expect(typeof capturedError).toBe('string');
      expect(capturedError.length).toBeGreaterThan(0);
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
