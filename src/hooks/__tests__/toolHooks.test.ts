// src/hooks/__tests__/toolHooks.test.ts
import { ToolCall } from '@langchain/core/messages/tool';
import { HumanMessage } from '@langchain/core/messages';
import { HookRegistry } from '../HookRegistry';
import { Run } from '@/run';
import {
  GraphEvents,
  Providers,
  ToolEndHandler,
  ModelEndHandler,
} from '@/index';
import type * as t from '@/types';
import type {
  HookCallback,
  PreToolUseHookOutput,
  PostToolUseHookOutput,
  PostToolUseFailureHookOutput,
  PermissionDeniedHookInput,
  PermissionDeniedHookOutput,
  PreToolUseHookInput,
  PostToolUseHookInput,
  PostToolUseFailureHookInput,
} from '../types';

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

const echoToolDef: t.LCTool = {
  name: 'echo',
  description: 'Echoes input',
  parameters: {
    type: 'object' as const,
    properties: { text: { type: 'string' } },
    required: ['text'],
  },
};

let callCounter = 0;

function makeToolCall(text = 'hello', name = 'echo'): ToolCall {
  return {
    name,
    args: { text },
    id: `call_${++callCounter}`,
    type: 'tool_call',
  };
}

function createToolExecuteHandler(): t.EventHandler {
  return {
    handle: async (_event: string, rawData: unknown): Promise<void> => {
      const data = rawData as t.ToolExecuteBatchRequest;
      const results: t.ToolExecuteResult[] = data.toolCalls.map(
        (tc: t.ToolCallRequest) => ({
          toolCallId: tc.id,
          content: `echo: ${(tc.args as Record<string, string>).text}`,
          status: 'success' as const,
        })
      );
      data.resolve(results);
    },
  };
}

function createErrorToolExecuteHandler(): t.EventHandler {
  return {
    handle: async (_event: string, rawData: unknown): Promise<void> => {
      const data = rawData as t.ToolExecuteBatchRequest;
      const results: t.ToolExecuteResult[] = data.toolCalls.map(
        (tc: t.ToolCallRequest) => ({
          toolCallId: tc.id,
          content: '',
          status: 'error' as const,
          errorMessage: `tool ${tc.name} failed deliberately`,
        })
      );
      data.resolve(results);
    },
  };
}

async function createEventDrivenRun(
  hooks: HookRegistry,
  toolHandler: t.EventHandler = createToolExecuteHandler(),
  runId = 'tool-hook-run'
): Promise<Run<t.IState>> {
  const customHandlers: Record<string, t.EventHandler> = {
    [GraphEvents.ON_TOOL_EXECUTE]: toolHandler,
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
  };

  return Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      llmConfig,
      toolDefinitions: [echoToolDef],
      instructions: 'Use the echo tool when asked.',
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers,
    hooks,
  });
}

describe('Tool-level hook integration (event-driven mode)', () => {
  beforeEach(() => {
    callCounter = 0;
  });
  jest.setTimeout(15000);

  describe('PreToolUse', () => {
    it('fires with toolName, toolInput, and toolUseId', async () => {
      const registry = new HookRegistry();
      let captured: PreToolUseHookInput | undefined;
      const hook: HookCallback<'PreToolUse'> = async (
        input
      ): Promise<PreToolUseHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PreToolUse', { hooks: [hook] });

      const tc = makeToolCall('world');
      const run = await createEventDrivenRun(registry);
      run.Graph!.overrideTestModel(['calling echo'], 5, [tc]);
      await run.processStream(
        { messages: [new HumanMessage('echo world')] },
        callerConfig
      );

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PreToolUse');
      expect(captured!.toolName).toBe('echo');
      expect(captured!.toolInput).toEqual({ text: 'world' });
      expect(captured!.toolUseId).toBe(tc.id);
    });

    it('deny blocks tool execution and produces error ToolMessage', async () => {
      const registry = new HookRegistry();
      let toolExecuted = false;
      const denyHook: HookCallback<
        'PreToolUse'
      > = async (): Promise<PreToolUseHookOutput> => ({
        decision: 'deny',
        reason: 'not allowed',
      });
      registry.register('PreToolUse', {
        pattern: '^echo$',
        hooks: [denyHook],
      });

      const spyHandler: t.EventHandler = {
        handle: async (_event: string, rawData: unknown): Promise<void> => {
          const data = rawData as t.ToolExecuteBatchRequest;
          toolExecuted = true;
          data.resolve(
            data.toolCalls.map((tc: t.ToolCallRequest) => ({
              toolCallId: tc.id,
              content: 'should not reach',
              status: 'success' as const,
            }))
          );
        },
      };

      const run = await createEventDrivenRun(registry, spyHandler);
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall()]);
      await run.processStream(
        { messages: [new HumanMessage('echo hello')] },
        callerConfig
      );

      expect(toolExecuted).toBe(false);
    });

    it('ask blocks tool execution in v1 (same as deny)', async () => {
      const registry = new HookRegistry();
      let toolExecuted = false;
      const askHook: HookCallback<
        'PreToolUse'
      > = async (): Promise<PreToolUseHookOutput> => ({
        decision: 'ask',
        reason: 'needs confirmation',
      });
      registry.register('PreToolUse', { hooks: [askHook] });

      const spyHandler: t.EventHandler = {
        handle: async (_event: string, rawData: unknown): Promise<void> => {
          const data = rawData as t.ToolExecuteBatchRequest;
          toolExecuted = true;
          data.resolve(
            data.toolCalls.map((tc: t.ToolCallRequest) => ({
              toolCallId: tc.id,
              content: 'x',
              status: 'success' as const,
            }))
          );
        },
      };

      const run = await createEventDrivenRun(registry, spyHandler);
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall()]);
      await run.processStream(
        { messages: [new HumanMessage('echo hello')] },
        callerConfig
      );

      expect(toolExecuted).toBe(false);
    });

    it('updatedInput rewrites tool args before dispatch', async () => {
      const registry = new HookRegistry();
      let receivedArgs: Record<string, unknown> | undefined;
      const rewriteHook: HookCallback<
        'PreToolUse'
      > = async (): Promise<PreToolUseHookOutput> => ({
        updatedInput: { text: 'sanitized' },
      });
      registry.register('PreToolUse', { hooks: [rewriteHook] });

      const captureHandler: t.EventHandler = {
        handle: async (_event: string, rawData: unknown): Promise<void> => {
          const data = rawData as t.ToolExecuteBatchRequest;
          receivedArgs = data.toolCalls[0]?.args;
          data.resolve(
            data.toolCalls.map((tc: t.ToolCallRequest) => ({
              toolCallId: tc.id,
              content: `echo: ${(tc.args as Record<string, string>).text}`,
              status: 'success' as const,
            }))
          );
        },
      };

      const run = await createEventDrivenRun(registry, captureHandler);
      run.Graph!.overrideTestModel(['calling echo'], 5, [
        makeToolCall('dangerous'),
      ]);
      await run.processStream(
        { messages: [new HumanMessage('echo')] },
        callerConfig
      );

      expect(receivedArgs).toEqual({ text: 'sanitized' });
    });

    it('hook errors are non-fatal — tool still executes', async () => {
      const registry = new HookRegistry();
      let toolExecuted = false;
      const throwingHook: HookCallback<
        'PreToolUse'
      > = async (): Promise<PreToolUseHookOutput> => {
        throw new Error('hook crash');
      };
      registry.register('PreToolUse', { hooks: [throwingHook] });

      const spyHandler: t.EventHandler = {
        handle: async (_event: string, rawData: unknown): Promise<void> => {
          const data = rawData as t.ToolExecuteBatchRequest;
          toolExecuted = true;
          data.resolve(
            data.toolCalls.map((tc: t.ToolCallRequest) => ({
              toolCallId: tc.id,
              content: 'ok',
              status: 'success' as const,
            }))
          );
        },
      };

      const run = await createEventDrivenRun(registry, spyHandler);
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall()]);
      await run.processStream(
        { messages: [new HumanMessage('echo')] },
        callerConfig
      );

      expect(toolExecuted).toBe(true);
    });
  });

  describe('PermissionDenied', () => {
    it('fires after PreToolUse deny with the reason', async () => {
      const registry = new HookRegistry();
      let pdResolve: () => void;
      const pdDone = new Promise<void>((r) => {
        pdResolve = r;
      });
      let captured: PermissionDeniedHookInput | undefined;
      const denyHook: HookCallback<
        'PreToolUse'
      > = async (): Promise<PreToolUseHookOutput> => ({
        decision: 'deny',
        reason: 'security policy',
      });
      const pdHook: HookCallback<'PermissionDenied'> = async (
        input
      ): Promise<PermissionDeniedHookOutput> => {
        captured = input;
        pdResolve();
        return {};
      };
      registry.register('PreToolUse', { hooks: [denyHook] });
      registry.register('PermissionDenied', { hooks: [pdHook] });

      const run = await createEventDrivenRun(registry);
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall()]);
      await run.processStream(
        { messages: [new HumanMessage('echo')] },
        callerConfig
      );

      await pdDone;
      expect(captured).toBeDefined();
      expect(captured!.reason).toBe('security policy');
      expect(captured!.toolName).toBe('echo');
    });
  });

  describe('PostToolUse', () => {
    it('fires after successful tool execution with output', async () => {
      const registry = new HookRegistry();
      let captured: PostToolUseHookInput | undefined;
      const hook: HookCallback<'PostToolUse'> = async (
        input
      ): Promise<PostToolUseHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PostToolUse', { hooks: [hook] });

      const run = await createEventDrivenRun(registry);
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall('hi')]);
      await run.processStream(
        { messages: [new HumanMessage('echo hi')] },
        callerConfig
      );

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PostToolUse');
      expect(captured!.toolName).toBe('echo');
      expect(captured!.toolOutput).toBe('echo: hi');
    });

    it('updatedOutput replaces the ToolMessage content', async () => {
      const registry = new HookRegistry();
      const replaceHook: HookCallback<
        'PostToolUse'
      > = async (): Promise<PostToolUseHookOutput> => ({
        updatedOutput: 'REDACTED',
      });
      registry.register('PostToolUse', { hooks: [replaceHook] });

      let resolvedContent: string | undefined;
      const captureHandler: t.EventHandler = {
        handle: async (_event: string, rawData: unknown): Promise<void> => {
          const data = rawData as t.ToolExecuteBatchRequest;
          const results = data.toolCalls.map(
            (tc: t.ToolCallRequest): t.ToolExecuteResult => ({
              toolCallId: tc.id,
              content: 'original secret output',
              status: 'success' as const,
            })
          );
          data.resolve(results);
        },
      };

      const run = await createEventDrivenRun(registry, captureHandler);
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall()]);
      await run.processStream(
        { messages: [new HumanMessage('echo')] },
        callerConfig
      );

      const messages = run.Graph!.getRunMessages() ?? [];
      const toolMsg = messages.find((m) => m.getType() === 'tool');
      expect(toolMsg).toBeDefined();
      if (toolMsg != null) {
        resolvedContent =
          typeof toolMsg.content === 'string'
            ? toolMsg.content
            : JSON.stringify(toolMsg.content);
      }

      expect(resolvedContent).toBe('REDACTED');
    });
  });

  describe('PostToolUseFailure', () => {
    it('fires when tool execution returns an error', async () => {
      const registry = new HookRegistry();
      let captured: PostToolUseFailureHookInput | undefined;
      const hook: HookCallback<'PostToolUseFailure'> = async (
        input
      ): Promise<PostToolUseFailureHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PostToolUseFailure', { hooks: [hook] });

      const run = await createEventDrivenRun(
        registry,
        createErrorToolExecuteHandler()
      );
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall()]);
      await run.processStream(
        { messages: [new HumanMessage('echo')] },
        callerConfig
      );

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PostToolUseFailure');
      expect(captured!.toolName).toBe('echo');
      expect(captured!.error).toContain('failed deliberately');
    });
  });

  describe('multi-call batch', () => {
    const mathToolDef: t.LCTool = {
      name: 'math',
      description: 'Does math',
      parameters: {
        type: 'object' as const,
        properties: { expr: { type: 'string' } },
        required: ['expr'],
      },
    };

    function createMultiToolRun(
      hooks: HookRegistry,
      runId = 'multi-run'
    ): Promise<Run<t.IState>> {
      const handler: t.EventHandler = {
        handle: async (_event: string, rawData: unknown): Promise<void> => {
          const data = rawData as t.ToolExecuteBatchRequest;
          data.resolve(
            data.toolCalls.map(
              (tc: t.ToolCallRequest): t.ToolExecuteResult => ({
                toolCallId: tc.id,
                content: `${tc.name}: ok`,
                status: 'success' as const,
              })
            )
          );
        },
      };
      return Run.create<t.IState>({
        runId,
        graphConfig: {
          type: 'standard',
          llmConfig,
          toolDefinitions: [echoToolDef, mathToolDef],
          instructions: 'Use tools.',
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers: {
          [GraphEvents.ON_TOOL_EXECUTE]: handler,
          [GraphEvents.TOOL_END]: new ToolEndHandler(),
          [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        },
        hooks,
      });
    }

    it('partial deny: denied call produces error, approved call executes, order preserved', async () => {
      const registry = new HookRegistry();
      const denyEcho: HookCallback<'PreToolUse'> = async (
        input
      ): Promise<PreToolUseHookOutput> =>
        input.toolName === 'echo'
          ? { decision: 'deny', reason: 'echo blocked' }
          : {};
      registry.register('PreToolUse', { hooks: [denyEcho] });

      const echoCall = makeToolCall('hi', 'echo');
      const mathCall = makeToolCall('1+1', 'math');
      const run = await createMultiToolRun(registry);
      run.Graph!.overrideTestModel(['calling tools'], 5, [echoCall, mathCall]);
      await run.processStream(
        { messages: [new HumanMessage('do both')] },
        callerConfig
      );

      const messages = run.Graph!.getRunMessages() ?? [];
      const toolMsgs = messages.filter((m) => m.getType() === 'tool');

      expect(toolMsgs).toHaveLength(2);
      const first = toolMsgs[0];
      const second = toolMsgs[1];
      expect(first.content).toContain('Blocked');
      expect(second.content).toContain('math: ok');
    });

    it('all denied: no ON_TOOL_EXECUTE dispatch, all error messages', async () => {
      const registry = new HookRegistry();
      let handlerCalled = false;
      const denyAll: HookCallback<
        'PreToolUse'
      > = async (): Promise<PreToolUseHookOutput> => ({
        decision: 'deny',
        reason: 'all blocked',
      });
      registry.register('PreToolUse', { hooks: [denyAll] });

      const handler: t.EventHandler = {
        handle: async (_event: string, rawData: unknown): Promise<void> => {
          handlerCalled = true;
          const data = rawData as t.ToolExecuteBatchRequest;
          data.resolve([]);
        },
      };

      const run = await Run.create<t.IState>({
        runId: 'all-denied-run',
        graphConfig: {
          type: 'standard',
          llmConfig,
          toolDefinitions: [echoToolDef, mathToolDef],
          instructions: 'Use tools.',
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers: {
          [GraphEvents.ON_TOOL_EXECUTE]: handler,
          [GraphEvents.TOOL_END]: new ToolEndHandler(),
          [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        },
        hooks: registry,
      });
      run.Graph!.overrideTestModel(['calling tools'], 5, [
        makeToolCall('a', 'echo'),
        makeToolCall('b', 'math'),
      ]);
      await run.processStream(
        { messages: [new HumanMessage('do both')] },
        callerConfig
      );

      expect(handlerCalled).toBe(false);
    });
  });

  describe('PostToolUse error resilience', () => {
    it('PostToolUse hook errors are non-fatal — original output preserved', async () => {
      const registry = new HookRegistry();
      const throwingHook: HookCallback<
        'PostToolUse'
      > = async (): Promise<PostToolUseHookOutput> => {
        throw new Error('post hook crash');
      };
      registry.register('PostToolUse', { hooks: [throwingHook] });

      const run = await createEventDrivenRun(registry);
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall('hi')]);
      await run.processStream(
        { messages: [new HumanMessage('echo hi')] },
        callerConfig
      );

      const messages = run.Graph!.getRunMessages() ?? [];
      const toolMsg = messages.find((m) => m.getType() === 'tool');
      expect(toolMsg).toBeDefined();
      const content =
        typeof toolMsg!.content === 'string'
          ? toolMsg!.content
          : JSON.stringify(toolMsg!.content);
      expect(content).toContain('echo: hi');
    });
  });

  describe('no-hooks baseline', () => {
    it('event-driven tool execution works identically without hooks', async () => {
      const run = await Run.create<t.IState>({
        runId: 'no-hooks-tool-run',
        graphConfig: {
          type: 'standard',
          llmConfig,
          toolDefinitions: [echoToolDef],
          instructions: 'Use echo.',
        },
        returnContent: true,
        skipCleanup: true,
        customHandlers: {
          [GraphEvents.ON_TOOL_EXECUTE]: createToolExecuteHandler(),
          [GraphEvents.TOOL_END]: new ToolEndHandler(),
          [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
        },
      });
      run.Graph!.overrideTestModel(['calling echo'], 5, [makeToolCall('test')]);
      const result = await run.processStream(
        { messages: [new HumanMessage('echo test')] },
        callerConfig
      );

      expect(result).toBeDefined();
    });
  });
});
