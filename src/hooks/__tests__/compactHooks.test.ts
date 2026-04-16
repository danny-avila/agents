// src/hooks/__tests__/compactHooks.test.ts
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { FakeListChatModel } from '@langchain/core/utils/testing';
import { HookRegistry } from '../HookRegistry';
import { Run } from '@/run';
import * as providers from '@/llm/providers';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { createTokenCounter } from '@/utils/tokens';
import { Providers, GraphEvents } from '@/common';
import type * as t from '@/types';
import type {
  HookCallback,
  PreCompactHookInput,
  PreCompactHookOutput,
  PostCompactHookInput,
  PostCompactHookOutput,
} from '../types';

const SUMMARY_RESPONSE = '## Summary\nUser asked a question and got an answer.';

const callerConfig = {
  configurable: { thread_id: 'compact-test' },
  streamMode: 'values' as const,
  version: 'v2' as const,
};

let getChatModelClassSpy: jest.SpyInstance;
const originalGetChatModelClass = providers.getChatModelClass;

function mockSummarizationModel(): void {
  getChatModelClassSpy = jest
    .spyOn(providers, 'getChatModelClass')
    .mockImplementation(((provider: Providers) => {
      if (provider === Providers.OPENAI) {
        return class extends FakeListChatModel {
          constructor(
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            _options: any
          ) {
            super({ responses: [SUMMARY_RESPONSE] });
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any;
      }
      return originalGetChatModelClass(provider);
    }) as typeof providers.getChatModelClass);
}

function buildConversation(): t.IState {
  const messages: import('@langchain/core/messages').BaseMessage[] = [];
  for (let i = 0; i < 20; i++) {
    messages.push(new HumanMessage(`Question ${i}: ` + 'padding '.repeat(50)));
    messages.push(new AIMessage(`Answer ${i}: ` + 'response '.repeat(50)));
  }
  return { messages };
}

async function createCompactingRun(
  tokenCounter: t.TokenCounter,
  hooks?: HookRegistry,
  runId = 'compact-run'
): Promise<Run<t.IState>> {
  const conversation = buildConversation();
  const indexTokenCountMap: Record<string, number> = {};
  for (let i = 0; i < conversation.messages.length; i++) {
    indexTokenCountMap[String(i)] = tokenCounter(conversation.messages[i]);
  }
  return Run.create<t.IState>({
    runId,
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENAI,
        streaming: true,
        streamUsage: false,
      },
      instructions: 'Be concise.',
      maxContextTokens: 200,
      summarizationEnabled: true,
      summarizationConfig: {
        provider: Providers.OPENAI,
      },
    },
    returnContent: true,
    skipCleanup: true,
    customHandlers: {
      [GraphEvents.TOOL_END]: new ToolEndHandler(),
      [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    },
    hooks,
    tokenCounter,
    indexTokenCountMap,
  });
}

describe('Compaction hook integration', () => {
  jest.setTimeout(30_000);

  let tokenCounter: t.TokenCounter;

  beforeAll(async () => {
    tokenCounter = await createTokenCounter();
  });

  beforeEach(() => {
    mockSummarizationModel();
  });

  afterEach(() => {
    getChatModelClassSpy.mockRestore();
  });

  describe('PreCompact', () => {
    it('fires with messagesBeforeCount and trigger', async () => {
      const registry = new HookRegistry();
      let captured: PreCompactHookInput | undefined;
      const hook: HookCallback<'PreCompact'> = async (
        input
      ): Promise<PreCompactHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PreCompact', { hooks: [hook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Final answer after compaction.']);
      const inputs = buildConversation();
      await run.processStream(inputs, callerConfig);

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PreCompact');
      expect(captured!.messagesBeforeCount).toBeGreaterThan(0);
      expect(captured!.runId).toBe('compact-run');
      expect(captured!.trigger).toBe('default');
      expect(captured!.agentId).toBeDefined();
    });
  });

  describe('PostCompact', () => {
    it('fires with summary text after compaction', async () => {
      const registry = new HookRegistry();
      let captured: PostCompactHookInput | undefined;
      const hook: HookCallback<'PostCompact'> = async (
        input
      ): Promise<PostCompactHookOutput> => {
        captured = input;
        return {};
      };
      registry.register('PostCompact', { hooks: [hook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Final answer after compaction.']);
      const inputs = buildConversation();
      await run.processStream(inputs, callerConfig);

      expect(captured).toBeDefined();
      expect(captured!.hook_event_name).toBe('PostCompact');
      expect(captured!.summary).toContain('Summary');
      expect(captured!.messagesAfterCount).toBe(0);
      expect(captured!.agentId).toBeDefined();
    });
  });

  describe('error resilience', () => {
    it('throwing PreCompact hook does not crash compaction', async () => {
      const registry = new HookRegistry();
      const throwingHook: HookCallback<
        'PreCompact'
      > = async (): Promise<PreCompactHookOutput> => {
        throw new Error('pre hook crash');
      };
      registry.register('PreCompact', { hooks: [throwingHook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Answer after compaction.']);
      const inputs = buildConversation();

      await expect(
        run.processStream(inputs, callerConfig)
      ).resolves.not.toThrow();
    });

    it('throwing PostCompact hook does not crash compaction', async () => {
      const registry = new HookRegistry();
      const throwingHook: HookCallback<
        'PostCompact'
      > = async (): Promise<PostCompactHookOutput> => {
        throw new Error('post hook crash');
      };
      registry.register('PostCompact', { hooks: [throwingHook] });

      const run = await createCompactingRun(tokenCounter, registry);
      run.Graph!.overrideTestModel(['Answer after compaction.']);
      const inputs = buildConversation();

      await expect(
        run.processStream(inputs, callerConfig)
      ).resolves.not.toThrow();
    });
  });

  describe('no-hooks baseline', () => {
    it('summarization works identically without hooks', async () => {
      const run = await createCompactingRun(tokenCounter);
      run.Graph!.overrideTestModel(['Answer.']);
      const inputs = buildConversation();
      const result = await run.processStream(inputs, callerConfig);

      expect(result).toBeDefined();
    });
  });
});
