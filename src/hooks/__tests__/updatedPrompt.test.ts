import type {
  HookCallback,
  UserPromptSubmitHookInput,
  UserPromptSubmitHookOutput,
} from '../types';
import {
  redactSensitiveText,
  type SensitivePattern,
} from '@/messageContentRedaction';
import { clearMatcherCache } from '../matchers';
import { HookRegistry } from '../HookRegistry';
import { executeHooks } from '../executeHooks';

const TEST_PATTERNS: SensitivePattern[] = [
  {
    id: 'anthropic_key',
    label: 'Anthropic key',
    pattern: /\b(sk-ant-)[A-Za-z0-9_-]{10,}/g,
  },
  {
    id: 'openai_key',
    label: 'OpenAI key',
    pattern: /\b(sk-proj-)[A-Za-z0-9_-]{10,}/g,
  },
  {
    id: 'jwt',
    label: 'JWT',
    pattern: /\b(eyJ)[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g,
  },
];

function promptSubmitInput(prompt: string): UserPromptSubmitHookInput {
  return {
    hook_event_name: 'UserPromptSubmit',
    runId: 'run-1',
    threadId: 'thread-1',
    agentId: 'agent-1',
    prompt,
  };
}

function userPromptMatcher(callback: HookCallback<'UserPromptSubmit'>): {
  hooks: HookCallback<'UserPromptSubmit'>[];
} {
  return { hooks: [callback] };
}

beforeEach(() => {
  clearMatcherCache();
});

describe('executeHooks — updatedPrompt fold', () => {
  it('aggregates a single hook updatedPrompt', async () => {
    const registry = new HookRegistry();
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(
        async (): Promise<UserPromptSubmitHookOutput> => ({
          updatedPrompt: 'scrubbed prompt',
        })
      )
    );

    const result = await executeHooks({
      registry,
      input: promptSubmitInput('original prompt with sk-ant-secret'),
    });

    expect(result.updatedPrompt).toBe('scrubbed prompt');
  });

  it('lets the last writer win in registration order', async () => {
    const registry = new HookRegistry();
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(
        async (): Promise<UserPromptSubmitHookOutput> => ({
          updatedPrompt: 'first rewrite',
        })
      )
    );
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(
        async (): Promise<UserPromptSubmitHookOutput> => ({
          updatedPrompt: 'second rewrite',
        })
      )
    );

    const result = await executeHooks({
      registry,
      input: promptSubmitInput('original'),
    });

    expect(result.updatedPrompt).toBe('second rewrite');
  });

  it('leaves updatedPrompt unset when no hook supplies one', async () => {
    const registry = new HookRegistry();
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(
        async (): Promise<UserPromptSubmitHookOutput> => ({
          additionalContext: 'noted',
        })
      )
    );

    const result = await executeHooks({
      registry,
      input: promptSubmitInput('original'),
    });

    expect(result.updatedPrompt).toBeUndefined();
    expect(result.additionalContexts).toEqual(['noted']);
  });

  it('does not collide with decision or preventContinuation', async () => {
    const registry = new HookRegistry();
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(
        async (): Promise<UserPromptSubmitHookOutput> => ({
          updatedPrompt: 'rewritten',
          decision: 'allow',
        })
      )
    );

    const result = await executeHooks({
      registry,
      input: promptSubmitInput('original'),
    });

    expect(result.updatedPrompt).toBe('rewritten');
    expect(result.decision).toBe('allow');
  });
});

describe('messageContentRedaction as a UserPromptSubmit hook', () => {
  it('produces an updatedPrompt with credentials redacted using caller-supplied patterns', async () => {
    const registry = new HookRegistry();
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(async (input): Promise<UserPromptSubmitHookOutput> => {
        const { text, matches } = redactSensitiveText(input.prompt, {
          patterns: TEST_PATTERNS,
        });
        return matches.length > 0 ? { updatedPrompt: text } : {};
      })
    );

    const result = await executeHooks({
      registry,
      input: promptSubmitInput('my key is sk-ant-FAKE1234567890 please debug'),
    });

    expect(result.updatedPrompt).toBe(
      'my key is sk-ant-[REDACTED] please debug'
    );
  });

  it('is a no-op (no updatedPrompt) when nothing matches', async () => {
    const registry = new HookRegistry();
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(async (input): Promise<UserPromptSubmitHookOutput> => {
        const { text, matches } = redactSensitiveText(input.prompt, {
          patterns: TEST_PATTERNS,
        });
        return matches.length > 0 ? { updatedPrompt: text } : {};
      })
    );

    const result = await executeHooks({
      registry,
      input: promptSubmitInput('plain conversational text'),
    });

    expect(result.updatedPrompt).toBeUndefined();
  });

  it('honors a configured pattern subset', async () => {
    const onlyJwt = [TEST_PATTERNS[2]];
    const registry = new HookRegistry();
    registry.register(
      'UserPromptSubmit',
      userPromptMatcher(async (input): Promise<UserPromptSubmitHookOutput> => {
        const { text, matches } = redactSensitiveText(input.prompt, {
          patterns: onlyJwt,
        });
        return matches.length > 0 ? { updatedPrompt: text } : {};
      })
    );

    const result = await executeHooks({
      registry,
      input: promptSubmitInput(
        'jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sig and key sk-ant-FAKE1234567890'
      ),
    });

    expect(result.updatedPrompt).toBe(
      'jwt eyJ[REDACTED] and key sk-ant-FAKE1234567890'
    );
  });
});
