import { ChatOpenRouter } from './index';
import type { OpenRouterReasoning, ChatOpenRouterCallOptions } from './index';
import type { OpenAIChatInput } from '@langchain/openai';

type CreateRouterOptions = Partial<
  ChatOpenRouterCallOptions & Pick<OpenAIChatInput, 'model' | 'apiKey'>
>;

function createRouter(overrides: CreateRouterOptions = {}): ChatOpenRouter {
  return new ChatOpenRouter({
    model: 'openrouter/test-model',
    apiKey: 'test-key',
    ...overrides,
  });
}

describe('ChatOpenRouter reasoning handling', () => {
  // ---------------------------------------------------------------
  // 1. Constructor reasoning config
  // ---------------------------------------------------------------
  describe('constructor reasoning config', () => {
    it('stores reasoning when passed directly', () => {
      const router = createRouter({ reasoning: { effort: 'high' } });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high' });
    });
  });

  // ---------------------------------------------------------------
  // 2. modelKwargs reasoning extraction
  // ---------------------------------------------------------------
  describe('modelKwargs reasoning extraction', () => {
    it('extracts reasoning from modelKwargs and places it into params.reasoning', () => {
      const router = createRouter({
        modelKwargs: { reasoning: { effort: 'medium' } },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'medium' });
    });

    it('does not leak reasoning into modelKwargs that reach the parent', () => {
      const router = createRouter({
        modelKwargs: {
          reasoning: { effort: 'medium' },
        },
      });
      const params = router.invocationParams();
      // reasoning should be the structured OpenRouter object, not buried in modelKwargs
      expect(params.reasoning).toEqual({ effort: 'medium' });
    });
  });

  // ---------------------------------------------------------------
  // 3. Reasoning merge precedence
  // ---------------------------------------------------------------
  describe('reasoning merge precedence', () => {
    it('constructor reasoning overrides modelKwargs.reasoning', () => {
      const router = createRouter({
        reasoning: { effort: 'high' },
        modelKwargs: { reasoning: { effort: 'low' } },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high' });
    });

    it('merges non-overlapping keys from modelKwargs.reasoning and constructor reasoning', () => {
      const router = createRouter({
        reasoning: { effort: 'high' },
        modelKwargs: { reasoning: { max_tokens: 5000 } },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high', max_tokens: 5000 });
    });
  });

  // ---------------------------------------------------------------
  // 4. invocationParams output
  // ---------------------------------------------------------------
  describe('invocationParams output', () => {
    it('includes reasoning object in params', () => {
      const router = createRouter({ reasoning: { effort: 'high' } });
      const params = router.invocationParams();
      expect(params.reasoning).toBeDefined();
      expect(params.reasoning).toEqual({ effort: 'high' });
    });

    it('does NOT include reasoning_effort in params', () => {
      const router = createRouter({ reasoning: { effort: 'high' } });
      const params = router.invocationParams();
      expect(params.reasoning_effort).toBeUndefined();
    });

    it('does not include reasoning when none is configured', () => {
      const router = createRouter();
      const params = router.invocationParams();
      expect(params.reasoning).toBeUndefined();
      expect(params.reasoning_effort).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------
  // 5. Legacy include_reasoning
  // ---------------------------------------------------------------
  describe('legacy include_reasoning', () => {
    it('produces { enabled: true } when only include_reasoning is true', () => {
      const router = createRouter({ include_reasoning: true });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ enabled: true });
    });

    it('does not produce reasoning when include_reasoning is false', () => {
      const router = createRouter({ include_reasoning: false });
      const params = router.invocationParams();
      expect(params.reasoning).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------
  // 6. Legacy include_reasoning ignored when reasoning is provided
  // ---------------------------------------------------------------
  describe('legacy include_reasoning ignored when reasoning provided', () => {
    it('reasoning wins over include_reasoning', () => {
      const router = createRouter({
        reasoning: { effort: 'medium' },
        include_reasoning: true,
      });
      const params = router.invocationParams();
      // Should use the structured reasoning, NOT fall back to { enabled: true }
      expect(params.reasoning).toEqual({ effort: 'medium' });
    });

    it('reasoning from modelKwargs also wins over include_reasoning', () => {
      const router = createRouter({
        modelKwargs: { reasoning: { effort: 'low' } },
        include_reasoning: true,
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'low' });
    });
  });

  // ---------------------------------------------------------------
  // 7. Various effort levels (OpenRouter-specific)
  // ---------------------------------------------------------------
  describe('various effort levels', () => {
    const efforts: Array<{
      effort: OpenRouterReasoning['effort'];
    }> = [
      { effort: 'xhigh' },
      { effort: 'none' },
      { effort: 'minimal' },
      { effort: 'high' },
      { effort: 'medium' },
      { effort: 'low' },
    ];

    it.each(efforts)('supports effort level "$effort"', ({ effort }) => {
      const router = createRouter({ reasoning: { effort } });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort });
      expect(params.reasoning_effort).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------
  // 8. max_tokens reasoning
  // ---------------------------------------------------------------
  describe('max_tokens reasoning', () => {
    it('passes max_tokens in reasoning object', () => {
      const router = createRouter({
        reasoning: { max_tokens: 8000 },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ max_tokens: 8000 });
    });

    it('combines max_tokens with effort', () => {
      const router = createRouter({
        reasoning: { effort: 'high', max_tokens: 8000 },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high', max_tokens: 8000 });
      expect(params.reasoning_effort).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------
  // 9. exclude reasoning
  // ---------------------------------------------------------------
  describe('exclude reasoning', () => {
    it('passes exclude flag in reasoning object', () => {
      const router = createRouter({
        reasoning: { effort: 'high', exclude: true },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ effort: 'high', exclude: true });
    });

    it('supports exclude without effort', () => {
      const router = createRouter({
        reasoning: { exclude: true },
      });
      const params = router.invocationParams();
      expect(params.reasoning).toEqual({ exclude: true });
    });
  });
});
