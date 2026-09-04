import OpenAI from 'openai';

import {
  ChatOpenAI,
  addChatCacheBreakpoints,
  addResponseCacheBreakpoints,
  isGpt6AstraModel,
  shouldIncludeEncryptedReasoning,
} from './index';

describe('managed GPT-5.6 request fields', () => {
  it('places cache breakpoints after instructions and the prior history prefix', () => {
    const messages = addChatCacheBreakpoints([
      { role: 'system', content: 'Stable instructions.' },
      { role: 'user', content: 'First question.' },
      { role: 'assistant', content: 'First answer.' },
      { role: 'user', content: 'Current question.' },
    ]);

    expect(messages[0]).toMatchObject({
      content: [
        {
          type: 'text',
          text: 'Stable instructions.',
          prompt_cache_breakpoint: { mode: 'explicit' },
        },
      ],
    });
    expect(messages[2]).toMatchObject({
      content: [
        {
          type: 'text',
          text: 'First answer.',
          prompt_cache_breakpoint: { mode: 'explicit' },
        },
      ],
    });
    expect(JSON.stringify(messages[1])).not.toContain(
      'prompt_cache_breakpoint'
    );
    expect(JSON.stringify(messages[3])).not.toContain(
      'prompt_cache_breakpoint'
    );
  });

  it('uses supported Responses content blocks for the same stable prefixes', () => {
    const input = [
      {
        type: 'message',
        role: 'developer',
        content: [{ type: 'input_text', text: 'Stable instructions.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Prior question.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Current question.' }],
      },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input);

    expect(result).toMatchObject([
      {
        content: [
          {
            prompt_cache_breakpoint: { mode: 'explicit' },
          },
        ],
      },
      {
        content: [
          {
            prompt_cache_breakpoint: { mode: 'explicit' },
          },
        ],
      },
      {
        content: [{ type: 'input_text', text: 'Current question.' }],
      },
    ]);
  });

  it('does not mark replayed assistant output blocks as breakpoints', () => {
    const input = [
      {
        type: 'message',
        role: 'developer',
        content: [{ type: 'input_text', text: 'Stable instructions.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'First question.' }],
      },
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'Prior answer.' }],
      },
      {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: 'Current question.' }],
      },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input) as unknown as Array<{
      content: Array<Record<string, unknown>>;
    }>;

    // output_text is rejected with a 400 by OpenAI, so it must stay unmarked;
    // the stable prefix falls back to the prior user input message.
    expect(result[2].content[0]).not.toHaveProperty('prompt_cache_breakpoint');
    expect(result[1].content[0]).toHaveProperty('prompt_cache_breakpoint');
    expect(result[0].content[0]).toHaveProperty('prompt_cache_breakpoint');
  });

  it('marks string-content Responses messages by wrapping them in input_text', () => {
    const input = [
      { type: 'message', role: 'system', content: 'Stable system prompt.' },
      { type: 'message', role: 'user', content: 'Prior turn.' },
      { type: 'message', role: 'user', content: 'Current question.' },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input) as unknown as Array<{
      content: unknown;
    }>;

    expect(result[0].content).toEqual([
      {
        type: 'input_text',
        text: 'Stable system prompt.',
        prompt_cache_breakpoint: { mode: 'explicit' },
      },
    ]);
  });

  it('does not rewrite assistant string content as input_text', () => {
    const input = [
      { type: 'message', role: 'system', content: 'Stable system prompt.' },
      { type: 'message', role: 'user', content: 'First question.' },
      { type: 'message', role: 'assistant', content: 'Prior answer.' },
      { type: 'message', role: 'user', content: 'Current question.' },
    ] as unknown as OpenAI.Responses.ResponseInput;
    const result = addResponseCacheBreakpoints(input) as unknown as Array<{
      content: unknown;
    }>;

    // input_text under role:assistant is rejected with a 400, so assistant
    // string content must stay a string and the breakpoint falls back to the
    // prior user turn.
    expect(result[2].content).toBe('Prior answer.');
    expect(result[1].content).toEqual([
      {
        type: 'input_text',
        text: 'First question.',
        prompt_cache_breakpoint: { mode: 'explicit' },
      },
    ]);
  });

  it('requests encrypted reasoning whenever persisted or stateless replay may be needed', () => {
    expect(shouldIncludeEncryptedReasoning('gpt-5.6', {})).toBe(true);
    expect(
      shouldIncludeEncryptedReasoning('gpt-5.6', {
        reasoning: { context: 'all_turns' },
      })
    ).toBe(true);
    expect(
      shouldIncludeEncryptedReasoning('gpt-5.6', {
        reasoning: { context: 'current_turn' },
      })
    ).toBe(false);
    expect(
      shouldIncludeEncryptedReasoning('gpt-5.6', {
        store: false,
        reasoning: { context: 'current_turn' },
      })
    ).toBe(true);
    expect(shouldIncludeEncryptedReasoning('gpt-5.5', {})).toBe(false);
  });

  it('requests encrypted reasoning for GPT-6 Astra on a first-party endpoint', () => {
    expect(shouldIncludeEncryptedReasoning('gpt-6-astra', {}, true)).toBe(true);
    expect(
      shouldIncludeEncryptedReasoning(
        'gpt-6-astra',
        { reasoning: { context: 'current_turn' } },
        true
      )
    ).toBe(false);
  });

  it('does not request encrypted reasoning for Astra behind a proxy', () => {
    expect(shouldIncludeEncryptedReasoning('gpt-6-astra', {}, false)).toBe(false);
  });
});

describe('GPT-6 Astra model detection', () => {
  it('matches the documented id and its snapshots', () => {
    expect(isGpt6AstraModel('gpt-6-astra')).toBe(true);
    expect(isGpt6AstraModel('gpt-6-astra-2026-04-30')).toBe(true);
    expect(isGpt6AstraModel('GPT-6-Astra')).toBe(true);
  });

  it('does not widen to the rest of the gpt-6 family or near-miss ids', () => {
    expect(isGpt6AstraModel('gpt-6')).toBe(false);
    expect(isGpt6AstraModel('gpt-6-mini')).toBe(false);
    expect(isGpt6AstraModel('gpt-6-astral')).toBe(false);
    expect(isGpt6AstraModel('gpt-5.6')).toBe(false);
    expect(isGpt6AstraModel('not-gpt-6-astra')).toBe(false);
    expect(isGpt6AstraModel(undefined)).toBe(false);
    expect(isGpt6AstraModel('')).toBe(false);
  });

  /**
   * A `provider/` prefix means a proxy owns the request contract. `ChatOpenRouter`
   * extends `ChatOpenAI`, so matching a prefixed id would apply OpenAI's rules to
   * OpenRouter — where `effort: 'none'` is a supported value meaning "disable
   * reasoning", not an error to substitute away.
   */
  it('does not match proxy-routed ids', () => {
    expect(isGpt6AstraModel('openai/gpt-6-astra')).toBe(false);
    expect(isGpt6AstraModel('openrouter/openai/gpt-6-astra')).toBe(false);
  });
});

describe('GPT-6 Astra request constraints', () => {
  /**
   * The endpoint gate falls back to `OPENAI_BASE_URL` when no `baseURL` is
   * configured, so a developer or CI shell pointing at a compatibility gateway
   * would otherwise turn every gate off and fail these tests for a reason that
   * has nothing to do with the implementation. Isolated the same way the
   * sequential-tool-call suite does.
   */
  const ISOLATED_ENV_VARS = ['OPENAI_BASE_URL'];
  const originalEnv = new Map(
    ISOLATED_ENV_VARS.map((name) => [name, process.env[name]])
  );

  beforeEach(() => {
    for (const name of ISOLATED_ENV_VARS) {
      delete process.env[name];
    }
  });

  afterAll(() => {
    for (const [name, value] of originalEnv) {
      if (value == null) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  });

  const astra = (fields: Record<string, unknown> = {}) =>
    new ChatOpenAI({ model: 'gpt-6-astra', apiKey: 'test-key', ...fields });

  const tool = {
    type: 'function' as const,
    function: {
      name: 'get_weather',
      description: 'Get the weather',
      parameters: { type: 'object', properties: {} },
    },
  };

  it('routes tool-bearing turns to the Responses API', () => {
    const model = astra();
    expect(model.invocationParams({ tools: [tool] })).toMatchObject({
      model: 'gpt-6-astra',
    });
    // Responses builds `max_output_tokens`; Completions builds
    // `max_completion_tokens`. The key present identifies the chosen path.
    expect(
      'max_output_tokens' in model.invocationParams({ tools: [tool] })
    ).toBe(true);
  });

  it('keeps non-tool turns on Chat Completions', () => {
    const params = astra().invocationParams({});
    expect('max_output_tokens' in params).toBe(false);
  });

  it('strips sampling parameters the model rejects', () => {
    const model = astra({ temperature: 0.7, topP: 0.9, topLogprobs: 3 });
    for (const options of [{}, { tools: [tool] }]) {
      const params = model.invocationParams(options) as Record<string, unknown>;
      expect(params).not.toHaveProperty('temperature');
      expect(params).not.toHaveProperty('top_p');
      expect(params).not.toHaveProperty('top_logprobs');
    }
  });

  it('strips logprobs on Chat Completions', () => {
    const params = astra({ logprobs: true }).invocationParams({}) as Record<
      string,
      unknown
    >;
    expect(params).not.toHaveProperty('logprobs');
  });

  it('drops the rejected logprobs include on Responses, keeping the rest', () => {
    const model = astra();
    const params = model.invocationParams({
      tools: [tool],
      include: ['message.output_text.logprobs', 'reasoning.encrypted_content'],
    } as never) as { include?: unknown[] };
    expect(params.include).toEqual(
      expect.not.arrayContaining(['message.output_text.logprobs'])
    );
    expect(params.include).toEqual(
      expect.arrayContaining(['reasoning.encrypted_content'])
    );
  });

  /**
   * The two paths carry effort under different keys: Chat Completions emits the
   * scalar `reasoning_effort`, Responses the nested `reasoning.effort`.
   */
  const effortOf = (options: Record<string, unknown>): string | undefined => {
    const params = astra().invocationParams(options as never) as {
      reasoning_effort?: string;
      reasoning?: { effort?: string };
    };
    return params.reasoning_effort ?? params.reasoning?.effort;
  };

  it('substitutes the reasoning efforts the model rejects with low', () => {
    for (const effort of ['none', 'minimal'] as const) {
      expect(effortOf({ reasoningEffort: effort })).toBe('low');
      expect(effortOf({ reasoningEffort: effort, tools: [tool] })).toBe('low');
    }
  });

  it('passes supported reasoning efforts through unchanged', () => {
    for (const effort of ['low', 'medium', 'high', 'xhigh', 'max'] as const) {
      expect(effortOf({ reasoningEffort: effort })).toBe(effort);
      expect(effortOf({ reasoningEffort: effort, tools: [tool] })).toBe(effort);
    }
  });

  it('leaves rejected efforts alone on other models', () => {
    const model = new ChatOpenAI({ model: 'gpt-5.6', apiKey: 'test-key' });
    const params = model.invocationParams({
      reasoningEffort: 'none',
    } as never) as { reasoning_effort?: string };
    expect(params.reasoning_effort).toBe('none');
  });

  it('leaves a proxy-routed Astra id alone on every gate', () => {
    const proxied = new ChatOpenAI({
      model: 'openai/gpt-6-astra',
      apiKey: 'test-key',
      temperature: 0.7,
    });
    const params = proxied.invocationParams({
      tools: [tool],
      reasoningEffort: 'none',
    } as never) as Record<string, unknown>;
    /** Chat Completions path retained, sampling kept, `none` not substituted. */
    expect('max_output_tokens' in params).toBe(false);
    expect(params.temperature).toBe(0.7);
    expect(params.reasoning_effort).toBe('none');
  });

  it('leaves Astra behind a custom baseURL on every gate', () => {
    const proxied = new ChatOpenAI({
      model: 'gpt-6-astra',
      apiKey: 'test-key',
      temperature: 0.7,
      configuration: { baseURL: 'https://gateway.internal/v1' },
    });
    const params = proxied.invocationParams({
      tools: [tool],
      reasoningEffort: 'none',
    } as never) as Record<string, unknown>;
    /** A bare Astra id still reaches a proxy whose contract is not OpenAI's. */
    expect('max_output_tokens' in params).toBe(false);
    expect(params.temperature).toBe(0.7);
    expect(params.reasoning_effort).toBe('none');
  });

  it('applies every gate on the first-party endpoint', () => {
    const direct = new ChatOpenAI({
      model: 'gpt-6-astra',
      apiKey: 'test-key',
      temperature: 0.7,
    });
    const params = direct.invocationParams({
      tools: [tool],
      reasoningEffort: 'none',
    } as never) as Record<string, unknown>;
    expect('max_output_tokens' in params).toBe(true);
    expect(params).not.toHaveProperty('temperature');
    expect((params.reasoning as { effort?: string } | undefined)?.effort).toBe('low');
  });

  it.each([
    ['the default port written explicitly', 'https://api.openai.com:443/v1'],
    ['a mixed-case host', 'https://API.OpenAI.com/v1'],
  ])('applies the gates for a first-party URL spelled with %s', (_label, baseURL) => {
    const model = astra({ configuration: { baseURL } });
    const params = model.invocationParams({
      tools: [tool],
      reasoningEffort: 'none',
    } as never) as Record<string, unknown>;
    expect('max_output_tokens' in params).toBe(true);
    expect(params).not.toHaveProperty('temperature');
  });

  it('leaves other models untouched', () => {
    const model = new ChatOpenAI({
      model: 'gpt-5.6',
      apiKey: 'test-key',
      temperature: 0.7,
    });
    const params = model.invocationParams({}) as Record<string, unknown>;
    expect(params.temperature).toBe(0.7);
  });
});
