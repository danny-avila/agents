import OpenAI from 'openai';

import {
  addChatCacheBreakpoints,
  addResponseCacheBreakpoints,
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
});
