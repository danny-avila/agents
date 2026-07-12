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
        role: 'assistant',
        content: [{ type: 'input_text', text: 'Prior answer.' }],
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
