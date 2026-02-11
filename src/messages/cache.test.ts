import type Anthropic from '@anthropic-ai/sdk';
import type { AnthropicMessages } from '@/types/messages';
import {
  stripAnthropicCacheControl,
  stripBedrockCacheControl,
  addBedrockCacheControl,
  addCacheControl,
} from './cache';
import { MessageContentComplex } from '@langchain/core/messages';
import { ContentTypes } from '@/common/enum';

describe('addCacheControl', () => {
  test('should add cache control to the last two user messages with array content', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'Hi there' }] },
      { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'I\'m doing well, thanks!' }],
      },
      { role: 'user', content: [{ type: 'text', text: 'Great!' }] },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content[0]).not.toHaveProperty('cache_control');
    expect(
      (result[2].content[0] as Anthropic.TextBlockParam).cache_control
    ).toEqual({ type: 'ephemeral' });
    expect(
      (result[4].content[0] as Anthropic.TextBlockParam).cache_control
    ).toEqual({ type: 'ephemeral' });
  });

  test('should add cache control to the last two user messages with string content', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
      { role: 'user', content: 'How are you?' },
      { role: 'assistant', content: 'I\'m doing well, thanks!' },
      { role: 'user', content: 'Great!' },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content).toBe('Hello');
    expect(result[2].content[0]).toEqual({
      type: 'text',
      text: 'How are you?',
      cache_control: { type: 'ephemeral' },
    });
    expect(result[4].content[0]).toEqual({
      type: 'text',
      text: 'Great!',
      cache_control: { type: 'ephemeral' },
    });
  });

  test('should handle mixed string and array content', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
      { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content[0]).toEqual({
      type: 'text',
      text: 'Hello',
      cache_control: { type: 'ephemeral' },
    });
    expect(
      (result[2].content[0] as Anthropic.TextBlockParam).cache_control
    ).toEqual({ type: 'ephemeral' });
  });

  test('should handle less than two user messages', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content[0]).toEqual({
      type: 'text',
      text: 'Hello',
      cache_control: { type: 'ephemeral' },
    });
    expect(result[1].content).toBe('Hi there');
  });

  test('should return original array if no user messages', () => {
    const messages: AnthropicMessages = [
      { role: 'assistant', content: 'Hi there' },
      { role: 'assistant', content: 'How can I help?' },
    ];

    const result = addCacheControl(messages);

    expect(result).toEqual(messages);
  });

  test('should handle empty array', () => {
    const messages: AnthropicMessages = [];
    const result = addCacheControl(messages);
    expect(result).toEqual([]);
  });

  test('should handle non-array input', () => {
    const messages = 'not an array';
    /** @ts-expect-error - This is a test */
    const result = addCacheControl(messages);
    expect(result).toBe('not an array');
  });

  test('should not modify assistant messages', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
      { role: 'user', content: 'How are you?' },
    ];

    const result = addCacheControl(messages);

    expect(result[1].content).toBe('Hi there');
  });

  test('should handle multiple content items in user messages', () => {
    const messages: AnthropicMessages = [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Hello' },
          {
            type: 'image',
            source: { type: 'url', url: 'http://example.com/image.jpg' },
          },
          { type: 'text', text: 'This is an image' },
        ],
      },
      { role: 'assistant', content: 'Hi there' },
      { role: 'user', content: 'How are you?' },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content[0]).not.toHaveProperty('cache_control');
    expect(result[0].content[1]).not.toHaveProperty('cache_control');
    expect(
      (result[0].content[2] as Anthropic.TextBlockParam).cache_control
    ).toEqual({ type: 'ephemeral' });
    expect(result[2].content[0]).toEqual({
      type: 'text',
      text: 'How are you?',
      cache_control: { type: 'ephemeral' },
    });
  });

  test('should handle an array with mixed content types', () => {
    const messages: AnthropicMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
      { role: 'user', content: [{ type: 'text', text: 'How are you?' }] },
      { role: 'assistant', content: 'I\'m doing well, thanks!' },
      { role: 'user', content: 'Great!' },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content).toEqual('Hello');
    expect(result[2].content[0]).toEqual({
      type: 'text',
      text: 'How are you?',
      cache_control: { type: 'ephemeral' },
    });
    expect(result[4].content).toEqual([
      {
        type: 'text',
        text: 'Great!',
        cache_control: { type: 'ephemeral' },
      },
    ]);
    expect(result[1].content).toBe('Hi there');
    expect(result[3].content).toBe('I\'m doing well, thanks!');
  });

  test('should handle edge case with multiple content types', () => {
    const messages: AnthropicMessages = [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/png',
              data: 'some_base64_string',
            },
          },
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/png',
              data: 'another_base64_string',
            },
          },
          { type: 'text', text: 'what do all these images have in common' },
        ],
      },
      { role: 'assistant', content: 'I see multiple images.' },
      { role: 'user', content: 'Correct!' },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content[0]).not.toHaveProperty('cache_control');
    expect(result[0].content[1]).not.toHaveProperty('cache_control');
    expect(
      (result[0].content[2] as Anthropic.ImageBlockParam).cache_control
    ).toEqual({ type: 'ephemeral' });
    expect(result[2].content[0]).toEqual({
      type: 'text',
      text: 'Correct!',
      cache_control: { type: 'ephemeral' },
    });
  });

  test('should handle user message with no text block', () => {
    const messages: AnthropicMessages = [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/png',
              data: 'some_base64_string',
            },
          },
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/png',
              data: 'another_base64_string',
            },
          },
        ],
      },
      { role: 'assistant', content: 'I see two images.' },
      { role: 'user', content: 'Correct!' },
    ];

    const result = addCacheControl(messages);

    expect(result[0].content[0]).not.toHaveProperty('cache_control');
    expect(result[0].content[1]).not.toHaveProperty('cache_control');
    expect(result[2].content[0]).toEqual({
      type: 'text',
      text: 'Correct!',
      cache_control: { type: 'ephemeral' },
    });
  });
});

type TestMsg = {
  role?: 'user' | 'assistant' | 'system';
  content?: string | MessageContentComplex[];
};

describe('addBedrockCacheControl (Bedrock cache checkpoints)', () => {
  it('returns input when not enough messages', () => {
    const empty: TestMsg[] = [];
    expect(addBedrockCacheControl(empty)).toEqual(empty);
    const single: TestMsg[] = [{ role: 'user', content: 'only' }];
    expect(addBedrockCacheControl(single)).toEqual(single);
  });

  it('wraps string content and appends separate cachePoint block', () => {
    const messages: TestMsg[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: [{ type: ContentTypes.TEXT, text: 'Hi' }] },
    ];
    const result = addBedrockCacheControl(messages);
    const last = result[1].content as MessageContentComplex[];
    expect(Array.isArray(last)).toBe(true);
    expect(last[0]).toEqual({ type: ContentTypes.TEXT, text: 'Hi' });
    expect(last[1]).toEqual({ cachePoint: { type: 'default' } });
  });

  it('inserts cachePoint after the last text when multiple text blocks exist', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          { type: ContentTypes.TEXT, text: 'Intro' },
          { type: ContentTypes.TEXT, text: 'Details' },
          {
            type: ContentTypes.IMAGE_FILE,
            image_file: { file_id: 'file_123' },
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Reply A' },
          { type: ContentTypes.TEXT, text: 'Reply B' },
        ],
      },
    ];

    const result = addBedrockCacheControl(messages);

    const first = result[0].content as MessageContentComplex[];
    const second = result[1].content as MessageContentComplex[];

    expect(first[0]).toEqual({ type: ContentTypes.TEXT, text: 'Intro' });
    expect(first[1]).toEqual({ type: ContentTypes.TEXT, text: 'Details' });
    expect(first[2]).toEqual({ cachePoint: { type: 'default' } });

    const img = first[3] as MessageContentComplex;
    expect(img.type).toBe(ContentTypes.IMAGE_FILE);
    if (img.type === ContentTypes.IMAGE_FILE) {
      expect('image_file' in img).toBe(true);
    }

    expect(second[0]).toEqual({ type: ContentTypes.TEXT, text: 'Reply A' });
    expect(second[1]).toEqual({ type: ContentTypes.TEXT, text: 'Reply B' });
    expect(second[2]).toEqual({ cachePoint: { type: 'default' } });
  });

  it('skips adding cachePoint when content is an empty array', () => {
    const messages: TestMsg[] = [
      { role: 'user', content: [] },
      { role: 'assistant', content: [] },
      { role: 'user', content: 'ignored because only last two are modified' },
    ];

    const result = addBedrockCacheControl(messages);

    const first = result[0].content as MessageContentComplex[];
    const second = result[1].content as MessageContentComplex[];

    expect(Array.isArray(first)).toBe(true);
    expect(first.length).toBe(0);

    expect(Array.isArray(second)).toBe(true);
    expect(second.length).toBe(0);
    expect(second[0]).not.toEqual({ cachePoint: { type: 'default' } });
  });

  it('skips adding cachePoint when content is an empty string', () => {
    const messages: TestMsg[] = [
      { role: 'user', content: '' },
      { role: 'assistant', content: '' },
      { role: 'user', content: 'ignored because only last two are modified' },
    ];

    const result = addBedrockCacheControl(messages);

    expect(result[0].content).toBe('');
    expect(result[1].content).toBe('');
  });

  /** (I don't think this will ever occur in actual use, but its the only branch left uncovered so I'm covering it */
  it('skips messages with non-string, non-array content and still modifies the previous to reach two edits', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Will be modified' }],
      },
      { role: 'assistant', content: undefined },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Also modified' }],
      },
    ];

    const result = addBedrockCacheControl(messages);

    const last = result[2].content as MessageContentComplex[];
    expect(last[0]).toEqual({ type: ContentTypes.TEXT, text: 'Also modified' });
    expect(last[1]).toEqual({ cachePoint: { type: 'default' } });

    expect(result[1].content).toBeUndefined();

    const first = result[0].content as MessageContentComplex[];
    expect(first[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Will be modified',
    });
    expect(first[1]).toEqual({ cachePoint: { type: 'default' } });
  });

  it('works with the example from the langchain pr (with multi-turn behavior)', () => {
    const messages: TestMsg[] = [
      {
        role: 'system',
        content: [
          { type: ContentTypes.TEXT, text: 'You\'re an advanced AI assistant.' },
        ],
      },
      {
        role: 'user',
        content: [
          { type: ContentTypes.TEXT, text: 'What is the capital of France?' },
        ],
      },
    ];

    const result = addBedrockCacheControl(messages);

    let system = result[0].content as MessageContentComplex[];
    let user = result[1].content as MessageContentComplex[];

    expect(system[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'You\'re an advanced AI assistant.',
    });
    expect(system[1]).toEqual({ cachePoint: { type: 'default' } });
    expect(user[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'What is the capital of France?',
    });
    expect(user[1]).toEqual({ cachePoint: { type: 'default' } });

    result.push({
      role: 'assistant',
      content: [
        {
          type: ContentTypes.TEXT,
          text: 'Sure! The capital of France is Paris.',
        },
      ],
    });

    const result2 = addBedrockCacheControl(result);

    system = result2[0].content as MessageContentComplex[];
    user = result2[1].content as MessageContentComplex[];
    const assistant = result2[2].content as MessageContentComplex[];

    expect(system[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'You\'re an advanced AI assistant.',
    });
    expect(system.length).toBe(1);

    expect(user[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'What is the capital of France?',
    });
    expect(user[1]).toEqual({ cachePoint: { type: 'default' } });

    expect(assistant[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Sure! The capital of France is Paris.',
    });
    expect(assistant[1]).toEqual({ cachePoint: { type: 'default' } });
  });

  it('is idempotent - calling multiple times does not add duplicate cache points', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'First message' }],
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'First response' }],
      },
    ];

    const result1 = addBedrockCacheControl(messages);
    const firstContent = result1[0].content as MessageContentComplex[];
    const secondContent = result1[1].content as MessageContentComplex[];

    expect(firstContent.length).toBe(2);
    expect(firstContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'First message',
    });
    expect(firstContent[1]).toEqual({ cachePoint: { type: 'default' } });

    expect(secondContent.length).toBe(2);
    expect(secondContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'First response',
    });
    expect(secondContent[1]).toEqual({ cachePoint: { type: 'default' } });

    const result2 = addBedrockCacheControl(result1);
    const firstContentAfter = result2[0].content as MessageContentComplex[];
    const secondContentAfter = result2[1].content as MessageContentComplex[];

    expect(firstContentAfter.length).toBe(2);
    expect(firstContentAfter[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'First message',
    });
    expect(firstContentAfter[1]).toEqual({ cachePoint: { type: 'default' } });

    expect(secondContentAfter.length).toBe(2);
    expect(secondContentAfter[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'First response',
    });
    expect(secondContentAfter[1]).toEqual({ cachePoint: { type: 'default' } });
  });

  it('skips messages that already have cache points in multi-agent scenarios', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Hello' }],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Response from agent 1' },
          { cachePoint: { type: 'default' } },
        ],
      },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Follow-up question' }],
      },
    ];

    const result = addBedrockCacheControl(messages);
    const lastContent = result[2].content as MessageContentComplex[];
    const secondLastContent = result[1].content as MessageContentComplex[];

    expect(lastContent.length).toBe(2);
    expect(lastContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Follow-up question',
    });
    expect(lastContent[1]).toEqual({ cachePoint: { type: 'default' } });

    expect(secondLastContent.length).toBe(2);
    expect(secondLastContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Response from agent 1',
    });
    expect(secondLastContent[1]).toEqual({ cachePoint: { type: 'default' } });
  });

  it('skips cachePoint on AI messages with only whitespace text and reasoning (tool-call scenario)', () => {
    const messages = [
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'What can you tell me about this PR?',
          },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: '\n\n' },
          {
            type: 'reasoning_content',
            reasoningText: { text: 'Let me look into that.', signature: 'abc' },
          },
        ] as MessageContentComplex[],
      },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'tool result here' }],
        getType: (): 'tool' => 'tool',
      },
    ];

    const result = addBedrockCacheControl(
      messages as Parameters<typeof addBedrockCacheControl>[0]
    );
    const aiContent = result[1].content as MessageContentComplex[];
    const hasCachePoint = aiContent.some((b) => 'cachePoint' in b);
    expect(hasCachePoint).toBe(false);

    const userContent = result[0].content as MessageContentComplex[];
    expect(userContent[userContent.length - 1]).toEqual({
      cachePoint: { type: 'default' },
    });
  });
});

describe('stripAnthropicCacheControl', () => {
  it('removes cache_control fields from content blocks', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'Hello',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'Hi there',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
    ];

    const result = stripAnthropicCacheControl(messages);

    const firstContent = result[0].content as MessageContentComplex[];
    const secondContent = result[1].content as MessageContentComplex[];

    expect(firstContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Hello',
    });
    expect('cache_control' in firstContent[0]).toBe(false);

    expect(secondContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Hi there',
    });
    expect('cache_control' in secondContent[0]).toBe(false);
  });

  it('handles messages without cache_control gracefully', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Hello' }],
      },
    ];

    const result = stripAnthropicCacheControl(messages);

    expect(result[0].content).toEqual([
      { type: ContentTypes.TEXT, text: 'Hello' },
    ]);
  });

  it('handles string content gracefully', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: 'Hello',
      },
    ];

    const result = stripAnthropicCacheControl(messages);

    expect(result[0].content).toBe('Hello');
  });

  it('returns non-array input unchanged', () => {
    const notArray = 'not an array';
    /** @ts-expect-error - Testing invalid input */
    const result = stripAnthropicCacheControl(notArray);
    expect(result).toBe('not an array');
  });
});

describe('stripBedrockCacheControl', () => {
  it('removes cachePoint blocks from content arrays', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          { type: ContentTypes.TEXT, text: 'Hello' },
          { cachePoint: { type: 'default' } },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Hi there' },
          { cachePoint: { type: 'default' } },
        ],
      },
    ];

    const result = stripBedrockCacheControl(messages);

    const firstContent = result[0].content as MessageContentComplex[];
    const secondContent = result[1].content as MessageContentComplex[];

    expect(firstContent.length).toBe(1);
    expect(firstContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Hello',
    });

    expect(secondContent.length).toBe(1);
    expect(secondContent[0]).toEqual({
      type: ContentTypes.TEXT,
      text: 'Hi there',
    });
  });

  it('handles messages without cachePoint blocks gracefully', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Hello' }],
      },
    ];

    const result = stripBedrockCacheControl(messages);

    expect(result[0].content).toEqual([
      { type: ContentTypes.TEXT, text: 'Hello' },
    ]);
  });

  it('handles string content gracefully', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: 'Hello',
      },
    ];

    const result = stripBedrockCacheControl(messages);

    expect(result[0].content).toBe('Hello');
  });

  it('preserves content with type field', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          { type: ContentTypes.TEXT, text: 'Hello' },
          {
            type: ContentTypes.IMAGE_FILE,
            image_file: { file_id: 'file_123' },
          },
          { cachePoint: { type: 'default' } },
        ],
      },
    ];

    const result = stripBedrockCacheControl(messages);

    const content = result[0].content as MessageContentComplex[];

    expect(content.length).toBe(2);
    expect(content[0]).toEqual({ type: ContentTypes.TEXT, text: 'Hello' });
    expect(content[1]).toEqual({
      type: ContentTypes.IMAGE_FILE,
      image_file: { file_id: 'file_123' },
    });
  });

  it('returns non-array input unchanged', () => {
    const notArray = 'not an array';
    /** @ts-expect-error - Testing invalid input */
    const result = stripBedrockCacheControl(notArray);
    expect(result).toBe('not an array');
  });
});

describe('Multi-agent provider interoperability', () => {
  it('strips Bedrock cache before applying Anthropic cache (single pass)', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          { type: ContentTypes.TEXT, text: 'First message' },
          { cachePoint: { type: 'default' } },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Response' },
          { cachePoint: { type: 'default' } },
        ],
      },
    ];

    /** @ts-expect-error - Testing cross-provider compatibility */
    const result = addCacheControl(messages);

    const firstContent = result[0].content as MessageContentComplex[];

    expect(firstContent.some((b) => 'cachePoint' in b)).toBe(false);
    expect('cache_control' in firstContent[0]).toBe(true);
  });

  it('strips Anthropic cache before applying Bedrock cache (single pass)', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'First message',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'Response',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
    ];

    const result = addBedrockCacheControl(messages);

    const firstContent = result[0].content as MessageContentComplex[];
    const secondContent = result[1].content as MessageContentComplex[];

    expect('cache_control' in firstContent[0]).toBe(false);
    expect('cache_control' in secondContent[0]).toBe(false);

    expect(firstContent.some((b) => 'cachePoint' in b)).toBe(true);
    expect(secondContent.some((b) => 'cachePoint' in b)).toBe(true);
  });

  it('strips Bedrock cache using separate function (backwards compat)', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          { type: ContentTypes.TEXT, text: 'First message' },
          { cachePoint: { type: 'default' } },
        ],
      },
    ];

    const stripped = stripBedrockCacheControl(messages);
    const firstContent = stripped[0].content as MessageContentComplex[];

    expect(firstContent.some((b) => 'cachePoint' in b)).toBe(false);
    expect(firstContent.length).toBe(1);
  });

  it('strips Anthropic cache using separate function (backwards compat)', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'First message',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
    ];

    const stripped = stripAnthropicCacheControl(messages);
    const firstContent = stripped[0].content as MessageContentComplex[];

    expect('cache_control' in firstContent[0]).toBe(false);
  });
});

describe('Immutability - addCacheControl does not mutate original messages', () => {
  it('should not mutate original messages when adding cache control to string content', () => {
    const originalMessages: TestMsg[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
      { role: 'user', content: 'How are you?' },
    ];

    const originalFirstContent = originalMessages[0].content;
    const originalThirdContent = originalMessages[2].content;

    const result = addCacheControl(originalMessages as never);

    expect(originalMessages[0].content).toBe(originalFirstContent);
    expect(originalMessages[2].content).toBe(originalThirdContent);
    expect(typeof originalMessages[0].content).toBe('string');
    expect(typeof originalMessages[2].content).toBe('string');

    expect(Array.isArray(result[0].content)).toBe(true);
    expect(Array.isArray(result[2].content)).toBe(true);
  });

  it('should not mutate original messages when adding cache control to array content', () => {
    const originalMessages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Hello' }],
      },
      { role: 'assistant', content: 'Hi there' },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'How are you?' }],
      },
    ];

    const originalFirstBlock = {
      ...(originalMessages[0].content as MessageContentComplex[])[0],
    };
    const originalThirdBlock = {
      ...(originalMessages[2].content as MessageContentComplex[])[0],
    };

    const result = addCacheControl(originalMessages as never);

    const firstContent = originalMessages[0].content as MessageContentComplex[];
    const thirdContent = originalMessages[2].content as MessageContentComplex[];

    expect('cache_control' in firstContent[0]).toBe(false);
    expect('cache_control' in thirdContent[0]).toBe(false);
    expect(firstContent[0]).toEqual(originalFirstBlock);
    expect(thirdContent[0]).toEqual(originalThirdBlock);

    const resultFirstContent = result[0].content as MessageContentComplex[];
    const resultThirdContent = result[2].content as MessageContentComplex[];
    expect('cache_control' in resultFirstContent[0]).toBe(true);
    expect('cache_control' in resultThirdContent[0]).toBe(true);
  });

  it('should not mutate original messages when stripping existing cache control', () => {
    const originalMessages: TestMsg[] = [
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'Hello',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
      { role: 'assistant', content: 'Hi there' },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'How are you?' }],
      },
    ];

    const originalFirstBlock = (
      originalMessages[0].content as MessageContentComplex[]
    )[0];

    addCacheControl(originalMessages as never);

    expect('cache_control' in originalFirstBlock).toBe(true);
  });

  it('should remove lc_kwargs to prevent serialization mismatch for LangChain messages', () => {
    type LangChainLikeMsg = TestMsg & {
      lc_kwargs?: { content?: MessageContentComplex[] };
    };

    const messagesWithLcKwargs: LangChainLikeMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'First user message' }],
        lc_kwargs: {
          content: [{ type: ContentTypes.TEXT, text: 'First user message' }],
        },
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'Assistant response' }],
        lc_kwargs: {
          content: [{ type: ContentTypes.TEXT, text: 'Assistant response' }],
        },
      },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Second user message' }],
        lc_kwargs: {
          content: [{ type: ContentTypes.TEXT, text: 'Second user message' }],
        },
      },
    ];

    const result = addCacheControl(messagesWithLcKwargs as never);

    const resultFirst = result[0] as LangChainLikeMsg;
    const resultThird = result[2] as LangChainLikeMsg;

    expect(resultFirst.lc_kwargs).toBeUndefined();
    expect(resultThird.lc_kwargs).toBeUndefined();

    const firstContent = resultFirst.content as MessageContentComplex[];
    expect('cache_control' in firstContent[0]).toBe(true);

    const originalFirst = messagesWithLcKwargs[0];
    const originalContent = originalFirst.content as MessageContentComplex[];
    const originalLcContent = originalFirst.lc_kwargs
      ?.content as MessageContentComplex[];
    expect('cache_control' in originalContent[0]).toBe(false);
    expect('cache_control' in originalLcContent[0]).toBe(false);
  });
});

describe('Immutability - addBedrockCacheControl does not mutate original messages', () => {
  it('should not mutate original messages when adding cache points to string content', () => {
    const originalMessages: TestMsg[] = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
    ];

    const originalFirstContent = originalMessages[0].content;
    const originalSecondContent = originalMessages[1].content;

    const result = addBedrockCacheControl(originalMessages);

    expect(originalMessages[0].content).toBe(originalFirstContent);
    expect(originalMessages[1].content).toBe(originalSecondContent);
    expect(typeof originalMessages[0].content).toBe('string');
    expect(typeof originalMessages[1].content).toBe('string');

    expect(Array.isArray(result[0].content)).toBe(true);
    expect(Array.isArray(result[1].content)).toBe(true);
  });

  it('should not mutate original messages when adding cache points to array content', () => {
    const originalMessages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Hello' }],
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'Hi there' }],
      },
    ];

    const originalFirstContentLength = (
      originalMessages[0].content as MessageContentComplex[]
    ).length;
    const originalSecondContentLength = (
      originalMessages[1].content as MessageContentComplex[]
    ).length;

    const result = addBedrockCacheControl(originalMessages);

    const firstContent = originalMessages[0].content as MessageContentComplex[];
    const secondContent = originalMessages[1]
      .content as MessageContentComplex[];

    expect(firstContent.length).toBe(originalFirstContentLength);
    expect(secondContent.length).toBe(originalSecondContentLength);
    expect(firstContent.some((b) => 'cachePoint' in b)).toBe(false);
    expect(secondContent.some((b) => 'cachePoint' in b)).toBe(false);

    const resultFirstContent = result[0].content as MessageContentComplex[];
    const resultSecondContent = result[1].content as MessageContentComplex[];
    expect(resultFirstContent.length).toBe(originalFirstContentLength + 1);
    expect(resultSecondContent.length).toBe(originalSecondContentLength + 1);
    expect(resultFirstContent.some((b) => 'cachePoint' in b)).toBe(true);
    expect(resultSecondContent.some((b) => 'cachePoint' in b)).toBe(true);
  });

  it('should not mutate original messages when stripping existing cache control', () => {
    const originalMessages: TestMsg[] = [
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'Hello',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Hi there' },
          { cachePoint: { type: 'default' } },
        ],
      },
    ];

    const originalFirstBlock = (
      originalMessages[0].content as MessageContentComplex[]
    )[0];
    const originalSecondContentLength = (
      originalMessages[1].content as MessageContentComplex[]
    ).length;

    addBedrockCacheControl(originalMessages);

    expect('cache_control' in originalFirstBlock).toBe(true);
    expect(
      (originalMessages[1].content as MessageContentComplex[]).length
    ).toBe(originalSecondContentLength);
  });

  it('should allow different providers to process same messages without cross-contamination', () => {
    const sharedMessages: TestMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Shared message 1' }],
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'Shared response 1' }],
      },
    ];

    const bedrockResult = addBedrockCacheControl(sharedMessages);

    const anthropicResult = addCacheControl(sharedMessages as never);

    const originalFirstContent = sharedMessages[0]
      .content as MessageContentComplex[];
    expect(originalFirstContent.some((b) => 'cachePoint' in b)).toBe(false);
    expect('cache_control' in originalFirstContent[0]).toBe(false);

    const bedrockFirstContent = bedrockResult[0]
      .content as MessageContentComplex[];
    expect(bedrockFirstContent.some((b) => 'cachePoint' in b)).toBe(true);
    expect('cache_control' in bedrockFirstContent[0]).toBe(false);

    const anthropicFirstContent = anthropicResult[0]
      .content as MessageContentComplex[];
    expect(anthropicFirstContent.some((b) => 'cachePoint' in b)).toBe(false);
    expect('cache_control' in anthropicFirstContent[0]).toBe(true);
  });

  it('should remove lc_kwargs to prevent serialization mismatch for LangChain messages', () => {
    type LangChainLikeMsg = TestMsg & {
      lc_kwargs?: { content?: MessageContentComplex[] };
    };

    const messagesWithLcKwargs: LangChainLikeMsg[] = [
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'User message' }],
        lc_kwargs: {
          content: [{ type: ContentTypes.TEXT, text: 'User message' }],
        },
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'Assistant response' }],
        lc_kwargs: {
          content: [{ type: ContentTypes.TEXT, text: 'Assistant response' }],
        },
      },
    ];

    const bedrockResult = addBedrockCacheControl(messagesWithLcKwargs);

    const resultFirst = bedrockResult[0] as LangChainLikeMsg;
    const resultSecond = bedrockResult[1] as LangChainLikeMsg;

    expect(resultFirst.lc_kwargs).toBeUndefined();
    expect(resultSecond.lc_kwargs).toBeUndefined();

    const firstContent = resultFirst.content as MessageContentComplex[];
    expect(firstContent.some((b) => 'cachePoint' in b)).toBe(true);

    const originalFirst = messagesWithLcKwargs[0];
    const originalContent = originalFirst.content as MessageContentComplex[];
    const originalLcContent = originalFirst.lc_kwargs
      ?.content as MessageContentComplex[];
    expect(originalContent.some((b) => 'cachePoint' in b)).toBe(false);
    expect(originalLcContent.some((b) => 'cachePoint' in b)).toBe(false);
  });
});

describe('Multi-turn cache cleanup', () => {
  it('strips stale Bedrock cache points from previous turns before applying new ones', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          { type: ContentTypes.TEXT, text: 'Turn 1 message 1' },
          { cachePoint: { type: 'default' } },
        ],
      },
      {
        role: 'assistant',
        content: [
          { type: ContentTypes.TEXT, text: 'Turn 1 response 1' },
          { cachePoint: { type: 'default' } },
        ],
      },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Turn 2 message 2' }],
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'Turn 2 response 2' }],
      },
    ];

    const result = addBedrockCacheControl(messages);

    const cachePointCount = result.reduce((count, msg) => {
      if (Array.isArray(msg.content)) {
        return (
          count +
          msg.content.filter(
            (block) => 'cachePoint' in block && !('type' in block)
          ).length
        );
      }
      return count;
    }, 0);

    expect(cachePointCount).toBe(2);

    const lastContent = result[3].content as MessageContentComplex[];
    const secondLastContent = result[2].content as MessageContentComplex[];

    expect(lastContent.some((b) => 'cachePoint' in b)).toBe(true);
    expect(secondLastContent.some((b) => 'cachePoint' in b)).toBe(true);

    const firstContent = result[0].content as MessageContentComplex[];
    const secondContent = result[1].content as MessageContentComplex[];

    expect(firstContent.some((b) => 'cachePoint' in b)).toBe(false);
    expect(secondContent.some((b) => 'cachePoint' in b)).toBe(false);
  });

  it('strips stale Anthropic cache_control from previous turns before applying new ones', () => {
    const messages: TestMsg[] = [
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'Turn 1 message 1',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'Turn 1 response 1' }],
      },
      {
        role: 'user',
        content: [
          {
            type: ContentTypes.TEXT,
            text: 'Turn 2 message 2',
            cache_control: { type: 'ephemeral' },
          } as MessageContentComplex,
        ],
      },
      {
        role: 'assistant',
        content: [{ type: ContentTypes.TEXT, text: 'Turn 2 response 2' }],
      },
      {
        role: 'user',
        content: [{ type: ContentTypes.TEXT, text: 'Turn 3 message 3' }],
      },
    ];

    /** @ts-expect-error - Testing cross-provider compatibility */
    const result = addCacheControl(messages);

    const cacheControlCount = result.reduce((count, msg) => {
      if (Array.isArray(msg.content)) {
        return (
          count +
          msg.content.filter(
            (block) => 'cache_control' in block && 'type' in block
          ).length
        );
      }
      return count;
    }, 0);

    expect(cacheControlCount).toBe(2);

    const lastContent = result[4].content as MessageContentComplex[];
    const thirdContent = result[2].content as MessageContentComplex[];

    expect('cache_control' in lastContent[0]).toBe(true);
    expect('cache_control' in thirdContent[0]).toBe(true);

    const firstContent = result[0].content as MessageContentComplex[];

    expect('cache_control' in firstContent[0]).toBe(false);
  });
});
