/* eslint-disable @typescript-eslint/no-explicit-any */
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { _convertMessagesToAnthropicPayload } from './message_inputs';

/**
 * Regression for cross-provider agent handoffs (e.g. Bedrock → Anthropic): a
 * Bedrock turn that used extended thinking leaves a `reasoning_content` content
 * block ({ reasoningText: { text, signature } }) in the history. The official
 * Anthropic converter has no branch for it and previously threw
 * "Unsupported message content format", crashing the handoff. Foreign reasoning
 * (Bedrock `reasoning_content`, Google `reasoning`, LibreChat `think`) is
 * dropped; an unknown block degrades gracefully on assistant turns but still
 * throws on user/tool input (real content must not be silently omitted); and a
 * tool call carried only on `tool_calls` survives dropping its reasoning sibling.
 */
describe('_convertMessagesToAnthropicPayload — cross-provider reasoning blocks', () => {
  const bedrockHandoffHistory = (): BaseMessage[] => [
    new HumanMessage('research Assort Health'),
    new AIMessage({
      content: [
        {
          type: 'reasoning_content',
          index: 0,
          reasoningText: {
            text: 'Let me search Notion then hand off to the data agent.',
            signature: 'bedrock-signature-not-valid-for-anthropic',
          },
        } as any,
        { type: 'text', text: 'Kicking off the searches now.' },
        {
          type: 'tool_use',
          id: 'tooluse_abc',
          name: 'notion-search',
          input: { query: 'Assort Health' },
        } as any,
      ],
      tool_calls: [
        {
          id: 'tooluse_abc',
          name: 'notion-search',
          args: { query: 'Assort Health' },
          type: 'tool_call',
        },
      ],
    }),
  ];

  it('does not throw on a Bedrock reasoning_content block', () => {
    expect(() =>
      _convertMessagesToAnthropicPayload(bedrockHandoffHistory())
    ).not.toThrow();
  });

  it('drops reasoning_content (incl. its foreign signature) but keeps text and tool_use', () => {
    const payload = _convertMessagesToAnthropicPayload(bedrockHandoffHistory());
    const assistant = payload.messages.find((m: any) => m.role === 'assistant');
    expect(assistant).toBeDefined();
    const blocks = assistant!.content as any[];

    expect(blocks.find((b) => b.type === 'reasoning_content')).toBeUndefined();
    expect(
      blocks.find(
        (b) => b.type === 'thinking' || b.type === 'redacted_thinking'
      )
    ).toBeUndefined();
    expect(JSON.stringify(blocks)).not.toContain(
      'bedrock-signature-not-valid-for-anthropic'
    );

    expect(
      blocks.some(
        (b) => b.type === 'text' && b.text === 'Kicking off the searches now.'
      )
    ).toBe(true);
    expect(blocks.find((b) => b.type === 'tool_use')).toMatchObject({
      type: 'tool_use',
      id: 'tooluse_abc',
      name: 'notion-search',
      input: { query: 'Assort Health' },
    });
  });

  it('drops a Google `reasoning` block without throwing', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning',
            reasoning: 'internal google chain of thought',
          } as any,
          { type: 'text', text: 'Hello!' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const assistant = _convertMessagesToAnthropicPayload(history).messages.find(
      (m: any) => m.role === 'assistant'
    );
    const blocks = assistant!.content as any[];
    expect(blocks.find((b) => b.type === 'reasoning')).toBeUndefined();
    expect(blocks.some((b) => b.type === 'text' && b.text === 'Hello!')).toBe(
      true
    );
  });

  it('drops a LibreChat `think` block without throwing', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'think', think: 'librechat serialized reasoning' } as any,
          { type: 'text', text: 'Done.' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const assistant = _convertMessagesToAnthropicPayload(history).messages.find(
      (m: any) => m.role === 'assistant'
    );
    const blocks = assistant!.content as any[];
    expect(blocks.find((b) => b.type === 'think')).toBeUndefined();
    expect(blocks.some((b) => b.type === 'text' && b.text === 'Done.')).toBe(
      true
    );
  });

  it('on an assistant turn, drops a genuinely unknown block instead of throwing', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'some_future_block_type', foo: 'bar' } as any,
          { type: 'text', text: 'Still here.' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const assistant = _convertMessagesToAnthropicPayload(history).messages.find(
      (m: any) => m.role === 'assistant'
    );
    const blocks = assistant!.content as any[];
    expect(
      blocks.find((b) => b.type === 'some_future_block_type')
    ).toBeUndefined();
    expect(
      blocks.some((b) => b.type === 'text' && b.text === 'Still here.')
    ).toBe(true);
  });

  it('on a user turn, throws on an unsupported block rather than silently dropping it', () => {
    const history: BaseMessage[] = [
      new HumanMessage({
        content: [
          {
            type: 'video_url',
            video_url: { url: 'https://example.com/v.mp4' },
          } as any,
          { type: 'text', text: 'what is in this video?' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).toThrow(
      'Unsupported message content format'
    );
  });

  it('preserves a tool call carried only on tool_calls when its reasoning sibling is dropped', () => {
    // Mirrors a Bedrock extended-thinking turn: the tool lives only on
    // `tool_calls`; `content` holds just the reasoning block (no tool_use).
    const history: BaseMessage[] = [
      new HumanMessage('research Assort Health'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { text: 'I should hand off now.', signature: 'sig' },
          } as any,
        ],
        tool_calls: [
          {
            id: 'tooluse_transfer',
            name: 'lc_transfer_to_data_agent',
            args: { reason: 'need consumption data' },
            type: 'tool_call',
          },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const assistant = _convertMessagesToAnthropicPayload(history).messages.find(
      (m: any) => m.role === 'assistant'
    );
    const blocks = assistant!.content as any[];
    expect(blocks.find((b) => b.type === 'reasoning_content')).toBeUndefined();
    const toolUse = blocks.find((b) => b.type === 'tool_use');
    expect(toolUse).toMatchObject({
      type: 'tool_use',
      id: 'tooluse_transfer',
      name: 'lc_transfer_to_data_agent',
      input: { reason: 'need consumption data' },
    });
    // The `_` placeholder must not linger once a real tool_use block is present.
    expect(blocks.some((b) => b.type === 'text' && b.text === '_')).toBe(false);
  });

  it('falls back to placeholder text when reasoning was the only content', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: {
              text: 'only thinking, no visible text',
              signature: 'sig',
            },
          } as any,
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const assistant = _convertMessagesToAnthropicPayload(history).messages.find(
      (m: any) => m.role === 'assistant'
    );
    const blocks = assistant!.content as any[];
    expect(blocks.find((b) => b.type === 'reasoning_content')).toBeUndefined();
    expect(blocks.length).toBeGreaterThan(0);
    expect(blocks.every((b) => b.type === 'text')).toBe(true);
  });
});
