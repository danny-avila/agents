import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { _convertMessagesToAnthropicPayload } from './message_inputs';

/**
 * Regression for cross-provider agent handoffs (Google → Anthropic): a Gemini
 * turn that used a server-side tool (URL context, Google Search) leaves
 * `toolCall`/`toolResponse` content blocks in history. The Anthropic converter
 * has no branch for them and previously threw
 * "Unsupported message content format", crashing the handoff. Only Google can
 * execute these blocks or validate their thought signatures, so they are
 * dropped on assistant turns; any other unknown block still throws.
 */
type AnthropicPayload = ReturnType<typeof _convertMessagesToAnthropicPayload>;

/** Minimal view of a converted Anthropic content block the assertions read. */
interface TestBlock {
  type?: string;
  text?: string;
}

const assistantBlocks = (payload: AnthropicPayload): TestBlock[] => {
  const content = payload.messages.find((m) => m.role === 'assistant')?.content;
  return Array.isArray(content) ? (content as TestBlock[]) : [];
};

describe('_convertMessagesToAnthropicPayload — Google server-side tool blocks', () => {
  it('drops toolCall/toolResponse on an assistant turn, keeping text', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('summarize this article'),
      new AIMessage({
        content: [
          {
            type: 'toolCall',
            thoughtSignature: 'google-signature-not-valid-for-anthropic',
            toolCall: {
              toolType: 'URL_CONTEXT',
              args: { urls: ['https://example.com/report'] },
              id: 'j7pfyr6k',
            },
          },
          {
            type: 'toolResponse',
            toolResponse: {
              toolType: 'URL_CONTEXT',
              id: 'j7pfyr6k',
              result: { status: 'SUCCESS' },
            },
          },
          {
            type: 'text',
            text: 'The article argues for dollar-cost averaging.',
          },
        ],
      }),
    ];

    expect(() => _convertMessagesToAnthropicPayload(messages)).not.toThrow();
    const blocks = assistantBlocks(
      _convertMessagesToAnthropicPayload(messages)
    );

    const serialized = JSON.stringify(blocks);
    expect(serialized).not.toContain('URL_CONTEXT');
    expect(serialized).not.toContain(
      'google-signature-not-valid-for-anthropic'
    );
    expect(
      blocks.some(
        (b) =>
          b.type === 'text' &&
          b.text === 'The article argues for dollar-cost averaging.'
      )
    ).toBe(true);
  });

  it('emits a placeholder (not empty content) when a server-tool-only turn is fully dropped', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'toolCall',
            toolCall: { toolType: 'URL_CONTEXT', args: { urls: [] }, id: 'x1' },
          },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(messages)).not.toThrow();
    const blocks = assistantBlocks(
      _convertMessagesToAnthropicPayload(messages)
    );
    expect(blocks.length).toBeGreaterThan(0);
  });

  it('still throws on a genuinely unknown assistant block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('run code'),
      new AIMessage({
        content: [
          { type: 'some_future_block_type', foo: 'bar' },
          { type: 'text', text: 'done' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(messages)).toThrow(
      'Unsupported message content format'
    );
  });
});
