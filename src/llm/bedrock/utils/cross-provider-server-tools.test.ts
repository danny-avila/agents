import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { convertToConverseMessages } from './message_inputs';

/**
 * Regression for cross-provider agent handoffs (Google → Bedrock): a Gemini
 * turn that used a server-side tool (URL context, Google Search) leaves
 * `toolCall`/`toolResponse` content blocks in history. The Bedrock Converse
 * converter has no branch for them and previously threw
 * "Unsupported content block type: toolCall", crashing the handoff. Only
 * Google can execute these blocks or validate their thought signatures, so
 * they are dropped on assistant turns; any other unknown block still throws.
 */
type ConverseResult = ReturnType<typeof convertToConverseMessages>;

/** Minimal view of a converted Bedrock Converse content block the assertions read. */
interface ConverseBlock {
  text?: string;
  toolUse?: {
    toolUseId?: string;
    name?: string;
    input?: Record<string, string>;
  };
}

const assistantContent = (result: ConverseResult): ConverseBlock[] => {
  const msg = result.converseMessages.find((m) => m.role === 'assistant');
  return (msg?.content ?? []) as ConverseBlock[];
};

describe('convertToConverseMessages — Google server-side tool blocks (Google → Bedrock)', () => {
  it('drops toolCall/toolResponse on an assistant turn, keeping text and tool calls', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('summarize this article'),
      new AIMessage({
        content: [
          {
            type: 'toolCall',
            thoughtSignature: 'google-signature-not-valid-for-bedrock',
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
        tool_calls: [
          {
            id: 'call_client_tool',
            name: 'save_note',
            args: { note: 'DCA summary' },
            type: 'tool_call',
          },
        ],
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));

    const serialized = JSON.stringify(content);
    expect(serialized).not.toContain('URL_CONTEXT');
    expect(serialized).not.toContain('google-signature-not-valid-for-bedrock');

    expect(
      content.some(
        (b) => b.text === 'The article argues for dollar-cost averaging.'
      )
    ).toBe(true);
    const toolUse = content.find((b) => b.toolUse != null);
    expect(toolUse?.toolUse).toMatchObject({
      toolUseId: 'call_client_tool',
      name: 'save_note',
      input: { note: 'DCA summary' },
    });
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
    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.length).toBeGreaterThan(0);
    expect(content.every((b) => typeof b.text === 'string')).toBe(true);
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
    expect(() => convertToConverseMessages(messages)).toThrow(
      'Unsupported content block type'
    );
  });
});
