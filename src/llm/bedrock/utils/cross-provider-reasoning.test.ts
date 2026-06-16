/* eslint-disable @typescript-eslint/no-explicit-any */
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { convertToConverseMessages } from './message_inputs';

/**
 * Mirror of the Anthropic-side cross-provider reasoning fix, for the reverse
 * handoff (Anthropic → Bedrock). An Anthropic extended-thinking turn leaves
 * `thinking`/`redacted_thinking` blocks in history; the Bedrock Converse
 * converter has no branch for them and previously threw
 * "Unsupported content block type: thinking", crashing the handoff. Bedrock's
 * native reasoning is `reasoning_content` (still converted); foreign reasoning
 * (`thinking`/`redacted_thinking`/`reasoning`/`think`) is dropped on assistant
 * turns, while any other unknown block still throws rather than being silently
 * omitted.
 */
describe('convertToConverseMessages — cross-provider reasoning (Anthropic → Bedrock)', () => {
  it('drops Anthropic thinking/redacted_thinking on an assistant turn, keeping text and tool calls', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('research Assort Health'),
      new AIMessage({
        content: [
          {
            type: 'thinking',
            thinking: 'Let me hand off to the data agent.',
            signature: 'anthropic-signature-not-valid-for-bedrock',
          } as any,
          { type: 'redacted_thinking', data: 'redacted-blob' } as any,
          { type: 'text', text: 'Handing off now.' },
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

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const { converseMessages } = convertToConverseMessages(messages);
    const assistant = converseMessages.find((m) => m.role === 'assistant');
    expect(assistant).toBeDefined();
    const content = assistant!.content as any[];

    expect(content.find((b) => 'reasoningContent' in b)).toBeUndefined();
    expect(JSON.stringify(content)).not.toContain(
      'anthropic-signature-not-valid-for-bedrock'
    );
    expect(JSON.stringify(content)).not.toContain('redacted-blob');

    expect(
      content.some((b) => 'text' in b && b.text === 'Handing off now.')
    ).toBe(true);
    const toolUse = content.find((b) => 'toolUse' in b);
    expect(toolUse?.toolUse).toMatchObject({
      toolUseId: 'tooluse_transfer',
      name: 'lc_transfer_to_data_agent',
      input: { reason: 'need consumption data' },
    });
  });

  it('emits a placeholder (not empty content) when a reasoning-only turn is fully dropped', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'thinking',
            thinking: 'only thinking, no other content',
          } as any,
        ],
      }),
    ];
    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const { converseMessages } = convertToConverseMessages(messages);
    const assistant = converseMessages.find((m) => m.role === 'assistant');
    const content = assistant!.content as any[];
    expect(content.length).toBeGreaterThan(0);
    expect(content.find((b) => 'reasoningContent' in b)).toBeUndefined();
    expect(content.every((b) => 'text' in b)).toBe(true);
  });

  it('still throws on a genuinely unknown assistant block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('run code'),
      new AIMessage({
        content: [
          { type: 'some_future_block_type', foo: 'bar' } as any,
          { type: 'text', text: 'done' },
        ],
      }),
    ];
    expect(() => convertToConverseMessages(messages)).toThrow(
      'Unsupported content block type'
    );
  });

  it('still converts Bedrock-native reasoning_content (not dropped)', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: {
              text: 'native bedrock reasoning',
              signature: 'sig',
            },
          } as any,
          { type: 'text', text: 'answer' },
        ],
      }),
    ];
    const { converseMessages } = convertToConverseMessages(messages);
    const assistant = converseMessages.find((m) => m.role === 'assistant');
    const content = assistant!.content as any[];
    const reasoning = content.find((b) => 'reasoningContent' in b);
    expect(reasoning).toBeDefined();
    expect(reasoning.reasoningContent.reasoningText.text).toBe(
      'native bedrock reasoning'
    );
  });
});
