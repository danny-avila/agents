import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { _convertMessagesToAnthropicPayload } from './message_inputs';

/**
 * Regression for cross-provider agent handoffs (e.g. Bedrock → Anthropic): a
 * Bedrock turn that used extended thinking leaves a `reasoning_content` content
 * block ({ reasoningText: { text, signature } }) in the history. The official
 * Anthropic converter has no branch for it and previously threw
 * "Unsupported message content format", crashing the handoff. Only known
 * foreign reasoning (Bedrock `reasoning_content`, Google `reasoning`, LibreChat
 * `think`) is dropped; any other unknown block still throws rather than being
 * silently omitted (real content — user media, Google code-execution — must be
 * surfaced); and a tool call carried only on `tool_calls` survives dropping its
 * reasoning sibling without being duplicated.
 */
type AnthropicPayload = ReturnType<typeof _convertMessagesToAnthropicPayload>;

/** Minimal view of a converted Anthropic content block the assertions read. */
interface TestBlock {
  type?: string;
  text?: string;
}

const findAssistant = (payload: AnthropicPayload) =>
  payload.messages.find((m) => m.role === 'assistant');

const assistantBlocks = (payload: AnthropicPayload): TestBlock[] => {
  const content = findAssistant(payload)?.content;
  return Array.isArray(content) ? (content as TestBlock[]) : [];
};

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
        },
        { type: 'text', text: 'Kicking off the searches now.' },
        {
          type: 'tool_use',
          id: 'tooluse_abc',
          name: 'notion-search',
          input: { query: 'Assort Health' },
        },
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
    expect(findAssistant(payload)).toBeDefined();
    const blocks = assistantBlocks(payload);

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
          { type: 'reasoning', reasoning: 'internal google chain of thought' },
          { type: 'text', text: 'Hello!' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const blocks = assistantBlocks(_convertMessagesToAnthropicPayload(history));
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
          { type: 'think', think: 'librechat serialized reasoning' },
          { type: 'text', text: 'Done.' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const blocks = assistantBlocks(_convertMessagesToAnthropicPayload(history));
    expect(blocks.find((b) => b.type === 'think')).toBeUndefined();
    expect(blocks.some((b) => b.type === 'text' && b.text === 'Done.')).toBe(
      true
    );
  });

  it('drops an unsigned `thinking` block (Google thinking-enabled output) on an assistant turn', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'thinking',
            thinking: 'google chain of thought, no signature',
          },
          { type: 'text', text: 'Answer.' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const blocks = assistantBlocks(_convertMessagesToAnthropicPayload(history));
    expect(blocks.find((b) => b.type === 'thinking')).toBeUndefined();
    expect(blocks.some((b) => b.type === 'text' && b.text === 'Answer.')).toBe(
      true
    );
  });

  it('forwards a signed `thinking` block (Anthropic-native) unchanged', () => {
    const history: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'thinking',
            thinking: 'native reasoning',
            signature: 'valid-sig',
          },
          { type: 'text', text: 'Answer.' },
        ],
      }),
    ];
    const blocks = assistantBlocks(_convertMessagesToAnthropicPayload(history));
    expect(blocks.find((b) => b.type === 'thinking')).toMatchObject({
      type: 'thinking',
      thinking: 'native reasoning',
      signature: 'valid-sig',
    });
  });

  it('throws (not silently drops) on an unknown assistant block such as Google code execution', () => {
    // executableCode/codeExecutionResult carry real visible content; silently
    // dropping them on a Google → Anthropic handoff would lose evidence.
    const history: BaseMessage[] = [
      new HumanMessage('run some code'),
      new AIMessage({
        content: [
          {
            type: 'executableCode',
            executableCode: { language: 'PYTHON', code: 'print(2+2)' },
          },
          { type: 'text', text: 'Here is the result.' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).toThrow(
      'Unsupported message content format'
    );
  });

  it('throws (not silently drops) on an unsupported user block such as media', () => {
    const history: BaseMessage[] = [
      new HumanMessage({
        content: [
          {
            type: 'video_url',
            video_url: { url: 'https://example.com/v.mp4' },
          },
          { type: 'text', text: 'what is in this video?' },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).toThrow(
      'Unsupported message content format'
    );
  });

  it('does not drop a reasoning-typed block on a user turn (only assistant reasoning is dropped)', () => {
    const history: BaseMessage[] = [
      new HumanMessage({
        content: [
          { type: 'reasoning_content', reasoningText: { text: 'user text' } },
          { type: 'text', text: 'hello' },
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
          },
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
    const blocks = assistantBlocks(_convertMessagesToAnthropicPayload(history));
    expect(blocks.find((b) => b.type === 'reasoning_content')).toBeUndefined();
    expect(blocks.find((b) => b.type === 'tool_use')).toMatchObject({
      type: 'tool_use',
      id: 'tooluse_transfer',
      name: 'lc_transfer_to_data_agent',
      input: { reason: 'need consumption data' },
    });
    // The `_` placeholder must not linger once a real tool_use block is present.
    expect(blocks.some((b) => b.type === 'text' && b.text === '_')).toBe(false);
  });

  it('does not duplicate a Google functionCall tool call already materialized by _formatContent', () => {
    // _formatContent converts the `functionCall` part into a tool_use; the
    // materialization must recognize it as represented and not append a second.
    const history: BaseMessage[] = [
      new HumanMessage('weather in SF?'),
      new AIMessage({
        content: [
          {
            type: 'functionCall',
            functionCall: { name: 'get_weather', args: { city: 'SF' } },
          },
        ],
        tool_calls: [
          {
            id: 'call_weather_1',
            name: 'get_weather',
            args: { city: 'SF' },
            type: 'tool_call',
          },
        ],
      }),
    ];
    const blocks = assistantBlocks(_convertMessagesToAnthropicPayload(history));
    const toolUses = blocks.filter((b) => b.type === 'tool_use');
    expect(toolUses).toHaveLength(1);
    expect(toolUses[0]).toMatchObject({
      type: 'tool_use',
      id: 'call_weather_1',
      name: 'get_weather',
    });
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
          },
        ],
      }),
    ];
    expect(() => _convertMessagesToAnthropicPayload(history)).not.toThrow();
    const blocks = assistantBlocks(_convertMessagesToAnthropicPayload(history));
    expect(blocks.find((b) => b.type === 'reasoning_content')).toBeUndefined();
    expect(blocks.length).toBeGreaterThan(0);
    expect(blocks.every((b) => b.type === 'text')).toBe(true);
  });
});
