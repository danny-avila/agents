import { AIMessage, HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import { convertToConverseMessages } from './message_inputs';

/**
 * Native-Bedrock reasoning serialization. A `reasoning_content` block whose
 * `reasoningText.text` is null/empty (e.g. a signature-only block that never
 * merged with its text) is invalid for Bedrock Converse — it rejects with
 * `...reasoningContent.reasoningText.text ... Member must not be null`. Such a
 * block must be dropped on replay rather than sent; a block carrying real text
 * is still converted.
 */
type ConverseResult = ReturnType<typeof convertToConverseMessages>;

/** Minimal view of a converted Bedrock Converse content block the assertions read. */
interface ConverseBlock {
  text?: string;
  reasoningContent?: { reasoningText?: { text?: string; signature?: string } };
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

describe('convertToConverseMessages — native Bedrock reasoning serialization', () => {
  it('drops a signature-only reasoning block, keeping text and tool calls', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('what data do you have?'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { signature: 'sig-abc' },
          },
          { type: 'text', text: 'Let me check your databases.' },
        ],
        tool_calls: [
          {
            id: 'tooluse_list',
            name: 'list_databases',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));

    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(JSON.stringify(content)).not.toContain('sig-abc');
    expect(content.some((b) => b.text === 'Let me check your databases.')).toBe(
      true
    );
    const toolUse = content.find((b) => b.toolUse != null);
    expect(toolUse?.toolUse).toMatchObject({
      toolUseId: 'tooluse_list',
      name: 'list_databases',
    });
  });

  it('drops a reasoning block whose text is empty', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { text: '', signature: 'sig' },
          },
          { type: 'text', text: 'answer' },
        ],
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });

  it('emits a placeholder (not empty content) when the only block is a signature-only reasoning block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning_content', reasoningText: { signature: 'sig' } },
        ],
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.length).toBeGreaterThan(0);
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.every((b) => typeof b.text === 'string')).toBe(true);
  });

  it('still converts a reasoning block that carries text (not dropped)', () => {
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
          },
          { type: 'text', text: 'answer' },
        ],
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    const reasoning = content.find((b) => b.reasoningContent != null);
    expect(reasoning).toBeDefined();
    expect(reasoning?.reasoningContent?.reasoningText?.text).toBe(
      'native bedrock reasoning'
    );
  });
});

/**
 * Same failure class, v1 converter path. Assistant messages carrying
 * `response_metadata.output_version === 'v1'` are converted by
 * `convertFromV1ToChatBedrockConverseMessage`, which serialized `reasoning` /
 * `reasoning_content` blocks without the null/empty-text guard applied to the
 * non-v1 path — a `reasoning` block whose `reasoning` is null/empty (e.g. a
 * model responding with `thinking.display: "omitted"`, the Opus 4.7+ /
 * Sonnet 5 default) reached Bedrock as `reasoningText: { text: null }` and the
 * whole request was rejected with `Member must not be null`.
 */
describe('convertToConverseMessages — v1 reasoning serialization', () => {
  const v1Metadata = {
    output_version: 'v1',
    model_provider: 'anthropic',
  } as const;

  it('drops a v1 reasoning block whose reasoning text is missing, keeping text and tool calls', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('what data do you have?'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: undefined } as never,
          { type: 'text', text: 'Let me check your databases.' },
        ],
        tool_calls: [
          {
            id: 'tooluse_list',
            name: 'list_databases',
            args: {},
            type: 'tool_call',
          },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));

    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.some((b) => b.text === 'Let me check your databases.')).toBe(
      true
    );
    const toolUse = content.find((b) => b.toolUse != null);
    expect(toolUse?.toolUse).toMatchObject({
      toolUseId: 'tooluse_list',
      name: 'list_databases',
    });
  });

  it('drops a v1 reasoning block whose reasoning text is empty', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: '' },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });

  it('drops a v1 signature-only reasoning_content block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { signature: 'sig-abc' },
          },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(JSON.stringify(content)).not.toContain('sig-abc');
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });

  it('emits a placeholder (not empty content) when dropping empties a v1 turn', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [{ type: 'reasoning', reasoning: undefined } as never],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).not.toThrow();
    const content = assistantContent(convertToConverseMessages(messages));
    expect(content.length).toBeGreaterThan(0);
    expect(content.find((b) => b.reasoningContent != null)).toBeUndefined();
    expect(content.every((b) => typeof b.text === 'string')).toBe(true);
  });

  it.each(['', ' \n '])(
    'emits a placeholder for invalid v1 text %j',
    (text) => {
      const messages: BaseMessage[] = [
        new HumanMessage('hi'),
        new AIMessage({
          content: [{ type: 'text', text }],
          response_metadata: v1Metadata,
        }),
      ];

      const content = assistantContent(convertToConverseMessages(messages));
      expect(content).toEqual([{ text: '_' }]);
    }
  );

  it('merges whitespace-only v1 text into the preceding text block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'text', text: 'answer' },
          { type: 'text', text: ' \n ' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(content).toEqual([{ text: 'answer \n ' }]);
  });

  it('merges split v1 reasoning_content text and signature blocks before serialization', () => {
    const splitContent = [
      {
        type: 'reasoning_content',
        reasoningText: { text: 'first ' },
      },
      {
        type: 'reasoning_content',
        reasoningText: { text: 'second' },
      },
      {
        type: 'reasoning_content',
        reasoningText: { signature: 'sig-abc' },
      },
      { type: 'text', text: 'answer' },
    ];
    const originalContent = structuredClone(splitContent);
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: splitContent,
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    const reasoning = content.filter((b) => b.reasoningContent != null);
    expect(reasoning).toHaveLength(1);
    expect(reasoning[0].reasoningContent?.reasoningText).toEqual({
      text: 'first second',
      signature: 'sig-abc',
    });
    expect(splitContent).toEqual(originalContent);
  });

  it('keeps independently signed v1 reasoning blocks separate', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'reasoning_content',
            reasoningText: { text: 'first', signature: 'sig-first' },
          },
          {
            type: 'reasoning_content',
            reasoningText: { text: 'second', signature: 'sig-second' },
          },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    expect(
      content
        .filter((block) => block.reasoningContent != null)
        .map((block) => block.reasoningContent?.reasoningText)
    ).toEqual([
      { text: 'first', signature: 'sig-first' },
      { text: 'second', signature: 'sig-second' },
    ]);
  });

  it('throws instead of returning empty assistant content for an unhandled v1 block', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          {
            type: 'image',
            source_type: 'base64',
            data: 'aGVsbG8=',
            mime_type: 'image/png',
          } as never,
        ],
        response_metadata: v1Metadata,
      }),
    ];

    expect(() => convertToConverseMessages(messages)).toThrow(
      'Unsupported v1 content block type: image'
    );
  });

  it('still converts v1 reasoning and reasoning_content blocks that carry text', () => {
    const messages: BaseMessage[] = [
      new HumanMessage('hi'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: 'v1 standard reasoning' },
          {
            type: 'reasoning_content',
            reasoningText: { text: 'native reasoning', signature: 'sig' },
          },
          { type: 'text', text: 'answer' },
        ],
        response_metadata: v1Metadata,
      }),
    ];

    const content = assistantContent(convertToConverseMessages(messages));
    const reasoningTexts = content
      .filter((b) => b.reasoningContent != null)
      .map((b) => b.reasoningContent?.reasoningText?.text);
    expect(reasoningTexts).toEqual([
      'v1 standard reasoning',
      'native reasoning',
    ]);
    expect(content.some((b) => b.text === 'answer')).toBe(true);
  });
});
