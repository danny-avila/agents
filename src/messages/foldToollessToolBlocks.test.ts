import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { ExtendedMessageContent } from '@/types';
import { foldToolBlocksForToollessAgent } from './format';

/** Concatenated text across a message's content (string or structured array). */
function getTextContent(msg: {
  content: string | ExtendedMessageContent[];
}): string {
  if (typeof msg.content === 'string') {
    return msg.content;
  }
  if (Array.isArray(msg.content)) {
    return (msg.content as ExtendedMessageContent[])
      .filter((b) => b.type === 'text')
      .map((b) => String(b.text ?? ''))
      .join('\n');
  }
  return '';
}

/** Any residual tool content that a tool-less agent cannot legally send. */
function hasResidualToolContent(messages: BaseMessage[]): boolean {
  return messages.some((m) => {
    if (m instanceof ToolMessage) {
      return true;
    }
    const ai = m as AIMessage;
    if (ai.tool_calls != null && ai.tool_calls.length > 0) {
      return true;
    }
    if (Array.isArray(m.content)) {
      return (m.content as ExtendedMessageContent[]).some(
        (b) => typeof b === 'object' && b.type === 'tool_use'
      );
    }
    return false;
  });
}

describe('foldToolBlocksForToollessAgent', () => {
  test('returns the same array reference when there is no tool content', () => {
    const messages = [
      new SystemMessage('You are helpful.'),
      new HumanMessage('Hi'),
      new AIMessage('Hello!'),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(result).toBe(messages);
  });

  test('folds an AI tool call plus its ToolMessage into one HumanMessage', () => {
    const messages = [
      new HumanMessage('Search my files for "roadmap"'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'file_search',
            args: { query: 'roadmap' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'Found roadmap.md',
        tool_call_id: 'call_1',
        name: 'file_search',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    // Human prompt kept, AI+Tool collapsed into a single HumanMessage.
    expect(result).toHaveLength(2);
    expect(result[0]).toBeInstanceOf(HumanMessage);
    const folded = getTextContent(result[1]);
    expect(folded).toContain('[Previous tool interaction]');
    expect(folded).toContain('file_search');
    expect(folded).toContain('roadmap');
    expect(folded).toContain('Found roadmap.md');
  });

  test('folds historical tool content that precedes the last human turn (the reported bug)', () => {
    const messages = [
      new HumanMessage('Search my files for "roadmap"'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'call_1',
            name: 'file_search',
            args: { query: 'roadmap' },
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: 'Found roadmap.md',
        tool_call_id: 'call_1',
        name: 'file_search',
      }),
      new AIMessage('Here is what I found in roadmap.md.'),
      new HumanMessage('thanks'),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    // The trailing plain-text turns survive untouched.
    const last = result[result.length - 1];
    expect(last).toBeInstanceOf(HumanMessage);
    expect(getTextContent(last)).toBe('thanks');
    expect(
      result.some((m) => getTextContent(m).includes('Here is what I found'))
    ).toBe(true);
  });

  test('folds parallel tool calls and all their results together', () => {
    const messages = [
      new HumanMessage('Look up A and B'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'a', name: 'lookup', args: { key: 'A' }, type: 'tool_call' },
          { id: 'b', name: 'lookup', args: { key: 'B' }, type: 'tool_call' },
        ],
      }),
      new ToolMessage({ content: 'A=1', tool_call_id: 'a', name: 'lookup' }),
      new ToolMessage({ content: 'B=2', tool_call_id: 'b', name: 'lookup' }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    expect(result).toHaveLength(2);
    const folded = getTextContent(result[1]);
    expect(folded).toContain('A=1');
    expect(folded).toContain('B=2');
  });

  test('detects Anthropic-style tool_use content blocks', () => {
    const messages = [
      new HumanMessage('Search'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me search.' },
          {
            type: 'tool_use',
            id: 'call_1',
            name: 'file_search',
            input: { query: 'roadmap' },
          },
        ],
      }),
      new ToolMessage({
        content: 'Found roadmap.md',
        tool_call_id: 'call_1',
        name: 'file_search',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    const folded = getTextContent(result[result.length - 1]);
    expect(folded).toContain('Let me search.');
    expect(folded).toContain('file_search');
  });

  test('preserves image blocks in a tool result instead of stringifying them', () => {
    const messages = [
      new HumanMessage('Render a chart'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'c', name: 'chart', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: [
          { type: 'text', text: 'chart:' },
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,AAAA' },
          },
        ],
        tool_call_id: 'c',
        name: 'chart',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    const foldedContent = result[result.length - 1].content;
    expect(Array.isArray(foldedContent)).toBe(true);
    expect(
      (foldedContent as ExtendedMessageContent[]).some(
        (b) => b.type === 'image_url'
      )
    ).toBe(true);
  });

  test('leaves non-tool conversations untouched', () => {
    const messages = [
      new SystemMessage('sys'),
      new HumanMessage('hi'),
      new AIMessage('hello'),
      new HumanMessage('bye'),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(result).toBe(messages);
    expect(hasResidualToolContent(result)).toBe(false);
  });
});
