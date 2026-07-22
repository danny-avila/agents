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
    const rawToolCalls = ai.additional_kwargs.tool_calls;
    if (Array.isArray(rawToolCalls) && rawToolCalls.length > 0) {
      return true;
    }
    if (Array.isArray(m.content)) {
      return (m.content as ExtendedMessageContent[]).some(
        (b) =>
          typeof b === 'object' &&
          (b.type === 'tool_use' ||
            b.type === 'tool_call' ||
            b.type === 'tool_result')
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

  test('folds an AI message whose tool call is only in additional_kwargs', () => {
    const messages = [
      new HumanMessage('Search my files'),
      // Parsed `tool_calls` is empty; the call survives only in the raw
      // additional_kwargs. The OpenAI converter still serializes it, so the
      // parent AI message must fold with its ToolMessage — otherwise the fold
      // would leave an orphan assistant(tool_calls) -> user(...) sequence.
      new AIMessage({
        content: '',
        additional_kwargs: {
          tool_calls: [
            {
              id: 'call_1',
              type: 'function',
              function: {
                name: 'file_search',
                arguments: '{"query":"roadmap"}',
              },
            },
          ],
        },
      }),
      new ToolMessage({
        content: 'Found roadmap.md',
        tool_call_id: 'call_1',
        name: 'file_search',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    expect(result).toHaveLength(2);
    const folded = getTextContent(result[1]);
    expect(folded).toContain('file_search');
    expect(folded).toContain('roadmap');
    expect(folded).toContain('Found roadmap.md');
  });

  test('folds a standard tool_result content block on a user message', () => {
    const messages = [
      new HumanMessage('Search'),
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
      // Tool result stored as a content block on a user message (the shape the
      // Anthropic converter produces/accepts) rather than a ToolMessage.
      new HumanMessage({
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call_1',
            content: 'Found roadmap.md',
          },
        ],
      }),
      new HumanMessage('thanks'),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    expect(result.map(getTextContent).join('\n')).toContain('Found roadmap.md');
  });

  test('detects v1 standard-content tool_call blocks (no AIMessage.tool_calls)', () => {
    const messages = [
      new HumanMessage('Search'),
      // LangChain v1 standard content: the tool call lives only as a
      // `tool_call` content block; @langchain/aws still serializes it to a
      // Converse toolUse, so a tool-less destination must fold it too.
      new AIMessage({
        content: [
          { type: 'text', text: 'Searching.' },
          {
            type: 'tool_call',
            id: 'call_1',
            name: 'file_search',
            args: { query: 'roadmap' },
          },
        ],
        response_metadata: { output_version: 'v1' },
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
    expect(folded).toContain('file_search');
    expect(folded).toContain('roadmap');
    expect(folded).toContain('Found roadmap.md');
  });

  test('preserves name/args/output of the nested ToolCallContent shape', () => {
    const messages = [
      new HumanMessage('Search'),
      // Shape produced by convertMessagesToContent / persisted LibreChat history:
      // the call (and its output) are nested under `tool_call`, not top level.
      new AIMessage({
        content: [
          {
            type: 'tool_call',
            tool_call: {
              type: 'tool_call',
              name: 'file_search',
              args: { query: 'roadmap' },
              output: 'Found roadmap.md',
            },
          },
        ],
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    const folded = result.map(getTextContent).join('\n');
    expect(folded).toContain('file_search');
    expect(folded).toContain('roadmap');
    // The embedded tool output is preserved, not dropped.
    expect(folded).toContain('Found roadmap.md');
  });

  test('folds a split AIMessage(tool_call) + tool_result user message as one turn', () => {
    const messages = [
      new HumanMessage('Search'),
      new AIMessage({
        content: [
          { type: 'text', text: 'Let me search.' },
          {
            type: 'tool_call',
            id: 'c1',
            name: 'file_search',
            args: { query: 'roadmap' },
          },
        ],
      }),
      new HumanMessage({
        content: [
          { type: 'tool_result', tool_use_id: 'c1', content: 'Found roadmap.md' },
        ],
      }),
      new HumanMessage('thanks'),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    // Call + result collapse into ONE folded turn (not split/mislabelled).
    expect(result).toHaveLength(3);
    const folded = getTextContent(result[1]);
    expect(folded).toContain('file_search');
    expect(folded).toContain('Found roadmap.md');
    expect(getTextContent(result[2])).toBe('thanks');
  });

  test('preserves image blocks nested inside a tool_result content block', () => {
    const messages = [
      new HumanMessage('Chart'),
      new AIMessage({
        content: [{ type: 'tool_call', id: 'c1', name: 'chart', args: {} }],
      }),
      new HumanMessage({
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'c1',
            content: [
              { type: 'text', text: 'chart:' },
              {
                type: 'image_url',
                image_url: { url: 'data:image/png;base64,AAAA' },
              },
            ],
          },
        ],
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(hasResidualToolContent(result)).toBe(false);
    const folded = result[result.length - 1].content;
    expect(Array.isArray(folded)).toBe(true);
    expect(
      (folded as ExtendedMessageContent[]).some((b) => b.type === 'image_url')
    ).toBe(true);
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
