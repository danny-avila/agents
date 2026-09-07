import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { ExtendedMessageContent } from '@/types';
import {
  compactSyntheticProviderContextMessage,
  foldToolBlocksForToollessAgent,
  isSyntheticProviderContextMessage,
} from './format';
import { HARD_MAX_TOOL_RESULT_CHARS } from '@/utils/truncation';
import { toLangChainContent } from './langchain';

test('folds computer actions as model calls rather than tool results', () => {
  const [folded] = foldToolBlocksForToollessAgent([
    new AIMessage({
      content: '',
      response_metadata: {
        output: [
          {
            type: 'computer_call',
            id: 'item',
            call_id: 'call',
            action: { type: 'click', x: 10, y: 20 },
          },
        ],
      },
    }),
  ]);
  expect(getTextContent(folded)).toContain(
    '[tool_call] computer({"type":"click","x":10,"y":20})'
  );
  expect(getTextContent(folded)).not.toContain('server_tool_output');
  expect(folded.additional_kwargs.provenance).toEqual({
    version: 1,
    parts: [{ attribution: 'synthetic' }, { attribution: 'model' }],
  });
});

test('retains a generated-image sidecar alongside a distinct native tool call', () => {
  const message = new AIMessage({
    content: [
      { type: 'text', text: 'drawing' },
      { type: 'tool_use', id: 'lookup', name: 'lookup', input: {} },
    ],
    additional_kwargs: {
      tool_outputs: [
        {
          type: 'image_generation_call',
          id: 'image-1',
          status: 'completed',
          result: '/9j/AA==',
        },
      ],
    },
  });
  const [folded] = foldToolBlocksForToollessAgent([message]);
  expect(folded.content).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        type: 'image',
        mimeType: 'image/jpeg',
        data: '/9j/AA==',
      }),
    ])
  );
  expect(getTextContent(folded)).toContain('lookup');
  expect(getTextContent(folded)).not.toContain('/9j/AA==');
});

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
        additional_kwargs: {
          sourceMessageId: 'assistant-row',
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'model',
                sourceMessageId: 'assistant-row',
                sourceContentPartIndices: [0],
              },
            ],
          },
        },
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
        additional_kwargs: {
          sourceMessageId: 'assistant-row',
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'tool',
                sourceMessageId: 'assistant-row',
                sourceContentPartIndices: [0],
              },
            ],
          },
        },
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
    expect(result[1].additional_kwargs.sourceMessageIds).toEqual([
      'assistant-row',
    ]);
    expect(result[1].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        {
          attribution: 'model',
          sourceMessageId: 'assistant-row',
          sourceContentPartIndices: [0],
        },
        {
          attribution: 'tool',
          sourceMessageId: 'assistant-row',
          sourceContentPartIndices: [0],
        },
      ],
    });
    expect(isSyntheticProviderContextMessage(result[1])).toBe(true);
    expect(
      isSyntheticProviderContextMessage(
        new HumanMessage('[Previous tool interaction] user-authored text')
      )
    ).toBe(false);
  });

  test('attributes unstamped retained sources by message role without ids', () => {
    const result = foldToolBlocksForToollessAgent([
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'call', name: 'lookup', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'legacy tool bytes',
        tool_call_id: 'call',
      }),
    ]);

    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        { attribution: 'model' },
        { attribution: 'tool' },
      ],
    });
  });

  test('attributes validated legacy folded lineage by message role', () => {
    const result = foldToolBlocksForToollessAgent([
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'call', name: 'lookup', args: {}, type: 'tool_call' },
        ],
        additional_kwargs: { sourceMessageId: 'assistant-row' },
      }),
      new ToolMessage({
        content: 'legacy tool bytes',
        tool_call_id: 'call',
        additional_kwargs: { sourceMessageId: 'tool-row' },
      }),
    ]);

    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        { attribution: 'model', sourceMessageId: 'assistant-row' },
        { attribution: 'tool', sourceMessageId: 'tool-row' },
      ],
    });
  });

  test.each([
    ['malformed provenance', { provenance: { version: 1, parts: [null] } }],
    [
      'malformed legacy lineage',
      { sourceMessageIds: { 0: 'forged', length: 1 } },
    ],
  ])(
    'preserves %s invalidity when folding retained bytes',
    (_, additional_kwargs) => {
      const sourceProvenance = (additional_kwargs as { provenance?: unknown })
        .provenance;
      const result = foldToolBlocksForToollessAgent([
        new AIMessage({
          content: '',
          tool_calls: [
            { id: 'call', name: 'lookup', args: {}, type: 'tool_call' },
          ],
        }),
        new ToolMessage({
          content: 'untrusted tool bytes',
          tool_call_id: 'call',
          additional_kwargs,
        }),
      ]);

      expect(result[0].additional_kwargs.provenance).toEqual({
        version: 1,
        parts: null,
      });
      expect(result[0].additional_kwargs.provenance).not.toBe(sourceProvenance);
      expect(result[0].lc_kwargs.additional_kwargs).toBe(
        result[0].additional_kwargs
      );
    }
  );

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

  test('folds Anthropic server-search blocks for a tool-less destination', () => {
    const messages = [
      new AIMessage({
        content: toLangChainContent([
          {
            type: 'server_tool_use',
            id: 'srvtoolu_1',
            name: 'web_search',
            input: { query: 'retained' },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: 'srvtoolu_1',
            content: {
              type: 'web_search_tool_result_error',
              error_code: 'max_uses_exceeded',
            },
          },
        ]),
        tool_calls: [
          {
            id: 'srvtoolu_1',
            name: 'web_search',
            args: { query: 'retained' },
            type: 'tool_call',
          },
        ],
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(result).not.toBe(messages);
    expect(result).toHaveLength(1);
    expect(result[0]).toBeInstanceOf(HumanMessage);
    expect(isSyntheticProviderContextMessage(result[0])).toBe(true);
    expect(getTextContent(result[0])).toContain('server_tool_use');
    expect(getTextContent(result[0])).toContain('web_search_tool_result');
    expect(getTextContent(result[0])).not.toContain('[tool_call]');
  });

  test('preserves authorship in mixed server-tool assistant turns', () => {
    const messages = [
      new AIMessage({
        content: toLangChainContent([
          { type: 'text', text: 'I found the answer.' },
          {
            type: 'web_search_tool_result',
            tool_use_id: 'srvtoolu_1',
            content: [],
          },
        ]),
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);
    const text = getTextContent(folded);

    expect(text).toContain('AI: I found the answer.');
    expect(text).toContain('Tool: [web_search_tool_result]');
    expect(text).not.toContain('Tool: I found the answer.');
  });

  test('folds standalone Google executable-code blocks', () => {
    const messages = [
      new AIMessage({
        content: [
          {
            type: 'executableCode',
            executableCode: { language: 'PYTHON', code: 'print(2 + 2)' },
          },
        ],
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(result).not.toBe(messages);
    expect(getTextContent(result[0])).toContain('AI: [executableCode]');
  });

  test('folds oversized standard tool-result arrays', () => {
    const content = Array.from({ length: 257 }, (_, index) => ({
      type: 'text',
      text: `result-${index}`,
    }));
    const messages = [
      new HumanMessage({
        content: [{ type: 'tool_result', tool_use_id: 'call-1', content }],
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(result).not.toBe(messages);
    expect(result[0]).toBeInstanceOf(HumanMessage);
    expect(isSyntheticProviderContextMessage(result[0])).toBe(true);
  });

  test('preserves native OpenAI server-tool turns without bound client tools', () => {
    const messages = [
      new AIMessage({
        content: [
          {
            type: 'server_tool_call',
            id: 'server-1',
            name: 'code_interpreter',
            args: { code: 'print(2 + 2)' },
          },
          {
            type: 'server_tool_call_result',
            toolCallId: 'server-1',
            status: 'success',
            output: { stdout: '4' },
          },
          { type: 'text', text: 'The result is 4.' },
        ],
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages, undefined, true);

    expect(result).toBe(messages);
  });

  test('folds authoritative Responses v0 output in source order', () => {
    const messages = [
      new AIMessage({
        content: 'The calculation completed.',
        response_metadata: {
          model_provider: 'openai',
          output: [
            {
              id: 'server-1',
              type: 'code_interpreter_call',
              status: 'completed',
              code: 'print(2 + 2)',
              outputs: [{ type: 'logs', logs: '4' }],
            },
            {
              id: 'message-1',
              type: 'message',
              role: 'assistant',
              content: [
                {
                  type: 'output_text',
                  text: 'The calculation completed.',
                },
              ],
            },
            {
              id: 'function-1',
              type: 'function_call',
              name: 'lookup',
              arguments: '{}',
            },
          ],
        },
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);

    expect(result).not.toBe(messages);
    expect(getTextContent(result[0])).toContain('server_tool_output');
    expect(getTextContent(result[0])).toContain('code_interpreter_call');
    expect(getTextContent(result[0])).toContain('The calculation completed.');
    const text = getTextContent(result[0]);
    expect(text).toContain('The calculation completed.');
    expect(text).toContain('[tool_call] lookup({})');
    expect(text.indexOf('code_interpreter_call')).toBeLessThan(
      text.indexOf('The calculation completed.')
    );
    expect(text.indexOf('The calculation completed.')).toBeLessThan(
      text.indexOf('[tool_call] lookup')
    );
    expect(result[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        { attribution: 'model' },
        { attribution: 'tool' },
      ],
    });
  });

  test('folds Responses custom tool calls from authoritative output', () => {
    const messages = [
      new AIMessage({
        content: 'Running the custom tool.',
        response_metadata: {
          output: [
            {
              id: 'message-1',
              type: 'message',
              role: 'assistant',
              content: [
                { type: 'output_text', text: 'Running the custom tool.' },
              ],
            },
            {
              id: 'custom-1',
              call_id: 'call-1',
              type: 'custom_tool_call',
              name: 'execute',
              input: 'print(2 + 2)',
            },
          ],
        },
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);
    const text = getTextContent(folded);

    expect(text).toContain('[tool_call] execute(');
    expect(text).toContain('execute');
    expect(text).toContain('print(2 + 2)');
  });

  test('folds streamed Responses MCP approval requests', () => {
    const messages = [
      new AIMessage({
        content: [],
        additional_kwargs: {
          tool_outputs: [
            {
              id: 'approval-1',
              type: 'mcp_approval_request',
              name: 'delete_file',
              arguments: '{"path":"draft.txt"}',
            },
          ],
        },
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);

    expect(getTextContent(folded)).toContain('mcp_approval_request');
    expect(getTextContent(folded)).toContain('delete_file');
  });

  test('preserves generated images from ordered Responses output', () => {
    const messages = [
      new AIMessage({
        content: [],
        response_metadata: {
          output: [
            {
              id: 'image-1',
              type: 'image_generation_call',
              status: 'completed',
              result: 'base64ImageData',
            },
          ],
        },
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);

    expect(folded.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: 'image',
          mimeType: 'image/png',
          data: 'base64ImageData',
          id: 'image-1',
        }),
      ])
    );
  });

  test('caps ordered Responses function-call arguments per block', () => {
    const messages = [
      new AIMessage({
        content: 'Done.',
        response_metadata: {
          output: [
            { type: 'web_search_call', status: 'completed' },
            {
              type: 'function_call',
              name: 'lookup',
              arguments: 'x'.repeat(20_000),
            },
            {
              type: 'message',
              content: [{ type: 'output_text', text: 'Done.' }],
            },
          ],
        },
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);
    const text = getTextContent(folded);

    expect(text).toContain('[tool_call] lookup(');
    expect(text).toContain('Done.');
    expect(text.length).toBeLessThan(10_000);
  });

  test('bounds nested ordered Responses message content', () => {
    const messages = [
      new AIMessage({
        content: [],
        response_metadata: {
          output: [
            {
              type: 'message',
              content: Array.from({ length: 100_005 }, () => ({
                type: 'output_text',
                text: '',
              })),
            },
            { type: 'web_search_call', status: 'completed' },
          ],
        },
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);

    expect(getTextContent(folded)).toContain(
      'additional folded context omitted'
    );
  });

  test('preserves parsed calls alongside Gemini executable code', () => {
    const messages = [
      new AIMessage({
        content: [
          {
            type: 'executableCode',
            executableCode: { language: 'PYTHON', code: 'print(2 + 2)' },
          },
        ],
        tool_calls: [
          {
            id: 'function-1',
            name: 'lookup',
            args: { query: 'roadmap' },
            type: 'tool_call',
          },
        ],
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);
    const text = getTextContent(folded);

    expect(text).toContain('[executableCode]');
    expect(text).toContain('[tool_call] lookup({"query":"roadmap"})');
  });

  test.each([
    'apply_patch_call_output',
    'local_shell_call_output',
    'mcp_list_tools',
    'program_output',
    'shell_call_output',
    'tool_search_output',
  ])('folds replayable Responses output type %s', (type) => {
    const messages = [
      new AIMessage({
        content: [],
        additional_kwargs: {
          tool_outputs: [{ type, output: 'result', result: 'result' }],
        },
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);

    expect(getTextContent(folded)).toContain('server_tool_output');
    expect(getTextContent(folded)).toContain(type);
  });

  test('preserves parsed calls alongside authoritative Responses output', () => {
    const messages = [
      new AIMessage({
        content: [{ type: 'text', text: 'I will use both tools.' }],
        tool_calls: [
          {
            id: 'function-1',
            name: 'lookup',
            args: { query: 'roadmap' },
            type: 'tool_call',
          },
        ],
        additional_kwargs: {
          tool_outputs: [
            {
              id: 'server-1',
              type: 'code_interpreter_call',
              status: 'completed',
              outputs: [{ type: 'logs', logs: '4' }],
            },
          ],
        },
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);
    const text = getTextContent(folded);

    expect(text).toContain('server_tool_output');
    expect(text).toContain('code_interpreter_call');
    expect(text).toContain('[tool_call] lookup({"query":"roadmap"})');
  });

  test('preserves user authorship in mixed tool-result turns', () => {
    const messages = [
      new HumanMessage({
        content: [
          { type: 'tool_result', tool_use_id: 'call-1', content: 'result' },
          { type: 'text', text: 'Please continue carefully.' },
        ],
      }),
    ];

    const [folded] = foldToolBlocksForToollessAgent(messages);
    const text = getTextContent(folded);

    expect(text).toContain('Tool: [tool_result] result');
    expect(text).toContain('User: Please continue carefully.');
    expect(text).not.toContain('Tool: Please continue carefully.');
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
          {
            type: 'tool_result',
            tool_use_id: 'c1',
            content: 'Found roadmap.md',
          },
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
        tool_calls: [{ id: 'c', name: 'chart', args: {}, type: 'tool_call' }],
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

  test('bounds non-portable media blocks instead of preserving or JSON-expanding them', () => {
    const blob = 'A'.repeat(20_000);
    const resource = {
      type: 'resource',
      resource: { uri: 'file:///report.bin', blob },
    };
    const messages = [
      new HumanMessage('Read the report'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'doc', name: 'read_document', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'report:' }, resource],
        tool_call_id: 'doc',
        name: 'read_document',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const foldedContent = result[result.length - 1].content;
    const foldedText = getTextContent(result[result.length - 1]);

    expect(
      Array.isArray(foldedContent) &&
        foldedContent.some(
          (block) => typeof block === 'object' && block.type === 'resource'
        )
    ).toBe(false);
    expect(foldedText).toContain('[resource]');
    expect(foldedText).toContain('report.bin');
    expect(foldedText.length).toBeLessThan(9_000);
    expect(foldedText).not.toContain(blob);
  });

  test('does not invoke toJSON while folding a small non-portable media block', () => {
    let toJSONCalls = 0;
    const resource = {
      type: 'resource',
      resource: { uri: 'file:///report.bin', blob: 'AAAA' },
      toJSON() {
        toJSONCalls++;
        return { expanded: 'x'.repeat(100_000) };
      },
    };
    const messages = [
      new HumanMessage('Read the report'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'doc', name: 'read_document', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: [resource],
        tool_call_id: 'doc',
        name: 'read_document',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const foldedText = getTextContent(result[result.length - 1]);

    expect(toJSONCalls).toBe(0);
    expect(foldedText).toContain('report.bin');
    expect(foldedText.length).toBeLessThan(9_000);
    expect(foldedText).not.toContain('x'.repeat(1_000));
  });

  test('bounds the aggregate folded context across many large text blocks', () => {
    const messages = [
      new HumanMessage('Summarize the query'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'query', name: 'run_query', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: Array.from({ length: 100 }, (_, index) => ({
          type: 'text',
          text: `${index}:${'x'.repeat(8_000)}`,
        })),
        tool_call_id: 'query',
        name: 'run_query',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const foldedText = getTextContent(result[result.length - 1]);

    expect(foldedText.length).toBeLessThanOrEqual(HARD_MAX_TOOL_RESULT_CHARS);
    expect(foldedText).toContain('additional folded context omitted');
  });

  test('folds a 150k-block tool result without call-argument spreading', () => {
    const blockCount = 150_000;
    const repeatedBlock = { type: 'text', text: 'x' } as const;
    const largeContent = new Array(blockCount).fill(repeatedBlock);
    const toolMessage = new ToolMessage({
      content: largeContent,
      tool_call_id: 'large-result',
      additional_kwargs: { sourceMessageId: 'large-tool-row' },
    });
    const messages = [
      new HumanMessage('Run'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'large-result',
            name: 'query',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      toolMessage,
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const folded = result.find(isSyntheticProviderContextMessage)!;

    expect(folded).toBeInstanceOf(HumanMessage);
    expect(getTextContent(folded).length).toBeLessThanOrEqual(
      HARD_MAX_TOOL_RESULT_CHARS
    );
    expect(toolMessage.content).toBe(largeContent);
  }, 30_000);

  test('omits source rows whose bytes do not survive the shared fold cap', () => {
    const messages = [
      new HumanMessage('Run both'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'first', name: 'query', args: {}, type: 'tool_call' },
          { id: 'second', name: 'query', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'x'.repeat(HARD_MAX_TOOL_RESULT_CHARS * 2),
        tool_call_id: 'first',
        additional_kwargs: {
          sourceMessageId: 'first-row',
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'tool',
                sourceMessageId: 'first-row',
                sourceContentPartIndices: [0],
              },
            ],
          },
        },
      }),
      new ToolMessage({
        content: 'OMITTED-SECOND-MARKER',
        tool_call_id: 'second',
        additional_kwargs: {
          sourceMessageId: 'second-row',
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'tool',
                sourceMessageId: 'second-row',
                sourceContentPartIndices: [0],
              },
            ],
          },
        },
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const folded = result.find(isSyntheticProviderContextMessage)!;

    expect(getTextContent(folded)).not.toContain('OMITTED-SECOND-MARKER');
    expect(folded.additional_kwargs.sourceMessageIds).toEqual(['first-row']);
  });

  test('retains only source part indices whose folded blocks survive the cap', () => {
    const messages = [
      new HumanMessage('Run'),
      new AIMessage({
        content: [
          {
            type: 'tool_call',
            tool_call: {
              id: 'first',
              name: 'query',
              args: {},
              output: 'x'.repeat(HARD_MAX_TOOL_RESULT_CHARS * 2),
            },
          },
          {
            type: 'tool_call',
            tool_call: {
              id: 'second',
              name: 'query',
              args: {},
              output: 'OMITTED-SECOND-PART',
            },
          },
        ],
        additional_kwargs: {
          sourceMessageId: 'assistant-row',
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'model',
                sourceMessageId: 'assistant-row',
                sourceContentPartIndices: [0, 1],
              },
            ],
          },
        },
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const folded = result.find(isSyntheticProviderContextMessage)!;

    expect(getTextContent(folded)).not.toContain('OMITTED-SECOND-PART');
    expect(folded.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        {
          attribution: 'model',
          sourceMessageId: 'assistant-row',
          sourceContentPartIndices: [0],
        },
      ],
    });
  });

  test.each([
    ['duplicate', [[0], [0]]],
    ['gapped/out-of-range', [[0], [2]]],
    ['reordered', [[1], [0]]],
  ])(
    'handles a %s folded content mapping without ordinal attribution loss',
    (mappingKind, sourceContentPartIndices) => {
      const messages = [
        new HumanMessage('Run'),
        new AIMessage({
          content: [
            {
              type: 'tool_call',
              tool_call: {
                id: 'first',
                name: 'query',
                args: {},
                output: 'x'.repeat(HARD_MAX_TOOL_RESULT_CHARS * 2),
              },
            },
            {
              type: 'tool_call',
              tool_call: {
                id: 'second',
                name: 'query',
                args: {},
                output: 'OMITTED-SECOND-PART',
              },
            },
          ],
          additional_kwargs: {
            sourceMessageId: 'source-row',
            provenance: {
              version: 1,
              parts: [
                {
                  attribution: 'model',
                  sourceMessageId: 'source-row',
                  sourceContentPartIndices: sourceContentPartIndices[0],
                },
                {
                  attribution: 'user',
                  sourceMessageId: 'source-row',
                  sourceContentPartIndices: sourceContentPartIndices[1],
                },
              ],
            },
          },
        }),
      ];

      const result = foldToolBlocksForToollessAgent(messages);
      const folded = result.find(isSyntheticProviderContextMessage)!;

      expect(getTextContent(folded)).not.toContain('OMITTED-SECOND-PART');
      expect(folded.additional_kwargs.provenance).toEqual({
        version: 1,
        parts:
          mappingKind === 'reordered'
            ? [
              { attribution: 'synthetic' },
              {
                attribution: 'user',
                sourceMessageId: 'source-row',
                sourceContentPartIndices: [0],
              },
            ]
            : [
              { attribution: 'synthetic' },
              { attribution: 'model', sourceMessageId: 'source-row' },
              { attribution: 'user', sourceMessageId: 'source-row' },
            ],
      });
    }
  );

  test('drops indexed claims when parsed tool calls omit malformed raw entries', () => {
    const messages = [
      new HumanMessage('Run'),
      new AIMessage({
        content: '',
        additional_kwargs: {
          sourceMessageId: 'assistant-row',
          provenance: {
            version: 1,
            parts: [
              {
                attribution: 'model',
                sourceMessageId: 'assistant-row',
                sourceContentPartIndices: [0, 1],
              },
            ],
          },
          tool_calls: [
            { id: 'malformed', type: 'function' },
            {
              id: 'valid',
              type: 'function',
              function: { name: 'query', arguments: '{}' },
            },
          ] as never,
        },
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const folded = result.find(isSyntheticProviderContextMessage)!;

    expect(getTextContent(folded)).toContain('[tool_call] query');
    expect(folded.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        { attribution: 'model', sourceMessageId: 'assistant-row' },
      ],
    });
  });

  test('drops ambiguous legacy source ids after partial folded compaction', () => {
    const messages = [
      new HumanMessage('Run'),
      new AIMessage({
        content: [
          {
            type: 'tool_call',
            tool_call: {
              id: 'first',
              name: 'query',
              args: {},
              output: 'x'.repeat(HARD_MAX_TOOL_RESULT_CHARS * 2),
            },
          },
          {
            type: 'tool_call',
            tool_call: {
              id: 'second',
              name: 'query',
              args: {},
              output: 'OMITTED-SECOND-PART',
            },
          },
        ],
        additional_kwargs: {
          sourceMessageIds: ['first-row', 'second-row'],
        },
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const folded = result.find(isSyntheticProviderContextMessage)!;

    expect(getTextContent(folded)).not.toContain('OMITTED-SECOND-PART');
    expect(folded.additional_kwargs.sourceMessageIds).toBeUndefined();
    expect(folded.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [{ attribution: 'synthetic' }, { attribution: 'model' }],
    });
  });

  test('drops ambiguous source ids when folded string content is truncated', () => {
    const messages = [
      new HumanMessage('Run'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'large', name: 'query', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'x'.repeat(HARD_MAX_TOOL_RESULT_CHARS * 2),
        tool_call_id: 'large',
        additional_kwargs: {
          sourceMessageIds: ['first-row', 'second-row'],
        },
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const folded = result.find(isSyntheticProviderContextMessage)!;

    expect(folded.additional_kwargs.sourceMessageIds).toBeUndefined();
    expect(folded.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        { attribution: 'model' },
        { attribution: 'tool' },
      ],
    });
  });

  test('fails closed when later overflow recovery compacts folded context', () => {
    const folded = foldToolBlocksForToollessAgent([
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'call', name: 'query', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'sensitive tool output'.repeat(100),
        tool_call_id: 'call',
        additional_kwargs: { sourceMessageId: 'tool-row' },
      }),
    ])[0] as HumanMessage;

    const compacted = compactSyntheticProviderContextMessage(folded, 20);

    expect(isSyntheticProviderContextMessage(compacted)).toBe(true);
    expect(compacted.additional_kwargs.sourceMessageIds).toBeUndefined();
    expect(compacted.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        { attribution: 'synthetic' },
        { attribution: 'user' },
        { attribution: 'tool' },
      ],
    });
  });

  test('does not emit an empty user message when an earlier fold exhausts the shared budget', () => {
    const messages = [
      new HumanMessage('Run both queries'),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'first', name: 'run_query', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: Array.from({ length: 60 }, () => ({
          type: 'text',
          text: 'x'.repeat(8_000),
        })),
        tool_call_id: 'first',
        name: 'run_query',
      }),
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'second', name: 'run_query', args: {}, type: 'tool_call' },
        ],
      }),
      new ToolMessage({
        content: 'second result',
        tool_call_id: 'second',
        name: 'run_query',
      }),
    ];

    const result = foldToolBlocksForToollessAgent(messages);
    const syntheticMessages = result.filter(isSyntheticProviderContextMessage);

    expect(hasResidualToolContent(result)).toBe(false);
    expect(syntheticMessages).toHaveLength(1);
    expect(getTextContent(syntheticMessages[0])).toContain(
      'additional folded context omitted'
    );
    expect(
      result
        .filter((message) => message instanceof HumanMessage)
        .every((message) =>
          typeof message.content === 'string'
            ? message.content.length > 0
            : message.content.length > 0
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
