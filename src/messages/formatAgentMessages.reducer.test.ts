import { convertMessagesToResponsesInput } from '@langchain/openai';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { MessageContentComplex } from '@/types';
import { Constants, ContentTypes, Providers } from '@/common';
import { getProviderMessageProvenance } from './provenance';
import { messagesStateReducer } from './reducer';
import { formatAgentMessages } from './format';

const formatToolTurn = () =>
  formatAgentMessages([
    {
      role: 'assistant',
      messageId: 'msg_assistant_1',
      content: [
        {
          type: ContentTypes.TEXT,
          [ContentTypes.TEXT]: 'Running tool',
          tool_call_ids: ['tool_1'],
        },
        {
          type: ContentTypes.TOOL_CALL,
          tool_call: {
            id: 'tool_1',
            name: 'search',
            args: '{"query":"hello"}',
            output: 'world',
          },
        },
      ],
    },
  ]).messages;

const pairedNativeToolResultCases: ReadonlyArray<{
  label: string;
  call: Record<string, unknown>;
  result: Record<string, unknown>;
}> = [
  {
    label: 'LangChain server_tool_call_result',
    call: {
      type: 'server_tool_call',
      id: 'lc-call-result',
      name: 'code_interpreter',
      args: { code: 'print(1)' },
    },
    result: {
      type: 'server_tool_call_result',
      toolCallId: 'lc-call-result',
      status: 'success',
      output: { stdout: '1' },
    },
  },
  {
    label: 'LangChain server_tool_result',
    call: {
      type: 'server_tool_call',
      id: 'lc-server-result',
      name: 'web_search',
      args: { query: 'docs' },
    },
    result: {
      type: 'server_tool_result',
      tool_call_id: 'lc-server-result',
      status: 'success',
      output: 'found',
    },
  },
  {
    label: 'Gemini codeExecutionResult',
    call: {
      type: 'executableCode',
      executableCode: { language: 'python', code: 'print(1)' },
    },
    result: {
      type: 'codeExecutionResult',
      codeExecutionResult: { outcome: 'OUTCOME_OK', output: '1' },
    },
  },
  {
    label: 'Gemini toolResponse',
    call: {
      type: 'toolCall',
      toolCall: { id: 'google-call', name: 'google_search', args: {} },
    },
    result: {
      type: 'toolResponse',
      toolResponse: {
        id: 'google-call',
        name: 'google_search',
        response: { results: [] },
      },
    },
  },
  {
    label: 'Anthropic MCP result',
    call: {
      type: 'mcp_tool_use',
      id: 'mcp-call',
      name: 'lookup',
      server_name: 'docs',
      input: {},
    },
    result: {
      type: 'mcp_tool_result',
      tool_use_id: 'mcp-call',
      is_error: false,
      content: [{ type: 'text', text: 'found' }],
    },
  },
  {
    label: 'Anthropic server result',
    call: {
      type: 'server_tool_use',
      id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}web-search`,
      name: 'web_search',
      input: { query: 'docs' },
    },
    result: {
      type: 'web_search_tool_result',
      tool_use_id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}web-search`,
      content: [
        {
          type: 'web_search_result',
          encrypted_content: 'ciphertext',
          title: 'Docs',
          url: 'https://example.com',
        },
      ],
    },
  },
  {
    label: 'Anthropic advisor result',
    call: {
      type: 'server_tool_use',
      id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}advisor-result`,
      name: 'advisor',
      input: {},
    },
    result: {
      type: 'advisor_tool_result',
      tool_use_id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}advisor-result`,
      content: {
        type: 'advisor_result',
        text: 'advice',
        stop_reason: 'end_turn',
      },
    },
  },
  {
    label: 'Anthropic advisor redacted result',
    call: {
      type: 'server_tool_use',
      id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}advisor-redacted`,
      name: 'advisor',
      input: {},
    },
    result: {
      type: 'advisor_tool_result',
      tool_use_id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}advisor-redacted`,
      content: {
        type: 'advisor_redacted_result',
        encrypted_content: 'ciphertext',
        stop_reason: null,
      },
    },
  },
  {
    label: 'Anthropic advisor error',
    call: {
      type: 'server_tool_use',
      id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}advisor-error`,
      name: 'advisor',
      input: {},
    },
    result: {
      type: 'advisor_tool_result',
      tool_use_id: `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}advisor-error`,
      content: {
        type: 'advisor_tool_result_error',
        error_code: 'unavailable',
      },
    },
  },
  {
    label: 'Anthropic tool_result',
    call: {
      type: 'tool_use',
      id: 'anthropic-tool-call',
      name: 'lookup',
      input: {},
    },
    result: {
      type: 'tool_result',
      tool_use_id: 'anthropic-tool-call',
      content: 'found',
    },
  },
  {
    label: 'Bedrock toolResult',
    call: {
      type: 'toolUse',
      toolUse: { toolUseId: 'bedrock-call', name: 'lookup', input: {} },
    },
    result: {
      type: 'toolResult',
      toolResult: {
        toolUseId: 'bedrock-call',
        content: [{ text: 'found' }],
        status: 'success',
      },
    },
  },
];

function attributionsForSourceIndex(
  messages: BaseMessage[],
  sourceContentPartIndex: number
): string[] {
  const attributions: string[] = [];
  for (const message of messages) {
    for (const part of getProviderMessageProvenance(message)?.parts ?? []) {
      if (
        part.sourceContentPartIndices?.includes(sourceContentPartIndex) === true
      ) {
        attributions.push(part.attribution);
      }
    }
  }
  return attributions;
}

describe('formatAgentMessages reducer compatibility', () => {
  it('does not expose derived message ids as OpenAI Responses item ids', () => {
    const messages = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'msg_provider_1',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Running tool',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"hello"}',
              output: 'world',
            },
          },
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Finished',
          },
        ],
      },
    ]).messages;
    const input = convertMessagesToResponsesInput({
      messages,
      zdrEnabled: false,
      model: 'gpt-5.6',
    });

    expect(
      input
        .filter((item) => item.type === 'message')
        .map((item) => ('id' in item ? item.id : undefined))
    ).toEqual(['msg_provider_1', undefined]);
  });

  it('does not let a later source id replace a derived message', () => {
    const messages = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'a',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'Running tool',
            tool_call_ids: ['tool_1'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'tool_1',
              name: 'search',
              args: '{"query":"hello"}',
              output: 'world',
            },
          },
        ],
      },
      {
        role: 'user',
        messageId: 'a:derived:1',
        content: 'Continue',
      },
    ]).messages;

    const merged = messagesStateReducer([], messages);

    expect(merged).toHaveLength(3);
    expect(merged[0]).toBeInstanceOf(AIMessage);
    expect(merged[1]).toBeInstanceOf(ToolMessage);
    expect(merged[2]).toBeInstanceOf(HumanMessage);
  });

  it('uses a valid id when the preferred messageId is blank', () => {
    const [message] = formatAgentMessages([
      {
        role: 'user',
        messageId: '   ',
        id: 'fallback-id',
        content: 'hello',
      },
    ]).messages;

    expect(message.additional_kwargs.sourceMessageId).toBe('fallback-id');
    expect(message.additional_kwargs.sourceMessageIds).toEqual(['fallback-id']);
  });

  it('lets the reducer identify derived messages while retaining source correlation', () => {
    const messages = formatToolTurn();

    expect(messages).toHaveLength(2);
    expect(messages[0]).toBeInstanceOf(AIMessage);
    expect(messages[1]).toBeInstanceOf(ToolMessage);
    expect(messages.map((message) => message.id)).toEqual([
      'msg_assistant_1',
      undefined,
    ]);
    expect(messages.map((message) => message.lc_kwargs.id)).toEqual([
      'msg_assistant_1',
      undefined,
    ]);
    expect(
      messages.map((message) => message.additional_kwargs.sourceMessageId)
    ).toEqual(['msg_assistant_1', 'msg_assistant_1']);
    expect(
      messages.map((message) => message.additional_kwargs.sourceMessageIds)
    ).toEqual([['msg_assistant_1'], ['msg_assistant_1']]);
    expect(messages[0].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'msg_assistant_1',
          sourceContentPartIndices: [0, 1],
        },
      ],
    });
    expect(messages[1].additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'tool',
          sourceMessageId: 'msg_assistant_1',
          sourceContentPartIndices: [1],
        },
      ],
    });

    const merged = messagesStateReducer([], messages);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toBeInstanceOf(AIMessage);
    expect(merged[1]).toBeInstanceOf(ToolMessage);
    expect(merged[0].id).toBe('msg_assistant_1');
    expect(merged[1].id).toEqual(expect.any(String));
    expect(merged[1].id).not.toBe(merged[0].id);
  });

  it('preserves the tool-call pair through a StateGraph input write', async () => {
    const State = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => [],
      }),
    });
    let observedMessages: BaseMessage[] = [];
    const graph = new StateGraph(State)
      .addNode('capture', (state) => {
        observedMessages = state.messages;
        return {};
      })
      .addEdge(START, 'capture')
      .addEdge('capture', END)
      .compile();

    await graph.invoke({ messages: formatToolTurn() });

    expect(observedMessages).toHaveLength(2);
    expect(observedMessages[0]).toBeInstanceOf(AIMessage);
    expect((observedMessages[0] as AIMessage).tool_calls).toHaveLength(1);
    expect(observedMessages[1]).toBeInstanceOf(ToolMessage);
  });

  it('keeps per-derived lineage when one assistant row emits model and user turns', () => {
    const messages = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'mixed-row',
        content: [
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model prefix' },
          { type: ContentTypes.STEER, [ContentTypes.STEER]: 'user steer' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model suffix' },
        ],
      },
    ]).messages;

    expect(messages.map((message) => message.getType())).toEqual([
      'ai',
      'human',
      'ai',
    ]);
    expect(
      messages.map((message) => message.additional_kwargs.provenance)
    ).toEqual([
      {
        version: 1,
        parts: [
          {
            attribution: 'model',
            sourceMessageId: 'mixed-row',
            sourceContentPartIndices: [0],
          },
        ],
      },
      {
        version: 1,
        parts: [
          {
            attribution: 'user',
            sourceMessageId: 'mixed-row',
            sourceContentPartIndices: [1],
          },
        ],
      },
      {
        version: 1,
        parts: [
          {
            attribution: 'model',
            sourceMessageId: 'mixed-row',
            sourceContentPartIndices: [2],
          },
        ],
      },
    ]);
  });

  it.each([
    ['user', 'user'],
    ['assistant', 'model'],
  ] as const)(
    'does not let an unpaired tool-result envelope override %s attribution',
    (role, attribution) => {
      const [message] = formatAgentMessages([
        {
          role,
          messageId: `${role}-tool-result-row`,
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'tool-result-id',
              content: 'tool bytes',
            },
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'ordinary prose' },
          ],
        },
      ]).messages;

      expect(message.additional_kwargs.provenance).toEqual({
        version: 1,
        parts: [
          {
            attribution,
            sourceMessageId: `${role}-tool-result-row`,
            sourceContentPartIndices: [0, 1],
          },
        ],
      });
    }
  );

  it.each([
    'server_tool_call_result',
    'server_tool_result',
    'codeExecutionResult',
    'code_execution_tool_result',
    'mcp_tool_result',
    'search_result',
    'web_search_result',
  ])('keeps a malformed or payload-only %s block model-attributed', (type) => {
    const [message] = formatAgentMessages([
      {
        role: 'assistant',
        messageId: `${type}-row`,
        content: [
          { type, content: 'tool bytes' },
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model prose' },
        ],
      },
    ]).messages;

    expect(message.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: `${type}-row`,
          sourceContentPartIndices: [0, 1],
        },
      ],
    });
  });

  it.each(pairedNativeToolResultCases)(
    'attributes a structurally valid paired $label block to the tool',
    ({ label, call, result: toolResult }) => {
      const { messages } = formatAgentMessages([
        {
          role: 'assistant',
          messageId: `paired-${label}`,
          content: [
            call,
            toolResult,
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model prose' },
          ] as MessageContentComplex[],
        },
      ]);

      expect(attributionsForSourceIndex(messages, 0)).toContain('model');
      expect(attributionsForSourceIndex(messages, 1)).toEqual(['tool']);
      expect(attributionsForSourceIndex(messages, 2)).toContain('model');
    }
  );

  it.each(pairedNativeToolResultCases)(
    'keeps an unpaired $label block from receiving tool attribution',
    ({ label, result: toolResult }) => {
      const { messages } = formatAgentMessages([
        {
          role: 'assistant',
          messageId: `unpaired-${label}`,
          content: [
            toolResult,
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'model prose' },
          ] as MessageContentComplex[],
        },
      ]);

      expect(
        messages.flatMap(
          (message) => getProviderMessageProvenance(message)?.parts ?? []
        )
      ).not.toEqual(
        expect.arrayContaining([expect.objectContaining({ attribution: 'tool' })])
      );
    }
  );

  it('does not pair an ordinary tool call with a server-tool result protocol', () => {
    const { messages } = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'cross-protocol-server-result',
        content: [
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'ordinary-call',
              name: 'lookup',
              args: '{}',
              output: '',
            },
          },
          {
            type: 'server_tool_call_result',
            toolCallId: 'ordinary-call',
            status: 'success',
            output: { attacker: 'bytes' },
          },
        ],
      },
    ]);

    expect(attributionsForSourceIndex(messages, 1)).toEqual(['model']);
  });

  it('consumes a provider call after one matching result', () => {
    const { call, result: toolResult } = pairedNativeToolResultCases[0];
    const { messages } = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'duplicate-server-result',
        content: [call, toolResult, { ...toolResult }] as MessageContentComplex[],
      },
    ]);

    expect(attributionsForSourceIndex(messages, 1)).toEqual(['tool']);
    expect(attributionsForSourceIndex(messages, 2)).toEqual(['model']);
  });

  it.each([
    {
      label: 'tool_result content',
      call: {
        type: 'tool_use',
        id: 'matched-tool-result',
        name: 'lookup',
        input: {},
      },
      result: {
        type: 'tool_result',
        tool_use_id: 'matched-tool-result',
        text: 'attacker bytes',
      },
    },
    {
      label: 'mcp_tool_result content',
      call: {
        type: 'mcp_tool_use',
        id: 'matched-mcp-result',
        name: 'lookup',
        server_name: 'docs',
        input: {},
      },
      result: {
        type: 'mcp_tool_result',
        tool_use_id: 'matched-mcp-result',
        is_error: false,
        text: 'attacker bytes',
      },
    },
    {
      label: 'server_tool_result output',
      call: {
        type: 'server_tool_call',
        id: 'matched-server-result',
        name: 'lookup',
        args: {},
      },
      result: {
        type: 'server_tool_result',
        tool_call_id: 'matched-server-result',
        status: 'success',
        text: 'attacker bytes',
      },
    },
  ])('rejects a matched block that substitutes text for $label', ({ call, result }) => {
    const formatted = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'matched-text-spoof',
        content: [call, result] as MessageContentComplex[],
      },
    ]);

    expect(attributionsForSourceIndex(formatted.messages, 1)).toEqual([
      'model',
    ]);
  });

  it('does not tool-attribute piggyback bytes on a valid paired envelope', () => {
    const { messages } = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'paired-piggyback-result',
        content: [
          {
            type: 'tool_use',
            id: 'piggyback-result',
            name: 'lookup',
            input: {},
          },
          {
            type: 'tool_result',
            tool_use_id: 'piggyback-result',
            content: 'safe tool bytes',
            text: 'attacker PII',
          },
        ] as MessageContentComplex[],
      },
    ]);

    expect(attributionsForSourceIndex(messages, 1)).toEqual(['model']);
  });

  it('keeps an unknown tool-result-like user block user-attributed', () => {
    const [message] = formatAgentMessages([
      {
        role: 'user',
        messageId: 'untrusted-type-row',
        content: [{ type: 'attacker_tool_result', text: 'submitted bytes' }],
      },
    ]).messages;

    expect(message.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'untrusted-type-row',
          sourceContentPartIndices: [0],
        },
      ],
    });
  });

  it('ignores null runtime content entries without shifting raw indices', () => {
    const [message] = formatAgentMessages([
      {
        role: 'user',
        messageId: 'nullable-row',
        content: [
          null,
          { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'visible' },
        ] as unknown as MessageContentComplex[],
      },
    ]).messages;

    expect(message.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'nullable-row',
          sourceContentPartIndices: [1],
        },
      ],
    });

    const [assistantMessage] = formatAgentMessages(
      [
        {
          role: 'assistant',
          messageId: 'nullable-assistant-row',
          content: [
            null,
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'visible' },
          ] as unknown as MessageContentComplex[],
        },
      ],
      undefined,
      new Set(['search'])
    ).messages;

    expect(assistantMessage.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'nullable-assistant-row',
          sourceContentPartIndices: [1],
        },
      ],
    });

    const [assistantMessageWithSkills] = formatAgentMessages(
      [
        {
          role: 'assistant',
          messageId: 'nullable-assistant-skills-row',
          content: [
            null,
            { type: ContentTypes.TEXT, [ContentTypes.TEXT]: 'visible' },
          ] as unknown as MessageContentComplex[],
        },
      ],
      undefined,
      undefined,
      new Map([['known-skill', 'body']])
    ).messages;

    expect(assistantMessageWithSkills.additional_kwargs.provenance).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'nullable-assistant-skills-row',
          sourceContentPartIndices: [1],
        },
      ],
    });
  });

  it('accumulates large homogeneous content provenance in one linear group', () => {
    const count = 4_000;
    const [message] = formatAgentMessages([
      {
        role: 'assistant',
        messageId: 'large-assistant-row',
        content: Array.from({ length: count }, (_, index) => ({
          type: ContentTypes.TEXT,
          [ContentTypes.TEXT]: String(index),
        })),
      },
    ]).messages;

    const parts = getProviderMessageProvenance(message)!.parts;
    expect(parts).toHaveLength(1);
    const sourceContentPartIndices = parts[0]?.sourceContentPartIndices;
    expect(sourceContentPartIndices).toHaveLength(count);
    expect(sourceContentPartIndices?.[0]).toBe(0);
    expect(sourceContentPartIndices?.[count - 1]).toBe(count - 1);
  });

  it(
    'formats a maximum paired server-result wave without deep output walks',
    () => {
      const pairCount = 2_048;
      const resultBlock = {
        type: 'web_search_result',
        encrypted_content: 'ciphertext',
        title: 'Result',
        url: 'https://example.com',
      };
      const opaqueOutput = Array.from(
        { length: 256 },
        () => resultBlock
      );
      const content: MessageContentComplex[] = [];
      for (let index = 0; index < pairCount; index++) {
        const id = `${Constants.ANTHROPIC_SERVER_TOOL_PREFIX}${index}`;
        content.push(
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id,
              name: 'web_search',
              args: { query: String(index) },
            },
          },
          {
            type: 'web_search_tool_result',
            tool_use_id: id,
            content: opaqueOutput,
          } as MessageContentComplex
        );
      }

      const startedAt = performance.now();
      const { messages } = formatAgentMessages(
        [{ role: 'assistant', messageId: 'server-wave', content }],
        undefined,
        new Set(['web_search']),
        undefined,
        { provider: Providers.ANTHROPIC }
      );
      const elapsedMs = performance.now() - startedAt;
      let resultCount = 0;
      for (let messageIndex = 0; messageIndex < messages.length; messageIndex++) {
        const messageContent = messages[messageIndex].content;
        if (!Array.isArray(messageContent)) {
          continue;
        }
        for (let partIndex = 0; partIndex < messageContent.length; partIndex++) {
          if (messageContent[partIndex].type === 'web_search_tool_result') {
            resultCount++;
          }
        }
      }

      expect(resultCount).toBe(pairCount);
      expect(elapsedMs).toBeLessThan(5_000);
    },
    10_000
  );

  it(
    'formats 150k tool derivatives without call-argument spreading',
    () => {
      const count = 150_000;
      const result = formatAgentMessages(
        [
          {
            role: 'assistant',
            messageId: 'large-tool-row',
            content: Array.from({ length: count }, (_, index) => ({
              type: ContentTypes.TOOL_CALL,
              tool_call: {
                id: `call-${index}`,
                name: 'allowed',
                args: {},
                output: 'x',
              },
            })),
          },
        ],
        undefined,
        new Set(['allowed'])
      );

      expect(result.messages).toHaveLength(count + 1);
      expect(result.messages[0]).toBeInstanceOf(AIMessage);
      expect(result.messages[count]).toBeInstanceOf(ToolMessage);
      expect(
        getProviderMessageProvenance(result.messages[count])?.parts[0]
          .sourceContentPartIndices
      ).toEqual([count - 1]);
    },
    30_000
  );
});
