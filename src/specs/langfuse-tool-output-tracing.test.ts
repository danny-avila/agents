import { AIMessage, ToolMessage, HumanMessage } from '@langchain/core/messages';
import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import type { BaseMessage } from '@langchain/core/messages';
import type { TPayload } from '@/types';
import {
  LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
  redactLangfuseSpanToolOutputs,
  resolveLangfuseConfig,
  shouldTraceToolNodeForLangfuse,
  type ResolvedLangfuseToolOutputTracingConfig,
} from '@/langfuseToolOutputTracing';
import { formatAgentMessages } from '@/messages/format';
import { ContentTypes } from '@/common';

type SerializedLangfuseChatMessage = {
  content: BaseMessage['content'];
  role?: string;
  additional_kwargs?: BaseMessage['additional_kwargs'];
  tool_calls?:
    | NonNullable<AIMessage['tool_calls']>
    | NonNullable<BaseMessage['additional_kwargs']['tool_calls']>;
};

type RedactedMessage = {
  role?: string;
  content?: string;
  tool_calls?: Array<{
    id?: string;
    name?: string;
    args?: {
      query?: string;
    };
  }>;
};

function createSpan(
  name: string,
  attributes: Record<string, unknown>
): ReadableSpan {
  return { name, attributes } as unknown as ReadableSpan;
}

function createConfig(
  overrides: Partial<ResolvedLangfuseToolOutputTracingConfig> = {}
): ResolvedLangfuseToolOutputTracingConfig {
  return {
    enabled: true,
    redactedToolNames: new Set<string>(),
    redactedToolNameMatchMode: 'exact',
    redactionText: LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT,
    ...overrides,
  };
}

function serializeMessageForLangfuse(
  message: BaseMessage
): SerializedLangfuseChatMessage {
  if (message instanceof HumanMessage) {
    return { content: message.content, role: 'user' };
  }

  if (message instanceof AIMessage) {
    const response: SerializedLangfuseChatMessage = {
      content: message.content,
      role: 'assistant',
    };
    if (message.tool_calls != null && message.tool_calls.length > 0) {
      response.tool_calls = message.tool_calls;
    }
    if (message.additional_kwargs.tool_calls != null) {
      response.tool_calls = message.additional_kwargs.tool_calls;
    }
    return response;
  }

  if (message instanceof ToolMessage) {
    return {
      content: message.content,
      additional_kwargs: message.additional_kwargs,
      role: message.name,
    };
  }

  return message.name != null
    ? { content: message.content, role: message.name }
    : { content: message.content };
}

function readJsonAttribute<T>(span: ReadableSpan, key: string): T {
  return JSON.parse(span.attributes[key] as string) as T;
}

describe('Langfuse tool output tracing redaction', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('enables ToolNode tracing only when Langfuse is active by default', () => {
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_BASE_URL;

    expect(shouldTraceToolNodeForLangfuse({})).toBe(false);
    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: {
          enabled: true,
          publicKey: 'pk-run',
          secretKey: 'sk-run',
        },
      })
    ).toBe(true);
    expect(
      shouldTraceToolNodeForLangfuse({
        agentLangfuse: {
          enabled: true,
          publicKey: 'pk-agent',
          secretKey: 'sk-agent',
          baseUrl: 'https://langfuse.test',
          toolNodeTracing: { enabled: true },
        },
      })
    ).toBe(true);

    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';
    process.env.LANGFUSE_BASE_URL = 'https://langfuse.test';

    expect(shouldTraceToolNodeForLangfuse({})).toBe(true);
    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: { toolNodeTracing: { enabled: true } },
      })
    ).toBe(true);
    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: { toolNodeTracing: { enabled: false } },
      })
    ).toBe(false);
  });

  it('lets agent Langfuse enablement override disabled run defaults for ToolNode tracing', () => {
    delete process.env.LANGFUSE_SECRET_KEY;
    delete process.env.LANGFUSE_PUBLIC_KEY;
    delete process.env.LANGFUSE_BASE_URL;

    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: {
          enabled: false,
        },
        agentLangfuse: {
          enabled: true,
          publicKey: 'pk-agent',
          secretKey: 'sk-agent',
          baseUrl: 'https://langfuse.test',
        },
      })
    ).toBe(true);
  });

  it('keeps ToolNode tracing disabled when resolved Langfuse is disabled', () => {
    process.env.LANGFUSE_SECRET_KEY = 'sk-test';
    process.env.LANGFUSE_PUBLIC_KEY = 'pk-test';

    expect(
      shouldTraceToolNodeForLangfuse({
        runLangfuse: {
          enabled: false,
          toolNodeTracing: { enabled: true },
        },
      })
    ).toBe(false);
  });

  it('classifies LangGraph tool-node spans as Langfuse tool observations', () => {
    const span = createSpan('tool_batch', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'span',
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langgraph_node`]:
        'tools=agent_1',
    });

    redactLangfuseSpanToolOutputs(span, createConfig());

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE]).toBe(
      'tool'
    );
  });

  it('does not reclassify non-tool LangGraph spans', () => {
    const span = createSpan('agent=agent_1', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'span',
      [`${LangfuseOtelSpanAttributes.OBSERVATION_METADATA}.langgraph_node`]:
        'agent=agent_1',
    });

    redactLangfuseSpanToolOutputs(span, createConfig());

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_TYPE]).toBe(
      'span'
    );
  });

  it('redacts raw tool observation output when tool output tracing is disabled', () => {
    const span = createSpan('execute_sql', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: '{"query":"select 1"}',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT]).toBe(
      '{"query":"select 1"}'
    );
  });

  it('redacts ToolMessage content inside serialized generation inputs', () => {
    const messages = [
      { role: 'user', content: 'show tables' },
      {
        role: 'execute_sql',
        content: 'private query result',
        additional_kwargs: {},
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    const redacted = JSON.parse(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_INPUT] as string
    ) as Array<{ role: string; content: string }>;
    expect(redacted[0].content).toBe('show tables');
    expect(redacted[1].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('redacts only configured tool names when output tracing stays enabled', () => {
    const messages = [
      { role: 'execute_sql', content: 'private query result' },
      { role: 'bash', content: 'public build log' },
    ];
    const span = createSpan('LangGraph', {
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: JSON.stringify({
        messages,
      }),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = JSON.parse(
      span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT] as string
    ) as { messages: Array<{ role: string; content: string }> };
    expect(redacted.messages[0].content).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(redacted.messages[1].content).toBe('public build log');
  });

  it('uses nested ToolMessage names instead of generic tool role', () => {
    const messages = [
      {
        role: 'tool',
        content: 'private query result',
        kwargs: {
          name: 'execute_sql',
          tool_call_id: 'call_1',
        },
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<Array<{ content: string }>>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
  });

  it('maps tool_call_id to the preceding tool call name for allowlisted redaction', () => {
    const messages = [
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'call_sql',
            name: 'execute_sql',
            args: { query: 'select * from private_table' },
          },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'call_sql',
        content: 'sensitive row output',
      },
      {
        role: 'tool',
        tool_call_id: 'call_bash',
        content: 'public build log',
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<RedactedMessage[]>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].tool_calls?.[0]?.args?.query).toBe(
      'select * from private_table'
    );
    expect(redacted[1].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[2].content).toBe('public build log');
  });

  it('does not redact partial tool name matches by default', () => {
    const span = createSpan('clickhouse_execute_sql_prod', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      'secret rows'
    );
  });

  it('redacts configured partial tool name matches when enabled', () => {
    const span = createSpan('clickhouse_execute_sql_prod', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'tool',
      [LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]: 'secret rows',
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
        redactedToolNameMatchMode: 'partial',
      })
    );

    expect(span.attributes[LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT]).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
  });

  it('redacts prior tool outputs from multi-turn generation inputs', () => {
    const messages = [
      { role: 'user', content: 'run the query' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [
          {
            id: 'call_sql',
            name: 'execute_sql',
            args: { query: 'select * from private_table' },
          },
        ],
      },
      {
        role: 'execute_sql',
        content: 'sensitive row output',
        additional_kwargs: {},
      },
      { role: 'assistant', content: 'I found the answer.' },
      { role: 'user', content: 'explain the first row' },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(span, createConfig({ enabled: false }));

    const redacted = readJsonAttribute<RedactedMessage[]>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[0].content).toBe('run the query');
    expect(redacted[1].tool_calls?.[0]?.args?.query).toBe(
      'select * from private_table'
    );
    expect(redacted[2].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted[3].content).toBe('I found the answer.');
    expect(redacted[4].content).toBe('explain the first row');
  });

  it('redacts tool outputs after formatAgentMessages rehydrates content parts', () => {
    const payload: TPayload = [
      { role: 'user', content: 'show me the private numbers' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I will query ClickHouse.',
            tool_call_ids: ['call_sql'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_sql',
              name: 'execute_sql',
              args: '{"query":"select secret_value from prod"}',
              output: 'secret_value: 12345',
            },
          },
        ],
      },
      { role: 'user', content: 'can you summarize it?' },
    ];
    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['execute_sql'])
    );
    const serialized = messages.map(serializeMessageForLangfuse);
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]:
        JSON.stringify(serialized),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<RedactedMessage[]>(
      span,
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    );
    expect(redacted[1].tool_calls?.[0]?.args?.query).toBe(
      'select secret_value from prod'
    );
    expect(redacted[2].role).toBe('execute_sql');
    expect(redacted[2].content).toBe(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(JSON.stringify(redacted)).not.toContain('secret_value: 12345');
  });

  it('redacts constructor-serialized ToolMessages from rehydrated content parts', () => {
    const payload: TPayload = [
      { role: 'user', content: 'show the stored result' },
      {
        role: 'assistant',
        content: [
          {
            type: ContentTypes.TEXT,
            [ContentTypes.TEXT]: 'I will query ClickHouse.',
            tool_call_ids: ['call_sql'],
          },
          {
            type: ContentTypes.TOOL_CALL,
            tool_call: {
              id: 'call_sql',
              name: 'execute_sql',
              args: '{"query":"select constructor_path from prod"}',
              output: 'constructor path secret',
            },
          },
        ],
      },
    ];
    const { messages } = formatAgentMessages(
      payload,
      undefined,
      new Set(['execute_sql'])
    );
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify([
        messages,
      ]),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = span.attributes[
      LangfuseOtelSpanAttributes.OBSERVATION_INPUT
    ] as string;
    expect(redacted).toContain('select constructor_path from prod');
    expect(redacted).toContain(LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT);
    expect(redacted).not.toContain('constructor path secret');
  });

  it('redacts ToolMessage artifacts because they are tool output', () => {
    const messages = [
      {
        id: ['langchain_core', 'messages', 'ToolMessage'],
        kwargs: {
          name: 'execute_sql',
          tool_call_id: 'call_sql',
          content: 'safe display content',
          artifact: {
            rows: ['artifact secret row'],
          },
        },
      },
    ];
    const span = createSpan('gpt-4o', {
      [LangfuseOtelSpanAttributes.OBSERVATION_TYPE]: 'generation',
      [LangfuseOtelSpanAttributes.OBSERVATION_INPUT]: JSON.stringify(messages),
    });

    redactLangfuseSpanToolOutputs(
      span,
      createConfig({
        redactedToolNames: new Set(['execute_sql']),
      })
    );

    const redacted = readJsonAttribute<
      Array<{
        kwargs: {
          artifact: string;
          content: string;
        };
      }>
    >(span, LangfuseOtelSpanAttributes.OBSERVATION_INPUT);
    expect(redacted[0].kwargs.content).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(redacted[0].kwargs.artifact).toBe(
      LANGFUSE_TOOL_OUTPUT_REDACTION_TEXT
    );
    expect(JSON.stringify(redacted)).not.toContain('artifact secret row');
  });

  it('merges run Langfuse defaults with agent redaction overrides', () => {
    const resolved = resolveLangfuseConfig(
      {
        enabled: true,
        publicKey: 'pk-run',
        secretKey: 'sk-run',
        baseUrl: 'https://langfuse.test',
        toolNodeTracing: { enabled: true },
        toolOutputTracing: {
          enabled: true,
          redactionText: '[redacted]',
        },
      },
      {
        toolOutputTracing: {
          enabled: false,
          redactedToolNames: ['execute_sql'],
        },
      }
    );

    expect(resolved).toMatchObject({
      enabled: true,
      publicKey: 'pk-run',
      secretKey: 'sk-run',
      baseUrl: 'https://langfuse.test',
      toolNodeTracing: { enabled: true },
      toolOutputTracing: {
        enabled: false,
        redactedToolNames: ['execute_sql'],
        redactionText: '[redacted]',
      },
    });
  });
});
