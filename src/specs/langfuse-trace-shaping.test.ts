import { LangfuseOtelSpanAttributes } from '@langfuse/tracing';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import {
  shapeLangfuseSpan,
  shouldDropLangfuseSpan,
} from '@/langfuseTraceShaping';

type TestSpan = ReadableSpan & {
  name: string;
  attributes: Record<string, unknown>;
};

function createSpan(
  name: string,
  attributes: Record<string, unknown> = {},
  parentSpanId?: string
): TestSpan {
  return {
    name,
    attributes,
    ...(parentSpanId != null ? { parentSpanId } : {}),
  } as unknown as TestSpan;
}

const INPUT = LangfuseOtelSpanAttributes.OBSERVATION_INPUT;
const OUTPUT = LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT;
const TRACE_INPUT = LangfuseOtelSpanAttributes.TRACE_INPUT;
const TRACE_OUTPUT = LangfuseOtelSpanAttributes.TRACE_OUTPUT;
const OBSERVATION_TYPE = LangfuseOtelSpanAttributes.OBSERVATION_TYPE;
const TRACE_TAGS = LangfuseOtelSpanAttributes.TRACE_TAGS;

describe('shouldDropLangfuseSpan', () => {
  it('drops langgraph __start__ seed spans', () => {
    expect(shouldDropLangfuseSpan('__start__')).toBe(true);
  });

  it('drops anonymous RunnableLambda pass-throughs', () => {
    expect(shouldDropLangfuseSpan('RunnableLambda')).toBe(true);
  });

  it('keeps named observations', () => {
    expect(shouldDropLangfuseSpan('GenerateTitle')).toBe(false);
    expect(shouldDropLangfuseSpan('agent=openAI__gpt-5.4')).toBe(false);
    expect(shouldDropLangfuseSpan('ChatOpenAI')).toBe(false);
    expect(shouldDropLangfuseSpan('tool_batch')).toBe(false);
  });
});

describe('shapeLangfuseSpan', () => {
  it('strips the ephemeral agent id (provider__model) from agent node names', () => {
    const span = createSpan('agent=openAI__gpt-5.4', {}, 'parent-1');
    shapeLangfuseSpan(span);
    expect(span.name).toBe('agent');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('agent');
  });

  it('shapes tool nodes as stable dispatch chains with scoped call inputs', () => {
    const messages = [
      { type: 'human', content: 'hello' },
      {
        type: 'ai',
        content: '',
        tool_calls: [
          {
            name: 'get_service_details',
            args: { path: 'organizations/1' },
            id: 'call_1',
          },
        ],
      },
    ];
    const span = createSpan(
      'tools=openAI__gpt-5.4',
      { [INPUT]: JSON.stringify({ messages }) },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('tool-dispatch');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
    expect(JSON.parse(span.attributes[INPUT] as string)).toEqual([
      { name: 'get_service_details', args: { path: 'organizations/1' } },
    ]);
  });

  it('preserves every pending call in a multi-tool dispatch input', () => {
    const messages = [
      {
        type: 'ai',
        tool_calls: [
          { name: 'web_search', args: { q: 'a' }, id: '1' },
          { name: 'web_search', args: { q: 'b' }, id: '2' },
          { name: 'execute_code', args: { code: '1+1' }, id: '3' },
        ],
      },
    ];
    const span = createSpan(
      'tools=openAI__gpt-5.4',
      { [INPUT]: JSON.stringify({ messages }) },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(JSON.parse(span.attributes[INPUT] as string)).toEqual([
      { name: 'web_search', args: { q: 'a' } },
      { name: 'web_search', args: { q: 'b' } },
      { name: 'execute_code', args: { code: '1+1' } },
    ]);
  });

  it('reads tool calls from serialized langchain message kwargs', () => {
    const messages = [
      {
        lc: 1,
        type: 'constructor',
        id: ['langchain_core', 'messages', 'AIMessage'],
        kwargs: {
          content: '',
          tool_calls: [{ name: 'lookup', args: { id: 7 }, id: 'call_7' }],
        },
      },
    ];
    const span = createSpan(
      'tools=agent_abc',
      { [INPUT]: JSON.stringify({ messages }) },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('tool-dispatch');
  });

  it('keeps a stable tool-dispatch shape when no tool calls are found', () => {
    const original = JSON.stringify({
      messages: [{ type: 'human', content: 'hi' }],
    });
    const span = createSpan(
      'tools=agent_abc',
      { [INPUT]: original },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('tool-dispatch');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
    expect(span.attributes[INPUT]).toBe(original);
  });

  it('sets root span and trace input/output to the question and answer', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({
        messages: [
          { type: 'system', content: 'You are helpful.' },
          { type: 'human', content: 'What is ClickHouse?' },
        ],
      }),
      [OUTPUT]: JSON.stringify({
        messages: [
          { type: 'human', content: 'What is ClickHouse?' },
          { type: 'ai', content: 'A columnar OLAP database.' },
        ],
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe('What is ClickHouse?');
    expect(span.attributes[TRACE_INPUT]).toBe('What is ClickHouse?');
    expect(span.attributes[OUTPUT]).toBe('A columnar OLAP database.');
    expect(span.attributes[TRACE_OUTPUT]).toBe('A columnar OLAP database.');
  });

  it('extracts answer text from content part arrays', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify([{ type: 'human', content: 'hi' }]),
      [OUTPUT]: JSON.stringify({
        messages: [
          {
            id: ['langchain_core', 'messages', 'AIMessage'],
            kwargs: {
              content: [
                { type: 'text', text: 'Hello ' },
                { type: 'text', text: 'there.' },
              ],
            },
          },
        ],
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe('hi');
    expect(span.attributes[OUTPUT]).toBe('Hello there.');
  });

  it('does not rewrite non-root spans with message payloads', () => {
    const original = JSON.stringify({
      messages: [{ type: 'human', content: 'hi' }],
    });
    const span = createSpan('ChatOpenAI', { [INPUT]: original }, 'parent-1');
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe(original);
  });

  it('preserves root attributes when extraction finds nothing', () => {
    const span = createSpan('LibreChat Agent', { [INPUT]: 'plain text' });
    shapeLangfuseSpan(span);
    expect(span.attributes[INPUT]).toBe('plain text');
    expect(span.attributes[TRACE_INPUT]).toBe('plain text');
  });

  it('renames generation spans to a provider-agnostic name', () => {
    const span = createSpan(
      'ChatOpenAI',
      { [OBSERVATION_TYPE]: 'generation' },
      'parent-1'
    );
    shapeLangfuseSpan(span);
    expect(span.name).toBe('llm');
  });

  it('marks only agent-tagged root spans as agent observations', () => {
    const span = createSpan('LibreChat Agent', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'agent']),
      [INPUT]: JSON.stringify({
        messages: [{ type: 'human', content: 'hi' }],
      }),
    });
    shapeLangfuseSpan(span);
    expect(span.attributes[OBSERVATION_TYPE]).toBe('agent');
  });

  it('marks title-tagged root spans as chain observations', () => {
    const span = createSpan('LibreChat Title', {
      [TRACE_TAGS]: JSON.stringify(['librechat', 'title']),
      [INPUT]: 'Conversation text',
      [OUTPUT]: 'Conversation title',
    });

    shapeLangfuseSpan(span);

    expect(span.attributes[OBSERVATION_TYPE]).toBe('chain');
    expect(span.attributes[TRACE_INPUT]).toBe('Conversation text');
    expect(span.attributes[TRACE_OUTPUT]).toBe('Conversation title');
  });

  it('does not classify untagged root spans as agents', () => {
    const span = createSpan('Custom root', { [INPUT]: 'input' });

    shapeLangfuseSpan(span);

    expect(span.attributes[OBSERVATION_TYPE]).toBeUndefined();
  });

  it('shapes standalone generation roots without replacing their type', () => {
    const span = createSpan('ChatOpenAI', {
      [OBSERVATION_TYPE]: 'generation',
      [TRACE_TAGS]: JSON.stringify(['librechat', 'title']),
      [INPUT]: 'Generate a title',
      [OUTPUT]: 'A useful title',
    });

    shapeLangfuseSpan(span);

    expect(span.name).toBe('llm');
    expect(span.attributes[OBSERVATION_TYPE]).toBe('generation');
    expect(span.attributes[TRACE_INPUT]).toBe('Generate a title');
    expect(span.attributes[TRACE_OUTPUT]).toBe('A useful title');
  });
});
