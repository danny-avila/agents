import {
  AIMessage,
  HumanMessage,
  ToolMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import { applyContextPruning } from './contextPruning';

function charCounter(message: BaseMessage): number {
  return (
    typeof message.content === 'string'
      ? message.content
      : JSON.stringify(message.content)
  ).length;
}

describe('applyContextPruning', () => {
  it('soft-trims old structured tool results and preserves metadata', () => {
    const toolCallId = 'tc-old';
    const messages: BaseMessage[] = [
      new HumanMessage('query the table'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: toolCallId,
            name: 'run_select_query',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: [
          {
            type: 'json',
            rows: Array.from({ length: 20 }, (_, index) => ({
              id: index,
              value: `${'x'.repeat(100)}-${index}`,
            })),
          },
        ],
        tool_call_id: toolCallId,
        name: 'run_select_query',
        status: 'success',
        artifact: { source: 'clickhouse' },
      }),
      new AIMessage('The query returned 20 rows.'),
      new HumanMessage('continue'),
      new AIMessage('Ready.'),
      new HumanMessage('next question'),
    ];
    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }

    const result = applyContextPruning({
      messages,
      indexTokenCountMap,
      tokenCounter: charCounter,
      config: {
        enabled: true,
        keepLastAssistants: 1,
        softTrimRatio: 0,
        minPrunableToolChars: 1,
        softTrim: { maxChars: 200, headChars: 80, tailChars: 40 },
        hardClear: { enabled: false },
      },
    });

    expect(result.softTrimmed).toBe(1);
    expect(typeof messages[2].content).toBe('string');
    expect(messages[2].content).toContain('soft-trimmed');
    const trimmed = messages[2] as ToolMessage;
    expect(trimmed.tool_call_id).toBe(toolCallId);
    expect(trimmed.name).toBe('run_select_query');
    expect(trimmed.status).toBe('success');
    expect(trimmed.artifact).toEqual({ source: 'clickhouse' });
    expect(indexTokenCountMap[2]).toBe(charCounter(trimmed));
  });

  it('preserves small non-image media blocks while trimming adjacent text', () => {
    const document = {
      type: 'document',
      source: { type: 'url', url: 'https://example.com/report.pdf' },
    };
    const messages: BaseMessage[] = [
      new HumanMessage('read the report'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'tc-document',
            name: 'read_document',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      new ToolMessage({
        content: [{ type: 'text', text: 'x'.repeat(2_000) }, document],
        tool_call_id: 'tc-document',
      }),
      new AIMessage('I read the report.'),
      new HumanMessage('continue'),
      new AIMessage('Ready.'),
      new HumanMessage('next question'),
    ];
    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }

    const result = applyContextPruning({
      messages,
      indexTokenCountMap,
      tokenCounter: charCounter,
      config: {
        enabled: true,
        keepLastAssistants: 1,
        softTrimRatio: 0,
        minPrunableToolChars: 1,
        softTrim: { maxChars: 400, headChars: 80, tailChars: 40 },
        hardClear: { enabled: false },
      },
    });

    expect(result.softTrimmed).toBe(1);
    expect(Array.isArray(messages[2].content)).toBe(true);
    expect(messages[2].content).toContain(document);
  });

  it('does not hard-clear native computer screenshots', () => {
    const screenshot = `data:image/png;base64,${'A'.repeat(2_000)}`;
    const computerOutput = new ToolMessage({
      content: screenshot,
      tool_call_id: 'tc-computer',
      additional_kwargs: { type: 'computer_call_output' },
    });
    const messages: BaseMessage[] = [
      new HumanMessage('take a screenshot'),
      new AIMessage({
        content: '',
        tool_calls: [
          {
            id: 'tc-computer',
            name: 'computer',
            args: {},
            type: 'tool_call',
          },
        ],
      }),
      computerOutput,
      new AIMessage('Done.'),
      new HumanMessage('continue'),
      new AIMessage('Ready.'),
      new HumanMessage('next question'),
    ];
    const indexTokenCountMap: Record<string, number | undefined> = {};
    for (let i = 0; i < messages.length; i++) {
      indexTokenCountMap[i] = charCounter(messages[i]);
    }

    const result = applyContextPruning({
      messages,
      indexTokenCountMap,
      tokenCounter: charCounter,
      config: {
        enabled: true,
        keepLastAssistants: 1,
        softTrimRatio: 0,
        minPrunableToolChars: 1,
        softTrim: { maxChars: 80, headChars: 20, tailChars: 20 },
        hardClear: { enabled: true },
      },
    });

    expect(result.softTrimmed).toBe(0);
    expect(result.hardCleared).toBe(0);
    expect(messages[2]).toBe(computerOutput);
    expect(messages[2].content).toBe(screenshot);
  });
});
