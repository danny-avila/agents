import { RunnableLambda } from '@langchain/core/runnables';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import {
  AIMessage,
  ChatMessage,
  FunctionMessage,
  HumanMessage,
  ToolMessage,
} from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { AgentLogEvent } from '@/types';
import type * as t from '@/types';
import { createExactTokenCountCache } from '@/llm/contextPressureMeter';
import { GraphEvents } from '@/common';
import { toLangChainContent } from './langchain';
import { syncBudgetDerivedFields } from './budget';

function snapshot(
  messageTokens = 1_000,
  calibrationRatio = 1
): t.ContextUsageEvent {
  return {
    breakdown: {
      maxContextTokens: 1_100,
      instructionTokens: 100,
      systemMessageTokens: 0,
      dynamicInstructionTokens: 0,
      toolSchemaTokens: 0,
      summaryTokens: 0,
      toolCount: 0,
      messageCount: 0,
      messageTokens,
      availableForMessages: 1_000,
    },
    contextBudget: 1_100,
    effectiveInstructionTokens: 100,
    remainingContextTokens: 1_000 - messageTokens,
    calibrationRatio,
  };
}

function toolResult(id: string, name?: string): ToolMessage {
  return new ToolMessage({
    content: 'result',
    tool_call_id: id,
    ...(name != null ? { name } : {}),
  });
}

describe('syncBudgetDerivedFields tool-message accounting', () => {
  it('counts retained results and tool-only invocations without double-counting mirrored calls', () => {
    const usage = snapshot();
    const messages = [
      new HumanMessage('question'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'a', name: 'read_file', args: {} }],
        additional_kwargs: {
          tool_calls: [
            {
              id: 'a',
              type: 'function',
              function: { name: 'read_file', arguments: '{}' },
            },
          ],
        },
      }),
      toolResult('a'),
      new AIMessage('answer'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ read_file: 10 });
  });

  it.each([
    { type: 'text', text: 'explaining the result' },
    {
      type: 'image_url',
      image_url: { url: 'https://example.com/image.png' },
    },
  ])('keeps $type assistant content in the conversation share', (part) => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: toLangChainContent([
          part,
          { type: 'tool_use', id: 'mixed-call', name: 'read_file', input: {} },
        ]),
        tool_calls: [{ id: 'mixed-call', name: 'read_file', args: {} }],
      }),
      toolResult('mixed-call'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(10);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ read_file: 10 });
  });

  it.each([
    { type: 'thinking', thinking: 'private reasoning', signature: 'sig' },
    { type: 'redacted_thinking', data: 'opaque' },
    { type: 'reasoning_content', reasoningText: { text: 'bedrock' } },
    { type: 'reasoning', reasoning: 'google' },
    { type: 'think', think: 'librechat' },
  ])('keeps $type reasoning with the tool-only turn it precedes', (part) => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: toLangChainContent([
          part,
          { type: 'tool_use', id: 'reasoned', name: 'read_file', input: {} },
        ]),
        tool_calls: [{ id: 'reasoned', name: 'read_file', args: {} }],
      }),
      toolResult('reasoned'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ read_file: 10 });
  });

  it('counts an Anthropic server-tool turn as a tool-only invocation', () => {
    const usage = snapshot();
    const turn = new AIMessage({
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
    });

    syncBudgetDerivedFields(usage, [turn, new AIMessage('answer')], () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(10);
    expect(usage.breakdown.toolMessageTokenCounts).toBeUndefined();
  });

  it('counts an MCP connector turn as a tool-only invocation', () => {
    const usage = snapshot();
    const turn = new AIMessage({
      content: toLangChainContent([
        {
          type: 'mcp_tool_use',
          id: 'mcptoolu_1',
          name: 'search',
          input: {},
          server_name: 'docs',
        },
        {
          type: 'mcp_tool_result',
          tool_use_id: 'mcptoolu_1',
          content: 'found',
          is_error: false,
        },
      ]),
    });

    syncBudgetDerivedFields(usage, [turn], () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(10);
  });

  it('attributes a user turn made only of tool results', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [
          { id: 'split-a', name: 'lookup', args: {} },
          { id: 'split-b', name: 'lookup', args: {} },
        ],
      }),
      new HumanMessage({
        content: toLangChainContent([
          { type: 'tool_result', tool_use_id: 'split-a', content: 'first' },
          { type: 'tool_result', tool_use_id: 'split-b', content: 'second' },
        ]),
      }),
      new HumanMessage({
        content: toLangChainContent([
          { type: 'tool_result', tool_use_id: 'split-a', content: 'again' },
          { type: 'text', text: 'and my question' },
        ]),
      }),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ lookup: 10 });
  });

  it('counts generic tool-role results', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'generic-result', name: 'lookup', args: {} }],
      }),
      new ChatMessage({ role: 'tool', content: 'result', name: 'lookup' }),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ lookup: 10 });
  });

  it('counts whitespace-only inline invocations without structured calls', () => {
    const usage = snapshot();
    const invocation = new AIMessage({
      content: toLangChainContent([
        { type: 'text', text: ' \n\t' },
        { type: 'tool_use', id: 'inline', name: 'lookup', input: {} },
      ]),
    });
    syncBudgetDerivedFields(
      usage,
      [invocation, toolResult('inline')],
      () => 10
    );
    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ lookup: 10 });
  });

  it('counts v1 standard-content tool_call invocations', () => {
    const usage = snapshot();
    const invocation = new AIMessage({
      content: toLangChainContent([
        { type: 'tool_call', id: 'standard', name: 'file_search', args: {} },
      ]),
      response_metadata: { output_version: 'v1' },
    });

    syncBudgetDerivedFields(
      usage,
      [invocation, toolResult('standard')],
      () => 10
    );

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ file_search: 10 });
  });

  it('attributes the nested tool_call content shape', () => {
    const usage = snapshot();
    const invocation = new AIMessage({
      content: toLangChainContent([
        {
          type: 'tool_call',
          tool_call: {
            type: 'tool_call',
            id: 'nested',
            name: 'lookup',
            args: {},
          },
        },
      ]),
    });

    syncBudgetDerivedFields(
      usage,
      [invocation, toolResult('nested')],
      () => 10
    );

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ lookup: 10 });
  });

  it('does not guess between conflicting names for one call id', () => {
    const usage = snapshot();
    const invocation = new AIMessage({
      content: toLangChainContent([
        { type: 'tool_call', id: 'mirrored', name: 'inline_name', args: {} },
      ]),
      tool_calls: [{ id: 'mirrored', name: 'structured_name', args: {} }],
    });

    syncBudgetDerivedFields(
      usage,
      [invocation, toolResult('mirrored')],
      () => 10
    );

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      unknown_tool: 10,
    });
  });

  it('counts generic assistant messages carrying a legacy call', () => {
    const usage = snapshot();
    const invocation = new ChatMessage({
      role: 'assistant',
      content: '',
      additional_kwargs: {
        function_call: { name: 'legacy_lookup', arguments: '{}' },
      },
    });

    syncBudgetDerivedFields(
      usage,
      [invocation, new FunctionMessage({ content: 'result', name: '' })],
      () => 10
    );

    expect(usage.breakdown.toolMessageTokens).toBe(20);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      legacy_lookup: 10,
    });
  });

  it('leaves non-assistant generic messages in the conversation share', () => {
    const usage = snapshot();
    const bystander = new ChatMessage({ role: 'user', content: 'aside' });

    syncBudgetDerivedFields(usage, [bystander, toolResult('a')], () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(10);
  });

  it('attributes structured, raw, inline, and legacy results with explicit precedence', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: toLangChainContent([
          {
            type: 'tool_use',
            id: 'inline',
            name: 'inline_tool',
            input: {},
          },
        ]),
        tool_calls: [{ id: 'structured', name: 'structured_tool', args: {} }],
        additional_kwargs: {
          tool_calls: [
            {
              id: 'raw',
              type: 'function',
              function: { name: 'raw_tool', arguments: '{}' },
            },
          ],
        },
      }),
      toolResult('structured'),
      toolResult('raw', 'ignored_result_name'),
      toolResult('inline'),
      toolResult('unmatched', 'explicit_tool'),
      new AIMessage({
        content: '',
        additional_kwargs: {
          function_call: { name: 'legacy_tool', arguments: '{}' },
        },
      }),
      new FunctionMessage({
        content: 'explicit legacy result',
        name: 'function_explicit',
      }),
      new AIMessage({
        content: '',
        additional_kwargs: {
          function_call: { name: 'legacy_tool', arguments: '{}' },
        },
      }),
      new FunctionMessage({ content: 'legacy result', name: '' }),
      new ToolMessage({ content: 'unknown result', tool_call_id: '' }),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(100);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      structured_tool: 10,
      raw_tool: 10,
      inline_tool: 10,
      explicit_tool: 10,
      function_explicit: 10,
      legacy_tool: 10,
      unknown_tool: 10,
    });
  });

  it('does not guess a name for an ID-less result from an unrelated call', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'unrelated', name: 'unrelated_tool', args: {} }],
      }),
      new ToolMessage({ content: 'result', tool_call_id: '' }),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      unknown_tool: 10,
    });
  });

  it('preserves prototype-sensitive tool names through JSON serialization', () => {
    const usage = snapshot();
    syncBudgetDerivedFields(
      usage,
      [
        toolResult('a', '__proto__'),
        toolResult('b', 'constructor'),
        toolResult('c', 'toString'),
      ],
      () => 10
    );

    const counts = JSON.parse(JSON.stringify(usage)).breakdown
      .toolMessageTokenCounts as Record<string, number>;
    expect(Object.hasOwn(counts, '__proto__')).toBe(true);
    expect(counts.__proto__).toBe(10);
    expect(counts.constructor).toBe(10);
    expect(counts.toString).toBe(10);
  });

  it('calibrates result shares while retaining invocation overhead when clamping', () => {
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'invoke', name: 'search', args: {} }],
      }),
      toolResult('a', 'search'),
      toolResult('b', 'search'),
      toolResult('c', 'search'),
    ];

    const unclamped = snapshot(10, 1.5);
    syncBudgetDerivedFields(unclamped, messages, () => 1);
    expect(unclamped.breakdown.toolMessageTokens).toBe(6);
    expect(unclamped.breakdown.toolMessageTokenCounts).toEqual({ search: 5 });

    const clamped = snapshot(4, 1.5);
    syncBudgetDerivedFields(clamped, messages, () => 1);
    expect(clamped.breakdown.toolMessageTokens).toBe(4);
    expect(
      Object.values(clamped.breakdown.toolMessageTokenCounts ?? {}).reduce(
        (sum, count) => sum + count,
        0
      )
    ).toBe(3);
  });

  it('apportions fractional counts across tools rather than rounding each independently', () => {
    const usage = snapshot(10, 1.5);
    syncBudgetDerivedFields(
      usage,
      [
        toolResult('a', 'first'),
        toolResult('b', 'second'),
        toolResult('c', 'third'),
      ],
      () => 1
    );
    expect(usage.breakdown.toolMessageTokens).toBe(5);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      first: 2,
      second: 2,
      third: 1,
    });
    usage.remainingContextTokens = 998;
    syncBudgetDerivedFields(usage);
    expect(usage.breakdown.toolMessageTokens).toBe(2);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      first: 1,
      second: 1,
      third: 0,
    });
  });

  it('distinguishes a known zero share from an unavailable counter', () => {
    const known = snapshot();
    syncBudgetDerivedFields(known, [new HumanMessage('hello')], () => 10);
    expect(known.breakdown.toolMessageTokens).toBe(0);
    expect(known.breakdown.toolMessageTokenCounts).toBeUndefined();

    const zero = snapshot();
    syncBudgetDerivedFields(zero, [toolResult('a')], () => 0);
    expect(zero.breakdown.toolMessageTokens).toBe(0);
    expect(zero.breakdown.toolMessageTokenCounts).toBeUndefined();

    const unavailable = snapshot();
    syncBudgetDerivedFields(unavailable, [toolResult('a')]);
    expect(unavailable.breakdown.toolMessageTokens).toBeUndefined();
  });

  it('drops the tool share on unsafe counts instead of failing the call', async () => {
    const logs: AgentLogEvent[] = [];
    const logHandler = BaseCallbackHandler.fromMethods({
      handleCustomEvent: (eventName: string, data: unknown): void => {
        if (eventName === GraphEvents.ON_AGENT_LOG) {
          logs.push(data as AgentLogEvent);
        }
      },
    });

    /** A config-less caller (the pre-send projection) has no channel to warn
     *  through, so it must not consume the live path's one warning. */
    const projected = snapshot();
    expect(() =>
      syncBudgetDerivedFields(projected, [toolResult('a')], () => Number.NaN)
    ).not.toThrow();
    expect(projected.breakdown.toolMessageTokens).toBeUndefined();

    await RunnableLambda.from(
      (_input: unknown, config?: RunnableConfig): void => {
        for (const value of [
          Number.NaN,
          Number.POSITIVE_INFINITY,
          -1,
          0.5,
          Number.MAX_SAFE_INTEGER + 1,
        ]) {
          const usage = snapshot();
          expect(() =>
            syncBudgetDerivedFields(
              usage,
              [toolResult('a')],
              () => value,
              config
            )
          ).not.toThrow();
          expect(usage.breakdown.toolMessageTokens).toBeUndefined();
          expect(usage.breakdown.toolMessageTokenCounts).toBeUndefined();
          /** Every other derived field still reconciles. */
          expect(usage.breakdown.instructionTokens).toBe(100);
          expect(usage.breakdown.messageTokens).toBe(1_000);
        }

        const overflow = snapshot();
        expect(() =>
          syncBudgetDerivedFields(
            overflow,
            [toolResult('a'), toolResult('b')],
            () => Number.MAX_SAFE_INTEGER,
            config
          )
        ).not.toThrow();
        expect(overflow.breakdown.toolMessageTokens).toBeUndefined();

        const unclampable = snapshot();
        unclampable.breakdown.messageTokens = Number.NaN;
        unclampable.effectiveInstructionTokens = undefined;
        expect(() =>
          syncBudgetDerivedFields(
            unclampable,
            [toolResult('a')],
            () => 10,
            config
          )
        ).not.toThrow();
        expect(unclampable.breakdown.toolMessageTokens).toBeUndefined();
        expect(unclampable.breakdown.toolMessageTokenCounts).toBeUndefined();
      }
    ).invoke({}, { callbacks: [logHandler] });

    expect(logs).toHaveLength(1);
    expect(logs[0].level).toBe('warn');
    expect(logs[0].message).toContain('Tool-message context share unavailable');
  });

  it('reuses exact cached counts and invalidates them after token-relevant mutation', () => {
    let counterCalls = 0;
    const tokenCounter: t.TokenCounter = (message) => {
      counterCalls += 1;
      return typeof message.content === 'string' ? message.content.length : 0;
    };
    const cachedCounter = createExactTokenCountCache(tokenCounter);
    const result = toolResult('cached', 'lookup');

    const first = snapshot();
    syncBudgetDerivedFields(first, [result], cachedCounter.count);
    expect(first.breakdown.toolMessageTokens).toBe(6);
    expect(counterCalls).toBe(1);

    const second = snapshot();
    syncBudgetDerivedFields(second, [result], cachedCounter.count);
    expect(second.breakdown.toolMessageTokens).toBe(6);
    expect(counterCalls).toBe(1);

    result.content = 'changed result';
    const third = snapshot();
    syncBudgetDerivedFields(third, [result], cachedCounter.count);
    expect(third.breakdown.toolMessageTokens).toBe(14);
    expect(counterCalls).toBe(2);
  });
});
