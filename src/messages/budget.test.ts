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
import {
  compactSyntheticProviderContextMessage,
  foldToolBlocksForToollessAgent,
} from './format';
import { createExactTokenCountCache } from '@/llm/contextPressureMeter';
import { stampSyntheticProviderMessage } from './provenance';
import { registerProvider } from '@/llm/providers';
import { syncBudgetDerivedFields } from './budget';
import { FakeChatModel } from '@/llm/fake';

declare module '../provider-registration' {
  interface CustomProviderOptionsMap {
    'bedrock-family-budget-test': BedrockFamilyBudgetOptions;
  }
}

interface BedrockFamilyBudgetOptions {
  endpoint?: string;
}

class BedrockFamilyBudgetTestModel extends FakeChatModel {
  constructor(_config: BedrockFamilyBudgetOptions) {
    super({ responses: ['ok'] });
  }
}
import { toLangChainContent } from './langchain';
import { GraphEvents } from '@/common';

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

  it('consumes an inline provider result so its id can be reused later', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: toLangChainContent([
          {
            type: 'mcp_tool_use',
            id: 'shared-id',
            name: 'connector_search',
            input: {},
            server_name: 'docs',
          },
          {
            type: 'mcp_tool_result',
            tool_use_id: 'shared-id',
            content: 'found',
            is_error: false,
          },
        ]),
      }),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'shared-id', name: 'local_tool', args: {} }],
      }),
      toolResult('shared-id'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(30);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ local_tool: 10 });
  });

  it('counts a Google code-execution exchange as a tool-only invocation', () => {
    const usage = snapshot();
    const turn = new AIMessage({
      content: toLangChainContent([
        {
          type: 'executableCode',
          executableCode: { language: 'PYTHON', code: 'print(1)' },
        },
        {
          type: 'codeExecutionResult',
          codeExecutionResult: { outcome: 'OUTCOME_OK', output: '1' },
        },
      ]),
    });

    syncBudgetDerivedFields(usage, [turn, new AIMessage('answer')], () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(10);
    expect(usage.breakdown.toolMessageTokenCounts).toBeUndefined();
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

  it('counts tool history a tool-less destination received folded into a user turn', () => {
    const folded = foldToolBlocksForToollessAgent([
      new HumanMessage('question'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'folded', name: 'lookup', args: { q: 'x' } }],
      }),
      new ToolMessage({
        content: 'r'.repeat(400),
        tool_call_id: 'folded',
        name: 'lookup',
      }),
      new AIMessage('done'),
    ]);
    expect(folded).toHaveLength(3);

    const usage = snapshot();
    syncBudgetDerivedFields(usage, folded, () => 10);
    expect(usage.breakdown.toolMessageTokens).toBe(10);
    expect(usage.breakdown.toolMessageTokenCounts).toBeUndefined();

    const compacted = folded.map((message) =>
      message.getType() === 'human' && message !== folded[0]
        ? compactSyntheticProviderContextMessage(message as HumanMessage, 80)
        : message
    );
    expect(compacted[1]).not.toBe(folded[1]);
    const afterCompaction = snapshot();
    syncBudgetDerivedFields(afterCompaction, compacted, () => 10);
    expect(afterCompaction.breakdown.toolMessageTokens).toBe(10);
  });

  it('leaves a synthetic user turn without tool lineage in the conversation share', () => {
    const usage = snapshot();
    const cue = stampSyntheticProviderMessage(new HumanMessage('handoff cue'));

    syncBudgetDerivedFields(usage, [cue, toolResult('a')], () => 10);

    expect(usage.breakdown.toolMessageTokens).toBe(10);
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

  it.each(['model', 'ai', 'supervisor'])(
    'treats a generic %s-role message as the model turn, as Google does',
    (role) => {
      const usage = snapshot();
      const invocation = new ChatMessage({
        role,
        content: '',
        additional_kwargs: {
          function_call: { name: 'gemini_lookup', arguments: '{}' },
        },
      });

      syncBudgetDerivedFields(
        usage,
        [invocation, new FunctionMessage({ content: 'result', name: '' })],
        () => 10
      );

      expect(usage.breakdown.toolMessageTokens).toBe(20);
      expect(usage.breakdown.toolMessageTokenCounts).toEqual({
        gemini_lookup: 10,
      });
    }
  );

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

  it('rounds an approximate counter instead of dropping the share', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'approx', name: 'lookup', args: {} }],
      }),
      toolResult('approx'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 12.5);

    expect(usage.breakdown.toolMessageTokens).toBe(25);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ lookup: 13 });
  });

  it('rounds approximate counts after aggregating the retained tool share', () => {
    const usage = snapshot();
    const messages = [
      toolResult('first'),
      toolResult('second'),
      toolResult('third'),
      toolResult('fourth'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 0.25);

    expect(usage.breakdown.toolMessageTokens).toBe(1);
    expect(usage.breakdown.toolMessageTokenCounts).toEqual({ unknown_tool: 1 });
  });

  it('uses the serving provider to interpret generic message roles', () => {
    const messages = [
      new ChatMessage({
        role: 'assistant',
        content: '',
        additional_kwargs: {
          tool_calls: [
            {
              id: 'generic',
              type: 'function',
              function: { name: 'lookup', arguments: '{}' },
            },
          ],
        },
      }),
    ];
    const openAIUsage = snapshot();
    const bedrockUsage = snapshot();
    const customBedrockUsage = snapshot();
    const dispose = registerProvider({
      provider: 'bedrock-family-budget-test',
      model: BedrockFamilyBudgetTestModel,
      family: 'bedrock',
    });

    syncBudgetDerivedFields(
      openAIUsage,
      messages,
      () => 10,
      undefined,
      'openai'
    );
    syncBudgetDerivedFields(
      bedrockUsage,
      messages,
      () => 10,
      undefined,
      'bedrock'
    );
    syncBudgetDerivedFields(
      customBedrockUsage,
      messages,
      () => 10,
      undefined,
      'bedrock-family-budget-test'
    );
    dispose();

    expect(openAIUsage.breakdown.toolMessageTokens).toBe(10);
    expect(bedrockUsage.breakdown.toolMessageTokens).toBe(0);
    expect(customBedrockUsage.breakdown.toolMessageTokens).toBe(0);
  });

  it('attributes a reused call id to the call it currently answers', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'reused', name: 'first_tool', args: {} }],
      }),
      toolResult('reused'),
      new HumanMessage('next'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'reused', name: 'second_tool', args: {} }],
      }),
      toolResult('reused'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      first_tool: 10,
      second_tool: 10,
    });
  });

  it('forgets an unanswered call at the next user turn', () => {
    const usage = snapshot();
    const messages = [
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'dangling', name: 'abandoned_tool', args: {} }],
      }),
      new HumanMessage('never mind, do this instead'),
      new AIMessage({
        content: '',
        tool_calls: [{ id: 'dangling', name: 'current_tool', args: {} }],
      }),
      toolResult('dangling'),
    ];

    syncBudgetDerivedFields(usage, messages, () => 10);

    expect(usage.breakdown.toolMessageTokenCounts).toEqual({
      current_tool: 10,
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
        const propagated = snapshot();
        propagated.breakdown.toolMessageTokens = 10;
        expect(() =>
          syncBudgetDerivedFields(
            propagated,
            undefined,
            undefined,
            config,
            undefined,
            new RangeError('Invalid tool context token count')
          )
        ).not.toThrow();
        expect(propagated.breakdown.toolMessageTokens).toBeUndefined();

        for (const value of [
          Number.NaN,
          Number.POSITIVE_INFINITY,
          -1,
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
