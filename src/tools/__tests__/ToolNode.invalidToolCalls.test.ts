import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { MemorySaver } from '@langchain/langgraph';
import { describe, it, expect } from '@jest/globals';
import {
  AIMessage,
  ToolMessage,
  HumanMessage,
} from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { ChatGenerationChunk } from '@langchain/core/outputs';
import type * as t from '@/types';
import { _convertMessagesToOpenAIResponsesParams } from '@/llm/openai/utils';
import { askUserQuestion } from '@/hitl/askUserQuestion';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import { ToolNode, toolsCondition } from '../ToolNode';
import { Run } from '@/run';

/**
 * `invalid_tool_calls` coverage: a streamed tool call whose accumulated args
 * never collapse into a JSON object is filed by `@langchain/core` under
 * `invalid_tool_calls` (never `tool_calls`), yet its `tool_use` block still
 * rides the AI message content the provider receives. ToolNode must
 * synthesize an error `ToolMessage` for it — skipping it leaves a `tool_use`
 * with no `tool_result` and the next model call is rejected (Anthropic 400
 * INVALID_TOOL_RESULTS). Fatal on HITL resume, where the paused AI message
 * is replayed from the checkpoint (observed with two parallel
 * `ask_user_question` calls, one malformed).
 */

function createEchoTool(name = 'echo'): StructuredToolInterface {
  return tool(async (input) => `ran:${(input as { command: string }).command}`, {
    name,
    description: 'Echo test tool',
    schema: z.object({ command: z.string() }),
  }) as unknown as StructuredToolInterface;
}

function resultMessages(result: unknown): BaseMessage[] {
  return Array.isArray(result)
    ? result
    : (result as { messages: BaseMessage[] }).messages;
}

function toToolMessages(result: unknown): ToolMessage[] {
  return resultMessages(result).filter(
    (msg): msg is ToolMessage => msg._getType() === 'tool'
  );
}

function toPromotedAiMessage(result: unknown): AIMessage | undefined {
  return resultMessages(result).find(
    (msg): msg is AIMessage => msg._getType() === 'ai'
  );
}

describe('ToolNode invalid_tool_calls handling', () => {
  it('synthesizes an error ToolMessage for an invalid call alongside real results (direct batch)', async () => {
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
      id: 'ai_mixed',
      content: '',
      tool_calls: [{ id: 'tc_valid', name: 'echo', args: { command: 'hi' } }],
      invalid_tool_calls: [
        {
          id: 'tc_invalid',
          name: 'echo',
          args: '"not an object"',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });

    const result = await node.invoke(
      { messages: [aiMsg] },
      { configurable: { run_id: 'invalid-mixed' } }
    );
    const toolMessages = toToolMessages(result);

    expect(toolMessages.map((m) => m.tool_call_id).sort()).toEqual([
      'tc_invalid',
      'tc_valid',
    ]);
    const invalidResult = toolMessages.find(
      (m) => m.tool_call_id === 'tc_invalid'
    )!;
    expect(String(invalidResult.content)).toContain('Malformed args.');
    expect(invalidResult.name).toBe('echo');

    /** Replacement AI message (reducer upsert-by-id): the answered invalid
     *  call is promoted into tool_calls so provider converters that rebuild
     *  the call side from tool_calls emit it alongside its synthesized
     *  result. */
    const promoted = toPromotedAiMessage(result);
    expect(promoted?.id).toBe('ai_mixed');
    expect(promoted?.tool_calls?.map((c) => c.id).sort()).toEqual([
      'tc_invalid',
      'tc_valid',
    ]);
    expect(promoted?.invalid_tool_calls).toHaveLength(0);
  });

  it('sanitizes the promoted call\'s Anthropic tool_use content block (raw string input → {})', async () => {
    /**
     * Anthropic formats array-content AI messages from the blocks verbatim; a
     * call whose streamed input never parsed leaves `input` as the raw
     * accumulated string, which the API rejects with "Input should be an
     * object" on replay. The replacement message must normalize the block to
     * match the promoted args; valid siblings' blocks stay untouched.
     */
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
      id: 'ai_blocks',
      content: [
        { type: 'text', text: 'Two calls.' },
        { type: 'tool_use', id: 'tc_ok', name: 'echo', input: { command: 'hi' } },
        { type: 'tool_use', id: 'tc_bad', name: 'echo', input: '"raw unparsed' },
      ],
      tool_calls: [{ id: 'tc_ok', name: 'echo', args: { command: 'hi' } }],
      invalid_tool_calls: [
        {
          id: 'tc_bad',
          name: 'echo',
          args: '"raw unparsed',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });

    const result = await node.invoke(
      { messages: [aiMsg] },
      { configurable: { run_id: 'invalid-blocks' } }
    );
    const promoted = toPromotedAiMessage(result)!;
    const blocks = promoted.content as Array<{
      type?: string;
      id?: string;
      input?: unknown;
    }>;
    expect(blocks.find((b) => b.id === 'tc_bad')?.input).toEqual({});
    expect(blocks.find((b) => b.id === 'tc_ok')?.input).toEqual({ command: 'hi' });
    expect(blocks[0]).toEqual({ type: 'text', text: 'Two calls.' });
  });

  it('leaves BaseMessage[] (array-input) callers at the status quo — no synthesized results, no replacement', async () => {
    /**
     * The array input form returns a plain output LIST the caller appends to
     * its own history: a replacement AI message would duplicate the assistant
     * turn, and synthesized results would reference calls the caller's
     * history formatting never emits. Both are reducer-shaped writes, so
     * they only apply to the messages-state form.
     */
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
      id: 'ai_array_input',
      content: '',
      tool_calls: [{ id: 'tc_ok', name: 'echo', args: { command: 'hi' } }],
      invalid_tool_calls: [
        {
          id: 'tc_bad',
          name: 'echo',
          args: 'garbage',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });

    const result = await node.invoke([aiMsg], {
      configurable: { run_id: 'invalid-array-input' },
    });

    expect(toToolMessages(result).map((m) => m.tool_call_id)).toEqual(['tc_ok']);
    expect(toPromotedAiMessage(result)).toBeUndefined();
  });

  it('toolsCondition routes a server-call + malformed-client-call mix to the tool node', () => {
    /**
     * `handleAnthropicSearchResults` marks completed server calls invoked, so
     * the valid-calls branch declines; every valid call is `srvtoolu_`-
     * prefixed (ToolNode's batch filter excludes those), so routing cannot
     * re-execute anything and the malformed call gets its result. A valid
     * NON-server call stays conservative: no routing, even when invoked.
     */
    const serverMix = new AIMessage({
      id: 'ai_server_mix',
      content: '',
      tool_calls: [{ id: 'srvtoolu_abc', name: 'web_search', args: { q: 'x' } }],
      invalid_tool_calls: [
        {
          id: 'tc_bad',
          name: 'echo',
          args: 'garbage',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });
    expect(
      toolsCondition({ messages: [serverMix] }, 'tools', new Set(['srvtoolu_abc']))
    ).toBe('tools');

    const clientMix = new AIMessage({
      id: 'ai_client_mix',
      content: '',
      tool_calls: [{ id: 'tc_regular', name: 'echo', args: { command: 'hi' } }],
      invalid_tool_calls: serverMix.invalid_tool_calls,
    });
    expect(
      toolsCondition({ messages: [clientMix] }, 'tools', new Set(['tc_regular']))
    ).toBe('__end__');

    /** Mirrors ToolNode's own gating: array-state graphs get a plain output
     *  list (invalid handling is skipped there), and an id-less message
     *  cannot take the replacement upsert — routing either would no-op. */
    expect(toolsCondition([serverMix], 'tools', new Set(['srvtoolu_abc']))).toBe(
      '__end__'
    );
    const noIdMix = new AIMessage({
      content: '',
      tool_calls: [],
      invalid_tool_calls: serverMix.invalid_tool_calls,
    });
    expect(toolsCondition({ messages: [noIdMix] }, 'tools')).toBe('__end__');
  });

  it('keeps the full status quo when the AI message has no id (results and replacement are all-or-nothing)', async () => {
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
      content: '',
      tool_calls: [],
      invalid_tool_calls: [
        {
          id: 'tc_no_promote',
          name: 'echo',
          args: 'garbage',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });

    const result = await node.invoke(
      { messages: [aiMsg] },
      { configurable: { run_id: 'invalid-no-id' } }
    );

    /** No replacement can upsert without an id, so the synthesized result is
     *  suppressed too — emitting it alone would strand an output whose call
     *  the provider converters never reconstruct. */
    expect(toToolMessages(result)).toHaveLength(0);
    expect(toPromotedAiMessage(result)).toBeUndefined();
  });

  it('synthesizes error ToolMessages when EVERY call in the batch is invalid', async () => {
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
      id: 'ai_only_invalid',
      content: '',
      tool_calls: [],
      invalid_tool_calls: [
        {
          id: 'tc_only_invalid',
          name: 'echo',
          args: 'garbage',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });

    const result = await node.invoke(
      { messages: [aiMsg] },
      { configurable: { run_id: 'invalid-only' } }
    );
    const toolMessages = toToolMessages(result);

    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0].tool_call_id).toBe('tc_only_invalid');
  });

  it('skips invalid calls that already have a ToolMessage or carry no id', async () => {
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
      content: '',
      tool_calls: [],
      invalid_tool_calls: [
        {
          id: 'tc_answered',
          name: 'echo',
          args: 'garbage',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
        {
          name: 'echo',
          args: 'garbage-no-id',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });
    const priorResult = new ToolMessage({
      content: 'already answered',
      tool_call_id: 'tc_answered',
      name: 'echo',
    });

    const result = await node.invoke(
      { messages: [aiMsg, priorResult] },
      { configurable: { run_id: 'invalid-skip' } }
    );

    expect(toToolMessages(result)).toHaveLength(0);
  });

  it('HITL resume regression: a malformed sibling of a paused ask_user_question gets a result instead of dangling', async () => {
    const ASK_TOOL = 'ask_user_question';
    const askTool = tool(
      async (input) => {
        const { answer } = askUserQuestion(
          input as { question: string }
        );
        return answer;
      },
      {
        name: ASK_TOOL,
        description: 'Ask the user a question.',
        schema: z.object({ question: z.string() }),
      }
    );

    /** Model invocation capture: the post-resume call's message list is the
     *  payload the real provider would validate tool_use/tool_result pairing
     *  on. */
    const modelInvocations: BaseMessage[][] = [];
    const buildModel = (responses: string[], emitCalls: boolean) => {
      const model = new FakeChatModel({
        responses,
        toolCalls: emitCalls
          ? [
            { name: ASK_TOOL, args: {}, id: 'tc_ask_invalid', type: 'tool_call' },
            {
              name: ASK_TOOL,
              args: { question: 'Which one?' },
              id: 'tc_ask_valid',
              type: 'tool_call',
            },
          ]
          : [],
      });
      const orig = model._streamResponseChunks.bind(model);
      model._streamResponseChunks = async function* (
        messages,
        options,
        runManager
      ): AsyncGenerator<ChatGenerationChunk> {
        modelInvocations.push(messages);
        for await (const chunk of orig(messages, options, runManager)) {
          /** Corrupt the first ask call's streamed args into a non-object
           *  JSON string so `collapseToolCallChunks` files it under
           *  `invalid_tool_calls` — the shape a malformed provider stream
           *  produces. */
          const chunkMessage = chunk.message as unknown as {
            tool_call_chunks?: Array<{ id?: string; args?: string }>;
          };
          for (const tc of chunkMessage.tool_call_chunks ?? []) {
            if (tc.id === 'tc_ask_invalid') {
              tc.args = '"malformed"';
            }
          }
          yield chunk;
        }
      };
      return model;
    };

    const saver = new MemorySaver();
    const buildRun = async (responses: string[], emitCalls: boolean) => {
      const run = await Run.create<t.IState>({
        runId: 'invalid-ask-resume',
        graphConfig: {
          type: 'standard',
          agents: [
            {
              agentId: 'agent-invalid-ask',
              provider: Providers.OPENAI,
              clientOptions: { model: 'gpt-4o-mini', streaming: true },
              instructions: 'noop',
              maxContextTokens: 8000,
              graphTools: [askTool],
            },
          ],
          compileOptions: { checkpointer: saver },
        },
        returnContent: true,
        customHandlers: {},
        tokenCounter: ((text: string) =>
          String(text).length) as unknown as t.RunConfig['tokenCounter'],
        indexTokenCountMap: {},
      });
      run.Graph!.overrideModel = buildModel(responses, emitCalls);
      return run;
    };
    const config = {
      configurable: { thread_id: 'invalid-ask-thread' },
      streamMode: 'values' as const,
      version: 'v2' as const,
    };

    const run = await buildRun(['Asking.'], true);
    await run.processStream(
      { messages: [new HumanMessage('go')] },
      config
    );
    expect(run.getInterrupt()?.payload).toMatchObject({
      type: 'ask_user_question',
      question: { question: 'Which one?' },
    });

    const resumed = await buildRun(['Done.'], false);
    await resumed.resume({ answer: 'the first one' }, config);
    expect(resumed.getInterrupt()).toBeUndefined();

    const finalCall = modelInvocations[modelInvocations.length - 1];
    const resultIds = new Set(
      finalCall
        .filter((msg) => msg._getType() === 'tool')
        .map((msg) => (msg as ToolMessage).tool_call_id)
    );
    /** Both tool_use blocks ride the paused AI message the provider
     *  replays; each must have a paired result or the call 400s. */
    expect(resultIds.has('tc_ask_valid')).toBe(true);
    expect(resultIds.has('tc_ask_invalid')).toBe(true);
  });

  it('routes an INVALID-ONLY turn through ToolNode at the graph level (toolsCondition)', async () => {
    /**
     * Codex P1: `toolsCondition` used to return END when `tool_calls` was
     * empty, so a turn whose only call was malformed never entered ToolNode —
     * the dangling `tool_use` was committed with no result and no promotion.
     * This drives the REAL graph routing (agent → toolsCondition → toolNode →
     * agent) via a scripted model, not a direct node.invoke.
     */
    const modelInvocations: BaseMessage[][] = [];
    const buildModel = (responses: string[], emitCalls: boolean) => {
      const model = new FakeChatModel({
        responses,
        toolCalls: emitCalls
          ? [{ name: 'echo', args: {}, id: 'tc_solo_invalid', type: 'tool_call' }]
          : [],
      });
      const orig = model._streamResponseChunks.bind(model);
      model._streamResponseChunks = async function* (
        messages,
        options,
        runManager
      ): AsyncGenerator<ChatGenerationChunk> {
        modelInvocations.push(messages);
        for await (const chunk of orig(messages, options, runManager)) {
          const chunkMessage = chunk.message as unknown as {
            tool_call_chunks?: Array<{ id?: string; args?: string }>;
          };
          for (const tc of chunkMessage.tool_call_chunks ?? []) {
            if (tc.id === 'tc_solo_invalid') {
              tc.args = '"malformed"';
            }
          }
          yield chunk;
        }
      };
      return model;
    };

    const run = await Run.create<t.IState>({
      runId: 'invalid-only-graph',
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId: 'agent-invalid-only',
            provider: Providers.OPENAI,
            clientOptions: { model: 'gpt-4o-mini', streaming: true },
            instructions: 'noop',
            maxContextTokens: 8000,
            graphTools: [createEchoTool()],
          },
        ],
      },
      returnContent: true,
      customHandlers: {},
      tokenCounter: ((text: string) =>
        String(text).length) as unknown as t.RunConfig['tokenCounter'],
      indexTokenCountMap: {},
    });
    run.Graph!.overrideModel = buildModel(['Calling.', 'Recovered.'], true);

    const invalidOnlyConfig = {
      configurable: { thread_id: 'invalid-only-thread' },
      streamMode: 'values' as const,
      version: 'v2' as const,
    };
    await run.processStream(
      { messages: [new HumanMessage('go')] },
      invalidOnlyConfig
    );

    /** A second model call happened at all (END would have stopped after one),
     *  and it sees the promoted call paired with its synthesized result. */
    expect(modelInvocations.length).toBeGreaterThan(1);
    const followUp = modelInvocations[1];
    const aiMsg = followUp.find(
      (msg): msg is AIMessage => msg._getType() === 'ai'
    )!;
    expect(aiMsg.tool_calls?.map((c) => c.id)).toEqual(['tc_solo_invalid']);
    expect(aiMsg.invalid_tool_calls).toHaveLength(0);
    const toolMsg = followUp.find(
      (msg): msg is ToolMessage => msg._getType() === 'tool'
    )!;
    expect(toolMsg.tool_call_id).toBe('tc_solo_invalid');
    expect(String(toolMsg.content)).toContain('Malformed');
  });

  it('round-trips through the REAL OpenAI Responses outbound converter with call/output pairing intact', async () => {
    /**
     * Codex P1: `_convertMessagesToOpenAIResponsesParams` rebuilds
     * `function_call` items from `tool_calls` only, so an un-promoted invalid
     * call would vanish while its synthesized `function_call_output` remained
     * — an output whose call_id has no matching call. The promotion keeps the
     * two sides agreeing; this exercises the real outbound converter over the
     * exact message shapes ToolNode emits for a mixed valid/invalid batch.
     */
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
      id: 'ai_responses',
      content: 'Two calls.',
      tool_calls: [{ id: 'tc_ok', name: 'echo', args: { command: 'hi' } }],
      invalid_tool_calls: [
        {
          id: 'tc_bad',
          name: 'echo',
          args: '"malformed"',
          error: 'Malformed args.',
          type: 'invalid_tool_call',
        },
      ],
    });
    const result = await node.invoke(
      { messages: [aiMsg] },
      { configurable: { run_id: 'invalid-responses' } }
    );
    const promoted = toPromotedAiMessage(result)!;
    const toolMessages = toToolMessages(result);

    const items = _convertMessagesToOpenAIResponsesParams(
      [new HumanMessage('go'), promoted, ...toolMessages],
      'gpt-4o-mini'
    ) as Array<{ type?: string; call_id?: string }>;

    const callIds = items
      .filter((item) => item.type === 'function_call')
      .map((item) => item.call_id)
      .sort();
    const outputIds = items
      .filter((item) => item.type === 'function_call_output')
      .map((item) => item.call_id)
      .sort();
    expect(callIds).toEqual(['tc_bad', 'tc_ok']);
    expect(outputIds).toEqual(['tc_bad', 'tc_ok']);
  });
});
