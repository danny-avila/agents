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
import { askUserQuestion } from '@/hitl/askUserQuestion';
import { FakeChatModel } from '@/llm/fake';
import { Providers } from '@/common';
import { ToolNode } from '../ToolNode';
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

function toToolMessages(result: unknown): ToolMessage[] {
  const messages = Array.isArray(result)
    ? result
    : (result as { messages: BaseMessage[] }).messages;
  return messages.filter(
    (msg): msg is ToolMessage => msg._getType() === 'tool'
  );
}

describe('ToolNode invalid_tool_calls handling', () => {
  it('synthesizes an error ToolMessage for an invalid call alongside real results (direct batch)', async () => {
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
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
  });

  it('synthesizes error ToolMessages when EVERY call in the batch is invalid', async () => {
    const node = new ToolNode({ tools: [createEchoTool()] });
    const aiMsg = new AIMessage({
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
});
