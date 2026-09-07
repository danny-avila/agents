import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage } from '@langchain/core/messages';
import { GraphInterrupt } from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import { HookRegistry } from '@/hooks';
import { ToolNode } from '../ToolNode';
import { restoreToolReplayConfig } from '../toolBatchReplay';
import * as events from '@/utils/events';
import { GraphEvents } from '@/common';

afterEach(() => jest.restoreAllMocks());

async function capturePause(node: ToolNode, input: object, config: RunnableConfig): Promise<unknown> {
  try {
    await node.invoke(input, config);
  } catch (error) {
    if (error instanceof GraphInterrupt) {
      return JSON.parse(JSON.stringify(error.interrupts[0].value));
    }
    throw error;
  }
  throw new Error('Expected an interrupt');
}

function replayConfig(payload: unknown): RunnableConfig {
  const configurable = { thread_id: 'checkpoint-replay' };
  restoreToolReplayConfig(configurable, 'pause', payload);
  return { configurable };
}

describe('ToolNode checkpoint authority', () => {
  it('replaces newer cached outputs when selecting an older checkpoint', async () => {
    let generation = 'original';
    let executions = 0;
    let shouldPause = true;
    const work = tool(async () => {
      executions += 1;
      return generation;
    }, { name: 'work', description: 'work', schema: z.object({}) });
    const pause = tool(async () => {
      if (shouldPause) {
        throw new GraphInterrupt([{ id: 'pause', value: 'confirm' }]);
      }
      return 'resumed';
    }, { name: 'pause', description: 'pause', schema: z.object({}) });
    const node = new ToolNode({ tools: [work, pause] });
    const input = { messages: [new AIMessage({ id: 'batch', content: '', tool_calls: [
      { id: 'work', name: 'work', args: {} }, { id: 'pause', name: 'pause', args: {} },
    ] })] };
    const older = await capturePause(node, input, { configurable: { thread_id: 'checkpoint-replay' } });
    const emptyCheckpoint = replayConfig(older);
    const state = emptyCheckpoint.configurable?.__librechat_tool_batch_replay as { records: object[] };
    state.records = [];
    generation = 'newer';
    await capturePause(node, input, emptyCheckpoint);
    expect(executions).toBe(2);
    shouldPause = false;
    const result = await node.invoke(input, replayConfig(older));
    expect(JSON.stringify(result)).toContain('original');
    expect(JSON.stringify(result)).not.toContain('newer');
    expect(executions).toBe(2);
  });

  it.each([false, true])('restores completed and pending turns in a fresh runtime (hooks=%s)', async (hooks) => {
    const observed: Array<{ id: string; turn: number }> = [];
    const completions: Array<{ index: number; tool_call: { id: string } }> = [];
    jest.spyOn(events, 'safeDispatchCustomEvent').mockImplementation(async (event, data) => {
      if (event === GraphEvents.ON_RUN_STEP_COMPLETED) {
        completions.push((data as { result: (typeof completions)[number] }).result);
      }
      return true;
    });
    let shouldPause = true;
    const work = tool(async ({ pause }, config) => {
      const call = config.toolCall as { id: string; turn: number };
      observed.push({ id: call.id, turn: call.turn });
      if (pause && shouldPause) {
        throw new GraphInterrupt([{ id: 'pause', value: 'confirm' }]);
      }
      return `turn-${call.turn}`;
    }, { name: 'work', description: 'work', schema: z.object({ pause: z.boolean() }) });
    const makeNode = () => {
      const registry = new HookRegistry();
      registry.register('PreToolUse', { hooks: [async () => ({ decision: 'allow' })] });
      return new ToolNode({ tools: [work], hookRegistry: hooks ? registry : undefined,
        toolCallStepIds: new Map(['earlier', 'completed', 'pending'].map((id) => [id, `step-${id}`])),
      });
    };
    const node = makeNode();
    const config = { configurable: { thread_id: 'checkpoint-replay' } };
    await node.invoke({ messages: [new AIMessage({ id: 'earlier', content: '', tool_calls: [
      { id: 'earlier', name: 'work', args: { pause: false } },
    ] })] }, config);
    const input = { messages: [new AIMessage({ id: 'batch', content: '', tool_calls: [
      { id: 'completed', name: 'work', args: { pause: false } },
      { id: 'pending', name: 'work', args: { pause: true } },
    ] })] };
    const checkpoint = await capturePause(node, input, config);
    expect(observed).toEqual([{ id: 'earlier', turn: 0 }, { id: 'completed', turn: 1 }, { id: 'pending', turn: 2 }]);
    shouldPause = false;
    const fresh = makeNode();
    const resumed = await fresh.invoke(input, replayConfig(checkpoint));
    expect(observed).toEqual([{ id: 'earlier', turn: 0 }, { id: 'completed', turn: 1 }, { id: 'pending', turn: 2 }, { id: 'pending', turn: 2 }]);
    expect(JSON.stringify(resumed)).toContain('turn-1');
    expect(JSON.stringify(resumed)).toContain('turn-2');
    expect(fresh.getToolUsageCounts().get('work')).toBe(3);
    expect(completions.map(({ index, tool_call }) => ({ id: tool_call.id, index }))).toEqual([
      { id: 'earlier', index: 0 }, { id: 'completed', index: 1 }, { id: 'pending', index: 2 },
    ]);
  });
});
