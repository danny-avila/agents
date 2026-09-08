import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import {
  END,
  START,
  StateGraph,
  MemorySaver,
  MessagesAnnotation,
} from '@langchain/langgraph';
import type { RunnableConfig } from '@langchain/core/runnables';
import type { IState } from '@/types';
import {
  TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY,
  HookRegistry,
} from '@/hooks';
import { TOOL_APPROVAL_REVIEW_CONFIG_KEY } from '@/hitl/approvalReview';
import { TOOL_BATCH_REPLAY_KEY } from '../toolBatchReplay';
import { Providers } from '@/common';
import { Run } from '@/run';
import { ToolNode } from '../ToolNode';

function createHarness(carryCheckpointConfig = false) {
  let carriedConfigurable: RunnableConfig['configurable'];
  const checkpointer = new MemorySaver();
  const executions: string[] = [];
  const namespaces: string[] = [];
  const echo = tool(
    async ({ value }) => {
      executions.push(value);
      return value;
    },
    {
      name: 'echo',
      description: 'echo',
      schema: z.object({ value: z.string() }),
    }
  );
  const createRun = async (agentId: string, runId: string, cycles = 1) => {
    const hooks = new HookRegistry();
    hooks.register('PreToolUse', {
      hooks: [async () => ({ decision: 'ask' })],
    });
    const node = new ToolNode({
      tools: [echo],
      agentId,
      hookRegistry: hooks,
      humanInTheLoop: { enabled: true },
      eventDrivenMode: true,
      directToolNames: new Set(['echo']),
    });
    const graph = new StateGraph(MessagesAnnotation)
      .addNode('agent', (state) => {
        const promptIndex = state.messages.findLastIndex(
          (message) => message instanceof HumanMessage
        );
        const completed = state.messages
          .slice(promptIndex)
          .filter((message) => message instanceof ToolMessage).length;
        return {
          messages: [
            new AIMessage(
              completed >= cycles
                ? 'done'
                : {
                  content: '',
                  tool_calls: [
                    {
                      id: `${runId}-${completed}`,
                      name: 'echo',
                      args: { value: `${runId}-${completed}` },
                    },
                  ],
                }
            ),
          ],
        };
      })
      .addNode('tools', async (state, config) => {
        namespaces.push(String(config.configurable?.checkpoint_ns));
        const nodeConfig = {
          ...config,
          configurable: { ...carriedConfigurable, ...config.configurable },
        };
        if (
          carryCheckpointConfig &&
          config.configurable?.[TOOL_APPROVAL_REVIEW_CONFIG_KEY] != null
        ) {
          carriedConfigurable = {
            [TOOL_APPROVAL_REVIEW_CONFIG_KEY]:
              config.configurable[TOOL_APPROVAL_REVIEW_CONFIG_KEY],
            [TOOL_BATCH_REPLAY_KEY]: config.configurable[TOOL_BATCH_REPLAY_KEY],
          };
        }
        return node.invoke(state, nodeConfig);
      })
      .addEdge(START, 'agent')
      .addConditionalEdges('agent', (state) =>
        ((state.messages.at(-1) as AIMessage).tool_calls?.length ?? 0) > 0
          ? 'tools'
          : END
      )
      .addEdge('tools', 'agent')
      .compile({ checkpointer });
    const run = await Run.create<IState>({
      runId,
      hooks,
      humanInTheLoop: { enabled: true },
      graphConfig: {
        type: 'standard',
        agents: [
          {
            agentId,
            provider: Providers.OPENAI,
            clientOptions: { modelName: 'gpt-4o-mini', apiKey: 'test-key' },
            instructions: 'noop',
            maxContextTokens: 8000,
          },
        ],
        compileOptions: { checkpointer },
      },
    });
    run.graphRunnable = graph as typeof run.graphRunnable;
    return run;
  };
  return { createRun, executions, namespaces };
}

function config(scope?: string): RunnableConfig & { version: 'v2' } {
  return {
    configurable: {
      thread_id: 'same-conversation',
      ...(scope == null
        ? {}
        : { [TOOL_APPROVAL_EXECUTION_SCOPE_CONFIG_KEY]: scope }),
    },
    version: 'v2',
  };
}

describe('approval execution scope', () => {
  it.each([
    { rebuilt: false, scoped: false },
    { rebuilt: false, scoped: true },
    { rebuilt: true, scoped: true },
  ])(
    'completes two approval cycles despite changing task namespaces (rebuilt=$rebuilt, scoped=$scoped)',
    async ({ rebuilt, scoped }) => {
      const { createRun, executions, namespaces } = createHarness();
      const run = await createRun('a', 'run-1', 2);
      const runConfig = config(scoped ? 'run-1' : undefined);
      await run.processStream(
        { messages: [new HumanMessage('start')] },
        runConfig
      );
      const firstInterrupt = run.getInterrupt()?.interruptId;
      expect(firstInterrupt).toBeTruthy();
      await run.resume([{ type: 'approve' }], runConfig);
      expect(executions).toEqual(['run-1-0']);
      expect(run.getInterrupt()?.interruptId).toBeTruthy();
      expect(run.getInterrupt()?.interruptId).not.toBe(firstInterrupt);
      const secondResume = rebuilt ? await createRun('a', 'run-1', 2) : run;
      await secondResume.resume([{ type: 'approve' }], runConfig);
      expect(executions).toEqual(['run-1-0', 'run-1-1']);
      expect(secondResume.getInterrupt()).toBeUndefined();
      expect(new Set(namespaces).size).toBeGreaterThan(1);
    }
  );

  it.each(['agent', 'scope'])(
    'fails closed when the %s changes during the pending resume',
    async (changedIdentity) => {
      const { createRun, executions } = createHarness();
      const original = await createRun('a', 'run-1');
      await original.processStream(
        { messages: [new HumanMessage('start')] },
        config('run-1')
      );
      const resumed = await createRun(
        changedIdentity === 'agent' ? 'b' : 'a',
        'run-1'
      );
      await expect(
        resumed.resume(
          [{ type: 'approve' }],
          config(changedIdentity === 'scope' ? 'run-2' : 'run-1')
        )
      ).rejects.toThrow('Tool approval execution owner changed');
      expect(executions).toEqual([]);
    }
  );

  it.each([false, true])(
    'allows an agent switch after a finalized run (explicit scope=%s)',
    async (scoped) => {
      const { createRun, executions } = createHarness(true);
      const first = await createRun('a', 'run-1');
      await first.processStream(
        { messages: [new HumanMessage('first')] },
        config(scoped ? 'run-1' : undefined)
      );
      await first.resume(
        [{ type: 'approve' }],
        config(scoped ? 'run-1' : undefined)
      );
      expect(first.getInterrupt()).toBeUndefined();
      const second = await createRun('b', 'run-2');
      await second.processStream(
        { messages: [new HumanMessage('second')] },
        config(scoped ? 'run-2' : undefined)
      );
      expect(second.getInterrupt()?.payload).toMatchObject({
        type: 'tool_approval',
      });
      expect(executions).toEqual(['run-1-0']);
      await second.resume(
        [{ type: 'approve' }],
        config(scoped ? 'run-2' : undefined)
      );
      expect(executions).toEqual(['run-1-0', 'run-2-0']);
    }
  );
});
