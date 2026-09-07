import { ToolMessage } from '@langchain/core/messages';
import { Command } from '@langchain/langgraph';
import {
  TOOL_BATCH_REPLAY_KEY,
  attachToolBatchReplayState,
  getToolBatchReplayOwner,
  getToolBatchReplayState,
  restoreToolBatchReplayState,
  stripToolBatchReplayState,
  restoreToolReplayConfig,
  getToolBatchReplayScope,
  getPublicToolInterruptPayload,
} from './toolBatchReplay';

const approval = {
  type: 'tool_approval',
  action_requests: [{ tool_call_id: 'call', name: 'echo', arguments: {} }],
  review_configs: [
    {
      tool_call_id: 'call',
      action_name: 'echo',
      allowed_decisions: ['approve'],
    },
  ],
};

describe('checkpoint-owned tool batch replay', () => {
  it.each([undefined, '', 0])('does not cache a missing scope %j', (scope) => {
    expect(getToolBatchReplayScope({ configurable: { thread_id: scope } })).toBeUndefined();
  });
  it('preserves objects resembling the primitive wrapper', async () => {
    const payload = { __librechat_tool_batch_wrapper: 1, __librechat_tool_batch_payload: 'user data' };
    const wrapped = await attachToolBatchReplayState(payload, 'owner', new Map([['batch', new Map()]]));
    expect(getPublicToolInterruptPayload(JSON.parse(JSON.stringify(wrapped)))).toEqual(payload);
  });
  it.each([undefined, {}, { name: 'echo' }])('rejects a missing or incomplete proposal %j', async (proposal) => {
    const wrapped = await attachToolBatchReplayState(approval, 'owner', new Map([['batch', new Map([['call', {
      proposal, output: new ToolMessage({ content: 'done', tool_call_id: 'call' }), additionalContexts: [],
    }]])]]));
    await expect(restoreToolBatchReplayState({ configurable: { [TOOL_BATCH_REPLAY_KEY]: getToolBatchReplayState(wrapped) } }, 'owner'))
      .rejects.toThrow('Invalid settled tool batch results');
  });
  it.each([null, 'confirm', ['one', 'two']])('preserves custom interrupt payload %j with settled results', async (payload) => {
    const wrapped = await attachToolBatchReplayState(payload, 'owner', new Map([['batch', new Map([['call', {
      output: new ToolMessage({ content: 'done', tool_call_id: 'call' }), additionalContexts: [],
    }]])]]));
    expect(getToolBatchReplayState(wrapped)?.records).toHaveLength(1);
    expect(stripToolBatchReplayState(wrapped)).toEqual(payload);
  });
  it('does not reinterpret corrupt approval evidence as permission to execute', () => {
    const configurable = {
      forged: 'retained',
      [TOOL_BATCH_REPLAY_KEY]: { version: 1, records: [] },
    };
    expect(() =>
      restoreToolReplayConfig(configurable, 'interrupt', {
        ...approval,
        action_requests: [null],
      })
    ).toThrow('Invalid tool approval checkpoint');
    expect(configurable).toEqual({ forged: 'retained' });
  });
  it.each([undefined, 'child:task'])(
    'round-trips complete results through the checkpoint codec (%s)',
    async (namespace) => {
      const config = {
        configurable: { thread_id: 'thread', checkpoint_ns: namespace },
      };
      const owner = getToolBatchReplayOwner(config, 'agent');
      const results = new Map([
        [
          'message',
          {
            proposal: { name: 'echo', args: {} },
            output: new ToolMessage({ content: 'saved', tool_call_id: 'call' }),
            additionalContexts: ['context'],
            completionHandled: true,
          },
        ],
        [
          'command',
          {
            proposal: { name: 'handoff', args: {} },
            output: new Command({
              update: {
                messages: [
                  new ToolMessage({
                    content: 'handoff',
                    tool_call_id: 'handoff',
                  }),
                ],
              },
            }),
            additionalContexts: [],
          },
        ],
      ]);
      const payload = await attachToolBatchReplayState(
        approval,
        owner,
        new Map([['batch', results]])
      );
      const state = getToolBatchReplayState(
        JSON.parse(JSON.stringify(payload))
      );
      const restored = await restoreToolBatchReplayState(
        { configurable: { [TOOL_BATCH_REPLAY_KEY]: state } },
        owner
      );
      expect(restored[0].batch).toBe('batch');
      expect(restored[0].results[0]).toEqual([
        'message',
        results.get('message'),
      ]);
      const command = restored[0].results[1][1].output;
      expect(command).toBeInstanceOf(Command);
      expect(command).toMatchObject({
        update: { messages: [expect.any(ToolMessage)] },
      });
      expect(JSON.stringify(command)).toEqual(
        JSON.stringify(results.get('command')?.output)
      );
      expect(stripToolBatchReplayState(payload)).toEqual(approval);
    }
  );

  it('keeps the child approval owner while accumulating distinct parent results', async () => {
    const child = await attachToolBatchReplayState(
      approval,
      'child',
      new Map()
    );
    const parent = await attachToolBatchReplayState(
      child,
      'parent',
      new Map([
        [
          'batch',
          new Map([
            [
              'call',
              {
                output: new ToolMessage({
                  content: 'result',
                  tool_call_id: 'call',
                }),
                additionalContexts: [],
              },
            ],
          ]),
        ],
      ])
    );
    const state = getToolBatchReplayState(parent);
    expect(state?.approvalOwner).toBe('child');
    expect(
      await restoreToolBatchReplayState(
        { configurable: { [TOOL_BATCH_REPLAY_KEY]: state } },
        'unrelated'
      )
    ).toEqual([]);
    expect(state?.records.map((record) => record.owner)).toEqual(['parent']);
  });

  it.each([
    null,
    {},
    { version: 2, records: [] },
    { version: 1, records: [null] },
  ])('fails closed for malformed checkpoint state %j', (state) => {
    expect(() =>
      getToolBatchReplayState({ [TOOL_BATCH_REPLAY_KEY]: state })
    ).toThrow('Invalid tool batch replay checkpoint');
  });

  it('does not require private state in legacy checkpoints', () => {
    expect(getToolBatchReplayState(approval)).toBeUndefined();
    expect(
      getToolBatchReplayState({ [TOOL_BATCH_REPLAY_KEY]: undefined })
    ).toBeUndefined();
  });
});
