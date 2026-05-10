import { GraphEvents, StepTypes } from '@/common';
import { composeEventHandlers } from '@/events';
import { createRunHandlers } from '@/session';
import type { AgentSessionStreamEvent } from '@/session';
import type * as t from '@/types';

describe('createRunHandlers', () => {
  it('emits live session events through the same graph handlers before user handlers', async () => {
    const liveEvents: AgentSessionStreamEvent[] = [];
    let liveEventCountSeenByUserHandler = 0;
    const handlerResult = createRunHandlers({
      runId: 'run_live',
      threadId: 'thread_live',
      onEvent: (event) => {
        liveEvents.push(event);
      },
      userHandlers: {
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (): void => {
            liveEventCountSeenByUserHandler = liveEvents.length;
          },
        },
      },
    });

    await handlerResult.handlers[GraphEvents.ON_RUN_STEP].handle(
      GraphEvents.ON_RUN_STEP,
      {
        stepIndex: 0,
        id: 'message_step',
        type: StepTypes.MESSAGE_CREATION,
        index: 0,
        stepDetails: {
          type: StepTypes.MESSAGE_CREATION,
          message_creation: { message_id: 'message_step' },
        },
        usage: null,
      } satisfies t.RunStep
    );
    await handlerResult.handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'message_step',
        delta: { content: [{ type: 'text', text: 'hello' }] },
      } satisfies t.MessageDeltaEvent
    );

    expect(liveEvents.map((event) => event.type)).toEqual([
      'run.started',
      'message.delta',
    ]);
    expect(liveEventCountSeenByUserHandler).toBe(2);
    expect(handlerResult.contentParts[0]).toEqual({
      type: 'text',
      text: 'hello',
    });
  });

  it('composes adapter and host handlers in order for the same graph event', async () => {
    const calls: string[] = [];
    const handlers = composeEventHandlers(
      {
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (): void => {
            calls.push('adapter');
          },
        },
      },
      {
        [GraphEvents.ON_MESSAGE_DELTA]: {
          handle: (): void => {
            calls.push('host');
          },
        },
      }
    );

    await handlers[GraphEvents.ON_MESSAGE_DELTA].handle(
      GraphEvents.ON_MESSAGE_DELTA,
      {
        id: 'message_step',
        delta: { content: [{ type: 'text', text: 'hello' }] },
      } satisfies t.MessageDeltaEvent
    );

    expect(calls).toEqual(['adapter', 'host']);
  });
});
