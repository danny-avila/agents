import type * as t from '@/types';
import {
  ChatModelStreamHandler,
  dispatchesChatModelStream,
  SDK_STREAM_DISPATCH,
} from '@/stream';
import { composeEventHandlers } from '@/events';
import { GraphEvents } from '@/common';

const inert: t.EventHandler = {
  handle: (): void => {
    /* renders nothing */
  },
};

describe('dispatchesChatModelStream', () => {
  it('recognizes the dispatcher itself', () => {
    expect(dispatchesChatModelStream(new ChatModelStreamHandler())).toBe(true);
  });

  it('is false for a handler that owns no content-part dispatch', () => {
    expect(dispatchesChatModelStream(inert)).toBe(false);
  });

  it('is false for nothing registered', () => {
    expect(dispatchesChatModelStream(undefined)).toBe(false);
  });

  /**
   * The reason identity is not a usable test: hosts wrap handlers, and every
   * wrapper fails `instanceof` while still driving the same dispatch. A run
   * that seals on that basis would be sealing against its own contract.
   */
  it('sees through a composed wrapper in either position', () => {
    const first = composeEventHandlers(
      { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler() },
      { [GraphEvents.CHAT_MODEL_STREAM]: inert }
    )[GraphEvents.CHAT_MODEL_STREAM];
    const second = composeEventHandlers(
      { [GraphEvents.CHAT_MODEL_STREAM]: inert },
      { [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler() }
    )[GraphEvents.CHAT_MODEL_STREAM];

    expect(first).not.toBeInstanceOf(ChatModelStreamHandler);
    expect(second).not.toBeInstanceOf(ChatModelStreamHandler);
    expect(dispatchesChatModelStream(first)).toBe(true);
    expect(dispatchesChatModelStream(second)).toBe(true);
  });

  it('does not brand a composition of inert handlers', () => {
    const composed = composeEventHandlers(
      { [GraphEvents.CHAT_MODEL_STREAM]: inert },
      { [GraphEvents.CHAT_MODEL_STREAM]: inert }
    )[GraphEvents.CHAT_MODEL_STREAM];
    expect(dispatchesChatModelStream(composed)).toBe(false);
  });

  it('exposes the brand as a cross-realm-safe registered symbol', () => {
    expect(SDK_STREAM_DISPATCH).toBe(
      Symbol.for('@librechat/agents:chatModelStreamDispatch')
    );
  });
});
