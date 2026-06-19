import { HumanMessage } from '@langchain/core/messages';
import { messagesStateReducer } from './reducer';

describe('messagesStateReducer', () => {
  it('drops null/undefined entries and preserves order', () => {
    const a = new HumanMessage({ id: 'a', content: 'a' });
    const b = new HumanMessage({ id: 'b', content: 'b' });
    const c = new HumanMessage({ id: 'c', content: 'c' });

    const result = messagesStateReducer(
      [a, null, b] as never,
      [undefined, c] as never
    );

    expect(result.map((m) => m.id)).toEqual(['a', 'b', 'c']);
  });

  it('does not throw when every entry is null/undefined', () => {
    expect(() =>
      messagesStateReducer(
        [null, undefined] as never,
        [undefined, null] as never
      )
    ).not.toThrow();

    const result = messagesStateReducer(
      [null, undefined] as never,
      [undefined, null] as never
    );
    expect(result).toEqual([]);
  });
});
