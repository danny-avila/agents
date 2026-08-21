import { HumanMessage } from '@langchain/core/messages';
import type { InjectedMessage } from '@/types/tools';
import { convertInjectedMessages } from './injected';

describe('convertInjectedMessages', () => {
  it('returns an empty array for no entries', () => {
    expect(convertInjectedMessages([])).toEqual([]);
  });

  it('emits one HumanMessage per entry, in order', () => {
    const messages: InjectedMessage[] = [
      { role: 'user', content: 'first' },
      { role: 'user', content: 'second' },
    ];
    const converted = convertInjectedMessages(messages);
    expect(converted).toHaveLength(2);
    expect(converted[0]).toBeInstanceOf(HumanMessage);
    expect(converted[0].content).toBe('first');
    expect(converted[1].content).toBe('second');
  });

  /**
   * Both 'user' and 'system' become `HumanMessage` — Anthropic and Google
   * reject a non-leading SystemMessage — with the original role preserved for
   * downstream consumers.
   */
  it('converts a system role to a HumanMessage carrying the original role', () => {
    const [converted] = convertInjectedMessages([
      { role: 'system', content: 'convention reminder' },
    ]);
    expect(converted).toBeInstanceOf(HumanMessage);
    expect(converted.additional_kwargs.role).toBe('system');
  });

  it('carries isMeta, source and skillName only when set', () => {
    const [bare] = convertInjectedMessages([{ role: 'user', content: 'x' }]);
    expect(bare.additional_kwargs).toEqual({
      role: 'user',
      injected: true,
      provenance: {
        version: 1,
        parts: [{ attribution: 'synthetic' }],
      },
    });

    const [full] = convertInjectedMessages([
      {
        role: 'user',
        content: 'x',
        isMeta: true,
        source: 'steer',
        skillName: 'writing',
      },
    ]);
    expect(full.additional_kwargs).toEqual({
      role: 'user',
      injected: true,
      isMeta: true,
      source: 'steer',
      skillName: 'writing',
      provenance: {
        version: 1,
        parts: [{ attribution: 'user' }],
      },
    });
  });

  /** Both marker fields are optional on `InjectedMessage`, so consumers that
   *  must tell in-run context from payload-replayed messages — compaction
   *  coverage anchors — cannot rely on them. `injected` is unconditional. */
  it('always records injected provenance, whatever the caller supplied', () => {
    const converted = convertInjectedMessages([
      { role: 'user', content: 'bare' },
      { role: 'system', content: 'hook output', source: 'hook' },
      { role: 'user', content: 'injected steer', source: 'steer' },
    ]);

    expect(converted).toHaveLength(3);
    for (const message of converted) {
      expect(message.additional_kwargs.injected).toBe(true);
    }
  });

  it('passes multimodal content through as a content array', () => {
    const [converted] = convertInjectedMessages([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'look at this' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } },
        ],
      },
    ]);
    expect(Array.isArray(converted.content)).toBe(true);
    expect(converted.content).toHaveLength(2);
  });

  /**
   * The provider-safety argument for sealing a stream mid-generation rests on
   * the preempt boundary emitting exactly what the already-shipped tool
   * boundary emits. Both call this function, so shape parity is structural —
   * this pins it against a future divergence.
   */
  it('produces identical shapes for the same entry across calls', () => {
    const entry: InjectedMessage = {
      role: 'user',
      content: 'Skip phase two.',
      source: 'steer',
    };
    const [atToolBoundary] = convertInjectedMessages([entry]);
    const [atPreemptBoundary] = convertInjectedMessages([entry]);
    expect(atPreemptBoundary.content).toEqual(atToolBoundary.content);
    expect(atPreemptBoundary.additional_kwargs).toEqual(
      atToolBoundary.additional_kwargs
    );
    expect(atPreemptBoundary.getType()).toBe(atToolBoundary.getType());
  });
});
