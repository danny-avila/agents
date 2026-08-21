import { HumanMessage } from '@langchain/core/messages';
import type { ProviderMessageProvenancePart } from './provenance';
import {
  appendProviderMessageProvenance,
  getProviderMessageProvenance,
  getProviderSourceMessageIds,
  setProviderMessageProvenance,
} from './provenance';

describe('provider message provenance', () => {
  it('synchronizes stable plural ids from ordered lineage', () => {
    const message = new HumanMessage({
      content: 'merged',
      additional_kwargs: { sourceMessageId: 'last' },
    });

    setProviderMessageProvenance(message, [
      { attribution: 'user', sourceMessageId: 'first' },
      { attribution: 'synthetic' },
      { attribution: 'user', sourceMessageId: 'last' },
      { attribution: 'user', sourceMessageId: 'first' },
    ]);

    expect(message.additional_kwargs.sourceMessageId).toBe('last');
    expect(message.additional_kwargs.sourceMessageIds).toEqual([
      'first',
      'last',
    ]);
    expect(getProviderSourceMessageIds(message)).toEqual(['first', 'last']);
  });

  it('coalesces adjacent contributions through the compatibility helper', () => {
    const message = new HumanMessage({ content: 'derived' });
    appendProviderMessageProvenance(message, {
      attribution: 'model',
      sourceMessageId: 'source',
      sourceContentPartIndices: [2],
    });
    appendProviderMessageProvenance(message, {
      attribution: 'model',
      sourceMessageId: 'source',
      sourceContentPartIndices: [4, 2],
    });

    expect(getProviderMessageProvenance(message)).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'model',
          sourceMessageId: 'source',
          sourceContentPartIndices: [2, 4],
        },
      ],
    });
  });

  it('migrates singular lineage when appending to a legacy message', () => {
    const message = new HumanMessage({
      content: 'legacy',
      additional_kwargs: {
        sourceMessageId: 'legacy-source',
        sourceMessageIds: ['legacy-source'],
      },
    });

    appendProviderMessageProvenance(message, {
      attribution: 'user',
      sourceContentPartIndices: [1],
    });

    expect(getProviderMessageProvenance(message)).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'legacy-source',
          sourceContentPartIndices: [1],
        },
      ],
    });
  });

  it('does not inherit a later contribution attribution for legacy lineage', () => {
    const message = new HumanMessage({
      content: 'legacy plus derived',
      additional_kwargs: { sourceMessageId: 'legacy-user-row' },
    });

    appendProviderMessageProvenance(message, {
      attribution: 'synthetic',
      sourceMessageId: 'derived-row',
    });

    expect(getProviderMessageProvenance(message)?.parts).toEqual([
      { attribution: 'user', sourceMessageId: 'legacy-user-row' },
      { attribution: 'synthetic', sourceMessageId: 'derived-row' },
    ]);

    const implicitSourceMessage = new HumanMessage({
      content: 'legacy plus derived',
      additional_kwargs: { sourceMessageId: 'legacy-user-row' },
    });
    appendProviderMessageProvenance(implicitSourceMessage, {
      attribution: 'synthetic',
      sourceContentPartIndices: [1],
    });
    expect(getProviderMessageProvenance(implicitSourceMessage)?.parts).toEqual([
      { attribution: 'user', sourceMessageId: 'legacy-user-row' },
      {
        attribution: 'synthetic',
        sourceMessageId: 'legacy-user-row',
        sourceContentPartIndices: [1],
      },
    ]);
  });

  it('does not collapse indexed and unindexed contributions', () => {
    const message = new HumanMessage({ content: 'mixed mapping' });
    appendProviderMessageProvenance(message, {
      attribution: 'tool',
      sourceMessageId: 'source',
      sourceContentPartIndices: [0],
    });
    appendProviderMessageProvenance(message, {
      attribution: 'tool',
      sourceMessageId: 'source',
    });

    expect(getProviderMessageProvenance(message)?.parts).toEqual([
      {
        attribution: 'tool',
        sourceMessageId: 'source',
        sourceContentPartIndices: [0],
      },
      { attribution: 'tool', sourceMessageId: 'source' },
    ]);
  });

  it('replaces stale source lineage when stamping a new origin', () => {
    const message = new HumanMessage({
      content: 'legacy merge',
      additional_kwargs: {
        sourceMessageId: 'second',
        sourceMessageIds: ['first', 'second'],
      },
    });

    setProviderMessageProvenance(message, [
      { attribution: 'user', sourceMessageId: 'replacement' },
    ]);

    expect(getProviderMessageProvenance(message)).toEqual({
      version: 1,
      parts: [{ attribution: 'user', sourceMessageId: 'replacement' }],
    });
    expect(message.additional_kwargs.sourceMessageId).toBe('replacement');
    expect(getProviderSourceMessageIds(message)).toEqual(['replacement']);
  });

  it('rejects malformed untrusted provenance and retains safe id fallback', () => {
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: {
        sourceMessageIds: ['safe-fallback'],
        provenance: { version: 1, parts: [null] },
      },
    });

    expect(getProviderMessageProvenance(message)).toBeUndefined();
    expect(getProviderSourceMessageIds(message)).toEqual(['safe-fallback']);
  });

  it('publishes deeply immutable metadata and rejects invalid setter attribution', () => {
    const message = new HumanMessage({ content: 'safe' });
    setProviderMessageProvenance(message, [
      {
        attribution: 'user',
        sourceMessageId: 'source',
        sourceContentPartIndices: [0],
      },
    ]);

    const provenance = getProviderMessageProvenance(message)!;
    expect(Object.isFrozen(provenance)).toBe(true);
    expect(Object.isFrozen(provenance.parts)).toBe(true);
    expect(Object.isFrozen(provenance.parts[0])).toBe(true);
    expect(Object.isFrozen(provenance.parts[0].sourceContentPartIndices)).toBe(
      true
    );
    expect(Object.isFrozen(message.additional_kwargs.sourceMessageIds)).toBe(
      true
    );
    expect(Reflect.set(provenance.parts[0], 'attribution', 'tool')).toBe(false);
    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'invalid' } as unknown as ProviderMessageProvenancePart,
      ])
    ).toThrow(TypeError);
    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'user', sourceContentPartIndices: [] },
      ])
    ).toThrow(TypeError);
    expect(getProviderMessageProvenance(message)).toEqual(provenance);
  });

  it('captures accessor-backed setter fields exactly once before publication', () => {
    const reads = { attribution: 0, sourceMessageId: 0, indices: 0 };
    const input = {
      get attribution() {
        reads.attribution++;
        return reads.attribution === 1 ? 'user' : 'attacker';
      },
      get sourceMessageId() {
        reads.sourceMessageId++;
        return reads.sourceMessageId === 1 ? 'source' : 'attacker';
      },
      get sourceContentPartIndices() {
        reads.indices++;
        return reads.indices === 1 ? [1] : [-1];
      },
    } as unknown as ProviderMessageProvenancePart;
    const message = new HumanMessage({ content: 'safe' });

    setProviderMessageProvenance(message, [input]);

    expect(reads).toEqual({ attribution: 1, sourceMessageId: 1, indices: 1 });
    expect(getProviderMessageProvenance(message)).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'source',
          sourceContentPartIndices: [1],
        },
      ],
    });
  });

  it('returns a canonical snapshot of accessor-backed external provenance', () => {
    const reads = { version: 0, parts: 0, attribution: 0, sourceId: 0 };
    const part = {
      get attribution() {
        reads.attribution++;
        return reads.attribution === 1 ? 'user' : 'attacker';
      },
      get sourceMessageId() {
        reads.sourceId++;
        return reads.sourceId === 1 ? 'source' : 'attacker';
      },
    };
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: {
        provenance: {
          get version() {
            reads.version++;
            return reads.version === 1 ? 1 : 2;
          },
          get parts() {
            reads.parts++;
            return reads.parts === 1 ? [part] : null;
          },
        },
      },
    });

    const provenance = getProviderMessageProvenance(message);

    expect(reads).toEqual({
      version: 1,
      parts: 1,
      attribution: 1,
      sourceId: 1,
    });
    expect(provenance).toEqual({
      version: 1,
      parts: [{ attribution: 'user', sourceMessageId: 'source' }],
    });
    expect(Object.isFrozen(provenance)).toBe(true);
    expect(Object.isFrozen(provenance?.parts)).toBe(true);
    expect(Object.isFrozen(provenance?.parts[0])).toBe(true);
  });

  it('reads hostile public arrays by captured index rather than callbacks', () => {
    const parts = [
      {
        attribution: 'user',
        sourceMessageId: 'source',
        sourceContentPartIndices: [0],
      },
    ];
    Object.defineProperty(parts, 'map', {
      value: () => [{ attribution: 'attacker' }],
    });
    Object.defineProperty(parts[0].sourceContentPartIndices, Symbol.iterator, {
      value: function* () {},
    });
    const message = new HumanMessage({ content: 'safe' });

    setProviderMessageProvenance(
      message,
      parts as unknown as ProviderMessageProvenancePart[]
    );

    expect(getProviderMessageProvenance(message)).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'source',
          sourceContentPartIndices: [0],
        },
      ],
    });
  });

  it('captures plural ids once and publishes metadata with one atomic assignment', () => {
    let pluralReads = 0;
    const message = new HumanMessage({ content: 'external' });
    Object.defineProperty(message.additional_kwargs, 'sourceMessageIds', {
      configurable: true,
      get() {
        pluralReads++;
        return pluralReads === 1 ? ['safe'] : null;
      },
    });

    expect(getProviderSourceMessageIds(message)).toEqual(['safe']);
    expect(pluralReads).toBe(1);

    const previousAdditionalKwargs = {
      sourceMessageId: 'legacy',
      retained: true,
    };
    const proxiedAdditionalKwargs = new Proxy(previousAdditionalKwargs, {
      set() {
        throw new Error('partial write');
      },
    });
    message.additional_kwargs = proxiedAdditionalKwargs;

    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).not.toThrow();
    expect(message.additional_kwargs).not.toBe(proxiedAdditionalKwargs);
    expect(message.additional_kwargs.retained).toBe(true);
    expect(previousAdditionalKwargs).toEqual({
      sourceMessageId: 'legacy',
      retained: true,
    });
    expect(getProviderSourceMessageIds(message)).toEqual(['replacement']);
  });

  it('preserves complete lineage above recommended consumer trust bounds', () => {
    const message = new HumanMessage({ content: 'large' });
    const sourceContentPartIndices = Array.from(
      { length: 257 },
      (_, index) => index
    );

    setProviderMessageProvenance(message, [
      {
        attribution: 'user',
        sourceMessageId: 'large-source',
        sourceContentPartIndices,
      },
    ]);

    expect(
      getProviderMessageProvenance(message)?.parts[0].sourceContentPartIndices
    ).toHaveLength(257);
  });

  it('rebuilds safely after external provenance replacement', () => {
    const message = new HumanMessage({ content: 'external' });
    setProviderMessageProvenance(message, [
      { attribution: 'user', sourceMessageId: 'source' },
    ]);
    message.additional_kwargs.provenance = null;

    appendProviderMessageProvenance(message, {
      attribution: 'user',
      sourceContentPartIndices: [1],
    });

    expect(getProviderMessageProvenance(message)).toEqual({
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'source',
          sourceContentPartIndices: [1],
        },
      ],
    });
  });
});
