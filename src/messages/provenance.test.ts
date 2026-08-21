import { HumanMessage } from '@langchain/core/messages';
import type { ProviderMessageProvenancePart } from './provenance';
import {
  appendProviderMessageProvenance,
  getProviderMessageProvenance,
  getProviderSourceMessageIds,
  PROVIDER_MESSAGE_PROVENANCE_LIMITS,
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
    expect(message.lc_kwargs.additional_kwargs).toBe(message.additional_kwargs);
    const serialized = message.toJSON() as unknown as {
      kwargs: { additional_kwargs: unknown };
    };
    expect(serialized.kwargs.additional_kwargs).toEqual(
      message.additional_kwargs
    );
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

    delete message.additional_kwargs.sourceMessageIds;
    message.additional_kwargs.sourceMessageId = 'legacy';
    const previousAdditionalKwargs = message.additional_kwargs;
    previousAdditionalKwargs.retained = true;

    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).not.toThrow();
    expect(message.additional_kwargs).not.toBe(previousAdditionalKwargs);
    expect(message.additional_kwargs.retained).toBe(true);
    expect(previousAdditionalKwargs.retained).toBe(true);
    expect(getProviderSourceMessageIds(message)).toEqual(['replacement']);
  });

  it('rejects proxied live kwargs without reading or partially publishing them', () => {
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: { sourceMessageId: 'legacy' },
    });
    const previousSerialized = message.lc_kwargs.additional_kwargs;
    let reads = 0;
    message.additional_kwargs = new Proxy(message.additional_kwargs, {
      ownKeys() {
        reads++;
        return [];
      },
    });

    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).toThrow('Invalid provider message additional kwargs');
    expect(reads).toBe(0);
    expect(message.lc_kwargs.additional_kwargs).toBe(previousSerialized);
  });

  it('does not publish live metadata when serialized metadata rejects the write', () => {
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: { sourceMessageId: 'legacy' },
    });
    const previousAdditionalKwargs = message.additional_kwargs;
    const previousSerializedAdditionalKwargs =
      message.lc_kwargs.additional_kwargs;
    const previousLcKwargs = message.lc_kwargs;
    Object.defineProperty(message, 'lc_kwargs', {
      configurable: true,
      get: () => previousLcKwargs,
      set: () => {
        throw new Error('serialized write rejected');
      },
    });

    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).toThrow('Invalid provider message serialization kwargs');
    expect(message.additional_kwargs).toBe(previousAdditionalKwargs);
    expect(message.lc_kwargs.additional_kwargs).toBe(
      previousSerializedAdditionalKwargs
    );
  });

  it('rolls serialized metadata back when the live message rejects the write', () => {
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: { sourceMessageId: 'legacy' },
    });
    const previousAdditionalKwargs = message.additional_kwargs;
    const previousSerializedAdditionalKwargs =
      message.lc_kwargs.additional_kwargs;
    let accessorReads = 0;
    Object.defineProperty(message, 'additional_kwargs', {
      configurable: true,
      get: () => {
        accessorReads++;
        return previousAdditionalKwargs;
      },
      set: () => {
        throw new Error('live write rejected');
      },
    });

    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).toThrow('Invalid provider message serialization kwargs');
    expect(accessorReads).toBe(0);
    expect(message.lc_kwargs.additional_kwargs).toBe(
      previousSerializedAdditionalKwargs
    );
  });

  it('fails closed before walking oversized sparse plural id arrays', () => {
    let elementReads = 0;
    const sourceMessageIds = new Array(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds + 1
    );
    Object.defineProperty(sourceMessageIds, '0', {
      get() {
        elementReads++;
        return 'attacker';
      },
    });
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: { sourceMessageIds },
    });

    expect(getProviderSourceMessageIds(message)).toEqual([]);
    expect(elementReads).toBe(0);
  });

  it('fails closed when a proxied plural array reports a non-finite length', () => {
    const sourceMessageIds = new Proxy(['hidden-user-source'], {
      get(target, property, receiver) {
        return property === 'length'
          ? Number.NaN
          : Reflect.get(target, property, receiver);
      },
    });
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: {
        sourceMessageId: 'visible-tool-source',
        sourceMessageIds,
      },
    });

    expect(getProviderSourceMessageIds(message)).toEqual([]);
  });

  it('fails closed when untrusted plural lineage is not an array', () => {
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: {
        sourceMessageId: 'must-not-fall-through',
        sourceMessageIds: { 0: 'forged', length: 1 },
      },
    });

    expect(getProviderSourceMessageIds(message)).toEqual([]);
  });

  it('fails closed when an untrusted plural source id exceeds the bound', () => {
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: {
        sourceMessageId: 'must-not-fall-through',
        sourceMessageIds: [
          'x'.repeat(
            PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIdLength + 1
          ),
        ],
      },
    });

    expect(getProviderSourceMessageIds(message)).toEqual([]);
  });

  it('rejects proxied messages and serialization kwargs without partial publication', () => {
    const target = new HumanMessage({
      content: 'external',
      additional_kwargs: { sourceMessageId: 'legacy' },
    });
    const previousLive = target.additional_kwargs;
    const previousSerialized = target.lc_kwargs.additional_kwargs;

    expect(() =>
      setProviderMessageProvenance(new Proxy(target, {}), [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).toThrow('Invalid provider message serialization kwargs');
    expect(target.additional_kwargs).toBe(previousLive);
    expect(target.lc_kwargs.additional_kwargs).toBe(previousSerialized);

    target.lc_kwargs = new Proxy(target.lc_kwargs, {});
    expect(() =>
      setProviderMessageProvenance(target, [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).toThrow('Invalid provider message serialization kwargs');
    expect(target.additional_kwargs).toBe(previousLive);
    expect(target.lc_kwargs.additional_kwargs).toBe(previousSerialized);
  });

  it('rejects accessor-backed serialization kwargs without invoking them', () => {
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: { sourceMessageId: 'legacy' },
    });
    const previousLive = message.additional_kwargs;
    let accessorReads = 0;
    Object.defineProperty(message.lc_kwargs, 'hostile', {
      configurable: true,
      enumerable: true,
      get() {
        accessorReads++;
        return 'attacker';
      },
    });

    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'user', sourceMessageId: 'replacement' },
      ])
    ).toThrow('Invalid provider message serialization kwargs');
    expect(accessorReads).toBe(0);
    expect(message.additional_kwargs).toBe(previousLive);
  });

  it('fails closed before walking oversized sparse provenance arrays', () => {
    let partReads = 0;
    const parts = new Array(PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts + 1);
    Object.defineProperty(parts, '0', {
      get() {
        partReads++;
        return { attribution: 'user' };
      },
    });
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: { provenance: { version: 1, parts } },
    });

    expect(getProviderMessageProvenance(message)).toBeUndefined();
    expect(partReads).toBe(0);
  });

  it('rejects proxied provenance arrays that report a non-finite length', () => {
    const parts = new Proxy([{ attribution: 'tool' as const }], {
      get(target, property, receiver) {
        return property === 'length'
          ? Number.NaN
          : Reflect.get(target, property, receiver);
      },
    });
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: { provenance: { version: 1, parts } },
    });

    expect(getProviderMessageProvenance(message)).toBeUndefined();
    expect(() => setProviderMessageProvenance(message, parts)).toThrow(
      'Provider message provenance parts must be an array'
    );
  });

  it('fails closed before walking oversized sparse part-index arrays', () => {
    let indexReads = 0;
    const sourceContentPartIndices = new Array(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxIndicesPerPart + 1
    );
    Object.defineProperty(sourceContentPartIndices, '0', {
      get() {
        indexReads++;
        return 0;
      },
    });
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: {
        provenance: {
          version: 1,
          parts: [{ attribution: 'user', sourceContentPartIndices }],
        },
      },
    });

    expect(getProviderMessageProvenance(message)).toBeUndefined();
    expect(indexReads).toBe(0);
  });

  it('does not broaden proxied part indices into whole-message tool attribution', () => {
    const sourceContentPartIndices = new Proxy([0], {
      get(target, property, receiver) {
        return property === 'length'
          ? Number.NaN
          : Reflect.get(target, property, receiver);
      },
    });
    const message = new HumanMessage({
      content: 'external',
      additional_kwargs: {
        provenance: {
          version: 1,
          parts: [{ attribution: 'tool', sourceContentPartIndices }],
        },
      },
    });

    expect(getProviderMessageProvenance(message)).toBeUndefined();
    expect(() =>
      setProviderMessageProvenance(message, [
        { attribution: 'tool', sourceContentPartIndices },
      ])
    ).toThrow('Provider source content part indices must be a non-empty array');
  });

  it('fails closed on excessive total refs, source indices, and id lengths', () => {
    const fullIndexList = Array.from(
      { length: PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxIndicesPerPart },
      (_, index) => index
    );
    const tooManyRefs = Array.from({ length: 17 }, () => ({
      attribution: 'user' as const,
      sourceContentPartIndices: fullIndexList,
    }));
    const message = new HumanMessage({ content: 'external' });

    message.additional_kwargs.provenance = {
      version: 1,
      parts: tooManyRefs,
    };
    expect(getProviderMessageProvenance(message)).toBeUndefined();

    message.additional_kwargs.provenance = {
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceContentPartIndices: [
            PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceContentPartIndex + 1,
          ],
        },
      ],
    };
    expect(getProviderMessageProvenance(message)).toBeUndefined();

    message.additional_kwargs.provenance = {
      version: 1,
      parts: [
        {
          attribution: 'user',
          sourceMessageId: 'x'.repeat(
            PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIdLength + 1
          ),
        },
      ],
    };
    expect(getProviderMessageProvenance(message)).toBeUndefined();
  });

  it('preserves complete lineage above recommended consumer trust bounds', () => {
    const message = new HumanMessage({ content: 'large' });
    const longSourceMessageId = 'x'.repeat(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIdLength + 1
    );
    const parts = Array.from(
      { length: PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts + 1 },
      (_, index) => ({
        attribution: 'user' as const,
        sourceMessageId:
          index === 0 ? longSourceMessageId : `large-source-${index}`,
        sourceContentPartIndices: [
          PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceContentPartIndex +
            index +
            1,
        ],
      })
    );

    setProviderMessageProvenance(message, parts);

    expect(getProviderMessageProvenance(message)?.parts).toHaveLength(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts + 1
    );
    expect(getProviderSourceMessageIds(message)).toHaveLength(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds + 1
    );
    expect(getProviderSourceMessageIds(message)[0]).toBe(longSourceMessageId);

    delete message.additional_kwargs.provenance;
    expect(getProviderSourceMessageIds(message)).toHaveLength(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds + 1
    );
  });

  it('fails closed when untrusted source fields are mixed into oversized trusted lineage', () => {
    const message = new HumanMessage({ content: 'large' });
    const parts = Array.from(
      { length: PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds + 1 },
      (_, index) => ({
        attribution: 'user' as const,
        sourceMessageId: `trusted-${index}`,
      })
    );
    setProviderMessageProvenance(message, parts);

    message.additional_kwargs.sourceMessageIds = Array.from(
      { length: PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds },
      (_, index) => `external-${index}`
    );
    expect(getProviderSourceMessageIds(message)).toEqual([]);

    setProviderMessageProvenance(message, parts);
    message.additional_kwargs.sourceMessageId = 'x'.repeat(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIdLength + 1
    );
    expect(getProviderSourceMessageIds(message)).toEqual([]);

    setProviderMessageProvenance(message, parts);
    message.additional_kwargs.sourceMessageId = { forged: true };
    expect(getProviderSourceMessageIds(message)).toEqual([]);
  });

  it('preflights an untrusted plural before walking oversized trusted provenance', () => {
    const message = new HumanMessage({ content: 'large' });
    setProviderMessageProvenance(
      message,
      Array.from(
        { length: PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts + 1 },
        (_, index) => ({
          attribution: 'user' as const,
          sourceMessageId: `trusted-${index}`,
        })
      )
    );
    let pluralElementReads = 0;
    const untrustedPlural = ['external'];
    Object.defineProperty(untrustedPlural, '0', {
      configurable: true,
      enumerable: true,
      get() {
        pluralElementReads++;
        return 'external';
      },
    });
    message.additional_kwargs.sourceMessageIds = untrustedPlural;
    const setHasSpy = jest.spyOn(Set.prototype, 'has');

    const result = getProviderSourceMessageIds(message);
    const sourceIdLookups = setHasSpy.mock.calls.length;
    setHasSpy.mockRestore();

    expect(result).toEqual([]);
    expect(pluralElementReads).toBe(0);
    expect(sourceIdLookups).toBe(0);
  });

  it.each([
    {
      label: 'invalid',
      singular: Object.defineProperty({}, 'value', {
        enumerable: true,
        get: jest.fn(() => 'attacker'),
      }),
    },
    {
      label: 'overlong',
      singular: 'x'.repeat(
        PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIdLength + 1
      ),
    },
  ])(
    'preflights a $label singular before walking oversized trusted provenance',
    ({ singular }) => {
      const message = new HumanMessage({ content: 'large' });
      setProviderMessageProvenance(
        message,
        Array.from(
          { length: PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts + 1 },
          (_, index) => ({
            attribution: 'user' as const,
            sourceMessageId: `trusted-${index}`,
          })
        )
      );
      message.additional_kwargs.sourceMessageId = singular;
      const setHasSpy = jest.spyOn(Set.prototype, 'has');

      const result = getProviderSourceMessageIds(message);
      const sourceIdLookups = setHasSpy.mock.calls.length;
      setHasSpy.mockRestore();

      expect(result).toEqual([]);
      expect(sourceIdLookups).toBe(0);
      const valueDescriptor = Object.getOwnPropertyDescriptor(
        singular,
        'value'
      );
      if (valueDescriptor?.get != null) {
        expect(valueDescriptor.get).not.toHaveBeenCalled();
      }
    }
  );

  it('does not combine independently trusted provenance and plural arrays above limits', () => {
    const first = new HumanMessage({ content: 'first' });
    const second = new HumanMessage({ content: 'second' });
    const half = Math.floor(
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds / 2
    );
    setProviderMessageProvenance(
      first,
      Array.from({ length: half + 1 }, (_, index) => ({
        attribution: 'user' as const,
        sourceMessageId: `first-${index}`,
      }))
    );
    setProviderMessageProvenance(
      second,
      Array.from({ length: half + 1 }, (_, index) => ({
        attribution: 'user' as const,
        sourceMessageId: `second-${index}`,
      }))
    );

    first.additional_kwargs.sourceMessageIds =
      second.additional_kwargs.sourceMessageIds;
    expect(getProviderSourceMessageIds(first)).toEqual([]);
  });

  it('rejects an oversized trusted pair splice before walking either side', () => {
    const first = new HumanMessage({ content: 'first' });
    const second = new HumanMessage({ content: 'second' });
    const oversizedLength =
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds + 1;
    setProviderMessageProvenance(
      first,
      Array.from({ length: oversizedLength }, (_, index) => ({
        attribution: 'user' as const,
        sourceMessageId: `first-${index}`,
      }))
    );
    setProviderMessageProvenance(
      second,
      Array.from({ length: oversizedLength }, (_, index) => ({
        attribution: 'user' as const,
        sourceMessageId: `second-${index}`,
      }))
    );
    first.additional_kwargs.sourceMessageIds =
      second.additional_kwargs.sourceMessageIds;
    const setHasSpy = jest.spyOn(Set.prototype, 'has');

    const result = getProviderSourceMessageIds(first);
    const sourceIdLookups = setHasSpy.mock.calls.length;
    setHasSpy.mockRestore();

    expect(result).toEqual([]);
    expect(sourceIdLookups).toBe(0);
  });

  it('applies source id length bounds to a mismatched trusted pair', () => {
    const first = new HumanMessage({ content: 'first' });
    const second = new HumanMessage({ content: 'second' });
    setProviderMessageProvenance(first, [
      {
        attribution: 'user',
        sourceMessageId: 'x'.repeat(
          PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIdLength + 1
        ),
      },
    ]);
    setProviderMessageProvenance(second, [
      { attribution: 'user', sourceMessageId: 'second' },
    ]);
    first.additional_kwargs.sourceMessageIds =
      second.additional_kwargs.sourceMessageIds;

    expect(getProviderSourceMessageIds(first)).toEqual([]);
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
