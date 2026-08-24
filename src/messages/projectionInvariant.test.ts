import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import {
  setInvalidProviderMessageProvenance,
  setProviderMessageProvenance,
  stampSyntheticProviderMessage,
} from './provenance';
import {
  inspectProviderMessageProjection,
  ProviderMessageProjectionInvariantError,
  resolveProviderMessageProjectionInvariantMode,
} from './projectionInvariant';

describe('provider message projection invariant', () => {
  it('classifies source-backed, synthetic, and mixed provenance', () => {
    const sourceBacked = new HumanMessage('source-backed');
    setProviderMessageProvenance(sourceBacked, [
      { attribution: 'user', sourceMessageId: 'user-row' },
    ]);
    const synthetic = stampSyntheticProviderMessage(
      new SystemMessage('synthetic')
    );
    const mixed = new HumanMessage('mixed');
    setProviderMessageProvenance(mixed, [
      { attribution: 'user', sourceMessageId: 'user-row' },
      { attribution: 'synthetic' },
    ]);

    expect(
      inspectProviderMessageProjection([sourceBacked, synthetic, mixed])
    ).toEqual({
      valid: true,
      messageCount: 3,
      sourceBackedMessageCount: 2,
      syntheticMessageCount: 1,
      gapMessageCount: 0,
      issues: [],
    });
  });

  it('reports absent, invalid, and unsourced non-synthetic provenance', () => {
    const absent = new HumanMessage('absent');
    const invalid = new HumanMessage('invalid');
    setInvalidProviderMessageProvenance(invalid);
    const unsourced = new HumanMessage('unsourced');
    setProviderMessageProvenance(unsourced, [{ attribution: 'user' }]);

    expect(
      inspectProviderMessageProjection([absent, invalid, unsourced])
    ).toEqual({
      valid: false,
      messageCount: 3,
      sourceBackedMessageCount: 0,
      syntheticMessageCount: 0,
      gapMessageCount: 3,
      issues: [
        {
          code: 'absent_provenance',
          messageIndex: 0,
          messageType: 'human',
        },
        {
          code: 'invalid_provenance',
          messageIndex: 1,
          messageType: 'human',
        },
        {
          code: 'unsourced_non_synthetic_part',
          messageIndex: 2,
          messageType: 'human',
        },
      ],
    });
  });

  it('fails closed on hostile provenance access without reading content', () => {
    const message = new HumanMessage('never-read-content');
    const hostile = new Proxy(message, {
      get(target, property, receiver) {
        if (property === 'additional_kwargs') {
          throw new Error('hostile provenance accessor');
        }
        return Reflect.get(target, property, receiver);
      },
    });

    expect(inspectProviderMessageProjection([hostile])).toMatchObject({
      valid: false,
      gapMessageCount: 1,
      issues: [{ code: 'invalid_provenance' }],
    });
  });

  it('bounds stable issues and never exposes content or source ids', () => {
    const secretContent = 'private-message-content';
    const secretId = 'private-source-id';
    const messages = Array.from({ length: 70 }, () => {
      const message = new HumanMessage(secretContent);
      setProviderMessageProvenance(message, [
        { attribution: 'user' },
        { attribution: 'synthetic', sourceMessageId: secretId },
      ]);
      return message;
    });

    const report = inspectProviderMessageProjection(messages);
    const serialized = JSON.stringify(report);
    expect(report.gapMessageCount).toBe(70);
    expect(report.issues).toHaveLength(64);
    expect(report.issues[0]).toEqual({
      code: 'unsourced_non_synthetic_part',
      messageIndex: 0,
      messageType: 'human',
    });
    expect(report.issues[63]?.messageIndex).toBe(63);
    expect(serialized).not.toContain(secretContent);
    expect(serialized).not.toContain(secretId);
  });

  it('resolves only supported opt-in modes', () => {
    expect(resolveProviderMessageProjectionInvariantMode('observe')).toBe(
      'observe'
    );
    expect(resolveProviderMessageProjectionInvariantMode('assert')).toBe(
      'assert'
    );
    expect(resolveProviderMessageProjectionInvariantMode('true')).toBe('off');
    expect(resolveProviderMessageProjectionInvariantMode(undefined)).toBe(
      'off'
    );
  });

  it('exposes only a privacy-safe report on assertion errors', () => {
    const message = new HumanMessage('private-message-content');
    const report = inspectProviderMessageProjection([message]);
    const error = new ProviderMessageProjectionInvariantError(report);

    expect(error.name).toBe('ProviderMessageProjectionInvariantError');
    expect(error.message).toBe('Provider message projection invariant failed');
    expect(JSON.stringify(error.report)).not.toContain('private-message-content');
  });
});
