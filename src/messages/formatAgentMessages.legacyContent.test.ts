import { describe, expect, it } from '@jest/globals';
import { HumanMessage } from '@langchain/core/messages';
import type { MessageContentComplex, TPayload } from '@/types';
import { formatContentStrings, isLegacyConvertible } from './content';
import {
  getProviderMessageProvenance,
  setFreshProviderMessageProvenance,
  setProviderMessageProvenance,
} from './provenance';
import { formatAgentMessages } from './format';
import { ContentTypes } from '@/common';

const textPart = (text: string): MessageContentComplex => ({
  type: ContentTypes.TEXT,
  [ContentTypes.TEXT]: text,
});

const buildPayload = (): TPayload => [
  { role: 'system', content: 'You are helpful.' },
  { role: 'user', content: [textPart('  hello '), textPart('there')] },
  {
    messageId: 'assistant-1',
    isCreatedByUser: false,
    content: [textPart('first block'), textPart('second block')],
  },
  { role: 'user', content: 'plain string turn' },
  {
    messageId: 'assistant-2',
    isCreatedByUser: false,
    content: [textPart('reply')],
  },
];

describe('formatAgentMessages legacyContent option', () => {
  it('produces byte-identical content to the legacy projection applied afterwards', () => {
    const { messages: eager } = formatAgentMessages(
      buildPayload(),
      undefined,
      undefined,
      undefined,
      { legacyContent: true }
    );
    const { messages: baseline } = formatAgentMessages(buildPayload());
    const projected = formatContentStrings(baseline);
    expect(eager.length).toBe(projected.length);
    for (let i = 0; i < eager.length; i++) {
      expect(eager[i].content).toEqual(projected[i].content);
      expect(eager[i].getType()).toBe(projected[i].getType());
      expect(eager[i].lc_kwargs.content).toEqual(projected[i].content);
    }
  });

  it('leaves nothing for the legacy projection to convert, preserving identity', () => {
    const { messages } = formatAgentMessages(
      buildPayload(),
      undefined,
      undefined,
      undefined,
      { legacyContent: true }
    );
    for (const message of messages) {
      expect(isLegacyConvertible(message)).toBe(false);
    }
    const projected = formatContentStrings(messages);
    for (let i = 0; i < messages.length; i++) {
      expect(projected[i]).toBe(messages[i]);
    }
  });

  it('keeps array content untouched without the option', () => {
    const { messages } = formatAgentMessages(buildPayload());
    expect(messages.some((message) => Array.isArray(message.content))).toBe(true);
  });
});

describe('setFreshProviderMessageProvenance', () => {
  const buildMessage = () =>
    new HumanMessage({
      content: 'hello',
      additional_kwargs: { existing: 'kept', sourceMessageId: 'stale' },
    });

  it('produces the same end state as the hardened publication', () => {
    const hardened = buildMessage();
    const fresh = buildMessage();
    const parts = [
      { attribution: 'user' as const, sourceMessageId: 'msg-1' },
      { attribution: 'model' as const, sourceMessageId: 'msg-2' },
    ];
    setProviderMessageProvenance(hardened, parts);
    setFreshProviderMessageProvenance(fresh, parts);
    expect(fresh.additional_kwargs).toEqual(hardened.additional_kwargs);
    expect(fresh.lc_kwargs.additional_kwargs).toBe(fresh.additional_kwargs);
    expect(hardened.lc_kwargs.additional_kwargs).toBe(hardened.additional_kwargs);
    expect(getProviderMessageProvenance(fresh)).toEqual(
      getProviderMessageProvenance(hardened)
    );
  });

  it('clears stale source ids when parts carry none', () => {
    const hardened = buildMessage();
    const fresh = buildMessage();
    const parts = [{ attribution: 'synthetic' as const }];
    setProviderMessageProvenance(hardened, parts);
    setFreshProviderMessageProvenance(fresh, parts);
    expect(fresh.additional_kwargs).toEqual(hardened.additional_kwargs);
    expect('sourceMessageId' in fresh.additional_kwargs).toBe(false);
    expect('sourceMessageIds' in fresh.additional_kwargs).toBe(false);
  });

  it('replaces the kwargs objects rather than mutating the originals', () => {
    const fresh = buildMessage();
    const originalKwargs = fresh.additional_kwargs;
    const originalLcKwargs = fresh.lc_kwargs;
    setFreshProviderMessageProvenance(fresh, [{ attribution: 'user' as const }]);
    expect(fresh.additional_kwargs).not.toBe(originalKwargs);
    expect(fresh.lc_kwargs).not.toBe(originalLcKwargs);
    expect('provenance' in originalKwargs).toBe(false);
  });
});
