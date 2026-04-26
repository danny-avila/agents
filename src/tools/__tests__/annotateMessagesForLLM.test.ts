import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { describe, it, expect } from '@jest/globals';
import {
  annotateMessagesForLLM,
  ToolOutputReferenceRegistry,
  TOOL_OUTPUT_REF_KEY,
} from '../toolOutputReferences';

function makeToolMessage(fields: {
  content: ToolMessage['content'];
  name?: string;
  tool_call_id?: string;
  status?: 'success' | 'error';
  additional_kwargs?: Record<string, unknown>;
}): ToolMessage {
  return new ToolMessage({
    name: fields.name ?? 'echo',
    tool_call_id: fields.tool_call_id ?? 'tc1',
    status: fields.status ?? 'success',
    additional_kwargs: fields.additional_kwargs,
    content: fields.content,
  });
}

describe('annotateMessagesForLLM', () => {
  it('returns the input array reference when registry is undefined', () => {
    const messages = [
      new HumanMessage('hi'),
      makeToolMessage({
        content: 'data',
        additional_kwargs: { _refKey: 'tool0turn0' },
      }),
    ];
    const out = annotateMessagesForLLM(messages, undefined, 'r1');
    expect(out).toBe(messages);
  });

  it('returns the input array reference when no ToolMessage carries metadata', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'stored');
    const messages = [
      new HumanMessage('hi'),
      makeToolMessage({ content: 'data' }),
      new AIMessage('answer'),
    ];
    const out = annotateMessagesForLLM(messages, registry, 'r1');
    expect(out).toBe(messages);
  });

  it('annotates string content when _refKey is live in the registry', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'stored-raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('[ref: tool0turn0]\noutput');
    expect(tm.content).toBe('output');
    expect(out).not.toBe([tm]);
    expect(out[0]).not.toBe(tm);
  });

  it('skips annotation when _refKey is stale (not in the registry)', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('output');
    expect(out[0]).toBe(tm);
  });

  it('always applies _unresolvedRefs even when there is no registry entry', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _unresolvedRefs: ['tool9turn9'] },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('output\n[unresolved refs: tool9turn9]');
  });

  it('injects _ref into JSON-object string content', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', '{"a":1}');
    const tm = makeToolMessage({
      content: '{"a":1,"b":"x"}',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const parsed = JSON.parse(out[0].content as string);
    expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
    expect(parsed.a).toBe(1);
    expect(parsed.b).toBe('x');
  });

  it('uses [ref: …] prefix for non-JSON string content', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'plain output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('[ref: tool0turn0]\nplain output');
  });

  it('prepends an unresolved-refs warning text block to multi-part content', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: [
        { type: 'text', text: 'data' },
        { type: 'image_url', image_url: { url: 'data:...' } },
      ] as unknown as ToolMessage['content'],
      additional_kwargs: { _unresolvedRefs: ['tool9turn9'] },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const blocks = out[0].content as Array<{ type: string; text?: string }>;
    expect(blocks).toHaveLength(3);
    expect(blocks[0].type).toBe('text');
    expect(blocks[0].text).toBe('[unresolved refs: tool9turn9]');
    expect(blocks[1].type).toBe('text');
    expect(blocks[1].text).toBe('data');
    expect(blocks[2].type).toBe('image_url');
  });

  it('does not mutate the original ToolMessage instance or its content', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const originalContent = tm.content;
    const originalKwargs = { ...tm.additional_kwargs };
    annotateMessagesForLLM([tm], registry, 'r1');
    expect(tm.content).toBe(originalContent);
    expect(tm.additional_kwargs).toEqual(originalKwargs);
  });

  it('strips ref metadata from the projected ToolMessage so annotation is not double-applied', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _unresolvedRefs: ['tool9turn9'],
        someOtherField: 'preserved',
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    const projected = out[0] as ToolMessage;
    expect(projected.additional_kwargs._refKey).toBeUndefined();
    expect(projected.additional_kwargs._unresolvedRefs).toBeUndefined();
    expect(projected.additional_kwargs.someOtherField).toBe('preserved');
  });

  it('passes through non-ToolMessages unchanged in the projected array', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const human = new HumanMessage('hi');
    const ai = new AIMessage('answer');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const out = annotateMessagesForLLM([human, ai, tm], registry, 'r1');
    expect(out[0]).toBe(human);
    expect(out[1]).toBe(ai);
    expect(out[2]).not.toBe(tm);
  });

  it('returns same array when only stale _refKey is present and no unresolvedRefs', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool1turn0', 'somethingelse');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: { _refKey: 'tool0turn0' },
    });
    const messages = [tm];
    const out = annotateMessagesForLLM(messages, registry, 'r1');
    expect(out).toBe(messages);
  });

  it('annotates only the live ref when both ref and unresolved are present', () => {
    const registry = new ToolOutputReferenceRegistry();
    registry.set('r1', 'tool0turn0', 'raw');
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _unresolvedRefs: ['tool9turn9'],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe(
      '[ref: tool0turn0]\noutput\n[unresolved refs: tool9turn9]'
    );
  });

  it('treats stale _refKey but live unresolved as unresolved-only', () => {
    const registry = new ToolOutputReferenceRegistry();
    const tm = makeToolMessage({
      content: 'output',
      additional_kwargs: {
        _refKey: 'tool0turn0',
        _unresolvedRefs: ['tool9turn9'],
      },
    });
    const out = annotateMessagesForLLM([tm], registry, 'r1');
    expect(out[0].content).toBe('output\n[unresolved refs: tool9turn9]');
  });
});
