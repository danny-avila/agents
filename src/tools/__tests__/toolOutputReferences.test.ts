import { describe, it, expect } from '@jest/globals';
import {
  ToolOutputReferenceRegistry,
  annotateToolOutputWithReference,
  buildReferenceKey,
  buildReferencePrefix,
  TOOL_OUTPUT_REF_KEY,
  TOOL_OUTPUT_REF_PATTERN,
} from '../toolOutputReferences';

describe('ToolOutputReferenceRegistry', () => {
  describe('buildReferenceKey', () => {
    it('formats keys as tool<idx>turn<turn>', () => {
      expect(buildReferenceKey(0, 0)).toBe('tool0turn0');
      expect(buildReferenceKey(3, 7)).toBe('tool3turn7');
    });
  });

  describe('set / get', () => {
    it('stores and retrieves outputs by key', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('tool0turn0', 'hello world');
      expect(reg.get('tool0turn0')).toBe('hello world');
      expect(reg.size).toBe(1);
    });

    it('clips stored values to the per-output limit', () => {
      const reg = new ToolOutputReferenceRegistry({ maxOutputSize: 5 });
      reg.set('tool0turn0', 'abcdefghij');
      expect(reg.get('tool0turn0')).toBe('abcde');
    });

    it('replaces existing entries under the same key without double-counting size', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 100,
        maxTotalSize: 20,
      });
      reg.set('tool0turn0', 'hello');
      reg.set('tool0turn0', 'world-longer');
      expect(reg.get('tool0turn0')).toBe('world-longer');
      expect(reg.size).toBe(1);
    });
  });

  describe('FIFO eviction', () => {
    it('evicts oldest entries when the aggregate cap is exceeded', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 10,
        maxTotalSize: 12,
      });
      reg.set('tool0turn0', '1234567'); // 7 chars
      reg.set('tool1turn0', '89'); // 9 total
      reg.set('tool2turn0', 'abc'); // 12 total — at limit
      reg.set('tool3turn0', 'XY'); // 14 → must evict oldest
      expect(reg.get('tool0turn0')).toBeUndefined();
      expect(reg.get('tool1turn0')).toBe('89');
      expect(reg.get('tool2turn0')).toBe('abc');
      expect(reg.get('tool3turn0')).toBe('XY');
    });

    it('keeps evicting oldest entries until the aggregate fits', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 10,
        maxTotalSize: 8,
      });
      reg.set('tool0turn0', 'aaa');
      reg.set('tool1turn0', 'bbb');
      reg.set('tool2turn0', 'ccccccc'); // total 3+3+7=13 > 8, evict aaa then bbb
      expect(reg.get('tool0turn0')).toBeUndefined();
      expect(reg.get('tool1turn0')).toBeUndefined();
      expect(reg.get('tool2turn0')).toBe('ccccccc');
    });
  });

  describe('resolve', () => {
    it('replaces placeholders in string args', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('tool0turn0', 'HELLO');
      const { resolved, unresolved } = reg.resolve('echo {{tool0turn0}}');
      expect(resolved).toBe('echo HELLO');
      expect(unresolved).toEqual([]);
    });

    it('replaces placeholders in nested object args', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('tool0turn0', 'DATA');
      const input = {
        command: 'cat {{tool0turn0}}',
        meta: { note: 'uses {{tool0turn0}} twice' },
      };
      const { resolved } = reg.resolve(input);
      expect(resolved).toEqual({
        command: 'cat DATA',
        meta: { note: 'uses DATA twice' },
      });
    });

    it('replaces placeholders inside array values', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('tool1turn2', '42');
      const { resolved } = reg.resolve({
        args: ['--id', '{{tool1turn2}}', 'plain'],
      });
      expect(resolved).toEqual({ args: ['--id', '42', 'plain'] });
    });

    it('reports unresolved references and leaves the placeholder in place', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('tool0turn0', 'known');
      const { resolved, unresolved } = reg.resolve(
        'use {{tool0turn0}} and {{tool5turn9}}'
      );
      expect(resolved).toBe('use known and {{tool5turn9}}');
      expect(unresolved).toEqual(['tool5turn9']);
    });

    it('deduplicates repeated unresolved keys', () => {
      const reg = new ToolOutputReferenceRegistry();
      const { unresolved } = reg.resolve(
        '{{tool7turn0}} and {{tool7turn0}} again'
      );
      expect(unresolved).toEqual(['tool7turn0']);
    });

    it('does not touch non-placeholder strings', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('tool0turn0', 'X');
      const { resolved } = reg.resolve('nothing to see here');
      expect(resolved).toBe('nothing to see here');
    });

    it('passes through primitive values untouched', () => {
      const reg = new ToolOutputReferenceRegistry();
      const { resolved } = reg.resolve({ count: 3, enabled: true, note: null });
      expect(resolved).toEqual({ count: 3, enabled: true, note: null });
    });
  });

  describe('annotateToolOutputWithReference', () => {
    it('injects _ref into plain JSON objects', () => {
      const content = '{"a":1,"b":"x"}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
      expect(parsed.a).toBe(1);
      expect(parsed.b).toBe('x');
    });

    it('preserves pretty-printed formatting when the original was pretty', () => {
      const content = '{\n  "a": 1\n}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated).toContain('\n  "');
      const parsed = JSON.parse(annotated);
      expect(parsed[TOOL_OUTPUT_REF_KEY]).toBe('tool0turn0');
    });

    it('uses the [ref: …] prefix for JSON arrays', () => {
      const content = '[1,2,3]';
      const annotated = annotateToolOutputWithReference(content, 'tool1turn0');
      expect(annotated).toBe(`${buildReferencePrefix('tool1turn0')}\n[1,2,3]`);
    });

    it('uses the [ref: …] prefix for JSON primitives', () => {
      expect(annotateToolOutputWithReference('42', 'tool0turn0')).toBe(
        '[ref: tool0turn0]\n42'
      );
    });

    it('uses the [ref: …] prefix for plain strings', () => {
      expect(annotateToolOutputWithReference('hello', 'tool0turn0')).toBe(
        '[ref: tool0turn0]\nhello'
      );
    });

    it('falls back to the prefix on JSON _ref collision', () => {
      const content = '{"_ref":"other-value","data":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated.startsWith('[ref: tool0turn0]\n')).toBe(true);
    });

    it('injects when the existing _ref matches the target key', () => {
      const content = '{"_ref":"tool0turn0","data":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed.data).toBe(1);
    });

    it('falls back to the prefix when parsing fails', () => {
      const content = '{ not actually json';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated).toBe(`[ref: tool0turn0]\n${content}`);
    });
  });

  describe('TOOL_OUTPUT_REF_PATTERN', () => {
    it('matches only braced tool<N>turn<M> tokens', () => {
      TOOL_OUTPUT_REF_PATTERN.lastIndex = 0;
      const matches = Array.from(
        'prefix {{tool0turn0}} and {{tool12turn34}} but not tool0turn0'.matchAll(
          TOOL_OUTPUT_REF_PATTERN
        )
      );
      expect(matches.map((m) => m[1])).toEqual(['tool0turn0', 'tool12turn34']);
    });
  });
});
