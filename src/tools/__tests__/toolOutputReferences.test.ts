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
      reg.set('r', 'tool0turn0', 'hello world');
      expect(reg.get('r', 'tool0turn0')).toBe('hello world');
      expect(reg.size).toBe(1);
    });

    it('clips stored values to the per-output limit', () => {
      const reg = new ToolOutputReferenceRegistry({ maxOutputSize: 5 });
      reg.set('r', 'tool0turn0', 'abcdefghij');
      expect(reg.get('r', 'tool0turn0')).toBe('abcde');
    });

    it('replaces existing entries under the same key without double-counting size', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 100,
        maxTotalSize: 20,
      });
      reg.set('r', 'tool0turn0', 'hello');
      reg.set('r', 'tool0turn0', 'world-longer');
      expect(reg.get('r', 'tool0turn0')).toBe('world-longer');
      expect(reg.size).toBe(1);
    });
  });

  describe('clear / releaseRun', () => {
    it('clear() drops every run bucket', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('run-A', 'tool0turn0', 'A');
      reg.set('run-B', 'tool0turn0', 'B');
      expect(reg.size).toBe(2);
      reg.clear();
      expect(reg.size).toBe(0);
      expect(reg.get('run-A', 'tool0turn0')).toBeUndefined();
      expect(reg.get('run-B', 'tool0turn0')).toBeUndefined();
    });

    it('releaseRun() drops only the named run', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('run-A', 'tool0turn0', 'A');
      reg.set('run-B', 'tool0turn0', 'B');
      reg.releaseRun('run-A');
      expect(reg.get('run-A', 'tool0turn0')).toBeUndefined();
      expect(reg.get('run-B', 'tool0turn0')).toBe('B');
    });
  });

  describe('FIFO eviction', () => {
    it('evicts oldest entries when the aggregate cap is exceeded', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 10,
        maxTotalSize: 12,
      });
      reg.set('r', 'tool0turn0', '1234567'); // 7 chars
      reg.set('r', 'tool1turn0', '89'); // 9 total
      reg.set('r', 'tool2turn0', 'abc'); // 12 total — at limit
      reg.set('r', 'tool3turn0', 'XY'); // 14 → must evict oldest
      expect(reg.get('r', 'tool0turn0')).toBeUndefined();
      expect(reg.get('r', 'tool1turn0')).toBe('89');
      expect(reg.get('r', 'tool2turn0')).toBe('abc');
      expect(reg.get('r', 'tool3turn0')).toBe('XY');
    });

    it('keeps evicting oldest entries until the aggregate fits', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 10,
        maxTotalSize: 8,
      });
      reg.set('r', 'tool0turn0', 'aaa');
      reg.set('r', 'tool1turn0', 'bbb');
      reg.set('r', 'tool2turn0', 'ccccccc'); // total 3+3+7=13 > 8, evict aaa then bbb
      expect(reg.get('r', 'tool0turn0')).toBeUndefined();
      expect(reg.get('r', 'tool1turn0')).toBeUndefined();
      expect(reg.get('r', 'tool2turn0')).toBe('ccccccc');
    });

    it('clamps the per-output cap to maxTotalSize so no entry exceeds the aggregate', () => {
      const reg = new ToolOutputReferenceRegistry({
        maxOutputSize: 1000,
        maxTotalSize: 10,
      });
      reg.set('r', 'tool0turn0', 'x'.repeat(500));
      const stored = reg.get('r', 'tool0turn0');
      expect(stored).toBeDefined();
      expect(stored!.length).toBeLessThanOrEqual(10);
    });

    it('clamps a caller-supplied maxTotalSize to the documented hard cap', () => {
      // 50 MB requested; should be clamped to the 5 MB hard cap.
      const reg = new ToolOutputReferenceRegistry({
        maxTotalSize: 50_000_000,
      });
      // `totalLimit` getter exposes the effective post-clamp value.
      expect(reg.totalLimit).toBeLessThanOrEqual(5_000_000);
      // Per-output is also bound by the same effective total.
      expect(reg.perOutputLimit).toBeLessThanOrEqual(5_000_000);
    });
  });

  describe('resolve', () => {
    it('replaces placeholders in string args', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'HELLO');
      const { resolved, unresolved } = reg.resolve('r', 'echo {{tool0turn0}}');
      expect(resolved).toBe('echo HELLO');
      expect(unresolved).toEqual([]);
    });

    it('replaces placeholders in nested object args', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'DATA');
      const input = {
        command: 'cat {{tool0turn0}}',
        meta: { note: 'uses {{tool0turn0}} twice' },
      };
      const { resolved } = reg.resolve('r', input);
      expect(resolved).toEqual({
        command: 'cat DATA',
        meta: { note: 'uses DATA twice' },
      });
    });

    it('replaces placeholders inside array values', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool1turn2', '42');
      const { resolved } = reg.resolve('r', {
        args: ['--id', '{{tool1turn2}}', 'plain'],
      });
      expect(resolved).toEqual({ args: ['--id', '42', 'plain'] });
    });

    it('reports unresolved references and leaves the placeholder in place', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'known');
      const { resolved, unresolved } = reg.resolve(
        'r',
        'use {{tool0turn0}} and {{tool5turn9}}'
      );
      expect(resolved).toBe('use known and {{tool5turn9}}');
      expect(unresolved).toEqual(['tool5turn9']);
    });

    it('deduplicates repeated unresolved keys', () => {
      const reg = new ToolOutputReferenceRegistry();
      const { unresolved } = reg.resolve(
        'r',
        '{{tool7turn0}} and {{tool7turn0}} again'
      );
      expect(unresolved).toEqual(['tool7turn0']);
    });

    it('does not touch non-placeholder strings', () => {
      const reg = new ToolOutputReferenceRegistry();
      reg.set('r', 'tool0turn0', 'X');
      const { resolved } = reg.resolve('r', 'nothing to see here');
      expect(resolved).toBe('nothing to see here');
    });

    it('passes through primitive values untouched', () => {
      const reg = new ToolOutputReferenceRegistry();
      const { resolved } = reg.resolve('r', {
        count: 3,
        enabled: true,
        note: null,
      });
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

    it('overwrites an existing _ref:null with the injected key', () => {
      const content = '{"_ref":null,"data":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed.data).toBe(1);
    });

    it('places the injected _ref as the first key in the serialized JSON', () => {
      const content = '{"a":1,"b":"x"}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated.indexOf('"_ref"')).toBe(1);
      const annotatedFromNull = annotateToolOutputWithReference(
        '{"_ref":null,"a":1}',
        'tool0turn0'
      );
      expect(annotatedFromNull.indexOf('"_ref"')).toBe(1);
    });

    it('falls back to the prefix when parsing fails', () => {
      const content = '{ not actually json';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      expect(annotated).toBe(`[ref: tool0turn0]\n${content}`);
    });

    it('carries unresolved refs as a JSON field on parseable objects', () => {
      const content = '{"a":1}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0', [
        'tool9turn9',
        'tool7turn0',
      ]);
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed._unresolved_refs).toEqual(['tool9turn9', 'tool7turn0']);
      expect(parsed.a).toBe(1);
    });

    it('appends unresolved refs as a trailer line on non-object content', () => {
      const annotated = annotateToolOutputWithReference(
        'plain text',
        'tool0turn0',
        ['tool9turn9']
      );
      expect(annotated).toBe(
        '[ref: tool0turn0]\nplain text\n[unresolved refs: tool9turn9]'
      );
    });

    it('supports unresolved-only annotation (no ref key)', () => {
      const annotated = annotateToolOutputWithReference(
        'error text',
        undefined,
        ['tool9turn9']
      );
      expect(annotated).toBe('error text\n[unresolved refs: tool9turn9]');
    });

    it('keeps JSON parseable for unresolved-only annotation on object content', () => {
      const annotated = annotateToolOutputWithReference(
        '{"error":"bad"}',
        undefined,
        ['tool9turn9']
      );
      const parsed = JSON.parse(annotated);
      expect(parsed._unresolved_refs).toEqual(['tool9turn9']);
      expect(parsed.error).toBe('bad');
      expect(parsed._ref).toBeUndefined();
    });

    it('returns content unchanged when there is nothing to annotate', () => {
      expect(annotateToolOutputWithReference('plain', undefined, [])).toBe(
        'plain'
      );
    });

    it('preserves an existing _unresolved_refs payload when only injecting a ref key', () => {
      const content = '{"data":1,"_unresolved_refs":["user-supplied"]}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0');
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('tool0turn0');
      expect(parsed._unresolved_refs).toEqual(['user-supplied']);
      expect(parsed.data).toBe(1);
    });

    it('preserves an existing _ref payload on the unresolved-only path', () => {
      const content = '{"data":1,"_ref":"preserved-value"}';
      const annotated = annotateToolOutputWithReference(content, undefined, [
        'tool9turn9',
      ]);
      const parsed = JSON.parse(annotated);
      expect(parsed._ref).toBe('preserved-value');
      expect(parsed._unresolved_refs).toEqual(['tool9turn9']);
      expect(parsed.data).toBe(1);
    });

    it('falls back to the prefix when _unresolved_refs conflicts with a non-matching array', () => {
      const content = '{"data":1,"_unresolved_refs":["legacy"]}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0', [
        'tool9turn9',
      ]);
      expect(annotated.startsWith('[ref: tool0turn0]\n')).toBe(true);
      expect(annotated).toContain('[unresolved refs: tool9turn9]');
    });

    it('accepts a deep-equal existing _unresolved_refs array', () => {
      const content = '{"data":1,"_unresolved_refs":["tool9turn9"]}';
      const annotated = annotateToolOutputWithReference(content, 'tool0turn0', [
        'tool9turn9',
      ]);
      const parsed = JSON.parse(annotated);
      expect(parsed._unresolved_refs).toEqual(['tool9turn9']);
      expect(parsed._ref).toBe('tool0turn0');
    });
  });

  describe('TOOL_OUTPUT_REF_PATTERN', () => {
    it('matches braced tool<N>turn<M> tokens and captures the key', () => {
      const match = '{{tool0turn0}}'.match(TOOL_OUTPUT_REF_PATTERN);
      expect(match?.[1]).toBe('tool0turn0');
    });

    it('rejects bare tool<N>turn<M> tokens without braces', () => {
      expect(TOOL_OUTPUT_REF_PATTERN.test('tool0turn0')).toBe(false);
    });

    it('is non-global so callers cannot trip on stale lastIndex', () => {
      expect(TOOL_OUTPUT_REF_PATTERN.flags).not.toContain('g');
    });
  });
});
