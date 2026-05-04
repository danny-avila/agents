import { describe, it, expect } from '@jest/globals';
import { applyEdit, locateEdit } from '../editStrategies';

/**
 * Focused unit tests for the fuzzy edit-matching chain. The chain is
 * what determines whether `edit_file` succeeds or silently corrupts a
 * file when the model's `oldString` doesn't byte-match the on-disk
 * content (different indentation, trailing whitespace, etc.). Pinning
 * the strategy boundaries here so a regression in any one strategy
 * surfaces fast.
 *
 * Order of strategies (defined in editStrategies.ts):
 *   1. exact
 *   2. line-trimmed
 *   3. whitespace-normalized
 *   4. indentation-flexible
 */

describe('editStrategies › locateEdit', () => {
  it('returns an exact-strategy match for a literal byte-equal substring', () => {
    const source = 'one\ntwo\nthree\n';
    const m = locateEdit(source, 'two');
    expect(m).not.toBeNull();
    expect(m?.strategy).toBe('exact');
    expect(applyEdit(source, m!, '2')).toBe('one\n2\nthree\n');
  });

  it('matches at the very start of the source', () => {
    const source = 'first\nsecond\n';
    const m = locateEdit(source, 'first');
    expect(m).not.toBeNull();
    expect(m?.strategy).toBe('exact');
    expect(m?.start).toBe(0);
  });

  it('matches at the very end of the source (no trailing newline)', () => {
    const source = 'a\nb\nlast';
    const m = locateEdit(source, 'last');
    expect(m).not.toBeNull();
    expect(m?.strategy).toBe('exact');
    expect(applyEdit(source, m!, 'LAST')).toBe('a\nb\nLAST');
  });

  it('falls back to line-trimmed when the model lost trailing whitespace inside a multi-line needle', () => {
    // Source has trailing tab on the second line; needle has none.
    // Multi-line needle so the exact strategy can't match (the source
    // line literally has the tab). Line-trimmed compares trimEnd vs
    // trimEnd and lets it through.
    const source = 'fn foo() {\n  return 42\t\n}\n';
    const m = locateEdit(source, 'fn foo() {\n  return 42\n}');
    expect(m).not.toBeNull();
    expect(['line-trimmed', 'whitespace-normalized']).toContain(m?.strategy);
  });

  it('falls back to whitespace-normalized when interior whitespace differs', () => {
    // Source uses tabs, needle uses spaces.
    const source = 'if\t(x)\t{\n\treturn\ty;\n}';
    const m = locateEdit(source, 'if (x) {\n  return y;\n}');
    expect(m).not.toBeNull();
    expect(['whitespace-normalized', 'line-trimmed', 'exact']).toContain(
      m?.strategy
    );
  });

  it('falls back to indentation-flexible when block leading indent differs', () => {
    // Source is indented by 4 spaces, needle by 2 — semantically the
    // same block, lexically different indent.
    const source = '    function go() {\n        return 1;\n    }\n';
    const m = locateEdit(source, '  function go() {\n    return 1;\n  }');
    expect(m).not.toBeNull();
    // Any of the looser strategies could legitimately win here; what
    // matters is that the chain finds *a* match instead of giving up.
    expect(m?.strategy).not.toBe('exact');
  });

  it('returns null when nothing in the chain matches', () => {
    const source = 'alpha\nbeta\ngamma\n';
    const m = locateEdit(source, 'this string is nowhere in the source');
    expect(m).toBeNull();
  });

  it('returns null on empty oldString (no anchor to locate)', () => {
    const source = 'a\nb\n';
    const m = locateEdit(source, '');
    expect(m).toBeNull();
  });

  it('handles multi-line needles spanning a blank line', () => {
    const source = 'before\n\nafter\n';
    const m = locateEdit(source, 'before\n\nafter');
    expect(m).not.toBeNull();
    expect(m?.strategy).toBe('exact');
  });

  it('handles unicode content (emoji, combining characters)', () => {
    const source = 'header\n  // ✅ done — café\nfooter\n';
    const m = locateEdit(source, '  // ✅ done — café');
    expect(m).not.toBeNull();
    expect(m?.strategy).toBe('exact');
  });
});

describe('editStrategies › applyEdit', () => {
  it('produces the new source by splicing newString into the matched span', () => {
    const source = 'aaa BAR ccc';
    const m = locateEdit(source, 'BAR');
    expect(m).not.toBeNull();
    expect(applyEdit(source, m!, 'baz')).toBe('aaa baz ccc');
  });

  it('is a no-op when newString equals the matched span', () => {
    const source = 'x y z';
    const m = locateEdit(source, 'y');
    expect(m).not.toBeNull();
    expect(applyEdit(source, m!, 'y')).toBe('x y z');
  });
});
