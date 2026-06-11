import { apportionTokenCounts } from '@/utils/tokens';

describe('apportionTokenCounts', () => {
  it('sums exactly to a ceil-of-sum aggregate despite per-entry fractions', () => {
    const raw = { add: 33, search: 33, fetch: 33 };
    const multiplier = 1.4;
    const target = Math.ceil((33 + 33 + 33) * multiplier);
    const result = apportionTokenCounts(raw, multiplier, target);
    const sum = Object.values(result).reduce((acc, count) => acc + count, 0);
    expect(sum).toBe(target);
    expect(Object.keys(result).sort()).toEqual(['add', 'fetch', 'search']);
  });

  it('gives larger remainders priority when distributing leftovers', () => {
    const result = apportionTokenCounts({ a: 10, b: 19 }, 1.05, 31);
    expect(result.a + result.b).toBe(31);
    expect(result.b).toBe(20);
    expect(result.a).toBe(11);
  });

  it('handles calibration-style rescaling to an arbitrary target', () => {
    const counts = { a: 100, b: 200, c: 300 };
    const target = 451;
    const result = apportionTokenCounts(counts, target / 600, target);
    const sum = Object.values(result).reduce((acc, count) => acc + count, 0);
    expect(sum).toBe(target);
  });

  it('returns an empty map for no entries', () => {
    expect(apportionTokenCounts({}, 1.4, 10)).toEqual({});
  });
});
