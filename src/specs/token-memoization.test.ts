import { HumanMessage } from '@langchain/core/messages';
import { createTokenCounter } from '@/utils/tokens';

describe('Token encoder memoization', () => {
  jest.setTimeout(45000);

  test('reuses the same tokenizer across counter calls', async () => {
    const counter1 = await createTokenCounter();
    const counter2 = await createTokenCounter();

    const m1 = new HumanMessage('hello world');
    const m2 = new HumanMessage('another short text');

    const c11 = counter1(m1);
    const c12 = counter1(m2);
    const c21 = counter2(m1);
    const c22 = counter2(m2);

    expect(c11).toBeGreaterThan(0);
    expect(c12).toBeGreaterThan(0);
    expect(c21).toBe(c11);
    expect(c22).toBe(c12);
  });
});
