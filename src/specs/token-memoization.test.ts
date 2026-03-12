import { HumanMessage } from '@langchain/core/messages';
import { createTokenCounter, TokenEncoderManager } from '@/utils/tokens';

jest.setTimeout(5000);

describe('Token encoder memoization', () => {
  beforeEach(() => {
    TokenEncoderManager.reset();
  });

  test('reuses the same tokenizer across counter calls', async () => {
    expect(TokenEncoderManager.isInitialized()).toBe(false);

    const counter1 = await createTokenCounter();
    expect(TokenEncoderManager.isInitialized()).toBe(true);

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

  test('reset clears cached tokenizers', async () => {
    await createTokenCounter();
    expect(TokenEncoderManager.isInitialized()).toBe(true);

    TokenEncoderManager.reset();
    expect(TokenEncoderManager.isInitialized()).toBe(false);
  });
});
