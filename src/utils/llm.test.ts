import { isAnthropicLike } from './llm';
import { Providers } from '@/common';

describe('isAnthropicLike', () => {
  it('treats the default Bedrock model as Claude', () => {
    expect(isAnthropicLike(Providers.BEDROCK)).toBe(true);
    expect(
      isAnthropicLike(Providers.BEDROCK, {
        model: 'anthropic.claude-sonnet-4-5',
      })
    ).toBe(true);
    expect(
      isAnthropicLike(Providers.BEDROCK, {
        model: 'amazon.nova-pro-v1:0',
      })
    ).toBe(false);
  });
});
