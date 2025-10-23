import { Providers } from '@/common';
import { validateClientOptions } from '@/utils/llm';

describe('validateClientOptions', () => {
  it('passes with valid OpenAI config', () => {
    const result = validateClientOptions({ apiKey: 'sk-test' }, Providers.OPENAI);
    expect(result).toEqual([]);
  });

  it('fails with missing OpenAI apiKey', () => {
    const result = validateClientOptions({}, Providers.OPENAI);
    expect(result).toContain('OpenAI requires an API key');
  });

  it('rejects invalid temperature', () => {
    const result = validateClientOptions({ apiKey: 'sk', temperature: 3 }, Providers.OPENAI);
    expect(result).toContain('Temperature must be between 0 and 2');
  });
});
