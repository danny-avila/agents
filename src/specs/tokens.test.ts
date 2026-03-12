import { HumanMessage } from '@langchain/core/messages';
import {
  encodingForModel,
  createTokenCounter,
  TokenEncoderManager,
} from '@/utils/tokens';

describe('encodingForModel', () => {
  test('returns claude for Claude model strings', () => {
    expect(encodingForModel('claude-3-5-sonnet-20241022')).toBe('claude');
    expect(encodingForModel('claude-3-haiku-20240307')).toBe('claude');
  });

  test('handles Bedrock Claude ARNs', () => {
    expect(encodingForModel('anthropic.claude-3-5-sonnet-20241022-v2:0')).toBe(
      'claude'
    );
  });

  test('is case-insensitive', () => {
    expect(encodingForModel('CLAUDE-3-HAIKU')).toBe('claude');
    expect(encodingForModel('Claude-3-Opus')).toBe('claude');
  });

  test('returns o200k_base for non-Claude models', () => {
    expect(encodingForModel('gpt-4o')).toBe('o200k_base');
    expect(encodingForModel('gemini-2.0-flash')).toBe('o200k_base');
    expect(encodingForModel('mistral-large')).toBe('o200k_base');
  });

  test('returns o200k_base for empty string', () => {
    expect(encodingForModel('')).toBe('o200k_base');
  });
});

describe('createTokenCounter with different encodings', () => {
  beforeEach(() => {
    TokenEncoderManager.reset();
  });

  test('claude encoding produces valid token counts', async () => {
    const counter = await createTokenCounter('claude');
    const msg = new HumanMessage('Hello, world!');
    const count = counter(msg);
    expect(count).toBeGreaterThan(0);
  });

  test('o200k_base encoding produces valid token counts', async () => {
    const counter = await createTokenCounter('o200k_base');
    const msg = new HumanMessage('Hello, world!');
    const count = counter(msg);
    expect(count).toBeGreaterThan(0);
  });

  test('both encodings can be initialized and used independently', async () => {
    const claudeCounter = await createTokenCounter('claude');
    const o200kCounter = await createTokenCounter('o200k_base');
    expect(TokenEncoderManager.isInitialized()).toBe(true);

    const msg = new HumanMessage('Test message for both encodings');
    expect(claudeCounter(msg)).toBeGreaterThan(0);
    expect(o200kCounter(msg)).toBeGreaterThan(0);
  });
});
