import { AIMessage } from '@langchain/core/messages';
import type { OpenAIClient } from '@langchain/openai';

import {
  getCacheWriteTokens,
  attachCacheWriteUsage,
  attachCacheWriteMetadata,
} from './index';

/**
 * Regression coverage for a crash reported against OpenAI-compatible
 * third-party servers (e.g. mlx_vlm.server) whose `/v1/responses` usage
 * payload omits `input_tokens_details` entirely — a shape the OpenAI API
 * itself always populates, but which `ResponsesUsageWithCacheWrite`
 * declares optional. Reading `.cache_write_tokens` off that missing object
 * without a second `?.` threw "Cannot read properties of undefined
 * (reading 'cache_write_tokens')" on every completion from such a server.
 */
describe('cache write token extraction (Responses API)', () => {
  describe('getCacheWriteTokens', () => {
    it('returns undefined without throwing when usage has no input_tokens_details', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          usage: {
            input_tokens: 10,
            output_tokens: 2,
            total_tokens: 12,
            // input_tokens_details intentionally omitted, mirroring a
            // minimal OpenAI-compatible server's usage payload.
          },
        },
      });

      expect(() => getCacheWriteTokens(message)).not.toThrow();
      expect(getCacheWriteTokens(message)).toBeUndefined();
    });

    it('returns undefined without throwing when there is no usage at all', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {},
      });

      expect(() => getCacheWriteTokens(message)).not.toThrow();
      expect(getCacheWriteTokens(message)).toBeUndefined();
    });

    it('still reports cache_write_tokens when the field is present', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          usage: {
            input_tokens: 10,
            output_tokens: 2,
            total_tokens: 12,
            input_tokens_details: { cache_write_tokens: 5 },
          },
        },
      });

      expect(getCacheWriteTokens(message)).toBe(5);
    });

    it('falls back to the serialized metadata key when usage is absent', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          metadata: { __librechat_cache_write_tokens: '7' },
        },
      });

      expect(getCacheWriteTokens(message)).toBe(7);
    });
  });

  describe('attachCacheWriteUsage', () => {
    it('leaves usage_metadata untouched without throwing when input_tokens_details is missing', () => {
      const message = new AIMessage({
        content: 'hi',
        response_metadata: {
          usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
        },
        usage_metadata: {
          input_tokens: 10,
          output_tokens: 2,
          total_tokens: 12,
        },
      });

      expect(() => attachCacheWriteUsage(message)).not.toThrow();
      expect(message.usage_metadata?.input_token_details).toBeUndefined();
    });
  });

  describe('attachCacheWriteMetadata', () => {
    it('returns the response unmodified without throwing when input_tokens_details is missing', () => {
      const response = {
        usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 },
      } as OpenAIClient.Responses.Response;

      expect(() => attachCacheWriteMetadata(response)).not.toThrow();
      expect(attachCacheWriteMetadata(response)).toBe(response);
    });

    it('returns the response unmodified without throwing when usage is missing entirely', () => {
      const response = {} as OpenAIClient.Responses.Response;

      expect(() => attachCacheWriteMetadata(response)).not.toThrow();
    });
  });
});
