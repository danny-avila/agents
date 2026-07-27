/**
 * Provider error signatures captured from live over-limit requests.
 *
 * Every entry below was produced by `src/scripts/context-overflow-probe.ts`
 * sending a prompt past the model's context window and recording what came
 * back. Account identifiers and request ids have been replaced with
 * placeholders; the wording, status codes, and nesting are verbatim.
 *
 * Treat these as evidence, not examples: if a provider changes its wording,
 * re-run the probe and update the fixture rather than loosening the matcher.
 */
import { Providers } from '@/common';

export interface OverflowSignature {
  provider: Providers;
  model: string;
  /** Class the SDK actually threw, for reference in review. */
  thrownAs: string;
  /**
   * Stand-in for the thrown error carrying exactly the fields the classifier
   * reads. Nested bodies are kept as the strings the SDKs attach.
   */
  error: Record<string, unknown>;
  expected: {
    kind: 'context_window' | 'request_too_large';
    limitTokens?: number;
    requestedTokens?: number;
  };
  /** Set when the signature is only decidable with caller-side corroboration. */
  requiresContextPressure?: boolean;
}

export const OVERFLOW_SIGNATURES: readonly OverflowSignature[] = [
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-haiku-4-5-20251001',
    thrownAs: 'ContextOverflowError',
    error: {
      name: 'ContextOverflowError',
      lc_error_code: 'CONTEXT_OVERFLOW',
      message:
        '400 {"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 274468 tokens > 200000 maximum"},"request_id":"req_test"}\n\nTroubleshooting URL: https://docs.langchain.com/oss/javascript/langchain/errors/CONTEXT_OVERFLOW/\n',
      cause:
        '{"status":400,"headers":{},"requestID":"req_test","error":{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 274468 tokens > 200000 maximum"},"request_id":"req_test"},"type":"invalid_request_error"}',
    },
    expected: {
      kind: 'context_window',
      limitTokens: 200_000,
      requestedTokens: 274_468,
    },
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-4o-mini',
    thrownAs: 'ContextOverflowError',
    error: {
      name: 'ContextOverflowError',
      message:
        '400 This model\'s maximum context length is 128000 tokens. However, your messages resulted in 149767 tokens. Please reduce the length of the messages.',
      cause:
        '{"status":400,"headers":{},"requestID":"req_test","error":{"message":"This model\'s maximum context length is 128000 tokens. However, your messages resulted in 149767 tokens. Please reduce the length of the messages.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"},"code":"context_length_exceeded","param":"messages","type":"invalid_request_error"}',
    },
    expected: {
      kind: 'context_window',
      limitTokens: 128_000,
      requestedTokens: 149_767,
    },
  },
  {
    /**
     * The common OpenAI symptom in practice: the request is large enough to
     * overrun the per-minute token bucket, so it is rejected as a 429 before
     * context validation ever runs. LangChain labels it MODEL_RATE_LIMIT.
     */
    provider: Providers.OPENAI,
    model: 'gpt-5-nano',
    thrownAs: 'RateLimitError',
    error: {
      name: 'Error',
      status: 429,
      code: 'rate_limit_exceeded',
      type: 'tokens',
      lc_error_code: 'MODEL_RATE_LIMIT',
      message:
        '429 Request too large for gpt-5-nano in organization org-test on tokens per min (TPM): Limit 200000, Requested 480002. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.\n\nTroubleshooting URL: https://docs.langchain.com/oss/javascript/langchain/errors/MODEL_RATE_LIMIT/\n',
      error:
        '{"message":"Request too large for gpt-5-nano in organization org-test on tokens per min (TPM): Limit 200000, Requested 480002. The input or output tokens must be reduced in order to run successfully.","type":"tokens","param":null,"code":"rate_limit_exceeded"}',
    },
    expected: {
      kind: 'request_too_large',
      limitTokens: 200_000,
      requestedTokens: 480_002,
    },
  },
  {
    provider: Providers.BEDROCK,
    model: 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    thrownAs: 'ValidationException',
    error: {
      name: 'ValidationException',
      message:
        'The model returned the following errors: prompt is too long: 207848 tokens > 200000 maximum',
      $metadata: { httpStatusCode: 400 },
    },
    expected: {
      kind: 'context_window',
      limitTokens: 200_000,
      requestedTokens: 207_848,
    },
  },
  {
    /** Same provider, same model family, no numbers reported. */
    provider: Providers.BEDROCK,
    model: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    thrownAs: 'ValidationException',
    error: {
      name: 'ValidationException',
      message:
        'The model returned the following errors: Input is too long for requested model.',
      $metadata: { httpStatusCode: 400 },
    },
    expected: { kind: 'context_window' },
  },
  {
    /**
     * Bedrock's most dangerous shape: the overflow arrives inside a
     * successful HTTP 200 stream, so status-code-based detection is useless.
     */
    provider: Providers.BEDROCK,
    model: 'us.amazon.nova-lite-v1:0',
    thrownAs: 'Error',
    error: {
      name: 'Error',
      message:
        'The model returned the following errors: Input Tokens Exceeded: Number of input tokens exceeds maximum length. Please update the input to try again.\n  Deserialization error: to see the raw response, inspect the hidden field {error}.$response on this object.',
      $metadata: { httpStatusCode: 200 },
    },
    expected: { kind: 'context_window' },
  },
  {
    provider: Providers.BEDROCK,
    model: 'us.meta.llama3-1-70b-instruct-v1:0',
    thrownAs: 'Error',
    error: {
      name: 'Error',
      message:
        'The model returned the following errors: This model\'s maximum context length is 131072 tokens. Please reduce the length of the prompt\n  Deserialization error: to see the raw response, inspect the hidden field {error}.$response on this object.',
      $metadata: { httpStatusCode: 200 },
    },
    expected: { kind: 'context_window', limitTokens: 131_072 },
  },
  {
    provider: Providers.GOOGLE,
    model: 'gemini-3.1-flash-image',
    thrownAs: 'GoogleGenerativeAIFetchError',
    error: {
      name: 'Error',
      status: 400,
      message:
        '[GoogleGenerativeAI Error]: Error fetching from https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:streamGenerateContent?alt=sse: [400 Bad Request] The input token count exceeds the maximum number of tokens allowed (65536).',
    },
    expected: { kind: 'context_window', limitTokens: 65_536 },
  },
  {
    /**
     * Vertex AI's gaxios path discards the API error document, leaving a bare
     * status line indistinguishable from any other 400 — the only signature
     * in this set that needs caller-side corroboration.
     */
    provider: Providers.VERTEXAI,
    model: 'gemini-2.5-flash-lite',
    thrownAs: 'Error',
    error: {
      name: 'Error',
      message: 'Google request failed with status code 400',
    },
    expected: { kind: 'context_window' },
    requiresContextPressure: true,
  },
  {
    provider: Providers.OPENROUTER,
    model: 'qwen/qwen-2.5-7b-instruct',
    thrownAs: 'ContextOverflowError',
    error: {
      name: 'ContextOverflowError',
      message:
        '400 This endpoint\'s maximum context length is 32768 tokens. However, you requested about 56827 tokens (56811 of text input, 16 in the output). Please reduce the length of either one, or use the context-compression plugin to compress your prompt automatically.',
      cause:
        '{"status":400,"headers":{},"requestID":null,"error":{"message":"This endpoint\'s maximum context length is 32768 tokens. However, you requested about 56827 tokens (56811 of text input, 16 in the output).","code":400,"metadata":{"provider_name":null}},"code":400}',
    },
    expected: {
      kind: 'context_window',
      limitTokens: 32_768,
      requestedTokens: 56_827,
    },
  },
  {
    provider: Providers.DEEPSEEK,
    model: 'deepseek-v4-flash',
    thrownAs: 'ContextOverflowError',
    error: {
      name: 'ContextOverflowError',
      message:
        '400 This model\'s maximum context length is 1048565 tokens. However, you requested 1179668 tokens (1179652 in the messages, 16 in the completion). Please reduce the length of the messages or completion.',
      cause:
        '{"status":400,"headers":{},"requestID":null,"error":{"message":"This model\'s maximum context length is 1048565 tokens. However, you requested 1179668 tokens (1179652 in the messages, 16 in the completion).","type":"invalid_request_error","param":null,"code":"invalid_request_error"},"code":"invalid_request_error"}',
    },
    expected: {
      kind: 'context_window',
      limitTokens: 1_048_565,
      requestedTokens: 1_179_668,
    },
  },
  {
    /** xAI says "prompt length", which LangChain's own matcher does not catch. */
    provider: Providers.XAI,
    model: 'grok-build-0.1',
    thrownAs: 'BadRequestError',
    error: {
      name: 'Error',
      status: 400,
      message:
        '400 "This model\'s maximum prompt length is 256000 but the request contains 332986 tokens."',
      error:
        'This model\'s maximum prompt length is 256000 but the request contains 332986 tokens.',
    },
    expected: {
      kind: 'context_window',
      limitTokens: 256_000,
      requestedTokens: 332_986,
    },
  },
  {
    provider: Providers.MISTRAL,
    model: 'mistral-tiny-latest',
    thrownAs: 'SDKError',
    error: {
      name: 'SDKError',
      status: 400,
      message:
        'API error occurred: Status 400 Content-Type application/json Body \n{"object":"error","message":"Prompt contains 170397 tokens and 0 draft tokens, too large for model with 131072 maximum context length","type":"invalid_request_invalid_args","param":null,"code":"3051","raw_status_code":400}',
      body: '{"object":"error","message":"Prompt contains 170397 tokens and 0 draft tokens, too large for model with 131072 maximum context length","type":"invalid_request_invalid_args","param":null,"code":"3051","raw_status_code":400}',
    },
    expected: {
      kind: 'context_window',
      limitTokens: 131_072,
      requestedTokens: 170_397,
    },
  },
] as const;

/**
 * Errors that mention limits, sizes, or tokens but are NOT fixed by
 * compaction. The first two were captured live alongside the signatures
 * above; the rest are the adjacent failures a loose matcher would swallow.
 */
export const NON_OVERFLOW_SIGNATURES: readonly {
  label: string;
  error: Record<string, unknown>;
}[] = [
  {
    label: 'bedrock invalid model identifier (captured live)',
    error: {
      name: 'ValidationException',
      message: 'The provided model identifier is invalid.',
      $metadata: { httpStatusCode: 400 },
    },
  },
  {
    label: 'bedrock legacy model access denied (captured live)',
    error: {
      name: 'ResourceNotFoundException',
      message:
        'Access denied. This Model is marked by provider as Legacy and you have not been actively using the model in the last 30 days. Please upgrade to an active model.',
      $metadata: { httpStatusCode: 404 },
    },
  },
  {
    label: 'openai genuine token throttling — request alone fits the bucket',
    error: {
      name: 'Error',
      status: 429,
      code: 'rate_limit_exceeded',
      type: 'tokens',
      message:
        '429 Rate limit reached for gpt-4o in organization org-test on tokens per min (TPM): Limit 30000, Used 28000, Requested 4000. Please try again in 4ms.',
    },
  },
  {
    label: 'openai request-per-minute throttling',
    error: {
      name: 'Error',
      status: 429,
      code: 'rate_limit_exceeded',
      type: 'requests',
      message:
        '429 Rate limit reached for gpt-4o in organization org-test on requests per min (RPM): Limit 500, Used 500, Requested 1. Please try again in 120ms.',
    },
  },
  {
    label: 'output token cap, not input',
    error: {
      name: 'BadRequestError',
      status: 400,
      message:
        '400 max_tokens is too large: 200000. This model supports at most 64000 completion tokens, whereas you provided 200000.',
    },
  },
  {
    label: 'anthropic max_tokens above model output limit',
    error: {
      name: 'BadRequestError',
      status: 400,
      message:
        '400 {"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: 100000 > 64000, which is the maximum allowed number of output tokens for claude-sonnet-4-5"}}',
    },
  },
  {
    label: 'quota exhausted',
    error: {
      name: 'Error',
      status: 429,
      code: 'insufficient_quota',
      message:
        '429 You exceeded your current quota, please check your plan and billing details.',
    },
  },
  {
    label: 'authentication failure',
    error: {
      name: 'AuthenticationError',
      status: 401,
      message: '401 Incorrect API key provided.',
    },
  },
];
