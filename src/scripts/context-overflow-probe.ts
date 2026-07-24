/**
 * Context-overflow signature probe.
 *
 * Deliberately sends an over-limit prompt to every provider the SDK
 * supports and records the raw error each one throws, so the overflow
 * classifier is grounded in observed provider behavior rather than
 * guessed phrases.
 *
 * Run:
 *   DOTENV_CONFIG_PATH=/path/to/.env node --loader ./tsconfig-paths-bootstrap.mjs \
 *     --experimental-specifier-resolution=node ./src/scripts/context-overflow-probe.ts
 *
 * Flags:
 *   --only <provider[,provider]>  restrict to given providers
 *   --model <substring>           restrict to matching model ids
 *   --tier <full|confirm>         restrict to a load tier
 *   --mode <stream|invoke|both>   which invocation path to probe (default: stream)
 *   --out <path>                  output JSON path
 *   --list                        print the resolved matrix and exit
 *
 * Cost note: an over-limit request is rejected at request validation, so
 * providers do not bill the prompt. The `confirm` tier still overshoots by a
 * wide margin so an expensive model can never accidentally accept the prompt.
 */
import { writeFileSync } from 'fs';
import { config as loadEnv } from 'dotenv';
import { HumanMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { initializeModel } from '@/llm/init';
import { Providers } from '@/common';

loadEnv({ path: process.env.DOTENV_CONFIG_PATH ?? '.env' });

/**
 * `full` targets are cheap enough to probe at a modest overshoot.
 * `confirm` targets are expensive or huge-context; they get a single
 * wide-margin attempt purely to confirm the API surface's signature.
 */
type ProbeTier = 'full' | 'confirm';

type ProbeMode = 'stream' | 'invoke';

interface ProbeTarget {
  provider: Providers;
  /** Model id as the provider expects it. */
  model: string;
  /** Documented input context window, in tokens. */
  contextWindow: number;
  tier: ProbeTier;
  /** Upstream API actually serving the request (OpenRouter routes to many). */
  upstream?: string;
}

interface SerializedError {
  constructorName: string;
  name?: string;
  message: string;
  status?: number;
  code?: string;
  type?: string;
  errorType?: string;
  errorCode?: string;
  errorStatus?: string;
  httpStatusCode?: number;
  awsErrorType?: string;
  requestId?: string;
  lcErrorCode?: string;
  /** gaxios-style clients (Vertex AI) hide the API error document here. */
  responseData?: string;
  /** Enumerable own properties, minus noisy/secret-bearing ones. */
  ownProperties: Record<string, string>;
  /** Best-effort raw JSON body if the SDK attached one. */
  rawBody?: string;
}

interface ProbeResult {
  provider: Providers;
  model: string;
  upstream?: string;
  tier: ProbeTier;
  mode: ProbeMode;
  contextWindow: number;
  approxTokensSent: number;
  outcome: 'rejected' | 'accepted' | 'skipped' | 'unavailable';
  skipReason?: string;
  durationMs: number;
  error?: SerializedError;
}

const OVERSHOOT_BY_TIER: Record<ProbeTier, number> = {
  full: 1.3,
  confirm: 2,
};

/** Absolute floor so tiny-context models still clear the window decisively. */
const MIN_OVERSHOOT_TOKENS = 4_000;

/**
 * Payload ceiling (~12MB of text). Beyond this the request stops testing the
 * token limit and starts testing the provider's body-size limit, which is a
 * different failure mode. Targets that would need more are skipped rather
 * than probed with a smaller payload, since an under-limit payload could be
 * *accepted* and billed.
 */
const MAX_PROBE_TOKENS = 2_400_000;

/**
 * Words chosen to be single tokens under BPE vocabularies, so an N-word
 * payload is guaranteed to be at least N tokens for every provider. Under-
 * counting would risk a request being *accepted* and billed; over-counting
 * only costs bandwidth.
 */
const FILLER_WORDS = [
  'the',
  'quick',
  'brown',
  'fox',
  'jumps',
  'over',
  'lazy',
  'dog',
  'and',
  'then',
  'runs',
  'past',
  'green',
  'river',
  'under',
  'bright',
  'morning',
  'sky',
];

function buildOverflowText(wordCount: number): string {
  const parts: string[] = new Array(wordCount);
  for (let i = 0; i < wordCount; i++) {
    parts[i] = FILLER_WORDS[i % FILLER_WORDS.length];
  }
  return parts.join(' ');
}

function overflowTokenTarget(
  target: ProbeTarget,
  factorOverride?: number
): number {
  const factor = factorOverride ?? OVERSHOOT_BY_TIER[target.tier];
  const scaled = Math.ceil(target.contextWindow * factor);
  if (factorOverride != null) {
    return scaled;
  }
  return Math.max(scaled, target.contextWindow + MIN_OVERSHOOT_TOKENS);
}

const SECRET_KEY_RE = /key|token|secret|credential|authorization|password/i;

function readString(
  source: Record<string, unknown>,
  key: string
): string | undefined {
  const value = source[key];
  return typeof value === 'string' && value !== '' ? value : undefined;
}

function readNumber(
  source: Record<string, unknown>,
  key: string
): number | undefined {
  const value = source[key];
  return typeof value === 'number' && Number.isFinite(value)
    ? value
    : undefined;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null
    ? (value as Record<string, unknown>)
    : undefined;
}

function stringifySafe(value: unknown, limit = 4_000): string {
  if (typeof value === 'string') {
    return value.slice(0, limit);
  }
  try {
    return JSON.stringify(value)?.slice(0, limit) ?? String(value);
  } catch {
    return String(value).slice(0, limit);
  }
}

function collectOwnProperties(error: object): Record<string, string> {
  const collected: Record<string, string> = {};
  for (const key of Object.keys(error)) {
    if (SECRET_KEY_RE.test(key)) {
      continue;
    }
    if (key === 'stack' || key === 'message') {
      continue;
    }
    collected[key] = stringifySafe(
      (error as Record<string, unknown>)[key],
      800
    );
  }
  return collected;
}

/**
 * Flattens a provider error into the fields a classifier could realistically
 * key on. Deliberately shallow-but-wide: the point of the probe is to learn
 * which of these fields providers actually populate.
 */
function serializeError(error: unknown): SerializedError {
  const record = asRecord(error) ?? {};
  const nestedError = asRecord(record.error);
  const metadata = asRecord(record.$metadata);
  const responseBody = asRecord(record.response);

  const message =
    error instanceof Error
      ? error.message
      : (readString(record, 'message') ?? stringifySafe(error));

  return {
    constructorName:
      typeof error === 'object' && error !== null
        ? error.constructor.name
        : typeof error,
    name: readString(record, 'name'),
    message: message.slice(0, 4_000),
    status: readNumber(record, 'status') ?? readNumber(record, 'statusCode'),
    code: readString(record, 'code'),
    type: readString(record, 'type'),
    errorType: nestedError ? readString(nestedError, 'type') : undefined,
    errorCode: nestedError ? readString(nestedError, 'code') : undefined,
    errorStatus: nestedError ? readString(nestedError, 'status') : undefined,
    httpStatusCode: metadata
      ? readNumber(metadata, 'httpStatusCode')
      : undefined,
    awsErrorType: readString(record, '__type'),
    requestId:
      readString(record, 'request_id') ?? readString(record, 'requestId'),
    lcErrorCode: readString(record, 'lc_error_code'),
    responseData:
      responseBody?.data != null
        ? stringifySafe(responseBody.data, 2_000)
        : undefined,
    ownProperties:
      typeof error === 'object' && error !== null
        ? collectOwnProperties(error)
        : {},
    rawBody: nestedError
      ? stringifySafe(nestedError)
      : responseBody
        ? stringifySafe(responseBody)
        : undefined,
  };
}

function envValue(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value != null && value !== ''
    ? value.replace(/^["']|["']$/g, '')
    : undefined;
}

interface CredentialCheck {
  ok: boolean;
  reason?: string;
}

function checkCredentials(provider: Providers): CredentialCheck {
  switch (provider) {
    case Providers.OPENAI:
      return envValue('OPENAI_API_KEY')
        ? { ok: true }
        : { ok: false, reason: 'OPENAI_API_KEY not set' };
    case Providers.AZURE:
      return envValue('AZURE_OPENAI_API_KEY') &&
        envValue('AZURE_OPENAI_API_INSTANCE')
        ? { ok: true }
        : { ok: false, reason: 'AZURE_OPENAI_API_* not set' };
    case Providers.ANTHROPIC:
      return envValue('ANTHROPIC_API_KEY')
        ? { ok: true }
        : { ok: false, reason: 'ANTHROPIC_API_KEY not set' };
    case Providers.BEDROCK:
      return (envValue('BEDROCK_AWS_ACCESS_KEY_ID') ??
        envValue('AWS_ACCESS_KEY_ID'))
        ? { ok: true }
        : { ok: false, reason: 'BEDROCK_AWS_* credentials not set' };
    case Providers.GOOGLE:
      return (envValue('GOOGLE_API_KEY') ?? envValue('GEMINI_API_KEY'))
        ? { ok: true }
        : { ok: false, reason: 'GOOGLE_API_KEY / GEMINI_API_KEY not set' };
    case Providers.VERTEXAI:
      return (envValue('GOOGLE_APPLICATION_CREDENTIALS') ??
        envValue('VERTEXAI_KEY_FILE'))
        ? { ok: true }
        : { ok: false, reason: 'Vertex credentials not set' };
    case Providers.OPENROUTER:
      return envValue('OPENROUTER_API_KEY')
        ? { ok: true }
        : { ok: false, reason: 'OPENROUTER_API_KEY not set' };
    case Providers.DEEPSEEK:
      return envValue('DEEPSEEK_API_KEY')
        ? { ok: true }
        : { ok: false, reason: 'DEEPSEEK_API_KEY not set' };
    case Providers.XAI:
      return envValue('XAI_API_KEY')
        ? { ok: true }
        : { ok: false, reason: 'XAI_API_KEY not set' };
    case Providers.MISTRAL:
    case Providers.MISTRALAI:
      return envValue('MISTRAL_API_KEY')
        ? { ok: true }
        : { ok: false, reason: 'MISTRAL_API_KEY not set' };
    case Providers.MOONSHOT:
      return envValue('MOONSHOT_API_KEY')
        ? { ok: true }
        : { ok: false, reason: 'MOONSHOT_API_KEY not set' };
    default:
      return { ok: false, reason: `no credential rule for ${provider}` };
  }
}

/** Output cap kept minimal — a probe never wants generated tokens. */
const MAX_OUTPUT_TOKENS = 16;

function buildClientOptions(target: ProbeTarget): t.ClientOptions {
  const { provider, model } = target;

  if (provider === Providers.ANTHROPIC) {
    return {
      model,
      apiKey: envValue('ANTHROPIC_API_KEY'),
      maxTokens: MAX_OUTPUT_TOKENS,
      streaming: true,
    } as t.AnthropicClientOptions;
  }

  if (provider === Providers.BEDROCK) {
    const accessKeyId =
      envValue('BEDROCK_AWS_ACCESS_KEY_ID') ?? envValue('AWS_ACCESS_KEY_ID');
    const secretAccessKey =
      envValue('BEDROCK_AWS_SECRET_ACCESS_KEY') ??
      envValue('AWS_SECRET_ACCESS_KEY');
    const sessionToken =
      envValue('BEDROCK_AWS_SESSION_TOKEN') ?? envValue('AWS_SESSION_TOKEN');
    return {
      model,
      region:
        envValue('BEDROCK_AWS_REGION') ??
        envValue('AWS_REGION') ??
        envValue('AWS_DEFAULT_REGION') ??
        'us-east-1',
      maxTokens: MAX_OUTPUT_TOKENS,
      credentials:
        accessKeyId != null && secretAccessKey != null
          ? {
              accessKeyId,
              secretAccessKey,
              ...(sessionToken != null ? { sessionToken } : {}),
            }
          : undefined,
    } as t.BedrockConverseClientOptions;
  }

  if (provider === Providers.GOOGLE) {
    return {
      model,
      apiKey: envValue('GOOGLE_API_KEY') ?? envValue('GEMINI_API_KEY'),
      maxOutputTokens: MAX_OUTPUT_TOKENS,
    } as t.GoogleClientOptions;
  }

  if (provider === Providers.VERTEXAI) {
    return {
      model,
      maxOutputTokens: MAX_OUTPUT_TOKENS,
      location:
        envValue('GOOGLE_CLOUD_LOCATION') ??
        envValue('GOOGLE_LOC') ??
        'us-central1',
    } as t.VertexAIClientOptions;
  }

  if (provider === Providers.OPENROUTER) {
    return {
      model,
      apiKey: envValue('OPENROUTER_API_KEY'),
      maxTokens: MAX_OUTPUT_TOKENS,
      configuration: {
        baseURL:
          envValue('OPENROUTER_BASE_URL') ?? 'https://openrouter.ai/api/v1',
      },
    } as t.OpenAIClientOptions;
  }

  if (provider === Providers.DEEPSEEK) {
    return {
      model,
      apiKey: envValue('DEEPSEEK_API_KEY'),
      maxTokens: MAX_OUTPUT_TOKENS,
    } as t.DeepSeekClientOptions;
  }

  if (provider === Providers.XAI) {
    return {
      model,
      apiKey: envValue('XAI_API_KEY'),
      maxTokens: MAX_OUTPUT_TOKENS,
    } as t.XAIClientOptions;
  }

  if (provider === Providers.MISTRAL || provider === Providers.MISTRALAI) {
    return {
      model,
      apiKey: envValue('MISTRAL_API_KEY'),
      maxTokens: MAX_OUTPUT_TOKENS,
    } as t.MistralAIClientOptions;
  }

  if (provider === Providers.MOONSHOT) {
    return {
      model,
      apiKey: envValue('MOONSHOT_API_KEY'),
      maxTokens: MAX_OUTPUT_TOKENS,
    } as t.OpenAIClientOptions;
  }

  if (provider === Providers.AZURE) {
    return {
      model,
      azureOpenAIApiKey: envValue('AZURE_OPENAI_API_KEY'),
      azureOpenAIApiInstanceName: envValue('AZURE_OPENAI_API_INSTANCE'),
      azureOpenAIApiDeploymentName: envValue('AZURE_OPENAI_API_DEPLOYMENT'),
      azureOpenAIApiVersion: envValue('AZURE_OPENAI_API_VERSION'),
      maxTokens: MAX_OUTPUT_TOKENS,
    } as t.AzureClientOptions;
  }

  return {
    model,
    apiKey: envValue('OPENAI_API_KEY'),
    maxTokens: MAX_OUTPUT_TOKENS,
  } as t.OpenAIClientOptions;
}

/**
 * The probe matrix. Context windows are the documented input limits; the
 * probe only needs them to be right enough to overshoot.
 */
const PROBE_MATRIX: ProbeTarget[] = [
  /* ---------------- OpenAI ---------------- */
  {
    provider: Providers.OPENAI,
    model: 'gpt-4',
    contextWindow: 8_192,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-3.5-turbo',
    contextWindow: 16_385,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-4o-mini',
    contextWindow: 128_000,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-4o',
    contextWindow: 128_000,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-4.1-nano',
    contextWindow: 1_047_576,
    tier: 'confirm',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-5-nano',
    contextWindow: 400_000,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-5-mini',
    contextWindow: 400_000,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-5.4-nano',
    contextWindow: 400_000,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-5.4-mini',
    contextWindow: 400_000,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-5.4',
    contextWindow: 400_000,
    tier: 'confirm',
  },
  {
    provider: Providers.OPENAI,
    model: 'gpt-5.5',
    contextWindow: 400_000,
    tier: 'confirm',
  },
  {
    provider: Providers.OPENAI,
    model: 'o4-mini',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.OPENAI,
    model: 'o3',
    contextWindow: 200_000,
    tier: 'confirm',
  },

  /* ---------------- Azure OpenAI ---------------- */
  {
    provider: Providers.AZURE,
    model: envValue('AZURE_MODEL_NAME') ?? 'gpt-4o-mini',
    contextWindow: 128_000,
    tier: 'full',
  },

  /* ---------------- Anthropic ---------------- */
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-haiku-4-5-20251001',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-sonnet-4-5-20250929',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-sonnet-4-6',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-fable-5',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-sonnet-5',
    contextWindow: 200_000,
    tier: 'confirm',
  },
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-opus-4-5-20251101',
    contextWindow: 200_000,
    tier: 'confirm',
  },
  {
    provider: Providers.ANTHROPIC,
    model: 'claude-opus-5',
    contextWindow: 200_000,
    tier: 'confirm',
  },

  /* ---------------- Bedrock ---------------- */
  {
    provider: Providers.BEDROCK,
    model: 'anthropic.claude-3-haiku-20240307-v1:0',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.BEDROCK,
    model: 'us.anthropic.claude-haiku-4-5-20251001-v1:0',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.BEDROCK,
    model: 'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    contextWindow: 200_000,
    tier: 'full',
  },
  {
    provider: Providers.BEDROCK,
    model: 'us.amazon.nova-lite-v1:0',
    contextWindow: 300_000,
    tier: 'full',
  },
  {
    provider: Providers.BEDROCK,
    model: 'us.meta.llama3-1-70b-instruct-v1:0',
    contextWindow: 128_000,
    tier: 'full',
  },
  {
    provider: Providers.BEDROCK,
    model: 'mistral.mistral-large-2407-v1:0',
    contextWindow: 128_000,
    tier: 'full',
  },
  {
    provider: Providers.BEDROCK,
    model: 'us.anthropic.claude-opus-4-5-20251101-v1:0',
    contextWindow: 200_000,
    tier: 'confirm',
  },

  /* ---------------- Google (Gemini API) ---------------- */
  {
    provider: Providers.GOOGLE,
    model: 'gemini-3.1-flash-image',
    contextWindow: 65_536,
    tier: 'full',
  },
  {
    provider: Providers.GOOGLE,
    model: 'gemini-omni-flash-preview',
    contextWindow: 131_072,
    tier: 'full',
  },
  {
    provider: Providers.GOOGLE,
    model: 'gemini-2.5-flash-lite',
    contextWindow: 1_048_576,
    tier: 'confirm',
  },
  {
    provider: Providers.GOOGLE,
    model: 'gemini-2.5-flash',
    contextWindow: 1_048_576,
    tier: 'confirm',
  },
  {
    provider: Providers.GOOGLE,
    model: 'gemini-3.5-flash',
    contextWindow: 1_048_576,
    tier: 'confirm',
  },
  {
    provider: Providers.GOOGLE,
    model: 'gemini-2.5-pro',
    contextWindow: 1_048_576,
    tier: 'confirm',
  },

  /* ---------------- Vertex AI ---------------- */
  {
    provider: Providers.VERTEXAI,
    model: 'gemini-2.5-flash-lite',
    contextWindow: 1_048_576,
    tier: 'confirm',
  },
  {
    provider: Providers.VERTEXAI,
    model: 'gemini-2.5-flash',
    contextWindow: 1_048_576,
    tier: 'confirm',
  },

  /* ---------------- OpenRouter (many upstreams) ---------------- */
  {
    provider: Providers.OPENROUTER,
    model: 'qwen/qwen-2.5-7b-instruct',
    contextWindow: 32_768,
    tier: 'full',
    upstream: 'qwen',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'mistralai/mistral-nemo',
    contextWindow: 131_072,
    tier: 'full',
    upstream: 'mistral',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'meta-llama/llama-3.1-8b-instruct',
    contextWindow: 131_072,
    tier: 'full',
    upstream: 'meta',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'openai/gpt-4o-mini',
    contextWindow: 128_000,
    tier: 'full',
    upstream: 'openai',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'anthropic/claude-haiku-4.5',
    contextWindow: 200_000,
    tier: 'full',
    upstream: 'anthropic',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'deepseek/deepseek-chat',
    contextWindow: 163_840,
    tier: 'full',
    upstream: 'deepseek',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'moonshotai/kimi-k2',
    contextWindow: 131_072,
    tier: 'full',
    upstream: 'moonshot',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'google/gemini-3.1-flash-lite-image',
    contextWindow: 65_536,
    tier: 'full',
    upstream: 'google',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'google/gemini-3.5-flash-lite',
    contextWindow: 1_048_576,
    tier: 'confirm',
    upstream: 'google',
  },
  {
    provider: Providers.OPENROUTER,
    model: 'x-ai/grok-build-0.1',
    contextWindow: 256_000,
    tier: 'full',
    upstream: 'xai',
  },

  /* ---------------- DeepSeek ---------------- */
  {
    provider: Providers.DEEPSEEK,
    model: 'deepseek-v4-flash',
    contextWindow: 131_072,
    tier: 'full',
  },
  {
    provider: Providers.DEEPSEEK,
    model: 'deepseek-v4-pro',
    contextWindow: 131_072,
    tier: 'confirm',
  },

  /* ---------------- xAI ---------------- */
  {
    provider: Providers.XAI,
    model: 'grok-build-0.1',
    contextWindow: 256_000,
    tier: 'full',
  },
  {
    provider: Providers.XAI,
    model: 'grok-4.5',
    contextWindow: 500_000,
    tier: 'confirm',
  },
  {
    provider: Providers.XAI,
    model: 'grok-4.3',
    contextWindow: 1_000_000,
    tier: 'confirm',
  },

  /* ---------------- Mistral ---------------- */
  {
    provider: Providers.MISTRAL,
    model: 'mistral-tiny-latest',
    contextWindow: 131_072,
    tier: 'full',
  },
  {
    provider: Providers.MISTRAL,
    model: 'mistral-medium-2508',
    contextWindow: 131_072,
    tier: 'confirm',
  },

  /* ---------------- Moonshot ---------------- */
  {
    provider: Providers.MOONSHOT,
    model: 'moonshot-v1-8k',
    contextWindow: 8_192,
    tier: 'full',
  },
];

interface CliOptions {
  only?: Set<string>;
  modelFilter?: string;
  tier?: ProbeTier;
  modes: ProbeMode[];
  out: string;
  list: boolean;
  /**
   * Overrides the tier overshoot. Useful for landing a payload *between* the
   * model's context window and the account's per-minute token allowance, which
   * is the only way to observe a provider's true context-window rejection on
   * accounts whose TPM ceiling sits below the window.
   */
  factor?: number;
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    modes: ['stream'],
    out: 'context-overflow-signatures.json',
    list: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--only') {
      options.only = new Set(
        argv[++i]?.split(',').map((value) => value.trim())
      );
    } else if (arg === '--model') {
      options.modelFilter = argv[++i];
    } else if (arg === '--tier') {
      options.tier = argv[++i] as ProbeTier;
    } else if (arg === '--mode') {
      const mode = argv[++i];
      options.modes =
        mode === 'both' ? ['stream', 'invoke'] : [mode as ProbeMode];
    } else if (arg === '--out') {
      options.out = argv[++i];
    } else if (arg === '--factor') {
      options.factor = Number(argv[++i]);
    } else if (arg === '--list') {
      options.list = true;
    }
  }
  return options;
}

function selectTargets(options: CliOptions): ProbeTarget[] {
  return PROBE_MATRIX.filter((target) => {
    if (options.only && !options.only.has(target.provider)) {
      return false;
    }
    if (options.modelFilter && !target.model.includes(options.modelFilter)) {
      return false;
    }
    if (options.tier && target.tier !== options.tier) {
      return false;
    }
    return true;
  });
}

async function runProbe(
  target: ProbeTarget,
  mode: ProbeMode,
  factor?: number
): Promise<ProbeResult> {
  const tokenTarget = overflowTokenTarget(target, factor);
  const base: Omit<ProbeResult, 'outcome' | 'durationMs'> = {
    provider: target.provider,
    model: target.model,
    upstream: target.upstream,
    tier: target.tier,
    mode,
    contextWindow: target.contextWindow,
    approxTokensSent: tokenTarget,
  };

  const credentials = checkCredentials(target.provider);
  if (!credentials.ok) {
    return {
      ...base,
      outcome: 'skipped',
      skipReason: credentials.reason,
      durationMs: 0,
    };
  }

  if (tokenTarget > MAX_PROBE_TOKENS) {
    return {
      ...base,
      outcome: 'skipped',
      skipReason: `payload would exceed ${MAX_PROBE_TOKENS} tokens`,
      durationMs: 0,
    };
  }

  const messages: BaseMessage[] = [
    new HumanMessage(buildOverflowText(tokenTarget)),
  ];
  const started = Date.now();

  try {
    const model = initializeModel({
      provider: target.provider,
      clientOptions: buildClientOptions(target),
    }) as t.ChatModel;

    if (mode === 'invoke' || model.stream == null) {
      await model.invoke(messages);
    } else {
      const stream = await model.stream(messages);
      for await (const _chunk of stream) {
        void _chunk;
      }
    }

    return { ...base, outcome: 'accepted', durationMs: Date.now() - started };
  } catch (error) {
    const serialized = serializeError(error);
    const unavailable =
      /not found|does not exist|access|not authorized|no such model|invalid model/i.test(
        serialized.message
      );
    return {
      ...base,
      outcome: unavailable ? 'unavailable' : 'rejected',
      durationMs: Date.now() - started,
      error: serialized,
    };
  }
}

function summarize(result: ProbeResult): string {
  const head = `${result.provider}/${result.model}${result.mode === 'invoke' ? ' [invoke]' : ''}`;
  if (result.outcome === 'skipped') {
    return `SKIP  ${head} — ${result.skipReason}`;
  }
  if (result.outcome === 'accepted') {
    return `ACCEPT ${head} — request was NOT rejected (${result.approxTokensSent} tokens)`;
  }
  const error = result.error;
  const status = error?.status ?? error?.httpStatusCode ?? '?';
  const code =
    error?.code ?? error?.errorType ?? error?.awsErrorType ?? error?.name ?? '';
  const label = result.outcome === 'unavailable' ? 'N/A  ' : 'HIT  ';
  return `${label} ${head} — ${status} ${code}: ${error?.message.slice(0, 160) ?? ''}`;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const targets = selectTargets(options);

  if (options.list) {
    for (const target of targets) {
      console.log(
        `${target.provider}\t${target.model}\tctx=${target.contextWindow}\ttier=${target.tier}\tprobe≈${overflowTokenTarget(target, options.factor)} tokens`
      );
    }
    return;
  }

  const results: ProbeResult[] = [];
  for (const target of targets) {
    for (const mode of options.modes) {
      const result = await runProbe(target, mode, options.factor);
      results.push(result);
      console.log(summarize(result));
    }
  }

  writeFileSync(
    options.out,
    JSON.stringify({ generatedAt: new Date().toISOString(), results }, null, 2)
  );
  console.log(`\nWrote ${results.length} probe results to ${options.out}`);
}

main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
