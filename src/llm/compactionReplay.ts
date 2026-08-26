import type { BaseMessage } from '@langchain/core/messages';
import type { ProviderMessageProjectionMode } from '@/llm/prepareProviderRequest';
import type * as t from '@/types';
import { inspectProviderSourceMessageIds } from '@/messages/provenance';
import { toJsonSchema } from '@/utils/schema';
import { Providers } from '@/common';

type CacheNamespaceValue = string;

export interface CompactionReplayRecipe {
  readonly provider: t.ProviderName;
  readonly modelId?: string;
  readonly projectionMode: ProviderMessageProjectionMode;
  readonly cacheNamespace: CompactionCacheNamespace;
  readonly promptCacheEnabled: boolean;
  readonly systemProjectionFingerprint?: string;
  readonly toolProjectionFingerprint?: string;
  readonly systemRevision: number;
  readonly toolRevision: number;
  readonly messages: readonly BaseMessage[];
  readonly sourceMessageFingerprints?: readonly string[];
}

export type CompactionReplayState = CompactionReplayRecipe | 'fallback';

export type CompactionReplayIneligibilityReason =
  | 'no_request_snapshot'
  | 'fallback_served_request'
  | 'summarizer_fallback_served_request'
  | 'provider_mismatch'
  | 'model_mismatch'
  | 'cache_namespace_unknown'
  | 'cache_namespace_mismatch'
  | 'prompt_cache_disabled'
  | 'system_projection_changed'
  | 'system_projection_unknown'
  | 'tool_projection_changed'
  | 'tool_projection_unknown'
  | 'projection_mode_mismatch'
  | 'restored_tool_substitution'
  | 'source_content_unknown'
  | 'source_content_mismatch'
  | 'ambiguous_lineage'
  | 'source_not_prefix';

export type CompactionReplayEligibility =
  | {
      readonly eligible: true;
      readonly replayMessageCount: number;
      readonly replaySourceCount: number;
      readonly requestSourceCount: number;
    }
  | {
      readonly eligible: false;
      readonly reason: CompactionReplayIneligibilityReason;
      readonly replaySourceCount: number;
      readonly requestSourceCount: number;
    };

export interface CompactionCacheNamespace {
  readonly complete: boolean;
  readonly entries: ReadonlyArray<
    readonly [key: string, value: CacheNamespaceValue]
  >;
}

interface CompactionReplayCandidate {
  readonly provider: t.ProviderName;
  readonly modelId?: string;
  readonly projectionMode?: ProviderMessageProjectionMode;
  readonly cacheNamespace: CompactionCacheNamespace;
  readonly promptCacheEnabled: boolean;
  readonly systemProjectionFingerprint?: string;
  readonly toolProjectionFingerprint?: string;
  readonly systemRevision: number;
  readonly toolRevision: number;
  readonly messages: readonly BaseMessage[];
  readonly restoredToolSubstitution: boolean;
  readonly summarizerFallbackServed?: boolean;
}

const CACHE_NAMESPACE_KEYS = [
  'apiKey',
  'anthropicApiKey',
  'anthropicApiUrl',
  'baseURL',
  'baseUrl',
  'organization',
  'project',
  'projectId',
  'region',
  'region_name',
  'endpoint',
  'deploymentName',
  'azureOpenAIApiDeploymentName',
  'azureOpenAIApiInstanceName',
  'azureOpenAIApiVersion',
  'azureOpenAIApiKey',
  'azureOpenAIBasePath',
  'openAIApiKey',
  'openAIBasePath',
  'openAIApiVersion',
  'apiVersion',
  'azureOpenAIEndpoint',
  'googleApiKey',
  'location',
  'credentials',
  'awsAccessKeyId',
  'awsSecretAccessKey',
  'awsSessionToken',
  'authOptions',
  'profile',
  'promptCache',
  'promptCacheTtl',
  'customHeaders',
  'defaultHeaders',
  'configuration',
  'useResponsesApi',
  'thinking',
  'thinkingBudget',
  'additionalModelRequestFields',
  'modelKwargs',
  'azureADTokenProvider',
] as const;

const BUILT_IN_PROVIDER_NAMES = new Set<string>(Object.values(Providers));

const PROVIDER_CREDENTIAL_KEYS: Partial<
  Record<Providers, readonly string[]>
> = {
  [Providers.OPENAI]: ['apiKey', 'openAIApiKey'],
  [Providers.AZURE]: [
    'apiKey',
    'openAIApiKey',
    'azureOpenAIApiKey',
    'azureADTokenProvider',
  ],
  [Providers.ANTHROPIC]: ['apiKey', 'anthropicApiKey'],
  [Providers.BEDROCK]: [
    'credentials',
    'awsAccessKeyId',
    'awsSecretAccessKey',
  ],
  [Providers.VERTEXAI]: ['credentials'],
  [Providers.GOOGLE]: ['apiKey', 'googleApiKey'],
  [Providers.MISTRALAI]: ['apiKey'],
  [Providers.MISTRAL]: ['apiKey'],
  [Providers.DEEPSEEK]: ['apiKey'],
  [Providers.OPENROUTER]: ['apiKey', 'openAIApiKey'],
  [Providers.XAI]: ['apiKey'],
  [Providers.MOONSHOT]: ['apiKey', 'openAIApiKey'],
};

interface EnvironmentRouteIdentity {
  readonly environmentKey: string;
  readonly optionPaths: readonly (readonly string[])[];
}

const PROVIDER_ENVIRONMENT_ROUTES: Partial<
  Record<Providers, readonly EnvironmentRouteIdentity[]>
> = {
  [Providers.OPENAI]: [
    {
      environmentKey: 'OPENAI_BASE_URL',
      optionPaths: [['baseURL'], ['baseUrl'], ['configuration', 'baseURL']],
    },
  ],
  [Providers.AZURE]: [
    {
      environmentKey: 'OPENAI_BASE_URL',
      optionPaths: [['baseURL'], ['baseUrl'], ['configuration', 'baseURL']],
    },
    {
      environmentKey: 'AZURE_OPENAI_ENDPOINT',
      optionPaths: [['azureOpenAIEndpoint'], ['endpoint']],
    },
    {
      environmentKey: 'AZURE_OPENAI_BASE_PATH',
      optionPaths: [
        ['azureOpenAIBasePath'],
        ['openAIBasePath'],
        ['configuration', 'baseURL'],
      ],
    },
  ],
  [Providers.BEDROCK]: [
    {
      environmentKey: 'BEDROCK_AWS_REGION',
      optionPaths: [['region'], ['region_name']],
    },
    {
      environmentKey: 'AWS_DEFAULT_REGION',
      optionPaths: [['region'], ['region_name']],
    },
  ],
  [Providers.VERTEXAI]: [
    {
      environmentKey: 'GOOGLE_CLOUD_PROJECT',
      optionPaths: [['projectId'], ['project']],
    },
    {
      environmentKey: 'GCLOUD_PROJECT',
      optionPaths: [['projectId'], ['project']],
    },
  ],
  [Providers.DEEPSEEK]: [
    {
      environmentKey: 'OPENAI_BASE_URL',
      optionPaths: [['baseURL'], ['baseUrl'], ['configuration', 'baseURL']],
    },
  ],
  [Providers.XAI]: [
    {
      environmentKey: 'OPENAI_BASE_URL',
      optionPaths: [['baseURL'], ['baseUrl'], ['configuration', 'baseURL']],
    },
  ],
  [Providers.MOONSHOT]: [
    {
      environmentKey: 'OPENAI_BASE_URL',
      optionPaths: [['baseURL'], ['baseUrl'], ['configuration', 'baseURL']],
    },
  ],
};

function hasExplicitCredentialIdentity(
  provider: t.ProviderName,
  options?: t.ClientOptions | Record<string, unknown>
): boolean {
  const keys = PROVIDER_CREDENTIAL_KEYS[provider as Providers];
  if (keys == null) {
    return false;
  }
  try {
    return keys.some(
      (key) =>
        (options as Record<string, unknown> | undefined)?.[key] !== undefined
    );
  } catch {
    return false;
  }
}

function hasDefinedOptionPath(
  options: t.ClientOptions | Record<string, unknown> | undefined,
  path: readonly string[]
): boolean | undefined {
  try {
    let current: unknown = options;
    for (const key of path) {
      if (current == null || typeof current !== 'object') {
        return false;
      }
      current = (current as Record<string, unknown>)[key];
    }
    return current !== undefined;
  } catch {
    return undefined;
  }
}

function serializeCacheNamespaceValue(
  value: unknown,
  seen: Set<object>
): string | undefined {
  if (value === null) {
    return 'null';
  }
  if (typeof value === 'string') {
    return `string:${JSON.stringify(value)}`;
  }
  if (typeof value === 'boolean') {
    return `boolean:${value}`;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return undefined;
    }
    return `number:${Object.is(value, -0) ? '-0' : value}`;
  }
  if (typeof value !== 'object' || seen.has(value)) {
    return undefined;
  }

  seen.add(value);
  const parts: string[] = [];
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      const serialized = serializeCacheNamespaceValue(value[i], seen);
      if (serialized == null) {
        seen.delete(value);
        return undefined;
      }
      parts.push(serialized);
    }
    seen.delete(value);
    return `array:[${parts.join(',')}]`;
  }

  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    seen.delete(value);
    return undefined;
  }
  if (Object.getOwnPropertySymbols(value).length > 0) {
    seen.delete(value);
    return undefined;
  }
  const keys = Object.getOwnPropertyNames(value).sort();
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (descriptor == null || !('value' in descriptor)) {
      seen.delete(value);
      return undefined;
    }
    const serialized = serializeCacheNamespaceValue(descriptor.value, seen);
    if (serialized == null) {
      seen.delete(value);
      return undefined;
    }
    parts.push(`${JSON.stringify(key)}:${serialized}`);
  }
  seen.delete(value);
  return `object:{${parts.join(',')}}`;
}

function fingerprintSerializedValue(serialized: string): string {
  let hashA = 0x811c9dc5;
  let hashB = 0x9e3779b9;
  for (let i = 0; i < serialized.length; i++) {
    const code = serialized.charCodeAt(i);
    hashA = Math.imul(hashA ^ code, 0x01000193);
    hashB = Math.imul(hashB ^ code, 0x5bd1e995);
    hashB ^= hashB >>> 13;
  }
  return `${serialized.length}:${hashA >>> 0}:${hashB >>> 0}`;
}

export const EMPTY_COMPACTION_SYSTEM_PROJECTION_FINGERPRINT =
  fingerprintSerializedValue('system:empty');

function fingerprintCacheNamespaceValue(value: unknown): string | undefined {
  try {
    const serialized = serializeCacheNamespaceValue(value, new Set<object>());
    return serialized == null
      ? undefined
      : fingerprintSerializedValue(serialized);
  } catch {
    return undefined;
  }
}

function projectToolForFingerprint(tool: unknown): object | undefined {
  try {
    if (tool == null || typeof tool !== 'object') {
      return undefined;
    }
    const candidate = tool as Record<string, unknown>;
    const projection: Record<string, unknown> = {};
    const name =
      typeof candidate.name === 'string' ? candidate.name : undefined;
    const description =
      typeof candidate.description === 'string'
        ? candidate.description
        : undefined;
    if (name != null) {
      projection.name = name;
    }
    if (description != null) {
      projection.description = description;
    }
    if (candidate.schema != null) {
      projection.schema = toJsonSchema(candidate.schema, name, description);
    }
    for (const key of [
      'type',
      'input_schema',
      'parameters',
      'function',
      'toolSpec',
      'cache_control',
      'cachePoint',
      'defer_loading',
      '__lc_bedrock_cache_point_after',
      '__lc_bedrock_skip_tool_cache',
    ] as const) {
      if (candidate[key] !== undefined) {
        projection[key] = candidate[key];
      }
    }
    const extras = candidate.extras as Record<string, unknown> | undefined;
    if (extras?.cache_control !== undefined) {
      projection.extrasCacheControl = extras.cache_control;
    }
    if (extras?.providerToolDefinition !== undefined) {
      projection.providerToolDefinition = extras.providerToolDefinition;
    }
    return Object.keys(projection).length === 0 ? undefined : projection;
  } catch {
    return undefined;
  }
}

export function createCompactionToolProjectionFingerprint(
  tools: readonly unknown[] | undefined
): string | undefined {
  const projections: object[] = [];
  for (const tool of tools ?? []) {
    const projection = projectToolForFingerprint(tool);
    if (projection == null) {
      return undefined;
    }
    projections.push(projection);
  }
  return fingerprintCacheNamespaceValue(projections);
}

export function isCompactionPromptCacheEnabled(
  provider: t.ProviderName,
  options?: t.ClientOptions | Record<string, unknown>
): boolean {
  if (
    provider !== Providers.ANTHROPIC &&
    provider !== Providers.BEDROCK &&
    provider !== Providers.OPENROUTER
  ) {
    return true;
  }
  try {
    return (
      (options as Record<string, unknown> | undefined)?.promptCache === true
    );
  } catch {
    return false;
  }
}

/** Captures only routing identity; values are never emitted in diagnostics. */
export function createCompactionCacheNamespace(
  provider: t.ProviderName,
  options?: t.ClientOptions | Record<string, unknown>,
  servingRouteKnown = true
): CompactionCacheNamespace {
  if (!BUILT_IN_PROVIDER_NAMES.has(provider) || !servingRouteKnown) {
    return Object.freeze({
      complete: false,
      entries: Object.freeze([]),
    });
  }
  const entries: Array<readonly [string, CacheNamespaceValue]> = [];
  let complete = hasExplicitCredentialIdentity(provider, options);
  for (const key of CACHE_NAMESPACE_KEYS) {
    let value: unknown;
    try {
      value = (options as Record<string, unknown> | undefined)?.[key];
    } catch {
      complete = false;
      continue;
    }
    if (value === undefined) {
      continue;
    }
    const fingerprint = fingerprintCacheNamespaceValue(value);
    if (fingerprint == null) {
      complete = false;
      continue;
    }
    entries.push(Object.freeze([key, fingerprint] as const));
  }
  for (const route of
    PROVIDER_ENVIRONMENT_ROUTES[provider as Providers] ?? []) {
    let overridden = false;
    for (const path of route.optionPaths) {
      const defined = hasDefinedOptionPath(options, path);
      if (defined == null) {
        complete = false;
        overridden = true;
        break;
      }
      if (defined) {
        overridden = true;
        break;
      }
    }
    if (overridden) {
      continue;
    }
    let value: string | undefined;
    try {
      value = process.env[route.environmentKey];
    } catch {
      complete = false;
      continue;
    }
    if (value === undefined) {
      continue;
    }
    const fingerprint = fingerprintCacheNamespaceValue(value);
    if (fingerprint == null) {
      complete = false;
      continue;
    }
    entries.push(
      Object.freeze([`env:${route.environmentKey}`, fingerprint] as const)
    );
  }
  return Object.freeze({
    complete,
    entries: Object.freeze(entries),
  });
}

function cacheNamespacesEqual(
  left: CompactionCacheNamespace,
  right: CompactionCacheNamespace
): boolean {
  if (left.entries.length !== right.entries.length) {
    return false;
  }
  for (let i = 0; i < left.entries.length; i++) {
    const [leftKey, leftValue] = left.entries[i];
    const [rightKey, rightValue] = right.entries[i];
    if (leftKey !== rightKey || !Object.is(leftValue, rightValue)) {
      return false;
    }
  }
  return true;
}

function isSyntheticMessage(message: BaseMessage): boolean {
  if (message.getType() === 'system') {
    return true;
  }
  const kwargs = message.additional_kwargs;
  if (kwargs.injected === true || kwargs.isMeta === true) {
    return true;
  }
  return kwargs.source != null && kwargs.source !== 'steer';
}

function fingerprintMessage(message: BaseMessage): string | undefined {
  try {
    const serialized = JSON.stringify(message.toDict());
    return fingerprintSerializedValue(serialized);
  } catch {
    return undefined;
  }
}

function fingerprintMessages(
  messages: readonly BaseMessage[]
): readonly string[] | undefined {
  const fingerprints: string[] = [];
  for (let i = 0; i < messages.length; i++) {
    const fingerprint = fingerprintMessage(messages[i]);
    if (fingerprint == null) {
      return undefined;
    }
    fingerprints.push(fingerprint);
  }
  return Object.freeze(fingerprints);
}

interface SourceLineage {
  readonly sourceMessageIds: readonly string[];
  readonly messageSourceCounts: readonly number[];
}

function appendUniqueSourceIds(
  target: string[],
  sourceIds: readonly string[]
): void {
  for (const sourceId of sourceIds) {
    if (target[target.length - 1] !== sourceId) {
      target.push(sourceId);
    }
  }
}

function inspectSourceLineage(
  messages: readonly BaseMessage[]
): SourceLineage | undefined {
  const sourceMessageIds: string[] = [];
  const messageSourceCounts: number[] = [];

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    const inspected = inspectProviderSourceMessageIds(message);
    if (inspected.status === 'invalid') {
      return undefined;
    }
    let ids = inspected.status === 'valid' ? inspected.sourceMessageIds : [];
    if (ids.length === 0 && !isSyntheticMessage(message)) {
      const fallbackId = message.id?.trim();
      if (fallbackId == null || fallbackId === '') {
        return undefined;
      }
      ids = [fallbackId];
    }
    if (ids.length === 0) {
      messageSourceCounts.push(sourceMessageIds.length);
      continue;
    }
    appendUniqueSourceIds(sourceMessageIds, ids);
    messageSourceCounts.push(sourceMessageIds.length);
  }

  return {
    sourceMessageIds: Object.freeze(sourceMessageIds),
    messageSourceCounts: Object.freeze(messageSourceCounts),
  };
}

export function createCompactionReplayRecipe(params: {
  provider: t.ProviderName;
  modelId?: string;
  projectionMode: ProviderMessageProjectionMode;
  cacheNamespace: CompactionCacheNamespace;
  promptCacheEnabled: boolean;
  systemProjectionFingerprint?: string;
  toolProjectionFingerprint?: string;
  systemRevision: number;
  toolRevision: number;
  messages: readonly BaseMessage[];
  sourceMessages: readonly BaseMessage[];
}): CompactionReplayRecipe {
  return Object.freeze({
    provider: params.provider,
    modelId: params.modelId,
    projectionMode: params.projectionMode,
    cacheNamespace: params.cacheNamespace,
    promptCacheEnabled: params.promptCacheEnabled,
    systemProjectionFingerprint: params.systemProjectionFingerprint,
    toolProjectionFingerprint: params.toolProjectionFingerprint,
    systemRevision: params.systemRevision,
    toolRevision: params.toolRevision,
    messages: params.messages,
    sourceMessageFingerprints: fingerprintMessages(params.sourceMessages),
  });
}

function ineligible(
  reason: CompactionReplayIneligibilityReason,
  replaySourceCount: number,
  requestSourceCount: number
): CompactionReplayEligibility {
  return {
    eligible: false,
    reason,
    replaySourceCount,
    requestSourceCount,
  };
}

export function inspectCompactionReplayEligibility(
  state: CompactionReplayState | undefined,
  candidate: CompactionReplayCandidate
): CompactionReplayEligibility {
  const candidateLineage = inspectSourceLineage(candidate.messages);
  const replaySourceCount = candidateLineage?.sourceMessageIds.length ?? 0;
  if (candidate.summarizerFallbackServed === true) {
    return ineligible(
      'summarizer_fallback_served_request',
      replaySourceCount,
      0
    );
  }
  if (state == null) {
    return ineligible('no_request_snapshot', replaySourceCount, 0);
  }
  if (state === 'fallback') {
    return ineligible('fallback_served_request', replaySourceCount, 0);
  }
  const recipe = state;
  const requestLineage = inspectSourceLineage(recipe.messages);
  const requestSourceCount = requestLineage?.sourceMessageIds.length ?? 0;
  if (recipe.provider !== candidate.provider) {
    return ineligible(
      'provider_mismatch',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (
    candidate.modelId != null &&
    recipe.modelId !== candidate.modelId
  ) {
    return ineligible('model_mismatch', replaySourceCount, requestSourceCount);
  }
  if (!recipe.cacheNamespace.complete || !candidate.cacheNamespace.complete) {
    return ineligible(
      'cache_namespace_unknown',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (!cacheNamespacesEqual(recipe.cacheNamespace, candidate.cacheNamespace)) {
    return ineligible(
      'cache_namespace_mismatch',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (!recipe.promptCacheEnabled || !candidate.promptCacheEnabled) {
    return ineligible(
      'prompt_cache_disabled',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (
    recipe.systemProjectionFingerprint == null ||
    candidate.systemProjectionFingerprint == null
  ) {
    return ineligible(
      'system_projection_unknown',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (
    recipe.systemProjectionFingerprint !==
    candidate.systemProjectionFingerprint
  ) {
    return ineligible(
      'system_projection_changed',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (
    recipe.toolProjectionFingerprint == null ||
    candidate.toolProjectionFingerprint == null
  ) {
    return ineligible(
      'tool_projection_unknown',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (
    recipe.toolProjectionFingerprint !== candidate.toolProjectionFingerprint
  ) {
    return ineligible(
      'tool_projection_changed',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (recipe.toolRevision !== candidate.toolRevision) {
    return ineligible(
      'tool_projection_changed',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (recipe.systemRevision !== candidate.systemRevision) {
    return ineligible(
      'system_projection_changed',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (
    candidate.projectionMode != null &&
    recipe.projectionMode !== candidate.projectionMode
  ) {
    return ineligible(
      'projection_mode_mismatch',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (candidate.restoredToolSubstitution) {
    return ineligible(
      'restored_tool_substitution',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (candidateLineage == null || replaySourceCount === 0) {
    return ineligible(
      'ambiguous_lineage',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (requestLineage == null || requestSourceCount === 0) {
    return ineligible(
      'ambiguous_lineage',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (replaySourceCount > requestSourceCount) {
    return ineligible(
      'source_not_prefix',
      replaySourceCount,
      requestSourceCount
    );
  }
  for (let i = 0; i < replaySourceCount; i++) {
    if (
      candidateLineage.sourceMessageIds[i] !==
      requestLineage.sourceMessageIds[i]
    ) {
      return ineligible(
        'source_not_prefix',
        replaySourceCount,
        requestSourceCount
      );
    }
  }
  if (recipe.sourceMessageFingerprints == null) {
    return ineligible(
      'source_content_unknown',
      replaySourceCount,
      requestSourceCount
    );
  }
  if (candidate.messages.length > recipe.sourceMessageFingerprints.length) {
    return ineligible(
      'source_content_mismatch',
      replaySourceCount,
      requestSourceCount
    );
  }
  for (let i = 0; i < candidate.messages.length; i++) {
    const fingerprint = fingerprintMessage(candidate.messages[i]);
    if (fingerprint == null) {
      return ineligible(
        'source_content_unknown',
        replaySourceCount,
        requestSourceCount
      );
    }
    if (fingerprint !== recipe.sourceMessageFingerprints[i]) {
      return ineligible(
        'source_content_mismatch',
        replaySourceCount,
        requestSourceCount
      );
    }
  }

  let replayMessageCount = 0;
  let reachedCandidateEnd = false;
  for (let i = 0; i < requestLineage.messageSourceCounts.length; i++) {
    const sourceCount = requestLineage.messageSourceCounts[i];
    if (sourceCount > replaySourceCount) {
      if (!reachedCandidateEnd) {
        return ineligible(
          'ambiguous_lineage',
          replaySourceCount,
          requestSourceCount
        );
      }
      break;
    }
    replayMessageCount = i + 1;
    if (sourceCount < replaySourceCount) {
      continue;
    }
    reachedCandidateEnd = true;
  }
  if (!reachedCandidateEnd) {
    return ineligible(
      'ambiguous_lineage',
      replaySourceCount,
      requestSourceCount
    );
  }

  return {
    eligible: true,
    replayMessageCount,
    replaySourceCount,
    requestSourceCount,
  };
}
