import type { BaseMessage } from '@langchain/core/messages';
import type { ProviderMessageProjectionMode } from '@/llm/prepareProviderRequest';
import type * as t from '@/types';
import { inspectProviderSourceMessageIds } from '@/messages/provenance';
import { Providers } from '@/common';

type CacheNamespaceValue = string;

export interface CompactionReplayRecipe {
  readonly provider: t.ProviderName;
  readonly modelId?: string;
  readonly projectionMode: ProviderMessageProjectionMode;
  readonly cacheNamespace: CompactionCacheNamespace;
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
  | 'system_projection_changed'
  | 'tool_projection_changed'
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
] as const;

const BUILT_IN_PROVIDER_NAMES = new Set<string>(Object.values(Providers));

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
  let complete = true;
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
