import { isProxy } from 'node:util/types';
import type { BaseMessage } from '@langchain/core/messages';

export const PROVIDER_MESSAGE_PROVENANCE_VERSION = 1 as const;

/** Recommended consumer trust bounds. Producers preserve complete lineage
 * above these limits; security-sensitive consumers must reject oversized
 * envelopes and take their fail-closed path rather than truncate attribution. */
export const PROVIDER_MESSAGE_PROVENANCE_LIMITS = Object.freeze({
  maxParts: 256,
  maxIndicesPerPart: 256,
  maxTotalIndexRefs: 4_096,
  maxSourceMessageIds: 256,
  maxSourceMessageIdLength: 512,
  maxSourceContentPartIndex: 4_095,
} as const);

/** Authorship of one logical contribution to a provider-bound message. */
export type ProviderMessageAttribution =
  | 'user'
  | 'model'
  | 'tool'
  | 'synthetic';

/**
 * Lineage for one logical contribution to a provider-bound message.
 * `sourceContentPartIndices` index the persisted source message's `content`
 * array before formatting, filtering, or summary-boundary slicing.
 */
export interface ProviderMessageProvenancePart {
  readonly attribution: ProviderMessageAttribution;
  readonly sourceMessageId?: string;
  readonly sourceContentPartIndices?: readonly number[];
}

/** Stable, versioned provenance carried in `BaseMessage.additional_kwargs`. */
export interface ProviderMessageProvenance {
  readonly version: typeof PROVIDER_MESSAGE_PROVENANCE_VERSION;
  readonly parts: readonly ProviderMessageProvenancePart[];
}

/** Inert marker used only when a derived message must retain invalidity. */
export interface InvalidProviderMessageProvenance {
  readonly version: typeof PROVIDER_MESSAGE_PROVENANCE_VERSION;
  readonly parts: null;
}

/** Distinguishes absent metadata from an explicitly malformed envelope. */
export type ProviderMessageProvenanceState =
  | { readonly status: 'absent' }
  | { readonly status: 'invalid' }
  | {
    readonly status: 'valid';
    readonly provenance: ProviderMessageProvenance;
  };

/** Distinguishes absent lineage from validated ids and malformed metadata. */
export type ProviderSourceMessageIdsState =
  | { readonly status: 'absent' }
  | { readonly status: 'invalid' }
  | {
    readonly status: 'valid';
    readonly sourceMessageIds: readonly string[];
  };

/** Typed subset of `additional_kwargs` exposed at provider callbacks. */
export interface ProviderMessageProvenanceAdditionalKwargs {
  readonly provenance?:
    | ProviderMessageProvenance
    | InvalidProviderMessageProvenance;
  readonly sourceMessageId?: string;
  readonly sourceMessageIds?: readonly string[];
}

interface UntrustedProviderMessageAdditionalKwargs
  extends Record<string, unknown> {
  provenance?: unknown;
  sourceMessageId?: unknown;
  sourceMessageIds?: unknown;
}

const PROVIDER_MESSAGE_ATTRIBUTIONS: ReadonlySet<ProviderMessageAttribution> =
  new Set(['user', 'model', 'tool', 'synthetic']);
/** Only envelopes built from copied/frozen inputs by the setter enter this
 * identity set. It avoids repeated O(n) canonicalization without trusting a
 * message or caller-owned envelope identity. */
const immutableProviderMessageProvenance = new WeakSet<object>();
/** Plural source-id arrays minted by the setter are copied and frozen before
 * entering this set. Their complete lineage may intentionally exceed the
 * public trust bounds without forcing repeated validation on reads. */
const immutableProviderSourceMessageIds = new WeakSet<object>();
/** Binds the two immutable objects produced by one setter call. Independently
 * valid envelopes from different messages must not be combined into a new
 * oversized lineage that no setter ever published. */
const immutableProviderMessageSourceIds = new WeakMap<
  object,
  readonly string[]
>();
const absentProviderMessageProvenanceState = Object.freeze({
  status: 'absent' as const,
});
const invalidProviderMessageProvenanceState = Object.freeze({
  status: 'invalid' as const,
});
const absentProviderSourceMessageIdsState = Object.freeze({
  status: 'absent' as const,
});
const invalidProviderSourceMessageIdsState = Object.freeze({
  status: 'invalid' as const,
});
/** Fresh projections use this inert envelope to preserve explicit invalidity
 * without retaining any hostile caller-owned object or array. */
const invalidProviderMessageProvenanceSentinel: InvalidProviderMessageProvenance =
  Object.freeze({
    version: PROVIDER_MESSAGE_PROVENANCE_VERSION,
    parts: null,
  });

function normalizeSourceMessageId(
  candidate: unknown,
  enforceTrustBounds = false
): string | undefined {
  if (typeof candidate !== 'string') {
    return undefined;
  }
  if (
    enforceTrustBounds &&
    candidate.length >
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIdLength
  ) {
    return undefined;
  }
  const normalized = candidate.trim();
  return normalized.length > 0 ? normalized : undefined;
}

function appendSourceMessageId(
  result: string[],
  seen: Set<string>,
  candidate: unknown,
  enforceTrustBounds = false
): boolean {
  const sourceMessageId = normalizeSourceMessageId(
    candidate,
    enforceTrustBounds
  );
  if (sourceMessageId == null) {
    return candidate === undefined;
  }
  if (seen.has(sourceMessageId)) {
    return true;
  }
  seen.add(sourceMessageId);
  result.push(sourceMessageId);
  return true;
}

function normalizeSourceContentPartIndices(
  indices: readonly number[] | undefined,
  trustState?: { totalIndexRefs: number }
): number[] | undefined {
  if (indices == null) {
    return undefined;
  }
  if (isProxy(indices) || !Array.isArray(indices)) {
    throw new TypeError(
      'Provider source content part indices must be a non-empty array'
    );
  }
  const length = indices.length;
  if (!Number.isSafeInteger(length) || length === 0) {
    throw new TypeError(
      'Provider source content part indices must be a non-empty array'
    );
  }
  if (
    trustState != null &&
    (length > PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxIndicesPerPart ||
      trustState.totalIndexRefs + length >
        PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxTotalIndexRefs)
  ) {
    throw new TypeError('Provider source content part indices exceed limits');
  }
  if (trustState != null) {
    trustState.totalIndexRefs += length;
  }
  const normalized: number[] = [];
  const seen = new Set<number>();
  for (let position = 0; position < length; position++) {
    const index = indices[position];
    if (
      !Number.isSafeInteger(index) ||
      index < 0 ||
      (trustState != null &&
        index > PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceContentPartIndex)
    ) {
      throw new TypeError('Invalid provider source content part index');
    }
    if (seen.has(index)) {
      continue;
    }
    seen.add(index);
    normalized.push(index);
  }
  return normalized.length > 0 ? normalized : undefined;
}

function normalizeProvenancePart(
  part: ProviderMessageProvenancePart | null | undefined,
  requireCanonicalSourceMessageId = false,
  trustState?: { totalIndexRefs: number; sourceMessageIdRefs: number }
): ProviderMessageProvenancePart {
  if (part == null || typeof part !== 'object') {
    throw new TypeError('Invalid provider message provenance attribution');
  }
  /** Capture each potentially accessor-backed public input exactly once before
   * validation so a changing getter cannot pass one value and publish another. */
  const attribution = part.attribution;
  const sourceMessageIdInput = part.sourceMessageId;
  const sourceContentPartIndicesInput = part.sourceContentPartIndices;
  if (!PROVIDER_MESSAGE_ATTRIBUTIONS.has(attribution)) {
    throw new TypeError('Invalid provider message provenance attribution');
  }
  const sourceMessageId = normalizeSourceMessageId(
    sourceMessageIdInput,
    trustState != null
  );
  if (
    sourceMessageIdInput !== undefined &&
    (sourceMessageId == null ||
      (requireCanonicalSourceMessageId &&
        sourceMessageId !== sourceMessageIdInput))
  ) {
    throw new TypeError('Invalid provider source message id');
  }
  if (sourceMessageId != null && trustState != null) {
    trustState.sourceMessageIdRefs++;
    if (
      trustState.sourceMessageIdRefs >
      PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds
    ) {
      throw new TypeError('Provider source message ids exceed limits');
    }
  }
  const sourceContentPartIndices = normalizeSourceContentPartIndices(
    sourceContentPartIndicesInput,
    trustState
  );
  return Object.freeze({
    attribution,
    ...(sourceMessageId != null && { sourceMessageId }),
    ...(sourceContentPartIndices != null && {
      sourceContentPartIndices: Object.freeze(sourceContentPartIndices),
    }),
  });
}

function normalizeProvenanceParts(
  parts: unknown,
  requireCanonicalSourceMessageId = false,
  enforceTrustBounds = false
): readonly ProviderMessageProvenancePart[] {
  if (isProxy(parts) || !Array.isArray(parts)) {
    throw new TypeError('Provider message provenance parts must be an array');
  }
  const length = parts.length;
  if (!Number.isSafeInteger(length) || length === 0) {
    throw new TypeError('Provider message provenance parts cannot be empty');
  }
  if (
    enforceTrustBounds &&
    length > PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts
  ) {
    throw new TypeError('Provider message provenance parts exceed limits');
  }
  const trustState = enforceTrustBounds
    ? { totalIndexRefs: 0, sourceMessageIdRefs: 0 }
    : undefined;
  const normalizedParts: ProviderMessageProvenancePart[] = [];
  for (let index = 0; index < length; index++) {
    const part = parts[index] as
      | ProviderMessageProvenancePart
      | null
      | undefined;
    normalizedParts.push(
      normalizeProvenancePart(part, requireCanonicalSourceMessageId, trustState)
    );
  }
  return Object.freeze(normalizedParts);
}

function normalizeProviderMessageProvenance(
  provenanceInput: unknown
): ProviderMessageProvenance | undefined {
  try {
    if (provenanceInput == null || typeof provenanceInput !== 'object') {
      return undefined;
    }
    if (immutableProviderMessageProvenance.has(provenanceInput)) {
      return provenanceInput as ProviderMessageProvenance;
    }
    const provenance = provenanceInput as {
      version?: unknown;
      parts?: unknown;
    };
    /** Capture accessor-backed envelope fields exactly once, then publish only
     * a canonical immutable copy rather than returning the untrusted object. */
    const version = provenance.version;
    const partsInput = provenance.parts;
    if (version !== PROVIDER_MESSAGE_PROVENANCE_VERSION) {
      return undefined;
    }
    const parts = normalizeProvenanceParts(partsInput, true, true);
    return Object.freeze({
      version: PROVIDER_MESSAGE_PROVENANCE_VERSION,
      parts,
    });
  } catch {
    return undefined;
  }
}

function getUntrustedAdditionalKwargs(
  message: BaseMessage
): UntrustedProviderMessageAdditionalKwargs | undefined {
  try {
    const additionalKwargs: unknown = message.additional_kwargs;
    return additionalKwargs != null && typeof additionalKwargs === 'object'
      ? (additionalKwargs as UntrustedProviderMessageAdditionalKwargs)
      : undefined;
  } catch {
    return undefined;
  }
}

function readAdditionalKwarg(
  additionalKwargs: UntrustedProviderMessageAdditionalKwargs | undefined,
  key: keyof UntrustedProviderMessageAdditionalKwargs
): unknown {
  try {
    return additionalKwargs?.[key];
  } catch {
    return undefined;
  }
}

function collectProviderSourceMessageIds(
  provenance: ProviderMessageProvenance | undefined,
  pluralInput: unknown,
  singularInput: unknown
): string[] | undefined {
  const result: string[] = [];
  const seen = new Set<string>();
  const trustedProvenance =
    provenance != null && immutableProviderMessageProvenance.has(provenance);
  const boundPlural = trustedProvenance
    ? immutableProviderMessageSourceIds.get(provenance!)
    : undefined;
  let hasUntrustedSourceMetadata = provenance != null && !trustedProvenance;
  let trustedPlural = false;
  const provenanceParts = provenance?.parts;

  /** Resolve trust across the setter-minted envelope/id pair before walking
   * either collection. A trusted object spliced from another setter call is
   * no longer a trusted combined lineage, so the public bounds apply to both
   * sides and oversized inputs fail before any per-item work. */
  let plural: readonly unknown[] | undefined;
  let pluralLength = 0;
  try {
    if (pluralInput !== undefined) {
      if (isProxy(pluralInput) || !Array.isArray(pluralInput)) {
        return undefined;
      }
      plural = pluralInput;
      pluralLength = pluralInput.length;
      if (!Number.isSafeInteger(pluralLength) || pluralLength === 0) {
        return undefined;
      }
      trustedPlural = immutableProviderSourceMessageIds.has(pluralInput);
      const bindingMismatch = trustedProvenance && boundPlural !== pluralInput;
      if (bindingMismatch) {
        trustedPlural = false;
      }
      if (!trustedPlural) {
        hasUntrustedSourceMetadata = true;
      }
    }
  } catch {
    return undefined;
  }

  /** A primitive singular id has no identity of its own. Treat it as the
   * setter's compatibility duplicate only when it exactly matches the last id
   * of an immutable plural array published by that setter. This classification
   * happens before either collection is walked so malformed singular metadata
   * cannot force work over an otherwise trusted oversized lineage. */
  const trustedSingularSourceIds =
    boundPlural ?? (trustedPlural ? plural : undefined);
  let trustedSingularSourceIdsLength = 0;
  if (trustedSingularSourceIds != null) {
    trustedSingularSourceIdsLength =
      trustedSingularSourceIds === plural
        ? pluralLength
        : trustedSingularSourceIds.length;
  }
  const unboundedSingular =
    singularInput === undefined
      ? undefined
      : normalizeSourceMessageId(singularInput);
  const trustedSingularDuplicate =
    singularInput !== undefined &&
    typeof singularInput === 'string' &&
    singularInput === unboundedSingular &&
    trustedSingularSourceIds != null &&
    trustedSingularSourceIdsLength > 0 &&
    trustedSingularSourceIds[trustedSingularSourceIdsLength - 1] ===
      singularInput;
  if (singularInput !== undefined && !trustedSingularDuplicate) {
    hasUntrustedSourceMetadata = true;
    if (normalizeSourceMessageId(singularInput, true) == null) {
      return undefined;
    }
  }

  /** Any untrusted lineage field revokes the setter-only size exemption for
   * every collection participating in the union. Preflight all public bounds
   * before reading a part or plural element. */
  if (hasUntrustedSourceMetadata) {
    try {
      if (
        provenanceParts != null &&
        provenanceParts.length > PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxParts
      ) {
        return undefined;
      }
      if (
        plural != null &&
        pluralLength > PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds
      ) {
        return undefined;
      }
    } catch {
      return undefined;
    }
  }
  const provenanceRequiresValidation =
    !trustedProvenance || hasUntrustedSourceMetadata;
  const pluralRequiresValidation = !trustedPlural || hasUntrustedSourceMetadata;

  if (provenanceParts != null) {
    const length = provenanceParts.length;
    for (let index = 0; index < length; index++) {
      if (
        !appendSourceMessageId(
          result,
          seen,
          provenanceParts[index].sourceMessageId,
          provenanceRequiresValidation
        )
      ) {
        return undefined;
      }
    }
  }

  try {
    if (plural != null) {
      for (let index = 0; index < pluralLength; index++) {
        if (
          !appendSourceMessageId(
            result,
            seen,
            plural[index],
            pluralRequiresValidation
          )
        ) {
          return undefined;
        }
      }
    }
  } catch {
    return undefined;
  }
  if (singularInput !== undefined) {
    if (trustedSingularDuplicate) {
      if (!seen.has(singularInput)) {
        return undefined;
      }
    } else {
      if (!appendSourceMessageId(result, seen, singularInput, true)) {
        return undefined;
      }
    }
  }
  if (
    hasUntrustedSourceMetadata &&
    result.length > PROVIDER_MESSAGE_PROVENANCE_LIMITS.maxSourceMessageIds
  ) {
    return undefined;
  }
  return result;
}

function copyOwnEnumerableDataProperties(
  input: object
): Record<PropertyKey, unknown> {
  if (isProxy(input)) {
    throw new TypeError('Invalid provider message serialization kwargs');
  }
  const result: Record<PropertyKey, unknown> = {};
  const descriptors = Object.getOwnPropertyDescriptors(input);
  for (const key of Reflect.ownKeys(descriptors)) {
    const descriptor = Reflect.get(descriptors, key) as PropertyDescriptor;
    if (descriptor.enumerable !== true) {
      continue;
    }
    if (!('value' in descriptor)) {
      throw new TypeError('Invalid provider message serialization kwargs');
    }
    Object.defineProperty(result, key, {
      configurable: true,
      enumerable: true,
      value: descriptor.value,
      writable: true,
    });
  }
  return result;
}

/** Legacy Human rows are treated as user-authored when no typed metadata is
 * available. Unknown custom message types take the same conservative path. */
function inferLegacyMessageAttribution(
  message: BaseMessage
): ProviderMessageAttribution {
  const messageType = message.type;
  if (messageType === 'ai') {
    return 'model';
  }
  if (messageType === 'tool') {
    return 'tool';
  }
  if (messageType === 'system') {
    return 'synthetic';
  }
  return 'user';
}

/** Returns explicitly stamped provenance without inferring from message role. */
export function getProviderMessageProvenance(
  message: BaseMessage
): ProviderMessageProvenance | undefined {
  const additionalKwargs = getUntrustedAdditionalKwargs(message);
  const provenanceInput = readAdditionalKwarg(additionalKwargs, 'provenance');
  return normalizeProviderMessageProvenance(provenanceInput);
}

/** Safely preserves the semantic difference between absent and invalid input. */
export function inspectProviderMessageProvenance(
  message: BaseMessage
): ProviderMessageProvenanceState {
  let additionalKwargs: unknown;
  try {
    additionalKwargs = message.additional_kwargs;
  } catch {
    return invalidProviderMessageProvenanceState;
  }
  if (additionalKwargs == null || typeof additionalKwargs !== 'object') {
    return absentProviderMessageProvenanceState;
  }
  let provenanceInput: unknown;
  try {
    provenanceInput = (additionalKwargs as { provenance?: unknown })
      .provenance;
  } catch {
    return invalidProviderMessageProvenanceState;
  }
  if (provenanceInput == null) {
    return absentProviderMessageProvenanceState;
  }
  const provenance = normalizeProviderMessageProvenance(provenanceInput);
  return provenance == null
    ? invalidProviderMessageProvenanceState
    : { status: 'valid', provenance };
}

function readProviderSourceMessageIds(
  message: BaseMessage
): string[] | null | undefined {
  let additionalKwargs: unknown;
  try {
    additionalKwargs = message.additional_kwargs;
  } catch {
    return null;
  }
  if (additionalKwargs == null || typeof additionalKwargs !== 'object') {
    return undefined;
  }
  if (isProxy(additionalKwargs)) {
    return null;
  }
  let provenanceInput: unknown;
  let pluralInput: unknown;
  let singularInput: unknown;
  try {
    const sourceMetadata =
      additionalKwargs as UntrustedProviderMessageAdditionalKwargs;
    provenanceInput = sourceMetadata.provenance;
    pluralInput = sourceMetadata.sourceMessageIds;
    singularInput = sourceMetadata.sourceMessageId;
  } catch {
    return null;
  }
  const hasProvenanceInput = provenanceInput != null;
  const hasLegacyInput =
    pluralInput !== undefined || singularInput !== undefined;
  if (!hasProvenanceInput && !hasLegacyInput) {
    return undefined;
  }
  const provenance = normalizeProviderMessageProvenance(provenanceInput);
  if (hasProvenanceInput && provenance == null) {
    return null;
  }
  return (
    collectProviderSourceMessageIds(
      provenance,
      pluralInput,
      singularInput
    ) ?? null
  );
}

/** Safely distinguishes missing source lineage from malformed metadata. */
export function inspectProviderSourceMessageIds(
  message: BaseMessage
): ProviderSourceMessageIdsState {
  const sourceMessageIds = readProviderSourceMessageIds(message);
  if (sourceMessageIds === undefined) {
    return absentProviderSourceMessageIdsState;
  }
  if (sourceMessageIds === null) {
    return invalidProviderSourceMessageIdsState;
  }
  return { status: 'valid', sourceMessageIds };
}

/** True when indexed contributions uniquely cover every current content part. */
export function hasBijectiveProviderContentPartMapping(
  parts: readonly ProviderMessageProvenancePart[],
  contentPartCount: number
): boolean {
  if (!Number.isSafeInteger(contentPartCount) || contentPartCount <= 0) {
    return false;
  }
  const seen = new Set<number>();
  for (const part of parts) {
    const indices = part.sourceContentPartIndices;
    if (indices == null) {
      return false;
    }
    for (const index of indices) {
      if (
        !Number.isSafeInteger(index) ||
        index < 0 ||
        index >= contentPartCount ||
        seen.has(index)
      ) {
        return false;
      }
      seen.add(index);
    }
  }
  return seen.size === contentPartCount;
}

/**
 * Returns every explicit persisted source id in stable content order.
 * Typed parts, plural lineage, and the legacy singular id are unioned in that
 * precedence order; duplicates are removed without reordering.
 */
export function getProviderSourceMessageIds(message: BaseMessage): string[] {
  const additionalKwargs = getUntrustedAdditionalKwargs(message);
  const provenanceInput = readAdditionalKwarg(additionalKwargs, 'provenance');
  const pluralInput = readAdditionalKwarg(additionalKwargs, 'sourceMessageIds');
  const singularInput = readAdditionalKwarg(
    additionalKwargs,
    'sourceMessageId'
  );
  return (
    collectProviderSourceMessageIds(
      normalizeProviderMessageProvenance(provenanceInput),
      pluralInput,
      singularInput
    ) ?? []
  );
}

function publishProviderMessageProvenance(
  message: BaseMessage,
  provenance: unknown,
  sourceMessageIds?: readonly string[]
): void {
  if (isProxy(message)) {
    throw new TypeError('Invalid provider message serialization kwargs');
  }
  const liveDescriptor = Object.getOwnPropertyDescriptor(
    message,
    'additional_kwargs'
  );
  const lcKwargsDescriptor = Object.getOwnPropertyDescriptor(
    message,
    'lc_kwargs'
  );
  if (
    liveDescriptor == null ||
    !('value' in liveDescriptor) ||
    liveDescriptor.writable !== true ||
    lcKwargsDescriptor == null ||
    !('value' in lcKwargsDescriptor) ||
    lcKwargsDescriptor.writable !== true
  ) {
    throw new TypeError('Invalid provider message serialization kwargs');
  }
  const currentAdditionalKwargsInput: unknown = liveDescriptor.value;
  if (
    currentAdditionalKwargsInput == null ||
    typeof currentAdditionalKwargsInput !== 'object'
  ) {
    throw new TypeError('Invalid provider message additional kwargs');
  }
  let replacement: UntrustedProviderMessageAdditionalKwargs;
  try {
    replacement = copyOwnEnumerableDataProperties(
      currentAdditionalKwargsInput
    ) as UntrustedProviderMessageAdditionalKwargs;
  } catch {
    throw new TypeError('Invalid provider message additional kwargs');
  }
  const lcKwargsInput: unknown = lcKwargsDescriptor.value;
  if (lcKwargsInput == null || typeof lcKwargsInput !== 'object') {
    throw new TypeError('Invalid provider message serialization kwargs');
  }
  let serializedReplacement: Record<PropertyKey, unknown>;
  try {
    serializedReplacement = copyOwnEnumerableDataProperties(lcKwargsInput);
  } catch {
    throw new TypeError('Invalid provider message serialization kwargs');
  }
  replacement.provenance = provenance;
  if (sourceMessageIds != null && sourceMessageIds.length > 0) {
    replacement.sourceMessageIds = sourceMessageIds;
    replacement.sourceMessageId = sourceMessageIds[sourceMessageIds.length - 1];
  } else {
    delete replacement.sourceMessageIds;
    delete replacement.sourceMessageId;
  }
  serializedReplacement.additional_kwargs = replacement;
  try {
    /** Both own data properties are prevalidated before one descriptor batch,
     * so custom accessors cannot observe or create a split publication. */
    Object.defineProperties(message, {
      additional_kwargs: { ...liveDescriptor, value: replacement },
      lc_kwargs: { ...lcKwargsDescriptor, value: serializedReplacement },
    });
  } catch {
    throw new TypeError('Invalid provider message serialization kwargs');
  }
}

interface ResolvedProvenancePublication {
  provenance: ProviderMessageProvenance;
  sourceMessageIds?: readonly string[];
}

function resolveProvenancePublication(
  parts: readonly ProviderMessageProvenancePart[]
): ResolvedProvenancePublication {
  const normalizedParts = normalizeProvenanceParts(parts);
  const provenance: ProviderMessageProvenance = Object.freeze({
    version: PROVIDER_MESSAGE_PROVENANCE_VERSION,
    parts: normalizedParts,
  });
  immutableProviderMessageProvenance.add(provenance);
  const sourceMessageIds: string[] = [];
  const seen = new Set<string>();
  for (const part of normalizedParts) {
    appendSourceMessageId(sourceMessageIds, seen, part.sourceMessageId);
  }
  if (sourceMessageIds.length === 0) {
    return { provenance };
  }
  const immutableSourceMessageIds = Object.freeze(sourceMessageIds);
  immutableProviderSourceMessageIds.add(immutableSourceMessageIds);
  immutableProviderMessageSourceIds.set(provenance, immutableSourceMessageIds);
  return { provenance, sourceMessageIds: immutableSourceMessageIds };
}

/** Replaces typed provenance and synchronizes its stable plural source ids. */
export function setProviderMessageProvenance(
  message: BaseMessage,
  parts: readonly ProviderMessageProvenancePart[]
): void {
  const resolved = resolveProvenancePublication(parts);
  publishProviderMessageProvenance(
    message,
    resolved.provenance,
    resolved.sourceMessageIds
  );
}

/**
 * {@link setProviderMessageProvenance} for a message the caller itself just
 * constructed from locally built plain objects. Such a message cannot carry
 * proxies, accessors, or foreign aliases in `additional_kwargs`/`lc_kwargs`,
 * so the publication skips the hardened descriptor walks while producing the
 * same end state: fresh replacement objects on both slots, with the
 * serialization mirror aliasing the live kwargs. Never call this with a
 * message received across a seam — that is what the hardened variant is for.
 */
export function setFreshProviderMessageProvenance(
  message: BaseMessage,
  parts: readonly ProviderMessageProvenancePart[]
): void {
  const resolved = resolveProvenancePublication(parts);
  const replacement: UntrustedProviderMessageAdditionalKwargs = {
    ...(message.additional_kwargs as Record<string, unknown>),
  };
  replacement.provenance = resolved.provenance;
  const sourceMessageIds = resolved.sourceMessageIds;
  if (sourceMessageIds != null && sourceMessageIds.length > 0) {
    replacement.sourceMessageIds = sourceMessageIds;
    replacement.sourceMessageId = sourceMessageIds[sourceMessageIds.length - 1];
  } else {
    delete replacement.sourceMessageIds;
    delete replacement.sourceMessageId;
  }
  const serializedReplacement: Record<PropertyKey, unknown> = {
    ...(message.lc_kwargs as Record<string, unknown>),
  };
  serializedReplacement.additional_kwargs = replacement;
  message.additional_kwargs = replacement as BaseMessage['additional_kwargs'];
  message.lc_kwargs = serializedReplacement;
}

/** Publishes the canonical fail-closed marker for malformed provenance. */
export function setInvalidProviderMessageProvenance(
  message: BaseMessage
): void {
  publishProviderMessageProvenance(
    message,
    invalidProviderMessageProvenanceSentinel
  );
}

/** Marks a provider-visible runtime message as host-generated context. */
export function stampSyntheticProviderMessage<T extends BaseMessage>(
  message: T
): T {
  setProviderMessageProvenance(message, [{ attribution: 'synthetic' }]);
  return message;
}

/**
 * Adds one logical lineage contribution, migrating legacy ids on first use.
 * This compatibility helper rebuilds validated public metadata; hot formatters
 * should accumulate locally and call `setProviderMessageProvenance` once.
 */
export function appendProviderMessageProvenance(
  message: BaseMessage,
  part: ProviderMessageProvenancePart
): void {
  let normalizedPart = normalizeProvenancePart(part);
  const additionalKwargs = getUntrustedAdditionalKwargs(message);
  const provenanceInput = readAdditionalKwarg(additionalKwargs, 'provenance');
  const pluralInput = readAdditionalKwarg(additionalKwargs, 'sourceMessageIds');
  const singularInput = readAdditionalKwarg(
    additionalKwargs,
    'sourceMessageId'
  );
  const provenance = normalizeProviderMessageProvenance(provenanceInput);
  const existing = provenance?.parts ?? [];
  const representedSourceIds = new Set<string>();
  for (const existingPart of existing) {
    if (existingPart.sourceMessageId != null) {
      representedSourceIds.add(existingPart.sourceMessageId);
    }
  }
  const sourceMessageIds =
    collectProviderSourceMessageIds(provenance, pluralInput, singularInput) ??
    [];
  const missingLegacySourceIds = sourceMessageIds.filter(
    (sourceMessageId) => !representedSourceIds.has(sourceMessageId)
  );
  const migratedParts: ProviderMessageProvenancePart[] = [...existing];
  const legacyAttribution = inferLegacyMessageAttribution(message);
  for (const sourceMessageId of missingLegacySourceIds) {
    migratedParts.push({
      attribution: legacyAttribution,
      sourceMessageId,
    });
  }
  if (normalizedPart.sourceMessageId == null && sourceMessageIds.length === 1) {
    normalizedPart = {
      ...normalizedPart,
      sourceMessageId: sourceMessageIds[0],
    };
    const migratedIndex = migratedParts.findIndex(
      (migratedPart, index) =>
        index >= existing.length &&
        migratedPart.attribution === normalizedPart.attribution &&
        migratedPart.sourceMessageId === sourceMessageIds[0]
    );
    if (migratedIndex >= 0) {
      migratedParts.splice(migratedIndex, 1);
    }
  }
  const lastPart = migratedParts[migratedParts.length - 1];
  if (
    migratedParts.length > 0 &&
    lastPart.attribution === normalizedPart.attribution &&
    lastPart.sourceMessageId === normalizedPart.sourceMessageId &&
    (lastPart.sourceContentPartIndices == null) ===
      (normalizedPart.sourceContentPartIndices == null)
  ) {
    const sourceContentPartIndices = normalizeSourceContentPartIndices([
      ...(lastPart.sourceContentPartIndices ?? []),
      ...(normalizedPart.sourceContentPartIndices ?? []),
    ]);
    migratedParts[migratedParts.length - 1] = {
      attribution: lastPart.attribution,
      ...(lastPart.sourceMessageId != null && {
        sourceMessageId: lastPart.sourceMessageId,
      }),
      ...(sourceContentPartIndices != null && { sourceContentPartIndices }),
    };
  } else {
    migratedParts.push(normalizedPart);
  }
  setProviderMessageProvenance(message, migratedParts);
}
