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

/** Typed subset of `additional_kwargs` exposed at provider callbacks. */
export interface ProviderMessageProvenanceAdditionalKwargs {
  readonly provenance?: ProviderMessageProvenance;
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

function normalizeSourceMessageId(candidate: unknown): string | undefined {
  if (typeof candidate !== 'string') {
    return undefined;
  }
  const normalized = candidate.trim();
  return normalized.length > 0 ? normalized : undefined;
}

function appendSourceMessageId(
  result: string[],
  seen: Set<string>,
  candidate: unknown
): void {
  const sourceMessageId = normalizeSourceMessageId(candidate);
  if (sourceMessageId == null || seen.has(sourceMessageId)) {
    return;
  }
  seen.add(sourceMessageId);
  result.push(sourceMessageId);
}

function normalizeSourceContentPartIndices(
  indices: readonly number[] | undefined
): number[] | undefined {
  if (indices == null) {
    return undefined;
  }
  if (!Array.isArray(indices)) {
    throw new TypeError(
      'Provider source content part indices must be a non-empty array'
    );
  }
  const length = indices.length;
  if (length === 0) {
    throw new TypeError(
      'Provider source content part indices must be a non-empty array'
    );
  }
  const normalized: number[] = [];
  const seen = new Set<number>();
  for (let position = 0; position < length; position++) {
    const index = indices[position];
    if (!Number.isSafeInteger(index) || index < 0) {
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
  requireCanonicalSourceMessageId = false
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
  const sourceMessageId = normalizeSourceMessageId(sourceMessageIdInput);
  if (
    sourceMessageIdInput !== undefined &&
    (sourceMessageId == null ||
      (requireCanonicalSourceMessageId &&
        sourceMessageId !== sourceMessageIdInput))
  ) {
    throw new TypeError('Invalid provider source message id');
  }
  const sourceContentPartIndices = normalizeSourceContentPartIndices(
    sourceContentPartIndicesInput
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
  requireCanonicalSourceMessageId = false
): readonly ProviderMessageProvenancePart[] {
  if (!Array.isArray(parts)) {
    throw new TypeError('Provider message provenance parts must be an array');
  }
  const length = parts.length;
  if (length === 0) {
    throw new TypeError('Provider message provenance parts cannot be empty');
  }
  const normalizedParts: ProviderMessageProvenancePart[] = [];
  for (let index = 0; index < length; index++) {
    const part = parts[index] as
      | ProviderMessageProvenancePart
      | null
      | undefined;
    normalizedParts.push(
      normalizeProvenancePart(part, requireCanonicalSourceMessageId)
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
    const parts = normalizeProvenanceParts(partsInput, true);
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
): string[] {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const part of provenance?.parts ?? []) {
    appendSourceMessageId(result, seen, part.sourceMessageId);
  }

  let pluralCandidates: unknown[] | undefined;
  try {
    if (Array.isArray(pluralInput)) {
      const length = pluralInput.length;
      pluralCandidates = [];
      for (let index = 0; index < length; index++) {
        pluralCandidates.push(pluralInput[index]);
      }
    }
  } catch {
    pluralCandidates = undefined;
  }
  for (const candidate of pluralCandidates ?? []) {
    appendSourceMessageId(result, seen, candidate);
  }
  appendSourceMessageId(result, seen, singularInput);
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
  const provenanceInput = readAdditionalKwarg(
    additionalKwargs,
    'provenance'
  );
  return normalizeProviderMessageProvenance(provenanceInput);
}

/**
 * Returns every explicit persisted source id in stable content order.
 * Typed parts, plural lineage, and the legacy singular id are unioned in that
 * precedence order; duplicates are removed without reordering.
 */
export function getProviderSourceMessageIds(message: BaseMessage): string[] {
  const additionalKwargs = getUntrustedAdditionalKwargs(message);
  const provenanceInput = readAdditionalKwarg(
    additionalKwargs,
    'provenance'
  );
  const pluralInput = readAdditionalKwarg(
    additionalKwargs,
    'sourceMessageIds'
  );
  const singularInput = readAdditionalKwarg(
    additionalKwargs,
    'sourceMessageId'
  );
  return collectProviderSourceMessageIds(
    normalizeProviderMessageProvenance(provenanceInput),
    pluralInput,
    singularInput
  );
}

/** Replaces typed provenance and synchronizes its stable plural source ids. */
export function setProviderMessageProvenance(
  message: BaseMessage,
  parts: readonly ProviderMessageProvenancePart[]
): void {
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
  const currentAdditionalKwargs = getUntrustedAdditionalKwargs(message);
  let replacement: UntrustedProviderMessageAdditionalKwargs;
  try {
    replacement = { ...(currentAdditionalKwargs ?? {}) };
  } catch {
    throw new TypeError('Invalid provider message additional kwargs');
  }
  replacement.provenance = provenance;
  if (sourceMessageIds.length > 0) {
    replacement.sourceMessageIds = Object.freeze(sourceMessageIds);
    replacement.sourceMessageId =
      sourceMessageIds[sourceMessageIds.length - 1];
  } else {
    delete replacement.sourceMessageIds;
    delete replacement.sourceMessageId;
  }
  message.additional_kwargs = replacement;
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
  const provenanceInput = readAdditionalKwarg(
    additionalKwargs,
    'provenance'
  );
  const pluralInput = readAdditionalKwarg(
    additionalKwargs,
    'sourceMessageIds'
  );
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
  const sourceMessageIds = collectProviderSourceMessageIds(
    provenance,
    pluralInput,
    singularInput
  );
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
