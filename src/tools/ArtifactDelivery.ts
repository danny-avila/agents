import type { ArtifactDeliveryFailure } from '@/types';

export const ARTIFACT_DELIVERY_WARNING_PREFIX = 'Artifact delivery warning:';

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

export function normalizeArtifactDeliveryFailure(
  value: unknown
): ArtifactDeliveryFailure | undefined {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }

  const candidate = value as Partial<ArtifactDeliveryFailure>;
  if (
    candidate.code !== 'artifact_delivery_failed' ||
    (candidate.status !== 'partial' && candidate.status !== 'failed') ||
    !isNonNegativeInteger(candidate.attempted) ||
    !isNonNegativeInteger(candidate.delivered) ||
    !isNonNegativeInteger(candidate.failed) ||
    candidate.failed === 0 ||
    candidate.attempted !== candidate.delivered + candidate.failed ||
    (candidate.status === 'failed' && candidate.delivered !== 0) ||
    (candidate.status === 'partial' && candidate.delivered === 0)
  ) {
    return undefined;
  }

  return candidate as ArtifactDeliveryFailure;
}

export function appendArtifactDeliveryWarning(
  output: string,
  delivery: ArtifactDeliveryFailure | undefined
): string {
  if (delivery == null) {
    return output;
  }

  const warning = `${ARTIFACT_DELIVERY_WARNING_PREFIX} ${delivery.failed} of ${delivery.attempted} generated files could not be persisted. Do not assume missing files are available to later calls or downloadable. The code itself ran; do not rerun automatically because it may have had side effects.`;
  return `${output.trimEnd()}\n${warning}\n`;
}
