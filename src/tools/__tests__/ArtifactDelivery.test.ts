import { describe, expect, it } from '@jest/globals';
import {
  ARTIFACT_DELIVERY_WARNING_PREFIX,
  appendArtifactDeliveryWarning,
  normalizeArtifactDeliveryFailure,
} from '../ArtifactDelivery';

describe('artifact delivery failures', () => {
  it('normalizes the bounded Code API failure contract', () => {
    expect(
      normalizeArtifactDeliveryFailure({
        code: 'artifact_delivery_failed',
        status: 'partial',
        attempted: 3,
        delivered: 2,
        failed: 1,
      })
    ).toEqual({
      code: 'artifact_delivery_failed',
      status: 'partial',
      attempted: 3,
      delivered: 2,
      failed: 1,
    });
  });

  it.each([
    null,
    {
      code: 'storage_error',
      status: 'failed',
      attempted: 1,
      delivered: 0,
      failed: 1,
    },
    {
      code: 'artifact_delivery_failed',
      status: 'failed',
      attempted: 2,
      delivered: 1,
      failed: 1,
    },
    {
      code: 'artifact_delivery_failed',
      status: 'partial',
      attempted: 1,
      delivered: 0,
      failed: 1,
    },
    {
      code: 'artifact_delivery_failed',
      status: 'failed',
      attempted: 1,
      delivered: 0,
      failed: -1,
    },
  ])('rejects malformed external values', (value) => {
    expect(normalizeArtifactDeliveryFailure(value)).toBeUndefined();
  });

  it('warns without claiming the code execution failed or recommending a retry', () => {
    const output = appendArtifactDeliveryWarning('stdout:\ndone\n', {
      code: 'artifact_delivery_failed',
      status: 'failed',
      attempted: 1,
      delivered: 0,
      failed: 1,
    });

    expect(output).toContain(ARTIFACT_DELIVERY_WARNING_PREFIX);
    expect(output).toContain('1 of 1 generated files could not be persisted');
    expect(output).toContain('The code itself ran');
    expect(output).toContain('do not rerun automatically');
  });
});
