import { getTraceIdSeed } from '@/langfuseRuntimeContext';
import {
  resolveLangfuseRuntimeScope,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';

/**
 * `generateActivityLabel` passes one label seed to both the Langfuse handler
 * and the runtime scope it invokes under. This proves the mechanism: a label
 * scope's seed must override an inherited (parent run) seed for the duration
 * of the label call, then restore — otherwise per-batch label generations
 * collapse into the main run trace under deterministic tracing.
 *
 * Asserted through the ALS runtime-context channel (`getTraceIdSeed`), which
 * the trace id generator consults alongside OTel context; tests run without
 * a registered OTel context manager, so the OTel channel is a no-op here.
 */
describe('activity-label trace seed scoping', () => {
  it('overrides an inherited run seed for the nested scope only', () => {
    const runScope = resolveLangfuseRuntimeScope({ traceIdSeed: 'run-seed' });
    withLangfuseRuntimeScope(runScope, () => {
      expect(getTraceIdSeed()).toBe('run-seed');

      const labelScope = resolveLangfuseRuntimeScope({
        traceIdSeed: 'run-1-activity-3',
      });
      withLangfuseRuntimeScope(labelScope, () => {
        expect(getTraceIdSeed()).toBe('run-1-activity-3');
      });

      expect(getTraceIdSeed()).toBe('run-seed');
    });
  });

  it('keeps distinct seeds for successive label scopes', () => {
    const seeds: Array<string | undefined> = [];
    for (const seed of ['run-1-activity-0', 'run-1-activity-1']) {
      withLangfuseRuntimeScope(
        resolveLangfuseRuntimeScope({ traceIdSeed: seed }),
        () => {
          seeds.push(getTraceIdSeed());
        }
      );
    }
    expect(seeds).toEqual(['run-1-activity-0', 'run-1-activity-1']);
  });
});
