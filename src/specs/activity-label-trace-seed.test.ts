import {
  resolveLangfuseRuntimeScope,
  withLangfuseRuntimeScope,
} from '@/langfuseRuntimeScope';
import { getTraceIdSeed } from '@/langfuseRuntimeContext';

/**
 * `generateActivityLabel` keeps a label-specific seed for its standalone
 * fallback when no source observation can be captured. An explicit source
 * parent wins during normal traced runs; this proves fallback seed scoping
 * does not leak into the surrounding agent run.
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
