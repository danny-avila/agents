import { createRequire } from 'node:module';

/**
 * Synchronous on-demand module loading that works from both build formats: provider
 * SDKs and other heavy dependencies load with their first request instead of at
 * import time. This is the only module that touches `import.meta`; jest maps it to
 * `test/stubs/lazyRequire.ts` so suites resolve source modules through their own
 * resolver (the same precedent as the `@langchain/mistralai` stub).
 */
const requireModule = createRequire(import.meta.url);

export function requireLazyModule<T>(specifier: string): T {
  return requireModule(specifier) as T;
}
