/**
 * Jest maps `@/lazyRequire` here: `import.meta` is not expressible under the CJS
 * transform, and the package self-references in production specifiers would load
 * stale `dist/` output instead of the source modules under test. Self-references
 * are rewritten onto the `@/` alias so every lazy load resolves through jest's
 * resolver — including the existing `@langchain/mistralai` stub.
 */
export function requireLazyModule<T>(specifier: string): T {
  const local = specifier.replace(/^@librechat\/agents\/llm\//, '@/llm/');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require(local) as T;
}
