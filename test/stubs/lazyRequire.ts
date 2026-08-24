/**
 * Jest maps `@/lazyRequire` here: `import.meta` is not expressible under the CJS
 * transform, and the production seam's format-matched dist siblings do not exist
 * for source under test. Internal paths are rewritten onto the `@/` alias so every
 * lazy load resolves through jest's resolver — including the existing
 * `@langchain/mistralai` stub.
 */
export function requireInternalModule<T>(relativePath: string): T {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require(`@/${relativePath.replace(/\/index$/, '')}`) as T;
}

export function requireLazyModule<T>(specifier: string): T {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require(specifier) as T;
}
