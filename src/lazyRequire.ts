import { createRequire } from 'node:module';

/**
 * Synchronous on-demand module loading: provider SDKs and other heavy dependencies
 * load with their first request instead of at import time. This is the only module
 * that touches `import.meta`; jest maps it to `test/stubs/lazyRequire.ts` so suites
 * resolve source modules through their own resolver (the same precedent as the
 * `@langchain/mistralai` stub).
 *
 * Internal modules load as format-matched siblings — `.cjs` neighbors from the CJS
 * build and `.mjs` neighbors from the ESM build (Node's `require(esm)`, safe on the
 * declared `>=24` engine) — so a lazily resolved provider shares one LangChain class
 * graph with the code that requested it. Under a source-mode runner such as `tsx`,
 * the seam resolves the TypeScript source directly through the active loader.
 */
const moduleUrl = import.meta.url;
const requireModule = createRequire(moduleUrl);

function detectBuildExtension(url: string): '.cjs' | '.mjs' | null {
  if (url.endsWith('.mjs')) {
    return '.mjs';
  }
  if (url.endsWith('.cjs')) {
    return '.cjs';
  }
  return null;
}

const buildExtension = detectBuildExtension(moduleUrl);

/** Source-mode commands run the TypeScript through an ESM loader that a synchronous
 *  CJS `require` can neither reach nor share module identity with, so lazily loadable
 *  modules are provided up front instead: `@/llm/providers.eager` imports them through
 *  the active loader and registers them here. */
const sourceModeModules = new Map<string, unknown>();

export function registerSourceModeModules(modules: Record<string, unknown>): void {
  for (const [relativePath, moduleExports] of Object.entries(modules)) {
    sourceModeModules.set(relativePath, moduleExports);
  }
}

/** Loads a module of this package by its src-relative path, e.g. `llm/openai/index`. */
export function requireInternalModule<T>(relativePath: string): T {
  if (buildExtension != null) {
    return requireModule(`./${relativePath}${buildExtension}`) as T;
  }
  const provided = sourceModeModules.get(relativePath);
  if (provided == null) {
    throw new Error(
      `Lazily loaded module "${relativePath}" is unavailable when running from source; ` +
        'import \'@/llm/providers.eager\' at the entrypoint before any model is used.'
    );
  }
  return provided as T;
}

/** Loads a third-party package on first use; identity-sensitive callers must not use this. */
export function requireLazyModule<T>(specifier: string): T {
  return requireModule(specifier) as T;
}
