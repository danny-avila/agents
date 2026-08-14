import { defineConfig } from 'tsdown';
import { isAbsolute } from 'node:path';

import { packageEntries } from './config/package-entries.mjs';

const shared = {
  entry: packageEntries,
  platform: 'node',
  // Declarations are emitted separately by `tsc -p tsconfig.build.json` (see the
  // `build` script); the source isn't isolatedDeclarations-clean, so oxc dts
  // isn't viable yet and tsc keeps the exact same dist/types output as before.
  dts: false,
  sourcemap: true,
  // Mirror Rollup's `preserveModules: true` — one output file per source module,
  // preserving the src-relative paths the package.json exports map points at.
  unbundle: true,
  outputOptions: {
    // Maps resolve against the `src/**/*.ts` the package ships, so inlining
    // `sourcesContent` would publish the same source a third time (once raw,
    // once per format). Stack traces still map to real source without it.
    sourcemapExcludeSources: true,
    // JSDoc is the editor-hover payload of `dist/types`, not of the runtime
    // JS — stripping it here leaves the `.d.ts` docs untouched. `annotation`
    // must stay so `@__PURE__` survives for consumer tree-shaking.
    comments: { legal: true, annotation: true, jsdoc: false },
  },
  // Force `.mjs`/`.cjs` regardless of package `type`, matching the previous
  // Rollup `entryFileNames` and the paths in the exports map.
  fixedExtension: true,
  alias: { '@': './src' },
  // Keep local build diagnostics aligned with the dedicated graph check.
  inputOptions: { checks: { circularDependency: true } },
  // Match the prior Rollup build (`external: [/node_modules/]`): bundle nothing
  // third-party, compile only this package's own modules.
  deps: {
    neverBundle: (id) =>
      !id.startsWith('.') && !id.startsWith('@/') && !isAbsolute(id),
    onlyBundle: false,
  },
};

export default defineConfig([
  { ...shared, format: 'esm', outDir: 'dist/esm' },
  { ...shared, format: 'cjs', outDir: 'dist/cjs' },
]);
