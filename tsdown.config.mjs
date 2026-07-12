import { defineConfig } from 'tsdown';
import { isAbsolute } from 'node:path';

const entry = {
  main: 'src/index.ts',
  'openai/index': 'src/openai/index.ts',
  'responses/index': 'src/responses/index.ts',
  'langchain/index': 'src/langchain/index.ts',
  'langchain/google-common': 'src/langchain/google-common.ts',
  'langchain/language_models/chat_models':
    'src/langchain/language_models/chat_models.ts',
  'langchain/messages': 'src/langchain/messages.ts',
  'langchain/messages/tool': 'src/langchain/messages/tool.ts',
  'langchain/openai': 'src/langchain/openai.ts',
  'langchain/prompts': 'src/langchain/prompts.ts',
  'langchain/runnables': 'src/langchain/runnables.ts',
  'langchain/tools': 'src/langchain/tools.ts',
  'langchain/utils/env': 'src/langchain/utils/env.ts',
};

const shared = {
  entry,
  platform: 'node',
  // Declarations are emitted separately by `tsc -p tsconfig.build.json` (see the
  // `build` script); the source isn't isolatedDeclarations-clean, so oxc dts
  // isn't viable yet and tsc keeps the exact same dist/types output as before.
  dts: false,
  sourcemap: true,
  // Mirror Rollup's `preserveModules: true` — one output file per source module,
  // preserving the src-relative paths the package.json exports map points at.
  unbundle: true,
  // Force `.mjs`/`.cjs` regardless of package `type`, matching the previous
  // Rollup `entryFileNames` and the paths in the exports map.
  fixedExtension: true,
  alias: { '@': './src' },
  // CI greps the build output for "Circular depend" (validate.yml); rolldown's
  // check is off by default, so enable it to keep that guard working.
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
