// jest.config.mjs
import { pathsToModuleNameMapper } from 'ts-jest';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const tsconfig = require('./tsconfig.json');

const config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/src/**/*.test.ts', '**/src/**/*.spec.ts'],
  moduleNameMapper: {
    /**
     * `@mistralai/mistralai` is published `"type": "module"` with no CommonJS
     * build, and `@langchain/mistralai` requires it from its own CJS output.
     * Jest's CJS runtime cannot load that chain, so every suite that
     * transitively imports `src/llm/providers.ts` — which pulls in
     * `ChatMistralAI` only to populate the provider constructor map — failed
     * at collection with `SyntaxError: Unexpected token 'export'`, regardless
     * of whether the test touched Mistral.
     *
     * Transforming the package does not work: Node decides module kind from
     * the package's `type` field, not from what a transform emits, so a
     * CommonJS rewrite is still evaluated as ESM and fails with
     * `ReferenceError: exports is not defined`.
     *
     * Mapped ahead of the tsconfig path aliases so the specific pattern wins.
     */
    '^@langchain/mistralai$': '<rootDir>/test/stubs/mistralai.ts',
    /**
     * `src/lazyRequire.ts` is the one module allowed to touch `import.meta`
     * (for `createRequire` in the ESM build); the CJS test transform cannot
     * express it, and its production package self-references would load stale
     * `dist/` output instead of the source under test. The stub resolves every
     * lazy load through jest's resolver instead.
     */
    '^@/lazyRequire$': '<rootDir>/test/stubs/lazyRequire.ts',
    ...pathsToModuleNameMapper(tsconfig.compilerOptions.paths, {
      prefix: '<rootDir>/'
    }),
  },
  modulePaths: [
    '<rootDir>'
  ],
  verbose: true,
  // setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  testEnvironmentOptions: {
    env: {
      NODE_ENV: 'test'
    }
  },
  // Limit concurrent test execution to avoid rate limits
  maxWorkers: '50%',
  maxConcurrency: 1,
  
  // Timeout for tests — E2E summarization tests hit real APIs and need more time.
  // Per-suite jest.setTimeout() calls can extend this further.
  testTimeout: 60000,  // 60 seconds
  
  // Optional: run tests serially (one at a time) - uncomment if needed
  // runInBand: true,
};

export default config;