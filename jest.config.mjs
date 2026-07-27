// jest.config.mjs
import { pathsToModuleNameMapper } from 'ts-jest';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const tsconfig = require('./tsconfig.json');

/**
 * Dependencies published as ESM-only, which Jest's CommonJS runtime cannot
 * `require`. They are excluded from `transformIgnorePatterns` below so the
 * transform rewrites them to CJS on the way in.
 *
 * `@mistralai/mistralai` is `"type": "module"` with no CJS build at all, and
 * `@langchain/mistralai` requires it from its own CJS output — so any suite
 * that transitively touches `src/llm/providers.ts` (which imports
 * `ChatMistralAI` to populate the provider map) dies at load with
 * `SyntaxError: Unexpected token 'export'`, whether or not the test has
 * anything to do with Mistral.
 */
const esmOnlyDependencies = ['@mistralai/mistralai', '@langchain/mistralai'];

const config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/src/**/*.test.ts', '**/src/**/*.spec.ts'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { tsconfig: './tsconfig.json' }],
    /**
     * `allowJs` so ts-jest will down-level the ESM-only packages above;
     * `isolatedModules` because they are already type-checked upstream and we
     * only need the syntax transform.
     */
    '^.+\\.m?js$': [
      'ts-jest',
      {
        tsconfig: {
          allowJs: true,
          module: 'commonjs',
          target: 'es2020',
          esModuleInterop: true,
        },
        isolatedModules: true,
        diagnostics: false,
      },
    ],
  },
  transformIgnorePatterns: [
    `/node_modules/(?!(${esmOnlyDependencies.join('|')})/)`,
  ],
  moduleNameMapper: pathsToModuleNameMapper(tsconfig.compilerOptions.paths, {
    prefix: '<rootDir>/'
  }),
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