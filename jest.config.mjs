// jest.config.mjs
import { pathsToModuleNameMapper } from 'ts-jest';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const tsconfig = require('./tsconfig.json');

const config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  testMatch: ['**/src/**/*.test.ts', '**/src/**/*.spec.ts'],
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
  maxWorkers: 7,  // Number of worker processes
  maxConcurrency: 1,  // Number of tests that can run in each worker
  // Alternative: use percentage of available CPUs
  // maxWorkers: '50%',
  
  // Optional: increase timeout for network requests
  testTimeout: 30000,  // 30 seconds (default is 5 seconds)
  
  // Optional: run tests serially (one at a time) - uncomment if needed
  // runInBand: true,
};

export default config;