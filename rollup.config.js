// rollup.config.js
import path from 'path';
import { fileURLToPath } from 'url';
import alias from '@rollup/plugin-alias';
import terser from '@rollup/plugin-terser';
import commonjs from '@rollup/plugin-commonjs';
import { cleandir } from 'rollup-plugin-cleandir';
import obfuscator from 'rollup-plugin-obfuscator';
import typescript from '@rollup/plugin-typescript';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import json from '@rollup/plugin-json';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isProduction = process.env.NODE_ENV === 'production';

const excludedDirsInProd = [
  'src/scripts/',
  'src/specs/',
  'src/proto/',
  'routes/',
  'config/',
];

function filterProdFiles(id) {
  if (isProduction) {
    return !excludedDirsInProd.some((dir) => id.includes(dir));
  }
  return true;
}

export default {
  input: {
    main: './src/index.ts',
  },
  output: [
    {
      dir: 'dist/esm',
      format: 'es',
      entryFileNames: '[name].mjs',
      // sourcemap: !isProduction,
      sourcemap: true,
      preserveModules: true,
      preserveModulesRoot: 'src',
    },
    {
      dir: 'dist/cjs',
      format: 'cjs',
      entryFileNames: '[name].cjs',
      // sourcemap: !isProduction,
      sourcemap: true,
      preserveModules: true,
      preserveModulesRoot: 'src',
      exports: 'named',
    },
  ],
  plugins: [
    cleandir('dist'),
    {
      name: 'filter-prod-files',
      resolveId(source, importer) {
        if (importer && !filterProdFiles(source)) {
          return false;
        }
      },
    },
    alias({
      entries: [{ find: '@', replacement: path.resolve(__dirname, 'src') }],
    }),
    nodeResolve({
      preferBuiltins: true,
      extensions: ['.mjs', '.js', '.json', '.node', '.ts'],
    }),
    commonjs({
      esmExternals: true,
      requireReturnsDefault: 'auto',
    }),
    json(),
    typescript({
      tsconfig: isProduction ? './tsconfig.build.json' : './tsconfig.json',
      /* enable source maps for testing with other production options */
      // sourceMap: !isProduction,
      // inlineSources: !isProduction,
      sourceMap: true,
      inlineSources: true,
      outDir: null,
      declaration: false,
      exclude: [
        'src/proto/**/*',
        'src/scripts/**/*',
        'src/specs/**/*',
        '**/*.test.ts',
        '**/*.spec.ts',
        'node_modules/**',
      ],
    }),
    /* Disable terser/obfuscator for now */
    // isProduction && terser(),
    // isProduction && obfuscator({
    //   exclude: [
    //     'node_modules/**',
    //     '**/*.spec.ts',
    //     'tsconfig-paths-bootstrap.mjs',
    //     'src/proto/**',
    //     'src/scripts/**',
    //     'dist/**',
    //     'config/**',
    //     'routes/**'
    //   ]
    // })
  ].filter(Boolean),
  external: [/node_modules/],
};
