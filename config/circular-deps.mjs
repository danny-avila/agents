import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { packageEntries } from './package-entries.mjs';

const root = path.resolve(fileURLToPath(import.meta.url), '../..');
const minimumModuleCount = 100;
/**
 * The public type barrels contain pre-existing declaration-only cycles. Runtime
 * edges are enforced now; type edges can be enabled after that graph is untangled.
 */
const includeTypeEdges = false;

/** Collect every type-only dependency specifier from a parsed TypeScript AST. */
const collectTypeSpecifiers = (node, specifiers) => {
  if (node === null || typeof node !== 'object') {
    return;
  }
  if (Array.isArray(node)) {
    for (const item of node) {
      collectTypeSpecifiers(item, specifiers);
    }
    return;
  }

  const typeOnly =
    node.type === 'TSImportType' ||
    (node.importKind ?? node.exportKind) === 'type' ||
    (Array.isArray(node.specifiers) &&
      node.specifiers.some((specifier) =>
        ['type', 'typeof'].includes(
          specifier.importKind ?? specifier.exportKind
        )
      ));
  if (typeOnly && typeof node.source?.value === 'string') {
    specifiers.add(node.source.value);
  }

  for (const value of Object.values(node)) {
    if (value !== null && typeof value === 'object') {
      collectTypeSpecifiers(value, specifiers);
    }
  }
};

/** Materialize type-only imports so Rolldown checks the declaration graph too. */
const typeEdgesPlugin = (parseAst) => ({
  name: 'type-edges',
  transform(code, id) {
    const extension = /\.([mc]?tsx?)(?:$|\?)/.exec(id)?.[1];
    if (!extension) {
      return null;
    }

    const { body } = parseAst(code, {
      lang: extension.endsWith('x') ? 'tsx' : 'ts',
    });
    const specifiers = new Set();
    collectTypeSpecifiers(body, specifiers);
    if (specifiers.size === 0) {
      return null;
    }

    const edges = [...specifiers]
      .map((specifier) => `\nimport ${JSON.stringify(specifier)};`)
      .join('');
    return { code: code + edges, map: null };
  },
});

async function loadRolldown() {
  const packageRequire = createRequire(path.join(root, 'package.json'));
  const tsdownRequire = createRequire(packageRequire.resolve('tsdown'));
  const { rolldown } = await import(
    pathToFileURL(tsdownRequire.resolve('rolldown')).href
  );
  const { parseAst } = await import(
    pathToFileURL(tsdownRequire.resolve('rolldown/parseAst')).href
  );
  return { rolldown, parseAst };
}

// eslint-disable-next-line no-control-regex
const stripAnsi = (message) => message.replace(/\u001B\[[0-9;]*m/g, '');
const relativize = (message) =>
  stripAnsi(message).replaceAll(root + path.sep, '');

async function scan({ rolldown, parseAst }) {
  const cycles = [];
  const unresolved = [];
  const isInternal = (id) =>
    id.startsWith('.') || path.isAbsolute(id) || id.startsWith('@/');

  try {
    const build = await rolldown({
      input: Object.values(packageEntries).map((entry) =>
        path.join(root, entry)
      ),
      platform: 'node',
      resolve: { alias: { '@': path.join(root, 'src') } },
      external: (id) => !isInternal(id),
      plugins: includeTypeEdges ? [typeEdgesPlugin(parseAst)] : [],
      checks: { circularDependency: true },
      onLog(_level, log) {
        if (log.code === 'CIRCULAR_DEPENDENCY') {
          cycles.push(relativize(log.message));
        } else if (log.code === 'UNRESOLVED_IMPORT') {
          unresolved.push(relativize(log.message));
        }
      },
    });
    const { output } = await build.generate({ format: 'cjs' });
    const modules = output.reduce(
      (sum, chunk) => sum + (chunk.moduleIds?.length ?? 0),
      0
    );
    await build.close();
    return { cycles, unresolved, modules, error: null };
  } catch (error) {
    return { cycles, unresolved, modules: 0, error };
  }
}

function report({ cycles, unresolved, modules, error }) {
  const problems = [];
  if (error) {
    problems.push(`build failed: ${relativize(error.message)}`);
  }
  problems.push(...cycles);
  problems.push(
    ...unresolved.map((message) => `unresolved first-party import: ${message}`)
  );
  if (!error && modules < minimumModuleCount) {
    problems.push(
      `graph has ${modules} modules, below the ${minimumModuleCount} floor; the scan is no longer resolving the full package`
    );
  }

  if (problems.length === 0) {
    const scope = includeTypeEdges
      ? 'runtime + type edges'
      : 'runtime edges only, type edges grandfathered';
    console.log(
      `✓ @librechat/agents: no circular dependencies (${modules} modules, ${scope})`
    );
    return true;
  }

  console.error('✗ @librechat/agents:');
  for (const problem of problems) {
    console.error(`    ${problem}`);
  }
  return false;
}

const passed = report(await scan(await loadRolldown()));
if (!passed) {
  console.error('\nCircular dependency check failed.');
  process.exit(1);
}
