import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { packageEntries } from './package-entries.mjs';

const root = path.resolve(fileURLToPath(import.meta.url), '../..');

/**
 * Type-only cycles in the public barrels predate this check. Runtime edges are
 * enforced now; the exclusion remains explicit until the type graph is
 * untangled in a dedicated change.
 */
export const agentsTarget = Object.freeze({
  name: '@librechat/agents',
  entries: Object.values(packageEntries).map((entry) => path.join(root, entry)),
  alias: { '@': path.join(root, 'src') },
  internalPrefixes: ['@/'],
  minModules: 100,
  typeEdges: false,
});

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

/** Materialize type-only imports when a target opts into declaration edges. */
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

/** Load the exact Rolldown installation resolved from tsdown. */
export async function loadRolldown() {
  const packageRequire = createRequire(path.join(root, 'package.json'));
  const tsdownRequire = createRequire(
    packageRequire.resolve('tsdown/package.json')
  );
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
const errorMessage = (error) =>
  error instanceof Error ? error.message : String(error);
const uniqueSorted = (values) => [...new Set(values)].sort();
const unresolvedCodes = new Set(['UNRESOLVED_IMPORT', 'RESOLVE_ERROR']);

const getNestedErrors = (error) => {
  if (
    error === null ||
    typeof error !== 'object' ||
    !Array.isArray(error.errors)
  ) {
    return [];
  }
  return error.errors;
};

const isInternal = (id, target) =>
  id.startsWith('.') ||
  path.isAbsolute(id) ||
  target.internalPrefixes.some((prefix) => id.startsWith(prefix));
const isUnresolvedFirstParty = (diagnostic, target) =>
  unresolvedCodes.has(diagnostic?.code) &&
  (typeof diagnostic.exporter !== 'string' ||
    isInternal(diagnostic.exporter, target));

export async function scan(engine, target = agentsTarget) {
  const cycles = [];
  const unresolved = [];
  const moduleIds = new Set();
  const graphCounterPlugin = {
    name: 'resolved-module-counter',
    async resolveId(source, importer) {
      const resolved = await this.resolve(source, importer, { skipSelf: true });
      if (resolved && !resolved.external && path.isAbsolute(resolved.id)) {
        moduleIds.add(resolved.id);
      }
      return resolved;
    },
  };

  try {
    const build = await engine.rolldown({
      input: target.entries,
      platform: 'node',
      resolve: { alias: target.alias },
      external: (id) => !isInternal(id, target),
      plugins: [
        ...(target.typeEdges ? [typeEdgesPlugin(engine.parseAst)] : []),
        graphCounterPlugin,
      ],
      checks: { circularDependency: true },
      onLog(_level, log) {
        if (log.code === 'CIRCULAR_DEPENDENCY') {
          cycles.push(relativize(log.message));
        } else if (isUnresolvedFirstParty(log, target)) {
          unresolved.push(relativize(log.message));
        }
      },
    });

    try {
      await build.generate({ format: 'cjs' });
      return {
        target,
        cycles: uniqueSorted(cycles),
        unresolved: uniqueSorted(unresolved),
        modules: moduleIds.size,
        error: null,
      };
    } finally {
      await build.close();
    }
  } catch (error) {
    const unresolvedErrors = getNestedErrors(error)
      .filter((nestedError) => isUnresolvedFirstParty(nestedError, target))
      .map((nestedError) => relativize(errorMessage(nestedError)));
    return {
      target,
      cycles: uniqueSorted(cycles),
      unresolved: uniqueSorted([...unresolved, ...unresolvedErrors]),
      modules: 0,
      error,
    };
  }
}

export function getProblems({ target, cycles, unresolved, modules, error }) {
  const problems = [];
  if (error) {
    problems.push(`build failed: ${relativize(errorMessage(error))}`);
  }
  problems.push(...cycles);
  problems.push(
    ...unresolved.map((message) => `unresolved first-party import: ${message}`)
  );
  if (!error && modules < target.minModules) {
    problems.push(
      `graph has ${modules} modules, below the ${target.minModules} floor; the scan is no longer resolving the full package`
    );
  }
  return problems;
}

export function report(result) {
  const { target, modules } = result;
  const problems = getProblems(result);
  if (problems.length === 0) {
    const scope = target.typeEdges
      ? 'runtime + type edges'
      : 'runtime edges only, type edges grandfathered';
    console.log(
      `✓ ${target.name}: no circular dependencies (${modules} modules, ${scope})`
    );
    return true;
  }

  console.error(`✗ ${target.name}:`);
  for (const problem of problems) {
    console.error(`    ${problem}`);
  }
  return false;
}

export async function main() {
  let result;
  try {
    result = await scan(await loadRolldown());
  } catch (error) {
    result = {
      target: agentsTarget,
      cycles: [],
      unresolved: [],
      modules: 0,
      error,
    };
  }

  const passed = report(result);
  if (!passed) {
    console.error('\nCircular dependency check failed.');
  }
  return passed;
}

const isDirectRun =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isDirectRun && !(await main())) {
  process.exitCode = 1;
}
