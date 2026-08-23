import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test, { after, before } from 'node:test';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';

import {
  agentsTarget,
  getProblems,
  loadRolldown,
  scan,
} from './circular-deps.mjs';
import { packageEntries } from './package-entries.mjs';
import tsdownConfig from '../tsdown.config.mjs';

let engine;
const fixtureRoots = [];

before(async () => {
  engine = await loadRolldown();
});

after(async () => {
  await Promise.all(
    fixtureRoots.map((fixtureRoot) =>
      rm(fixtureRoot, { recursive: true, force: true })
    )
  );
});

async function createTarget(files, options = {}) {
  const fixtureRoot = await mkdtemp(
    path.join(os.tmpdir(), 'agents-circular-deps-')
  );
  fixtureRoots.push(fixtureRoot);

  await Promise.all(
    Object.entries(files).map(async ([file, source]) => {
      const filePath = path.join(fixtureRoot, file);
      await mkdir(path.dirname(filePath), { recursive: true });
      await writeFile(filePath, source, 'utf8');
    })
  );

  return {
    name: 'fixture',
    entries: [path.join(fixtureRoot, 'entry.ts')],
    alias: { '@': fixtureRoot },
    internalPrefixes: ['@/'],
    minModules: options.minModules ?? 1,
    typeEdges: false,
  };
}

test('the package build and checker share all public entries', () => {
  assert.equal(
    packageEntries['provider-registration'],
    'src/provider-registration.ts'
  );
  assert.equal(tsdownConfig.length, 2);
  for (const config of tsdownConfig) {
    assert.equal(config.entry, packageEntries);
    assert.equal(config.inputOptions.checks.circularDependency, true);
  }
  assert.equal(agentsTarget.minModules, 100);
  assert.equal(agentsTarget.typeEdges, false);
  assert.deepEqual(
    agentsTarget.entries,
    Object.values(packageEntries).map((entry) =>
      path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', entry)
    )
  );
});

test('reports a runtime dependency cycle', async () => {
  const target = await createTarget({
    'entry.ts':
      "import { right } from './right';\nexport const left = right + 1;\n",
    'right.ts':
      "import { left } from './entry';\nexport const right = left + 1;\n",
  });

  const result = await scan(engine, target);

  assert.equal(result.error, null);
  assert.equal(result.cycles.length, 1);
  assert.match(result.cycles[0], /Circular dependency/);
});

test('reports unresolved relative and aliased first-party imports', async () => {
  for (const specifier of ['./missing', '@/missing']) {
    const target = await createTarget({
      'entry.ts': `import { missing } from '${specifier}';\nexport const value = missing;\n`,
    });

    const result = await scan(engine, target);

    assert.equal(result.unresolved.length, 1);
    assert.match(
      getProblems(result).join('\n'),
      /unresolved first-party import/
    );
  }
});

test('reports syntax errors as build failures', async () => {
  const target = await createTarget({
    'entry.ts': 'export const = ;\n',
  });

  const result = await scan(engine, target);

  assert.notEqual(result.error, null);
  assert.match(getProblems(result).join('\n'), /build failed/);
});

test('fails when the resolved module graph drops below its floor', async () => {
  const target = await createTarget(
    { 'entry.ts': 'export const value = 1;\n' },
    { minModules: 2 }
  );

  const result = await scan(engine, target);

  assert.equal(result.error, null);
  assert.equal(result.modules, 1);
  assert.match(getProblems(result).join('\n'), /below the 2 floor/);
});

test('counts resolved runtime modules even when tree-shaking removes them', async () => {
  const target = await createTarget(
    {
      'entry.ts': "import './unused';\nexport const value = 1;\n",
      'unused.ts': 'export const unused = 2;\n',
    },
    { minModules: 2 }
  );

  const result = await scan(engine, target);

  assert.equal(result.error, null);
  assert.equal(result.modules, 2);
  assert.deepEqual(getProblems(result), []);
});

test('keeps grandfathered type-only edges out of the runtime graph', async () => {
  const target = await createTarget({
    'entry.ts':
      "import type { Right } from './right';\nexport type Left = { right: Right };\nexport const value = 1;\n",
    'right.ts':
      "import type { Left } from './entry';\nexport type Right = { left: Left };\n",
  });

  const result = await scan(engine, target);

  assert.equal(result.error, null);
  assert.deepEqual(result.cycles, []);
  assert.equal(result.modules, 1);
});
