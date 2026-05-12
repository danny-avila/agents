import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

const dependencyGroups = [
  'dependencies',
  'devDependencies',
  'peerDependencies',
  'optionalDependencies',
];
const disallowedSpecPattern =
  /^(?:git(?:\+ssh|\+https|\+file)?:|github:|https?:|file:|link:|workspace:)/i;
const expectedPackageName = '@librechat/agents';
const blockSize = 512;

const [tarballPath] = process.argv.slice(2);
const failures = [];

function fail(message) {
  failures.push(message);
}

function readString(buffer, start, length) {
  const bytes = buffer.subarray(start, start + length);
  const nullIndex = bytes.indexOf(0);
  const end = nullIndex === -1 ? bytes.length : nullIndex;

  return bytes.subarray(0, end).toString('utf8');
}

function parseTarball(path) {
  const buffer = gunzipSync(readFileSync(path));
  const entries = [];

  for (let offset = 0; offset < buffer.length; ) {
    const header = buffer.subarray(offset, offset + blockSize);

    if (header.every((byte) => byte === 0)) {
      break;
    }

    const name = readString(header, 0, 100);
    const prefix = readString(header, 345, 155);
    const sizeText = readString(header, 124, 12).trim();
    const type = readString(header, 156, 1) || '0';
    const size = Number.parseInt(sizeText || '0', 8);
    const entryPath = prefix ? `${prefix}/${name}` : name;

    offset += blockSize;
    entries.push({
      type,
      path: entryPath,
      size,
      content: buffer.subarray(offset, offset + size),
    });
    offset += Math.ceil(size / blockSize) * blockSize;
  }

  return entries;
}

function hasEntries(value) {
  return value != null && Object.keys(value).length > 0;
}

function isAllowedPath(path) {
  return (
    path === 'package/package.json' ||
    path === 'package/LICENSE' ||
    path === 'package/README.md' ||
    path.startsWith('package/dist/') ||
    path.startsWith('package/src/')
  );
}

function checkDependencySpecs(source, dependencies) {
  if (!dependencies) {
    return;
  }

  for (const [name, spec] of Object.entries(dependencies)) {
    if (typeof spec === 'string' && disallowedSpecPattern.test(spec)) {
      fail(`${source} uses disallowed dependency spec ${name}@${spec}`);
    }
  }
}

if (!tarballPath) {
  fail('Expected path to a package tarball');
} else {
  const entries = parseTarball(tarballPath);
  const packageJsonEntry = entries.find(
    (entry) => entry.path === 'package/package.json'
  );

  for (const entry of entries) {
    if (
      entry.path.startsWith('/') ||
      entry.path.includes('..') ||
      entry.path.includes('\\')
    ) {
      fail(`Unsafe tarball path: ${entry.path}`);
    }

    if (entry.type !== '0') {
      fail(`Unsupported tarball entry type for ${entry.path}: ${entry.type}`);
    }

    if (!isAllowedPath(entry.path)) {
      fail(`Unexpected file in package tarball: ${entry.path}`);
    }
  }

  if (!packageJsonEntry) {
    fail('Package tarball is missing package/package.json');
  } else {
    const packageJson = JSON.parse(packageJsonEntry.content.toString('utf8'));

    if (packageJson.name !== expectedPackageName) {
      fail(
        `Expected package name ${expectedPackageName}, got ${packageJson.name}`
      );
    }

    if (hasEntries(packageJson.scripts)) {
      fail('Package tarball must not define lifecycle or development scripts');
    }

    if (hasEntries(packageJson.optionalDependencies)) {
      fail('Package tarball must not define optionalDependencies');
    }

    if (packageJson.bundleDependencies || packageJson.bundledDependencies) {
      fail('Package tarball must not define bundled dependencies');
    }

    for (const group of dependencyGroups) {
      checkDependencySpecs(`Package tarball ${group}`, packageJson[group]);
    }
  }
}

if (failures.length > 0) {
  console.error('Package tarball validation failed:');
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log(`Package tarball validation passed for ${tarballPath}.`);
