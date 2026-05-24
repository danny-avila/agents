import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { basename, join } from 'node:path';

const rootDir = new URL('../', import.meta.url).pathname;
const failures = [];
const workflowDir = join(rootDir, '.github', 'workflows');
const dependencyGroups = [
  'dependencies',
  'devDependencies',
  'peerDependencies',
  'optionalDependencies',
];
const blockedInstallScripts = ['preinstall', 'install', 'postinstall'];
const disallowedSpecPattern =
  /^(?:git(?:\+ssh|\+https|\+file)?:|github:|https?:|file:|link:|workspace:)/i;
const pinnedActionPattern = /^[0-9a-f]{40}$/i;

function readJson(path) {
  return JSON.parse(readFileSync(join(rootDir, path), 'utf8'));
}

function hasEntries(value) {
  return value != null && Object.keys(value).length > 0;
}

function fail(message) {
  failures.push(message);
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

function checkPackageJson() {
  const packageJson = readJson('package.json');

  for (const scriptName of blockedInstallScripts) {
    if (packageJson.scripts?.[scriptName]) {
      fail(
        `package.json defines install-time lifecycle script "${scriptName}"`
      );
    }
  }

  if (hasEntries(packageJson.optionalDependencies)) {
    fail('package.json must not define optionalDependencies');
  }

  if (packageJson.bundleDependencies || packageJson.bundledDependencies) {
    fail('package.json must not define bundled dependencies');
  }

  for (const group of dependencyGroups) {
    checkDependencySpecs(`package.json ${group}`, packageJson[group]);
  }
}

function checkPackageLock() {
  const packageLock = readJson('package-lock.json');

  for (const [path, pkg] of Object.entries(packageLock.packages ?? {})) {
    const resolved = typeof pkg.resolved === 'string' ? pkg.resolved : '';

    if (resolved && !resolved.startsWith('https://registry.npmjs.org/')) {
      fail(
        `${path || 'root package'} resolves from non-registry URL ${resolved}`
      );
    }

    for (const group of dependencyGroups) {
      checkDependencySpecs(
        `package-lock ${path || 'root package'} ${group}`,
        pkg[group]
      );
    }
  }
}

function checkWorkflows() {
  if (!existsSync(workflowDir)) {
    return;
  }

  for (const file of readdirSync(workflowDir).filter((name) =>
    name.endsWith('.yml')
  )) {
    const workflowPath = join(workflowDir, file);
    const workflow = readFileSync(workflowPath, 'utf8');

    if (/\bpull_request_target\s*:/.test(workflow)) {
      fail(`${file} uses pull_request_target`);
    }

    if (/^  id-token:\s*write\b/m.test(workflow)) {
      fail(
        `${file} grants id-token: write at workflow scope instead of job scope`
      );
    }

    if (file !== 'publish.yml' && /\bid-token:\s*write\b/.test(workflow)) {
      fail(`${file} grants id-token: write outside the publish workflow`);
    }

    if (
      /\bid-token:\s*write\b/.test(workflow) &&
      /actions\/cache@|cache:\s*['"]?(?:npm|yarn|pnpm|node_modules)/.test(
        workflow
      )
    ) {
      fail(`${file} combines id-token: write with dependency caching`);
    }

    if (file === 'publish.yml' && /secrets\./.test(workflow)) {
      fail('publish.yml must not expose repository or environment secrets');
    }

    for (const match of workflow.matchAll(/uses:\s*([^@\s]+)@([^\s#]+)/g)) {
      const [, action, ref] = match;

      if (!pinnedActionPattern.test(ref)) {
        fail(`${file} uses mutable action reference ${action}@${ref}`);
      }
    }

    for (const match of workflow.matchAll(
      /run:\s*npm ci(?![^\n]*--ignore-scripts)/g
    )) {
      fail(
        `${file} runs npm ci without --ignore-scripts near index ${match.index}`
      );
    }

    for (const match of workflow.matchAll(
      /run:\s*npm publish(?![^\n]*--ignore-scripts)/g
    )) {
      fail(
        `${file} runs npm publish without --ignore-scripts near index ${match.index}`
      );
    }
  }
}

checkPackageJson();
checkPackageLock();
checkWorkflows();

if (failures.length > 0) {
  console.error('Supply-chain hygiene check failed:');
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log(`Supply-chain hygiene check passed for ${basename(rootDir)}.`);
