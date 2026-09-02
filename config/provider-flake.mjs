import path from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { appendFileSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';

/**
 * Failure text that means an upstream provider broke rather than the code under
 * test: overload and throttle rejections, upstream 5xx, and transport resets.
 * Patterns stay specific so an assertion diff can never read as transient.
 */
const transientSignatures = [
  /overloaded_error/i,
  /\boverloaded\b/i,
  /rate_limit_error/i,
  /\brate[ _-]?limit(?:ed|ing)?\b/i,
  /ThrottlingException/,
  /TooManyRequests/i,
  /ServiceUnavailable/i,
  /InternalServerException/,
  /ModelNotReadyException/,
  /ModelTimeoutException/,
  /RESOURCE_EXHAUSTED/,
  /\bUNAVAILABLE\b/,
  /internal server error/i,
  /APIConnection(?:Timeout)?Error/,
  /\b(?:ECONNRESET|ECONNREFUSED|ETIMEDOUT|EAI_AGAIN|EPIPE)\b/,
  /socket hang up/i,
  /premature close/i,
  /fetch failed/i,
  /(?:status(?:\s*code)?|error|http)\D{0,12}\b(?:429|500|502|503|504|529)\b/i,
];

/** Jest assertion output — a real expectation failed, whatever else the message mentions. */
const assertionSignature =
  /(^|\n)\s*(?:expect\(|Expected(?: value)?:|Received:)/;

/** Titles are jest's own text and can quote any wording a test author chose. */
const withoutTitles = (message) =>
  message
    .split('\n')
    .filter((line) => !line.trimStart().startsWith('●'))
    .join('\n');

export const isTransientProviderFailure = (message) => {
  if (!message) {
    return false;
  }

  const body = withoutTitles(message);
  if (assertionSignature.test(body)) {
    return false;
  }

  return transientSignatures.some((signature) => signature.test(body));
};

/**
 * @typedef {{ name: string, message: string }} Failure
 * @param {{ testResults?: Array<{ name?: string, status?: string, message?: string, assertionResults?: Array<{ status?: string, fullName?: string, failureMessages?: string[] }> }> }} report
 * @returns {Failure[]}
 */
export const collectFailures = (report) => {
  const failures = [];

  for (const suite of report?.testResults ?? []) {
    const before = failures.length;

    for (const assertion of suite.assertionResults ?? []) {
      if (assertion.status !== 'failed') {
        continue;
      }
      failures.push({
        name: assertion.fullName || suite.name || 'unknown test',
        message: (assertion.failureMessages ?? []).join('\n'),
      });
    }

    if (failures.length === before && suite.status === 'failed') {
      failures.push({
        name: suite.name || 'unknown suite',
        message: suite.message ?? '',
      });
    }
  }

  return failures;
};

/**
 * A run is tolerable only when it failed and every failure came from a provider.
 * An empty failure list means jest died for its own reasons (bad config, no
 * tests matched, a crashed worker) and must stay red.
 */
export const classifyReport = (report) => {
  const failures = collectFailures(report);
  return {
    failures,
    tolerable:
      failures.length > 0 &&
      failures.every((failure) => isTransientProviderFailure(failure.message)),
  };
};

const firstLine = (message) =>
  message
    .split('\n')
    .find((line) => line.trim().length > 0)
    ?.trim() ?? 'no failure output';

const surface = (failures, jestArgs) => {
  const lines = failures.map(
    (failure) => `- \`${failure.name}\` — ${firstLine(failure.message)}`
  );

  console.log(
    `::warning title=Tolerated provider failures::${failures.length} test(s) failed on provider-side errors (overload, throttling, upstream 5xx). Treated as green; inspect the job logs before trusting this run.%0A${lines.join('%0A')}`
  );

  const summaryPath = process.env.GITHUB_STEP_SUMMARY;
  if (!summaryPath) {
    return;
  }

  appendFileSync(
    summaryPath,
    [
      '### Tolerated provider failures',
      '',
      `\`jest ${jestArgs.join(' ')}\` failed only on provider-side errors, so the job is green.`,
      '',
      ...lines,
      '',
    ].join('\n')
  );
};

const runJest = (args, reportPath) =>
  new Promise((resolve) => {
    const require = createRequire(import.meta.url);
    const child = spawn(
      process.execPath,
      [
        require.resolve('jest/bin/jest'),
        ...args,
        '--json',
        `--outputFile=${reportPath}`,
      ],
      { stdio: 'inherit' }
    );

    child.on('error', () => resolve(1));
    child.on('close', (code) => resolve(code ?? 1));
  });

const readReport = (reportPath) => {
  try {
    return JSON.parse(readFileSync(reportPath, 'utf8'));
  } catch {
    return null;
  }
};

const main = async (jestArgs) => {
  const reportDirectory = mkdtempSync(
    path.join(tmpdir(), 'agents-provider-flake-')
  );
  const reportPath = path.join(reportDirectory, 'jest.json');

  try {
    const code = await runJest(jestArgs, reportPath);
    if (code === 0) {
      return 0;
    }

    const { failures, tolerable } = classifyReport(readReport(reportPath));
    if (!tolerable) {
      return code;
    }

    surface(failures, jestArgs);
    return 0;
  } finally {
    rmSync(reportDirectory, { recursive: true, force: true });
  }
};

const invokedDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (invokedDirectly) {
  process.exitCode = await main(process.argv.slice(2));
}
