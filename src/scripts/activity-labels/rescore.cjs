/**
 * Offline rescore: recompute checks over a stored results JSON after a
 * metric change, without re-calling the API. Chains are rebuilt from the
 * stored labels in push order (steps within a case ran serially).
 *
 * Usage: node src/scripts/activity-labels/rescore.cjs [results/<file>.json]
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const { cases, stepEntries } = require('./corpus.cjs');
const { checkLabel } = require('./checks.cjs');
const { aggregate, markdownReport } = require('./report.cjs');

const RESULTS_DIR = path.join(__dirname, 'results');

function newestResults() {
  const files = fs
    .readdirSync(RESULTS_DIR)
    .filter((file) => file.endsWith('.json'))
    .sort();
  if (files.length === 0) {
    throw new Error('no stored results to rescore');
  }
  return path.join(RESULTS_DIR, files[files.length - 1]);
}

const sourcePath = process.argv[2]
  ? path.resolve(process.argv[2])
  : newestResults();
const { args, corpusFingerprint, records } = JSON.parse(
  fs.readFileSync(sourcePath, 'utf8')
);

/** Rescoring replays stored labels against the CURRENT corpus. If the
 *  corpus drifted since the run (renamed steps, changed tool names), the
 *  recomputed flags would grade against the wrong entries — refuse rather
 *  than silently produce a plausible-looking report. */
const currentFingerprint = crypto
  .createHash('sha256')
  .update(JSON.stringify(cases))
  .digest('hex');
if (corpusFingerprint == null) {
  console.warn(
    'WARN: stored results predate corpus fingerprinting — rescoring against the current corpus, which may have drifted'
  );
} else if (corpusFingerprint !== currentFingerprint) {
  throw new Error(
    'stored results came from a different corpus revision — re-run the sweep instead of rescoring'
  );
}

const stepsByCase = new Map(
  cases.map((testCase) => [
    testCase.id,
    new Map(testCase.steps.map((step) => [step.id ?? testCase.id, step])),
  ])
);

const chains = new Map();
for (const record of records) {
  if (record.error || record.label == null) {
    continue;
  }
  const key = `${record.variant}\0${record.sample}\0${record.caseId}`;
  if (!chains.has(key)) {
    chains.set(key, []);
  }
  const chain = chains.get(key);
  const step = stepsByCase.get(record.caseId)?.get(record.stepId);
  const { flags, wordCount, firstWord } = checkLabel(record.label, {
    entries: step != null ? stepEntries(step) : [],
    previousLabels: chain,
  });
  record.flags = flags;
  record.wordCount = wordCount;
  record.firstWord = firstWord;
  chain.push(record.label);
}

const variantNames = [...new Set(records.map((record) => record.variant))];
const runCases = cases.filter((testCase) =>
  records.some((r) => r.caseId === testCase.id)
);
const report = markdownReport({
  records,
  aggregates: aggregate(records, args.model),
  runCases,
  variantNames,
  model: args.model,
  samples: args.samples,
});
/** An archived JSON can be rescored by explicit path in a fresh checkout
 *  where the gitignored results/ directory does not exist yet. */
fs.mkdirSync(RESULTS_DIR, { recursive: true });
fs.writeFileSync(path.join(RESULTS_DIR, 'latest.md'), report);
console.log(`rescored ${path.basename(sourcePath)}`);
console.log(report.split('## Per-case')[0]);
console.log(
  'full per-case tables: src/scripts/activity-labels/results/latest.md'
);
