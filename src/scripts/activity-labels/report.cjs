/** Aggregation + markdown rendering, shared by the live runner and the
 *  offline rescorer so metric fixes never require re-spending on the API.
 *
 *  Ported from LibreChat #14527 with three fixes pending backport: numeric
 *  sample ordering, a keyed record index for the per-case cells, and
 *  case-folded opener tallies. */
const FLAG_TYPES = [
  'len',
  'punct',
  'quote',
  'md',
  'opener',
  'tool-echo',
  'count-echo',
  'restate',
  'template',
];

const PRICES = { 'claude-haiku-4-5': { input: 1, output: 5 } };

function flagType(flag) {
  return flag.split(':')[0];
}

function aggregate(records, model) {
  const byVariant = new Map();
  for (const record of records) {
    if (!byVariant.has(record.variant)) {
      byVariant.set(record.variant, {
        steps: 0,
        errors: 0,
        flagCounts: {},
        /** Null prototype: labels are model prose, and an opener like
         *  'Constructor' would otherwise hit Object.prototype.constructor
         *  and render a garbage tally ('__proto__' would vanish). */
        firstWords: Object.create(null),
        totalWords: 0,
        latencies: [],
        inputTokens: 0,
        outputTokens: 0,
      });
    }
    const agg = byVariant.get(record.variant);
    if (record.error) {
      agg.errors += 1;
      /** Some failures still billed (a 200 whose label normalized to
       *  empty) — dropping their usage would make an all-empty variant
       *  report $0. */
      agg.inputTokens += record.inputTokens ?? 0;
      agg.outputTokens += record.outputTokens ?? 0;
      continue;
    }
    agg.steps += 1;
    agg.totalWords += record.wordCount;
    agg.latencies.push(record.latencyMs);
    agg.inputTokens += record.inputTokens;
    agg.outputTokens += record.outputTokens;
    /** Case-folded: 'Found' and 'found' are one opener — counting them
     *  separately would make a sentence-case violation read as MORE
     *  register diversity, the opposite of what the tally detects. */
    const opener =
      typeof record.firstWord === 'string'
        ? record.firstWord.toLowerCase()
        : record.firstWord;
    agg.firstWords[opener] = (agg.firstWords[opener] ?? 0) + 1;
    for (const flag of record.flags) {
      const type = flagType(flag);
      agg.flagCounts[type] = (agg.flagCounts[type] ?? 0) + 1;
    }
  }
  const price = PRICES[model];
  return [...byVariant.entries()].map(([name, agg]) => {
    const sortedFirst = Object.entries(agg.firstWords).sort(
      (a, b) => b[1] - a[1]
    );
    const topOpener = sortedFirst[0] ?? ['—', 0];
    return {
      variant: name,
      steps: agg.steps,
      errors: agg.errors,
      flagCounts: agg.flagCounts,
      distinctOpeners: sortedFirst.length,
      topOpener: `${topOpener[0]} ×${topOpener[1]}`,
      avgWords: agg.steps > 0 ? (agg.totalWords / agg.steps).toFixed(1) : '—',
      meanLatencyMs: agg.latencies.length
        ? Math.round(
            agg.latencies.reduce((a, b) => a + b, 0) / agg.latencies.length
          )
        : 0,
      inputTokens: agg.inputTokens,
      outputTokens: agg.outputTokens,
      costUsd: price
        ? (
            (agg.inputTokens * price.input + agg.outputTokens * price.output) /
            1e6
          ).toFixed(4)
        : 'n/a',
    };
  });
}

function markdownReport({
  records,
  aggregates,
  runCases,
  variantNames,
  model,
  samples,
}) {
  const lines = [];
  lines.push(`# Activity-label eval — ${new Date().toISOString()}`);
  lines.push('');
  lines.push(
    `model: \`${model}\` · samples: ${samples} · cases: ${runCases.length}`
  );
  lines.push('');
  lines.push('## Aggregate');
  lines.push('');
  lines.push(
    `| variant | steps | ${FLAG_TYPES.join(' | ')} | distinct openers | top opener | avg words | mean ms | cost |`
  );
  lines.push(
    `|---|---:|${FLAG_TYPES.map(() => '---:').join('|')}|---:|---|---:|---:|---:|`
  );
  for (const agg of aggregates) {
    lines.push(
      `| ${agg.variant} | ${agg.steps}${agg.errors ? ` (+${agg.errors} err)` : ''} | ` +
        FLAG_TYPES.map((type) => agg.flagCounts[type] ?? 0).join(' | ') +
        ` | ${agg.distinctOpeners} | ${agg.topOpener} | ${agg.avgWords} | ${agg.meanLatencyMs} | $${agg.costUsd} |`
    );
  }
  lines.push('');
  lines.push('## Per-case');
  /** Sorted numerically — the default lexicographic sort orders samples
   *  1, 10, 11, …, 2 once a sweep reaches ten samples. */
  const sampleList = [...new Set(records.map((r) => r.sample))].sort(
    (a, b) => a - b
  );
  /** One keyed pass instead of a records.find per table cell, which is
   *  quadratic in paid results on large sweeps. */
  const recordIndex = new Map(
    records.map((record) => [
      `${record.variant}\0${record.sample}\0${record.caseId}\0${record.stepId}`,
      record,
    ])
  );
  for (const testCase of runCases) {
    lines.push('');
    lines.push(`### ${testCase.id}`);
    lines.push('');
    lines.push(`*${testCase.notes}*`);
    lines.push('');
    const header = ['step'];
    if (samples > 1) {
      header.push('s');
    }
    if (testCase.steps.some((step) => step.productionLabel)) {
      header.push('production');
    }
    header.push(...variantNames);
    lines.push(`| ${header.join(' | ')} |`);
    lines.push(`|${header.map(() => '---').join('|')}|`);
    for (const step of testCase.steps) {
      const stepId = step.id ?? testCase.id;
      for (const sample of sampleList) {
        const row = [stepId];
        if (samples > 1) {
          row.push(String(sample));
        }
        if (header.includes('production')) {
          row.push(step.productionLabel ?? '');
        }
        for (const variantName of variantNames) {
          const record = recordIndex.get(
            `${variantName}\0${sample}\0${testCase.id}\0${stepId}`
          );
          if (!record) {
            row.push('');
          } else if (record.error) {
            row.push(`⛔ ${record.error}`);
          } else {
            const flagNote =
              record.flags.length > 0 ? ` ⚠${record.flags.join(' ⚠')}` : '';
            row.push(`${record.label}${flagNote}`);
          }
        }
        lines.push(
          `| ${row.map((cell) => cell.replace(/\|/g, '\\|')).join(' | ')} |`
        );
      }
    }
  }
  return lines.join('\n') + '\n';
}

module.exports = { aggregate, markdownReport, FLAG_TYPES };
