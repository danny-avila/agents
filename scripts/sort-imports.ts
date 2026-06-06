#!/usr/bin/env bun
/**
 * Sorts imports in all .ts files under src/ per project convention
 * (AGENTS.md § Import Order):
 *
 * 1. Package value imports     — shortest line to longest
 * 2. import type from packages — longest line to shortest
 * 3. import type from local    — longest line to shortest
 * 4. Local value imports       — longest line to shortest
 *
 * Run:        bun run sort-imports
 * Check only: bun run sort-imports:check
 */

import { Glob } from 'bun';
import { join } from 'node:path';

const ROOT = join(import.meta.dir, '..');
const SRC = join(ROOT, 'src');
const CHECK = process.argv.includes('--check');

/** Directories under src/ excluded from linting (mirrors eslint.config.mjs). */
const EXCLUDED_DIRS = ['scripts/', 'proto/'];

function isLocal(spec: string): boolean {
  return (
    spec.startsWith('@/') ||
    spec.startsWith('./') ||
    spec.startsWith('../') ||
    spec.startsWith('#/')
  );
}

function isExcluded(rel: string): boolean {
  return EXCLUDED_DIRS.some((dir) => rel.startsWith(dir));
}

interface Stmt {
  raw: string;
  spec: string;
  isType: boolean;
  isLocal: boolean;
  len: number;
}

function extractSpec(raw: string): string | null {
  return raw.match(/from\s+['"]([^'"]+)['"]/)?.[1] ?? null;
}

function sortFileImports(content: string): string | null {
  const lines = content.split('\n');
  let i = 0;

  while (i < lines.length) {
    const t = lines[i].trimStart();
    if (
      t === '' ||
      t.startsWith('//') ||
      t.startsWith('/*') ||
      t.startsWith('*') ||
      t.startsWith('*/') ||
      t.startsWith('\'use ') ||
      t.startsWith('"use ')
    ) {
      i++;
    } else {
      break;
    }
  }

  const importStart = i;
  const sideEffects: string[] = [];
  const stmts: Stmt[] = [];
  let importEnd = i;

  while (i < lines.length) {
    const t = lines[i].trimStart();
    if (!t.startsWith('import ') && !t.startsWith('import{')) break;

    let raw = lines[i];
    let j = i;
    while (!raw.includes(';') && j + 1 < lines.length) {
      j++;
      raw += '\n' + lines[j];
    }
    i = j + 1;
    importEnd = i;

    const spec = extractSpec(raw);
    if (spec == null || spec === '') {
      sideEffects.push(raw);
      while (i < lines.length && lines[i].trim() === '') i++;
      continue;
    }

    stmts.push({
      raw,
      spec,
      isType: /^import\s+type[\s{]/.test(raw.trimStart()),
      isLocal: isLocal(spec),
      len: raw
        .split('\n')
        .map((l) => l.trim())
        .join(' ').length,
    });

    while (i < lines.length && lines[i].trim() === '') i++;
  }

  if (stmts.length < 2 && sideEffects.length === 0) return null;

  const g1 = stmts
    .filter((s) => !s.isType && !s.isLocal)
    .sort((a, b) => a.len - b.len);
  const g2 = stmts
    .filter((s) => s.isType && !s.isLocal)
    .sort((a, b) => b.len - a.len);
  const g3 = stmts
    .filter((s) => s.isType && s.isLocal)
    .sort((a, b) => b.len - a.len);
  const g4 = stmts
    .filter((s) => !s.isType && s.isLocal)
    .sort((a, b) => b.len - a.len);
  const sorted = [...sideEffects, ...g1, ...g2, ...g3, ...g4];

  const oldBlock = [...sideEffects, ...stmts.map((s) => s.raw)].join('\n');
  const newBlock = sorted
    .map((s) => (typeof s === 'string' ? s : s.raw))
    .join('\n');
  if (oldBlock === newBlock) return null;

  return [
    ...lines.slice(0, importStart),
    ...sorted.map((s) => (typeof s === 'string' ? s : s.raw)),
    ...lines.slice(importEnd),
  ].join('\n');
}

const glob = new Glob('**/*.ts');
let changed = 0;
let total = 0;

for await (const rel of glob.scan({ cwd: SRC })) {
  if (isExcluded(rel)) continue;
  const filePath = join(SRC, rel);
  const content = await Bun.file(filePath).text();
  const result = sortFileImports(content);
  total++;
  if (result === null) continue;
  changed++;
  if (CHECK) {
    console.log(`  ✗ ${rel}`);
  } else {
    await Bun.write(filePath, result);
    console.log(`  ✓ ${rel}`);
  }
}

if (CHECK && changed) {
  console.log(
    `\n${changed}/${total} files need sorting. Run: bun run sort-imports`
  );
  process.exit(1);
} else if (changed) {
  console.log(`\nSorted ${changed}/${total} files.`);
} else {
  console.log(`All ${total} files already sorted.`);
}
