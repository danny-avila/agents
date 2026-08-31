import type * as t from '@/types';

export type ProgrammaticRuntime = 'python' | 'bash';

const PYTHON_KEYWORDS = new Set([
  'False',
  'None',
  'True',
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
]);

/** Bash reserved words that get `_tool` suffix when used as function names */
const BASH_RESERVED = new Set([
  'if',
  'then',
  'else',
  'elif',
  'fi',
  'case',
  'esac',
  'for',
  'while',
  'until',
  'do',
  'done',
  'in',
  'function',
  'select',
  'time',
  'coproc',
  'declare',
  'typeset',
  'local',
  'readonly',
  'export',
  'unset',
]);

/**
 * Normalizes a tool name to a valid Python identifier.
 * 1. Replace hyphens and spaces with underscores
 * 2. Remove any other invalid characters
 * 3. Prefix with underscore if starts with number
 * 4. Append `_tool` if it's a Python keyword
 */
export function normalizeToPythonIdentifier(name: string): string {
  let normalized = name.replace(/[-\s]/g, '_');

  normalized = normalized.replace(/[^a-zA-Z0-9_]/g, '');

  if (/^[0-9]/.test(normalized)) {
    normalized = '_' + normalized;
  }

  if (PYTHON_KEYWORDS.has(normalized)) {
    normalized = normalized + '_tool';
  }

  return normalized;
}

/**
 * Normalizes a tool name to a valid bash function identifier.
 * 1. Replace hyphens, spaces, dots with underscores
 * 2. Remove any other invalid characters
 * 3. Prefix with underscore if starts with number
 * 4. Append `_tool` if it's a bash reserved word
 */
export function normalizeToBashIdentifier(name: string): string {
  let normalized = name.replace(/[-\s.]/g, '_');
  normalized = normalized.replace(/[^a-zA-Z0-9_]/g, '');

  if (/^[0-9]/.test(normalized)) {
    normalized = '_' + normalized;
  }

  if (BASH_RESERVED.has(normalized)) {
    normalized = normalized + '_tool';
  }

  return normalized;
}

export function normalizeToRuntimeIdentifier(
  name: string,
  runtime: ProgrammaticRuntime
): string {
  return runtime === 'bash'
    ? normalizeToBashIdentifier(name)
    : normalizeToPythonIdentifier(name);
}

/**
 * Extracts tool names that are actually called in the Python code.
 * Handles hyphen/underscore conversion since Python identifiers use underscores.
 * @param code - The Python code to analyze
 * @param toolNameMap - Map from normalized Python name to original tool name
 * @returns Set of original tool names found in the code
 */
export function extractUsedToolNames(
  code: string,
  toolNameMap: Map<string, string>
): Set<string> {
  const usedTools = new Set<string>();

  for (const [pythonName, originalName] of toolNameMap) {
    const escapedName = pythonName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escapedName}\\s*\\(`, 'g');

    if (pattern.test(code)) {
      usedTools.add(originalName);
    }
  }

  return usedTools;
}

/**
 * Extracts tool names that are actually called in the bash code.
 * Bash functions are invoked as commands (no parentheses), so we match
 * the normalized name as a word boundary.
 */
export function extractUsedBashToolNames(
  code: string,
  toolNameMap: Map<string, string>
): Set<string> {
  const usedTools = new Set<string>();

  for (const [bashName, originalName] of toolNameMap) {
    const escapedName = bashName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = new RegExp(`\\b${escapedName}\\b`, 'g');

    if (pattern.test(code)) {
      usedTools.add(originalName);
    }
  }

  return usedTools;
}

function matchesRuntimeIdentifier(
  code: string,
  identifier: string,
  runtime: ProgrammaticRuntime
): boolean {
  const escaped = identifier.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern =
    runtime === 'bash'
      ? new RegExp(`\\b${escaped}\\b`)
      : new RegExp(`\\b${escaped}\\s*\\(`);
  return pattern.test(code);
}

/**
 * Narrows `toolDefs` to the tools the submitted code actually references.
 *
 * Deliberately returns an EMPTY array when the code references nothing, unlike
 * `filterToolsByUsage`, which falls back to the full set. Callers use the empty
 * result to recognize a call that needs no tool stubs at all.
 *
 * Detection is a conservative regex over normalized identifiers, so a miss only
 * ever drops a stub and surfaces as a plain "not defined"/"command not found"
 * inside the sandbox. Callers stay responsible for the caller-capability
 * boundary: derive over the allowed and disallowed sets separately rather than
 * treating an unmatched name as absent.
 */
export function deriveReferencedToolDefs(
  toolDefs: readonly t.LCTool[],
  code: string,
  runtime: ProgrammaticRuntime
): t.LCTool[] {
  if (toolDefs.length === 0) {
    return [];
  }

  /* Names that normalize to the same identifier are kept together: the
   * reference is genuinely ambiguous, and dropping one would silently dispatch
   * by insertion order where sending both preserves existing behavior — the
   * Code API rejects bash collisions outright. */
  const namesByIdentifier = new Map<string, string[]>();
  for (const toolDef of toolDefs) {
    const identifier = normalizeToRuntimeIdentifier(toolDef.name, runtime);
    const names = namesByIdentifier.get(identifier);
    if (names != null) {
      names.push(toolDef.name);
      continue;
    }
    namesByIdentifier.set(identifier, [toolDef.name]);
  }

  const usedToolNames = new Set<string>();
  for (const [identifier, names] of namesByIdentifier) {
    if (!matchesRuntimeIdentifier(code, identifier, runtime)) {
      continue;
    }
    for (const name of names) {
      usedToolNames.add(name);
    }
  }

  if (usedToolNames.size === 0) {
    return [];
  }

  return toolDefs.filter((toolDef) => usedToolNames.has(toolDef.name));
}
