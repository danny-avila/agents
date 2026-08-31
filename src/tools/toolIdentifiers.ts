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

/* Interpolating constructs stay visible: an f-string replacement field and a
 * bash command substitution are executable code, and blanking them would hide a
 * real tool call. Everything else in the literal is prose and is erased, so
 * `f"example: write_file(path)"` does not read as a call. Bash double quotes
 * interpolate, so only single quotes are erased. */
const PYTHON_STRING_OR_COMMENT =
  /([A-Za-z]*)("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|#[^\n]*/g;
const BASH_STRING_OR_COMMENT = /'[^']*'|(?:^|\s)#[^\n]*/gm;
const PYTHON_REPLACEMENT_FIELD = /\{\{|\}\}|\{[^{}]*\}/g;

const blankOut = (match: string): string => ' '.repeat(match.length);

/** Blanks an f-string literal except for its `{...}` replacement fields. */
function blankOutsideReplacementFields(literal: string): string {
  let result = blankOut(literal);
  let match: RegExpExecArray | null;
  PYTHON_REPLACEMENT_FIELD.lastIndex = 0;
  while ((match = PYTHON_REPLACEMENT_FIELD.exec(literal)) != null) {
    if (match[0].length === 2) {
      continue;
    }
    result =
      result.slice(0, match.index) +
      match[0] +
      result.slice(match.index + match[0].length);
  }
  return result;
}

/**
 * Blanks comments and the literal text of strings so prose cannot be mistaken
 * for a call, while leaving interpolated expressions visible.
 *
 * Only reduces false positives — a name that survives may still not be a real
 * invocation. Runtimes enforce the caller boundary by defining stubs for the
 * selected tools alone, so a residual miss costs a stub, not a permission.
 */
export function stripNonCodeText(
  code: string,
  runtime: ProgrammaticRuntime
): string {
  if (runtime === 'bash') {
    return code.replace(BASH_STRING_OR_COMMENT, blankOut);
  }

  return code.replace(
    PYTHON_STRING_OR_COMMENT,
    (match: string, prefix?: string, literal?: string) => {
      if (literal == null) {
        return blankOut(match);
      }
      if (prefix == null || !/[fF]/.test(prefix)) {
        return blankOut(match);
      }
      return blankOut(prefix) + blankOutsideReplacementFields(literal);
    }
  );
}

/**
 * Returns the first pair of tool names that collapse to one runtime identifier.
 *
 * Stub generation binds by identifier, so two such tools are indistinguishable
 * once emitted — the local Python wrapper defines both and the later one wins.
 * Callers reject the selection rather than dispatch by declaration order.
 */
export function findNormalizedIdentifierCollision(
  toolDefs: readonly t.LCTool[],
  runtime: ProgrammaticRuntime
): { first: string; second: string; identifier: string } | null {
  const seen = new Map<string, string>();
  for (const toolDef of toolDefs) {
    const identifier = normalizeToRuntimeIdentifier(toolDef.name, runtime);
    const prior = seen.get(identifier);
    if (prior != null && prior !== toolDef.name) {
      return { first: prior, second: toolDef.name, identifier };
    }
    seen.set(identifier, toolDef.name);
  }
  return null;
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

  const executableCode = stripNonCodeText(code, runtime);
  const usedToolNames = new Set<string>();
  for (const [identifier, names] of namesByIdentifier) {
    if (!matchesRuntimeIdentifier(executableCode, identifier, runtime)) {
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
