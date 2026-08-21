import { config } from 'dotenv';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ProgrammaticToolCallingJsonSchema } from './ptcTimeout';
import type * as t from '@/types';
import {
  BASH_SHELL_GUIDANCE,
  CODE_ARTIFACT_PATH_GUIDANCE,
  appendFailedExecutionFileReminder,
  buildCodeApiExecutionErrorMessage,
  buildCodeApiEndpoint,
  CodeApiRequestError,
  getCodeBaseURL,
  selectRuntimeSessionHint,
} from './CodeExecutor';
import {
  makeRequest,
  executeTools,
  formatCompletedResponse,
  assertDisallowedToolUsage,
} from './ProgrammaticToolCalling';
import {
  clampCodeApiRunTimeoutMs,
  createCodeApiRunTimeoutSchema,
  resolveCodeApiRunTimeoutMs,
} from './ptcTimeout';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { Constants } from '@/common';

config();

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_MAX_ROUND_TRIPS = 20;
const DEFAULT_RUN_TIMEOUT_MS = resolveCodeApiRunTimeoutMs();

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

// ============================================================================
// Description Components
// ============================================================================

const STATELESS_WARNING = `CRITICAL - STATELESS EXECUTION:
Each call is a fresh bash shell. Variables and state do NOT persist between calls.
You MUST complete your entire workflow in ONE code block.
DO NOT split work across multiple calls expecting to reuse variables.`;

const CORE_RULES = `Rules:
- One call: state does not persist
- Tools are pre-defined as bash functions—DO NOT redefine them
- Each tool function accepts a JSON string argument
- Save tool output with raw=$(tool '{}'); printf '%s\n' "$raw" > /mnt/data/file.json; direct tool > file may be empty
- Tool stdout is normalized to one compact JSON value when possible; parse saved stdout once, then use fromjson? // . only for JSON-string fields
- Only echo/printf output returns to the model
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- timeout caps one sandbox run/replay iteration, not the total multi-round-trip workflow`;

const ADDITIONAL_RULES =
  '- Tool names normalized: hyphens→underscores, reserved words get `_tool` suffix';

const EXAMPLES = `Example (Complete workflow in one call):
  # Query data and process
  data=$(query_database '{"sql": "SELECT * FROM users"}')
  echo "$data" | jq '.[] | .name'

Example (Parallel calls):
  { sf=$(web_search '{"query": "SF weather"}'); printf '%s\n' "$sf" > /mnt/data/sf.json; } &
  { ny=$(web_search '{"query": "NY weather"}'); printf '%s\n' "$ny" > /mnt/data/ny.json; } &
  wait
  echo "SF: $(jq -r . /mnt/data/sf.json)"
  echo "NY: $(jq -r . /mnt/data/ny.json)"`;

const CODE_PARAM_DESCRIPTION = `Bash code that calls tools programmatically. Tools are available as bash functions.

${STATELESS_WARNING}

Each tool function accepts a JSON string as its argument.
Example: tool_name '{"key": "value"}'

${EXAMPLES}

${CORE_RULES}`;

// ============================================================================
// Schema
// ============================================================================

export function createBashProgrammaticToolCallingSchema(
  maxRunTimeoutMs = DEFAULT_RUN_TIMEOUT_MS
): ProgrammaticToolCallingJsonSchema {
  return {
    type: 'object',
    properties: {
      intent: { ...INTENT_PROPERTY },
      code: {
        type: 'string',
        minLength: 1,
        description: CODE_PARAM_DESCRIPTION,
      },
      timeout: createCodeApiRunTimeoutSchema(maxRunTimeoutMs),
    },
    required: ['code'],
  } as const;
}

export const BashProgrammaticToolCallingSchema =
  createBashProgrammaticToolCallingSchema();

export const BashProgrammaticToolCallingName =
  Constants.BASH_PROGRAMMATIC_TOOL_CALLING;

export const BashProgrammaticToolCallingDescription = `
Run tools via bash code. Tools are available as bash functions that accept JSON string arguments.

${STATELESS_WARNING}

${CORE_RULES}
${ADDITIONAL_RULES}

When to use: shell pipelines, parallel execution (& and wait), file processing, text manipulation.

${EXAMPLES}
`.trim();

export const BashProgrammaticToolCallingDefinition = {
  name: BashProgrammaticToolCallingName,
  description: BashProgrammaticToolCallingDescription,
  schema: BashProgrammaticToolCallingSchema,
} as const;

function maybeParseJsonResultString(result: unknown): unknown {
  if (typeof result !== 'string') {
    return result;
  }

  const trimmed = result.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) {
    return result;
  }

  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return result;
  }
}

export function normalizeBashToolResultsForReplay(
  toolResults: t.PTCToolResult[]
): t.PTCToolResult[] {
  return toolResults.map((toolResult) => {
    if (toolResult.is_error) {
      return toolResult;
    }

    return {
      ...toolResult,
      result: maybeParseJsonResultString(toolResult.result),
    };
  });
}

// ============================================================================
// Helper Functions
// ============================================================================

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
  const commandNames = extractBashCommandNames(code);

  for (const commandName of commandNames) {
    const originalName = toolNameMap.get(commandName);
    if (originalName != null) {
      usedTools.add(originalName);
    }
  }

  return usedTools;
}

const COMMAND_PREFIXES = new Set([
  'builtin',
  'command',
  'coproc',
  'env',
  'exec',
  'nohup',
  'sudo',
]);

const COMMAND_START_WORDS = new Set([
  'do',
  'elif',
  'else',
  'case',
  'if',
  'in',
  'then',
  'time',
  'until',
  'while',
]);

type BashToken =
  | { type: 'word'; value: string }
  | { type: 'separator' }
  | { type: 'redirect' }
  | { type: 'nested'; tokens: BashToken[] };

/** Returns shell words at command position, excluding arguments and comments. */
function extractBashCommandNames(code: string): Set<string> {
  const commands = new Set<string>();

  collectBashCommandNames(tokenizeBash(code), commands);
  return commands;
}

function collectBashCommandNames(
  tokens: BashToken[],
  commands: Set<string>
): void {
  let expectsCommand = true;
  let skipRedirectTarget = false;

  for (const token of tokens) {
    if (token.type === 'nested') {
      collectBashCommandNames(token.tokens, commands);
      if (skipRedirectTarget) {
        skipRedirectTarget = false;
      }
      continue;
    }
    if (token.type === 'separator') {
      expectsCommand = true;
      skipRedirectTarget = false;
      continue;
    }
    if (token.type === 'redirect') {
      skipRedirectTarget = true;
      continue;
    }
    if (skipRedirectTarget) {
      skipRedirectTarget = false;
      continue;
    }
    if (!expectsCommand) {
      continue;
    }

    const word = token.value;
    if (/^[A-Za-z_][A-Za-z0-9_]*\+?=/.test(word)) {
      continue;
    }
    if (word.startsWith('-')) {
      continue;
    }
    if (COMMAND_START_WORDS.has(word) || word === '!' || word === '{') {
      continue;
    }
    if (COMMAND_PREFIXES.has(word)) {
      continue;
    }

    commands.add(word);
    expectsCommand = false;
  }
}

/** Minimal shell lexer for locating command positions without executing code. */
function tokenizeBash(code: string): BashToken[] {
  code = maskBashHeredocBodies(code);
  const tokens: BashToken[] = [];
  let index = 0;

  while (index < code.length) {
    const char = code[index];
    if (char === ' ' || char === '\t' || char === '\r') {
      index += 1;
      continue;
    }
    if (code.slice(index, index + 2) === '&>') {
      tokens.push({ type: 'redirect' });
      index += code[index + 2] === '>' ? 3 : 2;
      continue;
    }
    if (char === '\n' || char === ';' || char === '|' || char === '&') {
      tokens.push({ type: 'separator' });
      index += code[index + 1] === char ? 2 : 1;
      continue;
    }
    if (/[0-9]/.test(char)) {
      let redirectIndex = index + 1;
      while (/[0-9]/.test(code[redirectIndex] ?? '')) {
        redirectIndex += 1;
      }
      if (code[redirectIndex] === '<' || code[redirectIndex] === '>') {
        index = redirectIndex;
        continue;
      }
    }
    if (code.slice(index, index + 3) === '$((') {
      const arithmeticEnd = findBashArithmeticExpansionEnd(code, index + 3);
      if (arithmeticEnd == null) {
        index = code.length;
      } else {
        tokens.push(
          ...tokenizeBashArithmeticSubstitutions(code, index + 3, arithmeticEnd)
        );
        index = arithmeticEnd + 1;
      }
      continue;
    }
    if (code.slice(index, index + 2) === '$(') {
      const bodyStart = index + 2;
      const bodyEnd = findBashCommandSubstitutionEnd(code, bodyStart);
      if (bodyEnd == null) {
        index = code.length;
      } else {
        tokens.push({
          type: 'nested',
          tokens: tokenizeBash(code.slice(bodyStart, bodyEnd)),
        });
        index = bodyEnd + 1;
      }
      continue;
    }
    if (char === '`') {
      const bodyEnd = findBashBacktickEnd(code, index + 1);
      if (bodyEnd == null) {
        index = code.length;
      } else {
        tokens.push({
          type: 'nested',
          tokens: tokenizeBash(code.slice(index + 1, bodyEnd)),
        });
        index = bodyEnd + 1;
      }
      continue;
    }
    if (char === '(') {
      tokens.push({ type: 'separator' });
      index += 1;
      continue;
    }
    if (char === ')') {
      tokens.push({ type: 'separator' });
      index += 1;
      continue;
    }
    if (char === '<' || char === '>') {
      tokens.push({ type: 'redirect' });
      index += code[index + 1] === char ? 2 : 1;
      continue;
    }
    if (char === '#') {
      while (index < code.length && code[index] !== '\n') {
        index += 1;
      }
      continue;
    }

    let value = '';
    while (index < code.length) {
      const current = code[index];
      if (/\s/.test(current) || ';|&()<>'.includes(current)) {
        break;
      }
      if (code.slice(index, index + 2) === '$(') {
        break;
      }
      if (current === '`') {
        break;
      }
      if (current === '\\' && index + 1 < code.length) {
        value += code[index + 1];
        index += 2;
        continue;
      }
      if (current === '\'' || current === '"') {
        const quote = current;
        index += 1;
        while (index < code.length && code[index] !== quote) {
          if (quote === '"' && code.slice(index, index + 3) === '$((') {
            const arithmeticEnd = findBashArithmeticExpansionEnd(
              code,
              index + 3
            );
            if (arithmeticEnd == null) {
              index = code.length;
            } else {
              tokens.push(
                ...tokenizeBashArithmeticSubstitutions(
                  code,
                  index + 3,
                  arithmeticEnd
                )
              );
              index = arithmeticEnd + 1;
            }
            continue;
          }
          if (quote === '"' && code.slice(index, index + 2) === '$(') {
            if (value !== '') {
              tokens.push({ type: 'word', value });
              value = '';
            }
            const bodyStart = index + 2;
            const bodyEnd = findBashCommandSubstitutionEnd(code, bodyStart);
            if (bodyEnd == null) {
              index = code.length;
            } else {
              tokens.push({
                type: 'nested',
                tokens: tokenizeBash(code.slice(bodyStart, bodyEnd)),
              });
              index = bodyEnd + 1;
            }
            continue;
          }
          if (quote === '"' && code[index] === '`') {
            if (value !== '') {
              tokens.push({ type: 'word', value });
              value = '';
            }
            const bodyEnd = findBashBacktickEnd(code, index + 1);
            if (bodyEnd == null) {
              index = code.length;
            } else {
              tokens.push({
                type: 'nested',
                tokens: tokenizeBash(code.slice(index + 1, bodyEnd)),
              });
              index = bodyEnd + 1;
            }
            continue;
          }
          if (
            quote === '"' &&
            code[index] === '\\' &&
            index + 1 < code.length
          ) {
            value += code[index + 1];
            index += 2;
          } else {
            value += code[index++];
          }
        }
        if (code[index] === quote) {
          index += 1;
        }
        continue;
      }
      value += current;
      index += 1;
    }
    if (value !== '') {
      tokens.push({ type: 'word', value });
    }
  }

  return tokens;
}

/** Finds the matching close parenthesis for a `$(` command substitution. */
function findBashCommandSubstitutionEnd(
  code: string,
  bodyStart: number
): number | undefined {
  let groupedParentheses = 0;
  let caseDepth = 0;
  let quote: '\'' | '"' | undefined;

  for (let index = bodyStart; index < code.length; index++) {
    const char = code[index];
    if (char === '\\') {
      index += 1;
      continue;
    }
    if (quote != null) {
      if (char === quote) {
        quote = undefined;
      }
      continue;
    }
    if (char === '\'' || char === '"') {
      quote = char;
      continue;
    }
    if (char === '`') {
      const backtickEnd = findBashBacktickEnd(code, index + 1);
      if (backtickEnd == null) {
        return undefined;
      }
      index = backtickEnd;
      continue;
    }
    if (
      char === '#' &&
      (index === bodyStart || /[\s;|&()]/.test(code[index - 1]))
    ) {
      while (index < code.length && code[index] !== '\n') {
        index += 1;
      }
      continue;
    }
    if (code.slice(index, index + 3) === '$((') {
      const arithmeticEnd = findBashArithmeticExpansionEnd(code, index + 3);
      if (arithmeticEnd == null) {
        return undefined;
      }
      index = arithmeticEnd;
      continue;
    }
    if (code.slice(index, index + 2) === '$(') {
      const nestedEnd = findBashCommandSubstitutionEnd(code, index + 2);
      if (nestedEnd == null) {
        return undefined;
      }
      index = nestedEnd;
      continue;
    }
    if (/[A-Za-z_]/.test(char)) {
      let wordEnd = index + 1;
      while (wordEnd < code.length && /[A-Za-z0-9_]/.test(code[wordEnd])) {
        wordEnd += 1;
      }
      const word = code.slice(index, wordEnd);
      if (word === 'case' && isBashKeywordPosition(code, index, bodyStart)) {
        caseDepth += 1;
      } else if (
        word === 'esac' &&
        caseDepth > 0 &&
        isBashKeywordPosition(code, index, bodyStart)
      ) {
        caseDepth -= 1;
      }
      index = wordEnd - 1;
      continue;
    }
    if (char === '(') {
      groupedParentheses += 1;
      continue;
    }
    if (char === ')') {
      if (groupedParentheses > 0) {
        groupedParentheses -= 1;
      } else if (caseDepth === 0) {
        return index;
      }
    }
  }

  return undefined;
}

/** Finds the closing delimiter for a legacy backtick command substitution. */
function findBashBacktickEnd(
  code: string,
  bodyStart: number
): number | undefined {
  for (let index = bodyStart; index < code.length; index++) {
    if (code[index] === '\\') {
      index += 1;
    } else if (code[index] === '`') {
      return index;
    }
  }
  return undefined;
}

/** Masks heredoc literals while preserving executable unquoted substitutions. */
function maskBashHeredocBodies(code: string): string {
  const masked = [...code];
  let lineStart = 0;

  while (lineStart < code.length) {
    const lineEnd = code.indexOf('\n', lineStart);
    const contentEnd = lineEnd === -1 ? code.length : lineEnd;
    const declaration = findBashHeredocDeclaration(
      code.slice(lineStart, contentEnd)
    );
    if (declaration == null || lineEnd === -1) {
      lineStart = lineEnd === -1 ? code.length : lineEnd + 1;
      continue;
    }

    const bodyStart = lineEnd + 1;
    let bodyLineStart = bodyStart;
    let delimiterStart = code.length;
    let afterDelimiter = code.length;
    while (bodyLineStart <= code.length) {
      const bodyLineEnd = code.indexOf('\n', bodyLineStart);
      const bodyContentEnd =
        bodyLineEnd === -1 ? code.length : bodyLineEnd;
      const bodyLine = code.slice(bodyLineStart, bodyContentEnd);
      const comparableLine = declaration.stripTabs
        ? bodyLine.replace(/^\t+/, '')
        : bodyLine;
      if (comparableLine === declaration.delimiter) {
        delimiterStart = bodyLineStart;
        afterDelimiter = bodyLineEnd === -1 ? code.length : bodyLineEnd + 1;
        break;
      }
      bodyLineStart = bodyLineEnd === -1 ? code.length + 1 : bodyLineEnd + 1;
    }

    for (let index = bodyStart; index < afterDelimiter; index++) {
      if (masked[index] !== '\n' && masked[index] !== '\r') {
        masked[index] = ' ';
      }
    }
    if (!declaration.quoted) {
      restoreBashHeredocSubstitutions(
        code,
        masked,
        bodyStart,
        delimiterStart
      );
    }
    lineStart = afterDelimiter;
  }

  return masked.join('');
}

function restoreBashHeredocSubstitutions(
  code: string,
  masked: string[],
  bodyStart: number,
  bodyEnd: number
): void {
  for (let index = bodyStart; index < bodyEnd; index++) {
    if (isBashCharacterEscaped(code, index, bodyStart)) {
      continue;
    }
    let substitutionEnd: number | undefined;
    if (code.slice(index, index + 2) === '$(') {
      substitutionEnd = findBashCommandSubstitutionEnd(code, index + 2);
    } else if (code[index] === '`') {
      substitutionEnd = findBashBacktickEnd(code, index + 1);
    }
    if (substitutionEnd == null || substitutionEnd >= bodyEnd) {
      continue;
    }
    for (let restore = index; restore <= substitutionEnd; restore++) {
      masked[restore] = code[restore];
    }
    index = substitutionEnd;
  }
}

function isBashCharacterEscaped(
  code: string,
  index: number,
  lowerBound: number
): boolean {
  let backslashes = 0;
  for (let cursor = index - 1; cursor >= lowerBound; cursor--) {
    if (code[cursor] !== '\\') {
      break;
    }
    backslashes += 1;
  }
  return backslashes % 2 === 1;
}

function findBashHeredocDeclaration(
  line: string
): { delimiter: string; stripTabs: boolean; quoted: boolean } | undefined {
  let quote: '\'' | '"' | undefined;

  for (let index = 0; index < line.length; index++) {
    const char = line[index];
    if (char === '\\') {
      index += 1;
      continue;
    }
    if (quote != null) {
      if (char === quote) {
        quote = undefined;
      }
      continue;
    }
    if (char === '\'' || char === '"') {
      quote = char;
      continue;
    }
    if (char === '#' && (index === 0 || /\s/.test(line[index - 1]))) {
      return undefined;
    }
    if (line.slice(index, index + 2) !== '<<') {
      continue;
    }

    let targetStart = index + 2;
    const stripTabs = line[targetStart] === '-';
    targetStart += stripTabs ? 1 : 0;
    while (/[ \t]/.test(line[targetStart] ?? '')) {
      targetStart += 1;
    }
    const delimiterQuote = line[targetStart];
    if (delimiterQuote === '\'' || delimiterQuote === '"') {
      const targetEnd = line.indexOf(delimiterQuote, targetStart + 1);
      if (targetEnd === -1) {
        continue;
      }
      return {
        delimiter: line.slice(targetStart + 1, targetEnd),
        stripTabs,
        quoted: true,
      };
    }
    let targetEnd = targetStart;
    while (
      targetEnd < line.length &&
      !/[ \t;&|()<>]/.test(line[targetEnd])
    ) {
      targetEnd += 1;
    }
    if (targetEnd > targetStart) {
      return {
        delimiter: line.slice(targetStart, targetEnd),
        stripTabs,
        quoted: false,
      };
    }
  }

  return undefined;
}

/** Returns whether a reserved word begins where a shell command may begin. */
function isBashKeywordPosition(
  code: string,
  wordStart: number,
  bodyStart: number
): boolean {
  let index = wordStart - 1;
  while (index >= bodyStart && /[ \t\r]/.test(code[index])) {
    index -= 1;
  }

  if (index < bodyStart || /[\n;|&(){}]/.test(code[index])) {
    return true;
  }

  if (!/[A-Za-z0-9_]/.test(code[index])) {
    return false;
  }
  const wordEnd = index + 1;
  while (index >= bodyStart && /[A-Za-z0-9_]/.test(code[index])) {
    index -= 1;
  }
  return COMMAND_START_WORDS.has(code.slice(index + 1, wordEnd));
}

/** Extracts command substitutions nested inside a `$((...))` expansion. */
function tokenizeBashArithmeticSubstitutions(
  code: string,
  bodyStart: number,
  arithmeticEnd: number
): BashToken[] {
  const tokens: BashToken[] = [];
  const bodyEnd = arithmeticEnd - 1;
  let quote: '\'' | '"' | undefined;

  for (let index = bodyStart; index < bodyEnd; index++) {
    const char = code[index];
    if (char === '\\') {
      index += 1;
      continue;
    }
    if (quote === '\'') {
      if (char === quote) {
        quote = undefined;
      }
      continue;
    }
    if (quote === '"' && char === '"') {
      quote = undefined;
      continue;
    }
    if (quote == null && char === '\'') {
      quote = char;
      continue;
    }
    if (quote == null && char === '"') {
      quote = '"';
      continue;
    }
    if (code.slice(index, index + 3) === '$((') {
      const nestedArithmeticEnd = findBashArithmeticExpansionEnd(
        code,
        index + 3
      );
      if (nestedArithmeticEnd == null || nestedArithmeticEnd > arithmeticEnd) {
        break;
      }
      tokens.push(
        ...tokenizeBashArithmeticSubstitutions(
          code,
          index + 3,
          nestedArithmeticEnd
        )
      );
      index = nestedArithmeticEnd;
      continue;
    }
    if (code.slice(index, index + 2) !== '$(') {
      continue;
    }

    const nestedEnd = findBashCommandSubstitutionEnd(code, index + 2);
    if (nestedEnd == null || nestedEnd >= arithmeticEnd) {
      break;
    }
    tokens.push({
      type: 'nested',
      tokens: tokenizeBash(code.slice(index + 2, nestedEnd)),
    });
    index = nestedEnd;
  }

  return tokens;
}

/** Finds the second close parenthesis terminating a `$((...))` expansion. */
function findBashArithmeticExpansionEnd(
  code: string,
  bodyStart: number
): number | undefined {
  let groupedParentheses = 0;

  for (let index = bodyStart; index < code.length; index++) {
    const char = code[index];
    if (char === '\\') {
      index += 1;
      continue;
    }
    if (char === '(') {
      groupedParentheses += 1;
      continue;
    }
    if (char !== ')') {
      continue;
    }
    if (groupedParentheses > 0) {
      groupedParentheses -= 1;
      continue;
    }
    if (code[index + 1] === ')') {
      return index + 1;
    }
  }

  return undefined;
}

/** Rejects direct-only tools before any bash sandbox request is made. */
export function assertBashToolsAllowProgrammaticCalling(
  toolDefs: t.LCTool[] | undefined,
  code: string,
  programmaticToolName: string = Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
  allowedToolDefs?: t.LCTool[]
): void {
  if (toolDefs == null || toolDefs.length === 0) {
    return;
  }

  const toolNameMap = new Map<string, string>();
  const allowedNames = new Set(
    allowedToolDefs?.map((toolDef) => normalizeToBashIdentifier(toolDef.name)) ??
      []
  );
  for (const toolDef of toolDefs) {
    const normalizedName = normalizeToBashIdentifier(toolDef.name);
    if (!allowedNames.has(normalizedName)) {
      toolNameMap.set(normalizedName, toolDef.name);
    }
  }

  assertDisallowedToolUsage(
    extractUsedBashToolNames(code, toolNameMap),
    programmaticToolName
  );
}

/**
 * Filters tool definitions to only include tools actually used in the bash code.
 */
export function filterBashToolsByUsage(
  toolDefs: t.LCTool[],
  code: string,
  debug = false
): t.LCTool[] {
  const toolNameMap = new Map<string, string>();
  for (const def of toolDefs) {
    const bashName = normalizeToBashIdentifier(def.name);
    toolNameMap.set(bashName, def.name);
  }

  const usedToolNames = extractUsedBashToolNames(code, toolNameMap);

  if (debug) {
    // eslint-disable-next-line no-console
    console.log(
      `[BashPTC Debug] Tool filtering: found ${usedToolNames.size}/${toolDefs.length} tools in code`
    );
    if (usedToolNames.size > 0) {
      // eslint-disable-next-line no-console
      console.log(
        `[BashPTC Debug] Matched tools: ${Array.from(usedToolNames).join(', ')}`
      );
    }
  }

  if (usedToolNames.size === 0) {
    if (debug) {
      // eslint-disable-next-line no-console
      console.log(
        '[BashPTC Debug] No tools detected in code - sending all tools as fallback'
      );
    }
    return toolDefs;
  }

  return toolDefs.filter((def) => usedToolNames.has(def.name));
}

// ============================================================================
// Tool Factory
// ============================================================================

/**
 * Creates a Bash Programmatic Tool Calling tool for multi-tool orchestration.
 *
 * This tool enables AI agents to write bash scripts that orchestrate multiple
 * tool calls programmatically via the remote Code API, reducing LLM round-trips.
 *
 * The tool map must be provided at runtime via config.toolCall (injected by ToolNode).
 */
export function createBashProgrammaticToolCallingTool(
  initParams: t.BashProgrammaticToolCallingParams = {}
): DynamicStructuredTool {
  const baseUrl = initParams.baseUrl ?? getCodeBaseURL();
  const maxRoundTrips = initParams.maxRoundTrips ?? DEFAULT_MAX_ROUND_TRIPS;
  const maxRunTimeoutMs = resolveCodeApiRunTimeoutMs(initParams.runTimeoutMs);
  const proxy = initParams.proxy ?? process.env.PROXY;
  const debug = initParams.debug ?? process.env.BASH_PTC_DEBUG === 'true';
  const EXEC_ENDPOINT = buildCodeApiEndpoint(baseUrl, 'exec/programmatic');

  return tool(
    async (rawParams, config) => {
      const params = rawParams as { code: string; timeout?: number };
      const { code } = params;
      const timeout = clampCodeApiRunTimeoutMs(params.timeout, maxRunTimeoutMs);

      const toolCall = (config.toolCall ?? {}) as ToolCall &
        Partial<t.ProgrammaticCache> & {
          session_id?: string;
          _injected_files?: t.CodeEnvFile[];
          _runtime_session_hint?: string;
        };
      const {
        toolMap,
        toolDefs,
        disallowedToolDefs,
        session_id,
        _injected_files,
        _runtime_session_hint,
      } = toolCall;

      assertBashToolsAllowProgrammaticCalling(
        disallowedToolDefs,
        code,
        toolCall.programmaticToolName ??
          (typeof toolCall.name === 'string' && toolCall.name !== ''
            ? toolCall.name
            : Constants.BASH_PROGRAMMATIC_TOOL_CALLING),
        toolDefs
      );

      if (toolMap == null || toolMap.size === 0) {
        throw new Error(
          'No toolMap provided. ' +
            'ToolNode should inject this from AgentContext when invoked through the graph.'
        );
      }

      if (toolDefs == null || toolDefs.length === 0) {
        throw new Error(
          'No tool definitions provided. ' +
            'Either pass tools in the input or ensure ToolNode injects toolDefs.'
        );
      }

      let roundTrip = 0;

      try {
        // ====================================================================
        // Phase 1: Filter tools and make initial request
        // ====================================================================

        const effectiveTools = filterBashToolsByUsage(toolDefs, code, debug);

        if (debug) {
          // eslint-disable-next-line no-console
          console.log(
            `[BashPTC Debug] Sending ${effectiveTools.length} tools to API ` +
              `(filtered from ${toolDefs.length})`
          );
        }

        /* `/files/<session_id>` HTTP fallback removed — codeapi's
         * sessionAuth requires kind/id query params unavailable at
         * this point. See `CodeExecutor.ts` for full rationale. */
        let files: t.CodeEnvFile[] | undefined;
        if (_injected_files && _injected_files.length > 0) {
          files = _injected_files;
        } else if (session_id != null && session_id.length > 0) {
          // eslint-disable-next-line no-console
          console.debug(
            `[BashProgrammaticToolCalling] No injected files for session_id=${session_id} — exec will run without input files`
          );
        }

        /* The hint rides the INITIAL request only; continuation_token binds
         * later round-trips. Prefer trusted per-agent factory context over
         * legacy ToolNode injection. Explicit default profiles always drop it.
         * BashPTC keeps its stateless runtime prompt in v1. */
        const selectedRuntimeSessionHint = selectRuntimeSessionHint(
          initParams.runtimeSessionHint,
          _runtime_session_hint
        );
        const runtimeSessionHint =
          initParams.executionProfile !== 'default' &&
          typeof selectedRuntimeSessionHint === 'string' &&
          selectedRuntimeSessionHint !== ''
            ? selectedRuntimeSessionHint
            : undefined;

        let response = await makeRequest(
          EXEC_ENDPOINT,
          {
            lang: 'bash',
            code,
            tools: effectiveTools,
            session_id,
            timeout,
            ...(files && files.length > 0 ? { files } : {}),
            ...(runtimeSessionHint != null
              ? { runtime_session_hint: runtimeSessionHint }
              : {}),
          },
          proxy,
          initParams.authHeaders,
          initParams.executionProfile
        );

        // ====================================================================
        // Phase 2: Handle response loop
        // ====================================================================

        while (response.status === 'tool_call_required') {
          roundTrip++;

          if (roundTrip > maxRoundTrips) {
            throw new Error(
              `Exceeded maximum round trips (${maxRoundTrips}). ` +
                'This may indicate an infinite loop, excessive tool calls, ' +
                'or a logic error in your code.'
            );
          }

          if (debug) {
            // eslint-disable-next-line no-console
            console.log(
              `[BashPTC Debug] Round trip ${roundTrip}: ${response.tool_calls?.length ?? 0} tool(s) to execute`
            );
          }

          const toolResults = normalizeBashToolResultsForReplay(
            await executeTools(
              response.tool_calls ?? [],
              toolMap,
              Constants.BASH_PROGRAMMATIC_TOOL_CALLING
            )
          );

          response = await makeRequest(
            EXEC_ENDPOINT,
            {
              continuation_token: response.continuation_token,
              tool_results: toolResults,
            },
            proxy,
            initParams.authHeaders,
            initParams.executionProfile
          );
        }

        // ====================================================================
        // Phase 3: Handle final state
        // ====================================================================

        if (response.status === 'completed') {
          return formatCompletedResponse(response, code);
        }

        if (response.status === 'error') {
          throw new Error(buildCodeApiExecutionErrorMessage(response));
        }

        throw new CodeApiRequestError();
      } catch (error) {
        const messageWithReminder = appendFailedExecutionFileReminder(
          (error as Error).message,
          code
        );
        throw new Error(
          `Bash programmatic execution failed: ${messageWithReminder}`
        );
      }
    },
    {
      name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      description: BashProgrammaticToolCallingDescription,
      schema: createBashProgrammaticToolCallingSchema(maxRunTimeoutMs),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
