import type * as t from '@/types';

/**
 * Operator diagnostics for the Code API tools.
 *
 * Every failure these tools surface to the model is reduced to a fixed
 * sentence, so this log is the only surviving account of what went wrong —
 * which also makes it the one place a credential can escape by accident. It
 * escaped repeatedly while any field was quoted from something the tools did
 * not produce: a response body echoing the header that was sent, an error
 * message quoting an excerpt of a malformed signing key, a `stack` whose first
 * line is that message, a writable `name`, a configured base URL carrying a
 * capability in its path or its authority.
 *
 * Filtering those carriers case by case never converged, because the shape of
 * a credential inside free text is not decidable. The rule here is structural
 * instead: a diagnostic NAMES what happened using values this module owns, and
 * QUOTES nothing it received. `CodeApiDiagnosticDetail` is a closed union of
 * such values, so host-authored text cannot reach a log without failing to
 * compile — and adding a field means widening that union in this file, which
 * is the point at which the question gets asked.
 */

export type CodeApiMethod = 'GET' | 'POST';

type CodeApiProfileLabel = t.CodeApiExecutionProfile | 'unset';

const KNOWN_ERROR_TYPES: ReadonlyArray<
  readonly [CodeApiErrorLabel, new () => Error]
> = [
  ['SyntaxError', SyntaxError],
  ['TypeError', TypeError],
  ['RangeError', RangeError],
  ['ReferenceError', ReferenceError],
  ['URIError', URIError],
  ['EvalError', EvalError],
];

/** Built-in error types, the `typeof` results for a non-Error rejection, and
 * the two labels this module supplies when neither applies. */
type CodeApiErrorLabel =
  | 'SyntaxError'
  | 'TypeError'
  | 'RangeError'
  | 'ReferenceError'
  | 'URIError'
  | 'EvalError'
  | 'Error'
  | 'UndescribableError'
  | 'string'
  | 'number'
  | 'bigint'
  | 'boolean'
  | 'symbol'
  | 'undefined'
  | 'object'
  | 'function';

export type CodeApiDiagnosticDetail =
  | { type: CodeApiErrorLabel }
  | { method: CodeApiMethod; profile: CodeApiProfileLabel; status: number }
  | { files: 'none' };

/**
 * Classifies a rejection without reading anything off it. `instanceof` walks
 * the prototype chain, so an accessor trap is never reached; the guard covers
 * a `getPrototypeOf` trap, which would otherwise cost both the log and the
 * rejection.
 */
export function describeCodeApiError(error: unknown): {
  type: CodeApiErrorLabel;
} {
  try {
    if (!(error instanceof Error)) {
      return { type: typeof error };
    }
    for (const [label, constructor] of KNOWN_ERROR_TYPES) {
      if (error instanceof constructor) {
        return { type: label };
      }
    }
    return { type: 'Error' };
  } catch {
    return { type: 'UndescribableError' };
  }
}

type CodeApiDiagnosticSource =
  | 'CodeExecutor'
  | 'BashExecutor'
  | 'ProgrammaticToolCalling'
  | 'BashProgrammaticToolCalling';

/**
 * Console is the SDK's diagnostic channel; a winston logger is the host's
 * concern. Hosts correlate these lines with their own request-scoped logs,
 * which is where received identifiers such as a session id belong.
 */
export function logCodeApiDiagnostic(
  source: CodeApiDiagnosticSource,
  level: 'debug' | 'warn' | 'error',
  message: string,
  detail: CodeApiDiagnosticDetail
): void {
  // eslint-disable-next-line no-console
  console[level](`[${source}] ${message}`, detail);
}
