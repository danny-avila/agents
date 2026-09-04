import { config } from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import { appendCodeSessionFileSummary } from '@/tools/CodeSessionFileSummary';
import { resolveFetchProxyAgent } from '@/utils/proxy';
import { redactSecrets } from '@/utils/redactSecrets';
import { INTENT_PROPERTY } from '@/tools/intentArg';
import { EnvVar, Constants } from '@/common';

export {
  appendCodeSessionFileSummary,
  stripCodeSessionFileSummary,
} from '@/tools/CodeSessionFileSummary';

config();

export const getCodeBaseURL = (): string =>
  getEnvironmentVariable(EnvVar.CODE_BASEURL) ??
  Constants.OFFICIAL_CODE_BASEURL;

export function buildCodeApiEndpoint(baseUrl: string, route: string): string {
  return `${baseUrl.replace(/\/+$/, '')}/${route.replace(/^\/+/, '')}`;
}

export function selectRuntimeSessionHint(
  factoryHint?: string,
  injectedHint?: string
): string | undefined {
  return factoryHint != null && factoryHint !== '' ? factoryHint : injectedHint;
}

export const emptyOutputMessage =
  'stdout: Empty. Ensure you\'re writing output explicitly.\n';

export const CODE_ARTIFACT_PATH_GUIDANCE =
  'Anything a later call needs (data, helper scripts/modules, partial results) MUST be written under `/mnt/data` in the same call that produces it; `/tmp` never survives the call. `/mnt/data` keeps files with recognized extensions, covering common source, text, data, document, image, and archive formats (.py/.sh/.sql/.md/.json/.csv/.parquet/.png/.pdf/.zip and similar); extensionless or unusual extensions are not kept. Failed executions register nothing; fix the error and rerun before relying on new files.';

export const BASH_SHELL_GUIDANCE =
  'Bash: multi-line files use heredoc/printf; run Python via python3 -c/heredoc, not bare Python.';

const TMP_PATH_PATTERN = /(^|[^A-Za-z0-9_])\/tmp(?:\/|\b)/;
const MNT_DATA_PATH_PATTERN = /(^|[^A-Za-z0-9_])\/mnt\/data(?:\/|\b)/;

export const TMP_SCRATCH_OUTPUT_REMINDER =
  'Note: /tmp files are same-call scratch only and were not persisted; use /mnt/data for files needed later.';

export const FAILED_EXECUTION_FILE_REMINDER =
  'Note: any files written during this failed call were not registered for later calls; fix the error and rerun before relying on them.';

export function appendTmpScratchReminder(output: string, code: string): string {
  if (!TMP_PATH_PATTERN.test(code)) {
    return output;
  }
  return `${output.trimEnd()}\n${TMP_SCRATCH_OUTPUT_REMINDER}\n`;
}

export function appendFailedExecutionFileReminder(
  output: string,
  code: string
): string {
  if (
    !MNT_DATA_PATH_PATTERN.test(code) ||
    output.includes(FAILED_EXECUTION_FILE_REMINDER)
  ) {
    return output;
  }
  return `${output.trimEnd()}\n${FAILED_EXECUTION_FILE_REMINDER}\n`;
}

const SUPPORTED_LANGUAGES = [
  'py',
  'js',
  'ts',
  'c',
  'cpp',
  'java',
  'php',
  'rs',
  'go',
  'd',
  'f90',
  'r',
  'bash',
] as const;

export const CodeExecutionToolSchema = {
  type: 'object',
  properties: {
    intent: { ...INTENT_PROPERTY },
    lang: {
      type: 'string',
      enum: SUPPORTED_LANGUAGES,
      description:
        'The programming language or runtime to execute the code in.',
    },
    code: {
      type: 'string',
      description: `The complete, self-contained code to execute, without any truncation or minimization.
- The environment is stateless; variables and imports don't persist between executions.
- Prior /mnt/data files are available and can be modified in place.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- Input code **IS ALREADY** displayed to the user, so **DO NOT** repeat it in your response unless asked.
- Output code **IS NOT** displayed to the user, so **DO** write all desired output explicitly.
- IMPORTANT: You MUST explicitly print/output ALL results you want the user to see.
- py: This is not a Jupyter notebook environment. Use \`print()\` for all outputs.
- py: Matplotlib: Use \`plt.savefig()\` to save plots as files.
- js: use the \`console\` or \`process\` methods for all outputs.
- r: IMPORTANT: No X11 display available. ALL graphics MUST use Cairo library (library(Cairo)).
- Other languages: use appropriate output functions.`,
    },
    args: {
      type: 'array',
      items: { type: 'string' },
      description:
        'Additional arguments to execute the code with. This should only be used if the input code requires additional arguments to run.',
    },
  },
  required: ['lang', 'code'],
} as const;

export const CODE_API_EXPECTED_PROFILE_HEADER = 'X-CodeAPI-Expected-Profile';

type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

const MAX_RETRY_AFTER_SECONDS = 3600;

export const CODE_API_UNAVAILABLE_ERROR_MESSAGE =
  'Code execution is temporarily unavailable. Please retry.';
export const CODE_API_AUTHORIZATION_ERROR_MESSAGE =
  'Code execution is not authorized. Verify access before trying again.';
export const CODE_API_EXECUTION_FAILED_ERROR_MESSAGE = 'Code execution failed.';
export const CODE_API_INVALID_REQUEST_ERROR_MESSAGE =
  'The code execution request was rejected. Please check the tool input and try again.';
export const CODE_API_RATE_LIMITED_ERROR_MESSAGE =
  'Code execution is temporarily rate-limited. Please retry shortly.';

const MAX_LOGGED_STACK_FRAMES = 10;

type CodeApiErrorDiagnostic = {
  name: string;
  frames?: string;
};

type CodeApiRejectionDiagnostic = {
  method: string;
  host: string;
  status: number;
};

/** Error names are class identifiers; anything else is host-authored text. */
const ERROR_NAME_PATTERN = /^[A-Za-z_$][A-Za-z0-9_$]{0,63}$/;

/**
 * Names the failing error type and the code path that produced it, and nothing
 * else. An error's message is host-authored free text that can carry the very
 * credential this module exists to keep out of reach — a malformed signing key,
 * for instance, surfaces as a `SyntaxError` whose message quotes an excerpt of
 * the key source. `name` is kept only when it is shaped like the class
 * identifier it is supposed to be, and stack frames are code locations rather
 * than payload; the leading line is dropped because V8 begins a stack with the
 * message. Both fields are still read from the rejection, so a host that
 * deliberately forges a stack can put text in `frames` — that is the same trust
 * boundary as a host that throws its credential in the first place, and it is
 * the reason nothing here is claimed to be secret-free by construction. Reads
 * are guarded because an exotic rejection can throw from an accessor, which
 * would otherwise lose the log and the rejection both.
 */
function describeCodeApiError(error: unknown): CodeApiErrorDiagnostic {
  try {
    if (!(error instanceof Error)) {
      return { name: typeof error };
    }
    const name = error.name;
    const frames = error.stack
      ?.split('\n')
      .filter((line) => /^\s+at /.test(line))
      .slice(0, MAX_LOGGED_STACK_FRAMES)
      .join('\n');
    return {
      name:
        typeof name === 'string' && ERROR_NAME_PATTERN.test(name)
          ? name
          : 'UnnamedError',
      ...(frames != null && frames !== '' ? { frames } : {}),
    };
  } catch {
    return { name: 'UndescribableError' };
  }
}

/**
 * Only the authority is logged. A configured base URL can carry a capability in
 * its path, and the files route puts a session id in one, so the full endpoint
 * is host text; the authority is what actually answers the question a rejection
 * raises — which backend refused this call.
 */
function describeEndpointHost(endpoint: string): string {
  try {
    return new URL(endpoint).host;
  } catch {
    return 'unparsed';
  }
}

/**
 * The messages above are deliberately detail-free so infrastructure never
 * reaches the model, which also means a failure's cause survives nowhere else.
 * This is the operator's only copy: console is the SDK's diagnostic channel (a
 * winston logger is the host's concern). What is recorded is confined to values
 * this module produced or chose — never upstream or host free text — so the
 * diagnostic cannot become the leak the sanitization prevents. `redactSecrets`
 * still runs, since a configured base URL may embed credentials.
 */
function logCodeApiDiagnostic(
  level: 'warn' | 'error',
  message: string,
  detail: CodeApiErrorDiagnostic | CodeApiRejectionDiagnostic
): void {
  // eslint-disable-next-line no-console
  console[level](`[CodeExecutor] ${message}`, redactSecrets(detail));
}

const SAFE_CODE_API_EXECUTION_ERROR_DETAILS: Readonly<
  Partial<Record<string, string>>
> = {
  'Execution failed or timed out': 'Execution failed or timed out.',
  'Out of memory': 'Execution exceeded the memory limit.',
  'Time limit exceeded': 'Execution exceeded the time limit.',
  'sandbox emitted an empty pending tool call block; aborting to avoid a tight retry loop':
    'Generated code emitted an invalid empty tool request.',
  'stderr length exceeded': 'Execution error output exceeded the size limit.',
  'stdout length exceeded': 'Execution output exceeded the size limit.',
};

export class CodeApiRequestError extends Error {
  constructor(message = CODE_API_UNAVAILABLE_ERROR_MESSAGE) {
    super(message);
    this.name = 'CodeApiRequestError';
  }
}

function getRetryAfterSeconds(responseBody: string): number | undefined {
  try {
    const parsed = JSON.parse(responseBody) as {
      error?: unknown;
      retry_after_seconds?: unknown;
    };
    if (
      parsed.error !== 'rate_limited' ||
      typeof parsed.retry_after_seconds !== 'number' ||
      !Number.isFinite(parsed.retry_after_seconds) ||
      parsed.retry_after_seconds <= 0
    ) {
      return undefined;
    }
    return Math.min(
      Math.ceil(parsed.retry_after_seconds),
      MAX_RETRY_AFTER_SECONDS
    );
  } catch {
    return undefined;
  }
}

export function normalizeCodeApiRequestError(
  error: unknown
): CodeApiRequestError {
  return error instanceof CodeApiRequestError
    ? error
    : new CodeApiRequestError();
}

function getSafeCodeApiExecutionErrorDetail(
  error: unknown
): string | undefined {
  if (typeof error !== 'string') {
    return undefined;
  }
  const exactMatch = SAFE_CODE_API_EXECUTION_ERROR_DETAILS[error];
  if (exactMatch != null) {
    return exactMatch;
  }
  if (/^Sandbox exited with code (?:-?\d{1,4}|unknown)$/.test(error)) {
    return error.replace(/^Sandbox/, 'Execution');
  }
  if (/^Sandbox requested an unregistered tool: .+$/.test(error)) {
    return 'Generated code requested a tool that is not available.';
  }
  return undefined;
}

export function buildCodeApiExecutionErrorMessage(response: {
  error?: unknown;
  stderr?: unknown;
}): string {
  const safeDetail = getSafeCodeApiExecutionErrorDetail(response.error);
  const message =
    safeDetail != null
      ? `${CODE_API_EXECUTION_FAILED_ERROR_MESSAGE} ${safeDetail}`
      : CODE_API_EXECUTION_FAILED_ERROR_MESSAGE;
  if (typeof response.stderr === 'string' && response.stderr !== '') {
    return `${message}\n\nStderr:\n${response.stderr}`;
  }
  return message;
}

export async function resolveCodeApiAuthHeaders(
  authHeaders?: t.CodeApiAuthHeaders
): Promise<t.CodeApiAuthHeaderMap> {
  if (authHeaders == null) {
    return {};
  }
  if (typeof authHeaders === 'function') {
    try {
      const resolvedHeaders = await authHeaders();
      return resolvedHeaders;
    } catch (error) {
      logCodeApiDiagnostic(
        'error',
        'auth header resolution failed; the Code API request was never sent',
        describeCodeApiError(error)
      );
      throw new CodeApiRequestError(CODE_API_AUTHORIZATION_ERROR_MESSAGE);
    }
  }
  return authHeaders;
}

export function addCodeApiExecutionProfileHeader(
  headers: t.CodeApiAuthHeaderMap,
  executionProfile?: t.CodeApiExecutionProfile
): t.CodeApiAuthHeaderMap {
  if (executionProfile == null) {
    return headers;
  }
  return {
    ...headers,
    [CODE_API_EXPECTED_PROFILE_HEADER]: executionProfile,
  };
}

export async function buildCodeApiHttpErrorMessage(
  method: string,
  endpoint: string,
  response: { status: number; text: () => Promise<string> },
  options?: { recoverable?: boolean }
): Promise<string> {
  let responseBody = '';
  try {
    responseBody = await response.text();
  } catch {
    responseBody = '';
  }
  /* The response body is deliberately absent: it is upstream free text that
     can echo the header that was sent, and the status distinguishes the
     failure modes on its own. A recoverable caller reports its own outcome —
     logging here too would raise an error-level alert for a lookup that
     succeeds by returning nothing. */
  if (options?.recoverable !== true) {
    logCodeApiDiagnostic(
      response.status === 429 ? 'warn' : 'error',
      'Code API rejected the request',
      {
        method,
        host: describeEndpointHost(endpoint),
        status: response.status,
      }
    );
  }
  if (response.status === 429) {
    const retryAfterSeconds = getRetryAfterSeconds(responseBody);
    return retryAfterSeconds != null
      ? `Code execution is temporarily rate-limited. Retry after ${retryAfterSeconds} seconds.`
      : CODE_API_RATE_LIMITED_ERROR_MESSAGE;
  }
  if (response.status === 401 || response.status === 403) {
    return CODE_API_AUTHORIZATION_ERROR_MESSAGE;
  }
  if (response.status === 400 || response.status === 422) {
    return CODE_API_INVALID_REQUEST_ERROR_MESSAGE;
  }
  return CODE_API_UNAVAILABLE_ERROR_MESSAGE;
}

export const CodeExecutionToolDescription = `
Runs code and returns stdout/stderr output from a stateless execution environment, similar to running scripts in a command-line interface. Each execution is isolated and independent.

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- NEVER use this tool to execute malicious code.
`.trim();

/**
 * Statefulness here is FILESYSTEM-tier, not runtime-tier. Executions in a
 * session reuse one warm machine, so `/mnt/data` carries across calls — but
 * every execution is a brand-new interpreter process in a fresh sandbox, so
 * variables and imports never survive. The note must not imply otherwise: a
 * model told its in-memory state persists writes `df = ...` in one call and
 * `df.head()` in the next, then hits a NameError it was told to treat as rare.
 */
export const STATEFUL_ENV_NOTE =
  'Session state: executions in this conversation run on the same warm machine, so files persist between calls — but each execution is a NEW process. Variables, imports, and in-memory data NEVER carry over: every call must re-import and rebuild the state it needs. Only /mnt/data is durable (the machine itself may also be reset at any time), so write anything that must survive there and read it back next call.';

export const StatefulCodeExecutionToolDescription = `
Runs code and returns stdout/stderr output. Executions in this conversation share one warm machine with a persistent /mnt/data, but each execution runs as a separate process (not a notebook-style kernel).

${STATEFUL_ENV_NOTE}

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- NEVER use this tool to execute malicious code.
`.trim();

export function buildCodeExecutionToolDescription(opts?: {
  statefulSessions?: boolean;
}): string {
  return opts?.statefulSessions === true
    ? StatefulCodeExecutionToolDescription
    : CodeExecutionToolDescription;
}

const STATELESS_CODE_PARAM_NOTE =
  'The environment is stateless; variables and imports don\'t persist between executions.';
const STATEFUL_CODE_PARAM_NOTE =
  'Executions in this conversation share one warm machine, so files written to /mnt/data persist between calls. Each execution is a new process: variables and imports do NOT carry over — re-import and reload from /mnt/data every call.';

export function buildCodeExecutionToolSchema(opts?: {
  statefulSessions?: boolean;
}): typeof CodeExecutionToolSchema {
  const note =
    opts?.statefulSessions === true
      ? STATEFUL_CODE_PARAM_NOTE
      : STATELESS_CODE_PARAM_NOTE;
  const codeDescription =
    CodeExecutionToolSchema.properties.code.description.replace(
      STATELESS_CODE_PARAM_NOTE,
      note
    );
  return {
    ...CodeExecutionToolSchema,
    properties: {
      ...CodeExecutionToolSchema.properties,
      code: {
        ...CodeExecutionToolSchema.properties.code,
        description: codeDescription,
      },
    },
  } as typeof CodeExecutionToolSchema;
}

export const CodeExecutionToolName = Constants.EXECUTE_CODE;

export const CodeExecutionToolDefinition = {
  name: CodeExecutionToolName,
  description: CodeExecutionToolDescription,
  schema: CodeExecutionToolSchema,
} as const;

function createCodeExecutionTool(
  params: t.CodeExecutionToolParams | null = {}
): DynamicStructuredTool {
  const execEndpoint = buildCodeApiEndpoint(
    params?.baseUrl ?? getCodeBaseURL(),
    'exec'
  );

  return tool(
    async (rawInput, config) => {
      /* `statefulSessions` drives the description and gates trusted runtime
       * affinity hints. Keep the flag itself out of the wire body. */
      const {
        authHeaders,
        baseUrl: _baseUrl,
        executionProfile,
        runtimeSessionHint,
        statefulSessions,
        ...executionParams
      } = params ?? {};
      void _baseUrl;
      /* Drop any model-supplied `runtime_session_hint` from the raw args: the
       * hint is host-controlled and must only ever come from ToolNode's
       * injected `_runtime_session_hint` (below). Spreading `...rest` into
       * postData would otherwise let a tool call opt itself into / pick a
       * stateful runtime even when statefulSessions is off. */
      /* `intent` is a UI display label — never part of the wire body. */
      const {
        lang,
        code,
        intent: _ignoredIntent,
        runtime_session_hint: _ignoredModelHint,
        ...rest
      } = rawInput as {
        lang: SupportedLanguage;
        code: string;
        intent?: unknown;
        runtime_session_hint?: unknown;
        args?: string[];
      };
      void _ignoredModelHint;
      void _ignoredIntent;
      /**
       * Extract session context from config.toolCall (injected by ToolNode).
       * - session_id: associates with the previous run.
       * - _injected_files: File refs to pass directly (avoids /files endpoint race condition).
       */
      const { session_id, _injected_files, _runtime_session_hint } =
        (config.toolCall ?? {}) as {
          session_id?: string;
          _injected_files?: t.CodeEnvFile[];
          _runtime_session_hint?: string;
        };

      const postData: Record<string, unknown> = {
        lang,
        code,
        ...rest,
        ...executionParams,
      };

      /* Stateful sessions: prefer the per-agent factory hint, falling back to
       * legacy ToolNode injection. Explicit default-profile tools always drop
       * the hint so a mixed graph cannot promote them to the stateful path. */
      const effectiveRuntimeSessionHint = selectRuntimeSessionHint(
        runtimeSessionHint,
        _runtime_session_hint
      );
      if (
        statefulSessions === true &&
        executionProfile !== 'default' &&
        typeof effectiveRuntimeSessionHint === 'string' &&
        effectiveRuntimeSessionHint !== ''
      ) {
        postData.runtime_session_hint = effectiveRuntimeSessionHint;
      }

      /* File injection: `_injected_files` from ToolNode (set when host
       * primes a CodeSessionContext) or `params.files` from tool
       * factory (set by hosts that pre-resolve at construction time).
       * The legacy `/files/<session_id>` HTTP fallback was removed —
       * codeapi's `sessionAuth` middleware now requires kind/id query
       * params the tool can't supply at this point, so the fetch 400'd
       * silently and the catch swallowed the failure. */
      if (_injected_files && _injected_files.length > 0) {
        postData.files = _injected_files;
      } else if (
        session_id != null &&
        session_id.length > 0 &&
        !Array.isArray(postData.files)
      ) {
        // eslint-disable-next-line no-console
        console.debug(
          `[CodeExecutor] No injected files for session_id=${session_id} — exec will run without input files`
        );
      }

      try {
        const resolvedAuthHeaders =
          await resolveCodeApiAuthHeaders(authHeaders);
        const fetchOptions: RequestInit = {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'LibreChat/1.0',
            ...addCodeApiExecutionProfileHeader(
              resolvedAuthHeaders,
              executionProfile
            ),
          },
          body: JSON.stringify(postData),
        };

        const proxyAgent = resolveFetchProxyAgent(execEndpoint);
        if (proxyAgent) {
          fetchOptions.agent = proxyAgent;
        }
        const response = await fetch(execEndpoint, fetchOptions);
        if (!response.ok) {
          throw new CodeApiRequestError(
            await buildCodeApiHttpErrorMessage('POST', execEndpoint, response)
          );
        }

        const result: t.ExecuteResult = await response.json();
        let formattedOutput = '';
        if (result.stdout) {
          formattedOutput += `stdout:\n${result.stdout}\n`;
        } else {
          formattedOutput += emptyOutputMessage;
        }
        if (result.stderr) formattedOutput += `stderr:\n${result.stderr}\n`;

        const outputWithReminder = appendTmpScratchReminder(
          formattedOutput,
          code
        );
        const hasFiles = result.files != null && result.files.length > 0;
        /* Echo the durable runtime session (stateful backends only) so hosts
         * can surface a "session active / was reset" signal later. Additive:
         * absent on stateless servers. */
        const runtimeEcho =
          result.runtime_session_id != null
            ? {
              runtime_session_id: result.runtime_session_id,
              runtime_status: result.runtime_status,
            }
            : {};
        return [
          appendCodeSessionFileSummary(outputWithReminder, result.files),
          (hasFiles
            ? {
              session_id: result.session_id,
              files: result.files,
              ...runtimeEcho,
            }
            : {
              session_id: result.session_id,
              ...runtimeEcho,
            }) satisfies t.CodeExecutionArtifact,
        ];
      } catch (error) {
        const messageWithReminder = appendFailedExecutionFileReminder(
          normalizeCodeApiRequestError(error).message,
          code
        );
        throw new Error(`Execution error:\n\n${messageWithReminder}`);
      }
    },
    {
      name: CodeExecutionToolName,
      description: buildCodeExecutionToolDescription(params ?? undefined),
      schema: buildCodeExecutionToolSchema(params ?? undefined),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export { createCodeExecutionTool };
