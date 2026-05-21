import { config } from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import type * as t from '@/types';
import { appendCodeSessionFileSummary } from '@/tools/CodeSessionFileSummary';
import { EnvVar, Constants } from '@/common';

export {
  appendCodeSessionFileSummary,
  stripCodeSessionFileSummary,
} from '@/tools/CodeSessionFileSummary';

config();

export const getCodeBaseURL = (): string =>
  getEnvironmentVariable(EnvVar.CODE_BASEURL) ??
  Constants.OFFICIAL_CODE_BASEURL;

export const emptyOutputMessage =
  'stdout: Empty. Ensure you\'re writing output explicitly.\n';

export const CODE_ARTIFACT_PATH_GUIDANCE =
  'Persist handoff artifacts in `/mnt/data` with standard extensions (.json/.txt/.csv/.tsv/.log/.parquet/.png/.jpg/.pdf/.xlsx); failed executions do not register new files; `/tmp` and odd extensions are same-call scratch only, not later-call storage.';

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

const baseEndpoint = getCodeBaseURL();
const EXEC_ENDPOINT = `${baseEndpoint}/exec`;

type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

export async function resolveCodeApiAuthHeaders(
  authHeaders?: t.CodeApiAuthHeaders
): Promise<t.CodeApiAuthHeaderMap> {
  if (authHeaders == null) {
    return {};
  }
  if (typeof authHeaders === 'function') {
    return authHeaders();
  }
  return authHeaders;
}

export async function buildCodeApiHttpErrorMessage(
  method: string,
  endpoint: string,
  response: { status: number; text: () => Promise<string> }
): Promise<string> {
  let responseBody = '';
  try {
    responseBody = await response.text();
  } catch {
    responseBody = '';
  }
  const body = responseBody.trim();
  const bodySuffix = body === '' ? '' : `, body: ${body.slice(0, 1000)}`;
  return `CodeAPI request failed: ${method} ${endpoint} returned ${response.status}${bodySuffix}`;
}

export const CodeExecutionToolDescription = `
Runs code and returns stdout/stderr output from a stateless execution environment, similar to running scripts in a command-line interface. Each execution is isolated and independent.

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- NEVER use this tool to execute malicious code.
`.trim();

export const CodeExecutionToolName = Constants.EXECUTE_CODE;

export const CodeExecutionToolDefinition = {
  name: CodeExecutionToolName,
  description: CodeExecutionToolDescription,
  schema: CodeExecutionToolSchema,
} as const;

function createCodeExecutionTool(
  params: t.CodeExecutionToolParams | null = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput, config) => {
      const { authHeaders, ...executionParams } = params ?? {};
      const { lang, code, ...rest } = rawInput as {
        lang: SupportedLanguage;
        code: string;
        args?: string[];
      };
      /**
       * Extract session context from config.toolCall (injected by ToolNode).
       * - session_id: associates with the previous run.
       * - _injected_files: File refs to pass directly (avoids /files endpoint race condition).
       */
      const { session_id, _injected_files } = (config.toolCall ?? {}) as {
        session_id?: string;
        _injected_files?: t.CodeEnvFile[];
      };

      const postData: Record<string, unknown> = {
        lang,
        code,
        ...rest,
        ...executionParams,
      };

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
            ...resolvedAuthHeaders,
          },
          body: JSON.stringify(postData),
        };

        if (process.env.PROXY != null && process.env.PROXY !== '') {
          fetchOptions.agent = new HttpsProxyAgent(process.env.PROXY);
        }
        const response = await fetch(EXEC_ENDPOINT, fetchOptions);
        if (!response.ok) {
          throw new Error(
            await buildCodeApiHttpErrorMessage('POST', EXEC_ENDPOINT, response)
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
        return [
          appendCodeSessionFileSummary(outputWithReminder, result.files),
          (hasFiles
            ? { session_id: result.session_id, files: result.files }
            : {
              session_id: result.session_id,
            }) satisfies t.CodeExecutionArtifact,
        ];
      } catch (error) {
        const messageWithReminder = appendFailedExecutionFileReminder(
          (error as Error | undefined)?.message ?? '',
          code
        );
        throw new Error(
          `Execution error:\n\n${messageWithReminder}`
        );
      }
    },
    {
      name: CodeExecutionToolName,
      description: CodeExecutionToolDescription,
      schema: CodeExecutionToolSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export { createCodeExecutionTool };
