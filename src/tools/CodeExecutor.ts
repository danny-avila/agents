import { config } from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import type * as t from '@/types';
import { EnvVar, Constants } from '@/common';

config();

export const getCodeBaseURL = (): string =>
  getEnvironmentVariable(EnvVar.CODE_BASEURL) ??
  Constants.OFFICIAL_CODE_BASEURL;

export const emptyOutputMessage =
  'stdout: Empty. Ensure you\'re writing output explicitly.\n';

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
- Generated files from previous executions are automatically available in "/mnt/data/".
- Files from previous executions are automatically available and can be modified in place.
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

export const CodeExecutionToolDescription = `
Runs code and returns stdout/stderr output from a stateless execution environment, similar to running scripts in a command-line interface. Each execution is isolated and independent.

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- NEVER use this tool to execute malicious code.
`.trim();

export const CodeExecutionToolName = Constants.EXECUTE_CODE;

export const CodeExecutionToolDefinition = {
  name: CodeExecutionToolName,
  description: CodeExecutionToolDescription,
  schema: CodeExecutionToolSchema,
} as const;

function createCodeExecutionTool(
  params: t.CodeExecutionToolParams = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput, config) => {
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
        ...params,
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
        const fetchOptions: RequestInit = {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'LibreChat/1.0',
          },
          body: JSON.stringify(postData),
        };

        if (process.env.PROXY != null && process.env.PROXY !== '') {
          fetchOptions.agent = new HttpsProxyAgent(process.env.PROXY);
        }
        const response = await fetch(EXEC_ENDPOINT, fetchOptions);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result: t.ExecuteResult = await response.json();
        /* Output is stdout/stderr only — file listings were removed
         * because the LLM-facing summary (split inherited/generated
         * with prescriptive notes) caused more confusion than help,
         * especially for bash where models naturally explore
         * `/mnt/data/` themselves. The artifact still carries every
         * file so the host's session map stays in sync; the LLM
         * doesn't see them in the tool result text. */
        let formattedOutput = '';
        if (result.stdout) {
          formattedOutput += `stdout:\n${result.stdout}\n`;
        } else {
          formattedOutput += emptyOutputMessage;
        }
        if (result.stderr) formattedOutput += `stderr:\n${result.stderr}\n`;

        const hasFiles = result.files != null && result.files.length > 0;
        return [
          formattedOutput.trim(),
          (hasFiles
            ? { session_id: result.session_id, files: result.files }
            : {
              session_id: result.session_id,
            }) satisfies t.CodeExecutionArtifact,
        ];
      } catch (error) {
        throw new Error(
          `Execution error:\n\n${(error as Error | undefined)?.message}`
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
