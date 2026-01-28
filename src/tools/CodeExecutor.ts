import { config } from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import type * as t from '@/types';
import { EnvVar, Constants } from '@/common';

config();

export const imageExtRegex = /\.(jpg|jpeg|png|gif|webp)$/i;
export const getCodeBaseURL = (): string =>
  getEnvironmentVariable(EnvVar.CODE_BASEURL) ??
  Constants.OFFICIAL_CODE_BASEURL;

const imageMessage = 'Image is already displayed to the user';
const otherMessage = 'File is already downloaded by the user';
const accessMessage =
  'Note: Files from previous executions are automatically available and can be modified.';
const emptyOutputMessage =
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
  const apiKey =
    params[EnvVar.CODE_API_KEY] ??
    params.apiKey ??
    getEnvironmentVariable(EnvVar.CODE_API_KEY) ??
    '';
  if (!apiKey) {
    throw new Error('No API key provided for code execution tool.');
  }

  return tool(
    async (rawInput, config) => {
      const { lang, code, ...rest } = rawInput as {
        lang: SupportedLanguage;
        code: string;
        args?: string[];
      };
      /**
       * Extract session context from config.toolCall (injected by ToolNode).
       * - session_id: For API to associate with previous session
       * - _injected_files: File refs to pass directly (avoids /files endpoint race condition)
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

      /**
       * File injection priority:
       * 1. Use _injected_files from ToolNode (avoids /files endpoint race condition)
       * 2. Fall back to fetching from /files endpoint if session_id provided but no injected files
       */
      if (_injected_files && _injected_files.length > 0) {
        postData.files = _injected_files;
      } else if (session_id != null && session_id.length > 0) {
        /** Fallback: fetch from /files endpoint (may have race condition issues) */
        try {
          const filesEndpoint = `${baseEndpoint}/files/${session_id}?detail=full`;
          const fetchOptions: RequestInit = {
            method: 'GET',
            headers: {
              'User-Agent': 'LibreChat/1.0',
              'X-API-Key': apiKey,
            },
          };

          if (process.env.PROXY != null && process.env.PROXY !== '') {
            fetchOptions.agent = new HttpsProxyAgent(process.env.PROXY);
          }

          const response = await fetch(filesEndpoint, fetchOptions);
          if (!response.ok) {
            throw new Error(
              `Failed to fetch files for session: ${response.status}`
            );
          }

          const files = await response.json();
          if (Array.isArray(files) && files.length > 0) {
            const fileReferences: t.CodeEnvFile[] = files.map((file) => {
              const nameParts = file.name.split('/');
              const id = nameParts.length > 1 ? nameParts[1].split('.')[0] : '';

              return {
                session_id,
                id,
                name: file.metadata['original-filename'],
              };
            });

            postData.files = fileReferences;
          }
        } catch {
          // eslint-disable-next-line no-console
          console.warn(`Failed to fetch files for session: ${session_id}`);
        }
      }

      try {
        const fetchOptions: RequestInit = {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'LibreChat/1.0',
            'X-API-Key': apiKey,
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
        let formattedOutput = '';
        if (result.stdout) {
          formattedOutput += `stdout:\n${result.stdout}\n`;
        } else {
          formattedOutput += emptyOutputMessage;
        }
        if (result.stderr) formattedOutput += `stderr:\n${result.stderr}\n`;
        if (result.files && result.files.length > 0) {
          formattedOutput += 'Generated files:\n';

          const fileCount = result.files.length;
          for (let i = 0; i < fileCount; i++) {
            const file = result.files[i];
            const isImage = imageExtRegex.test(file.name);
            formattedOutput += `- /mnt/data/${file.name} | ${isImage ? imageMessage : otherMessage}`;

            if (i < fileCount - 1) {
              formattedOutput += fileCount <= 3 ? ', ' : ',\n';
            }
          }

          formattedOutput += `\n\n${accessMessage}`;
          return [
            formattedOutput.trim(),
            {
              session_id: result.session_id,
              files: result.files,
            },
          ];
        }

        return [formattedOutput.trim(), { session_id: result.session_id }];
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
