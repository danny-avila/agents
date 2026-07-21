import { config } from 'dotenv';
import fetch, { RequestInit } from 'node-fetch';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  BASH_SHELL_GUIDANCE,
  CODE_ARTIFACT_PATH_GUIDANCE,
  appendFailedExecutionFileReminder,
  appendTmpScratchReminder,
  appendCodeSessionFileSummary,
  emptyOutputMessage,
  buildCodeApiHttpErrorMessage,
  getCodeBaseURL,
  resolveCodeApiAuthHeaders,
} from './CodeExecutor';
import { Constants } from '@/common';

config();

const baseEndpoint = getCodeBaseURL();
const EXEC_ENDPOINT = `${baseEndpoint}/exec`;

export const BashExecutionToolSchema = {
  type: 'object',
  properties: {
    command: {
      type: 'string',
      description: `The bash command or script to execute.
- The environment is stateless; variables and state don't persist between executions.
- Prior /mnt/data files are available and can be modified in place.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- Input code **IS ALREADY** displayed to the user, so **DO NOT** repeat it in your response unless asked.
- Output code **IS NOT** displayed to the user, so **DO** write all desired output explicitly.
- IMPORTANT: You MUST explicitly print/output ALL results you want the user to see.
- Use \`echo\`, \`printf\`, or \`cat\` for all outputs.`,
    },
    args: {
      type: 'array',
      items: { type: 'string' },
      description:
        'Additional arguments to execute the command with. This should only be used if the input command requires additional arguments to run.',
    },
  },
  required: ['command'],
} as const;

export const BashExecutionToolDescription = `
Runs bash commands and returns stdout/stderr output from a stateless execution environment, similar to running scripts in a command-line interface. Each execution is isolated and independent.

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- NEVER use this tool to execute malicious commands.
`.trim();

/**
 * Bash statefulness is filesystem-tier and scoped to `/mnt/data`. The machine
 * is warm across calls, but each call runs in a fresh sandbox (new process
 * tree + private /tmp), so background processes are reaped when the call ends
 * and anything written outside /mnt/data is discarded. The note must not
 * promise otherwise: a model told background processes survive will start a
 * server in one call and assume it is listening in the next.
 */
export const STATEFUL_BASH_NOTE =
  'Session state: commands in this conversation run on the same warm machine, so files written to /mnt/data persist between calls. Each call runs in a fresh, isolated sandbox: shell variables, the working directory, /tmp, and background processes do NOT survive after the call returns — a process started in one call is terminated when that call ends. Only /mnt/data is durable (the machine itself may also be reset at any time).';

export const StatefulBashExecutionToolDescription = `
Runs bash commands and returns stdout/stderr output. Commands in this conversation share one warm machine with a persistent /mnt/data, but each command runs in its own isolated sandbox (not a persistent shell session).

${STATEFUL_BASH_NOTE}

Usage:
- No network access available.
- Generated files are automatically delivered; **DO NOT** provide download links.
- ${CODE_ARTIFACT_PATH_GUIDANCE}
- ${BASH_SHELL_GUIDANCE}
- NEVER use this tool to execute malicious commands.
`.trim();

/**
 * Supplemental prompt documenting the tool-output reference feature.
 *
 * Hosts should append this (separated by a blank line) to the base
 * {@link BashExecutionToolDescription} only when
 * `RunConfig.toolOutputReferences.enabled` is `true`. When the feature
 * is disabled, including this text would tell the LLM to emit
 * `{{tool0turn0}}` placeholders that pass through unsubstituted and
 * leak into the shell.
 */
export const BashToolOutputReferencesGuide = `
Referencing previous tool outputs:
- Every successful tool result is tagged with a reference key of the form \`tool<idx>turn<turn>\` (e.g., \`tool0turn0\`). The key appears either as a \`[ref: tool0turn0]\` prefix line or, when the output is a JSON object, as a \`_ref\` field on the object.
- To pipe a previous tool output into this tool, embed the placeholder \`{{tool<idx>turn<turn>}}\` literally anywhere in the \`command\` string (or any string arg). It will be substituted with the stored output verbatim before the command runs.
- The substituted value is the original output string (no \`[ref: …]\` prefix, no \`_ref\` key), so it is safe to pipe directly into \`jq\`, \`grep\`, \`awk\`, etc.
- Example (simple ASCII output): \`echo '{{tool0turn0}}' | jq '.foo'\` takes the full output of the first tool from the first turn and pipes it into jq.
- For payloads that may contain quotes, parentheses, backticks, or arbitrary bytes (random/binary data, JSON with embedded quotes, multi-line strings), prefer a quoted-delimiter heredoc over \`echo '…'\`. The heredoc body is not interpreted by the shell, so substituted payloads pass through unchanged.
- Heredoc example: \`wc -c << 'EOF'\\n{{tool0turn0}}\\nEOF\` (the quotes around \`'EOF'\` disable interpolation inside the body).
- Unknown reference keys are left in place and surfaced as \`[unresolved refs: …]\` after the output.
`.trim();

/**
 * Composes the bash tool description, optionally appending the
 * tool-output references guide. Hosts that enable
 * `RunConfig.toolOutputReferences` should pass `enableToolOutputReferences: true`
 * when registering the tool so the LLM learns the `{{…}}` syntax it
 * will actually be able to use.
 */
export function buildBashExecutionToolDescription(options?: {
  enableToolOutputReferences?: boolean;
  statefulSessions?: boolean;
}): string {
  const base =
    options?.statefulSessions === true
      ? StatefulBashExecutionToolDescription
      : BashExecutionToolDescription;
  if (options?.enableToolOutputReferences === true) {
    return `${base}\n\n${BashToolOutputReferencesGuide}`;
  }
  return base;
}

const STATELESS_BASH_PARAM_NOTE =
  'The environment is stateless; variables and state don\'t persist between executions.';
const STATEFUL_BASH_PARAM_NOTE =
  'Files written to /mnt/data persist between calls on the same warm machine. Each call runs in a fresh sandbox: shell variables, cwd, /tmp, and background processes do NOT survive the call. Only /mnt/data is durable.';

export function buildBashExecutionToolSchema(opts?: {
  statefulSessions?: boolean;
}): typeof BashExecutionToolSchema {
  const note =
    opts?.statefulSessions === true
      ? STATEFUL_BASH_PARAM_NOTE
      : STATELESS_BASH_PARAM_NOTE;
  const commandDescription =
    BashExecutionToolSchema.properties.command.description.replace(
      STATELESS_BASH_PARAM_NOTE,
      note
    );
  return {
    ...BashExecutionToolSchema,
    properties: {
      ...BashExecutionToolSchema.properties,
      command: {
        ...BashExecutionToolSchema.properties.command,
        description: commandDescription,
      },
    },
  } as typeof BashExecutionToolSchema;
}

export const BashExecutionToolName = Constants.BASH_TOOL;

/**
 * Default bash tool definition using the base description.
 *
 * When `RunConfig.toolOutputReferences.enabled` is `true`, build a
 * reference-aware description with
 * {@link buildBashExecutionToolDescription}
 * (`{ enableToolOutputReferences: true }`) and construct a custom
 * definition using it — using this constant as-is leaves the LLM
 * unaware of the `{{tool<i>turn<n>}}` syntax.
 */
export const BashExecutionToolDefinition = {
  name: BashExecutionToolName,
  description: BashExecutionToolDescription,
  schema: BashExecutionToolSchema,
} as const;

function createBashExecutionTool(
  params: t.BashExecutionToolParams | null = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput, config) => {
      /* `statefulSessions` is prompt-only — keep it out of the wire body. */
      const {
        authHeaders,
        statefulSessions: _statefulSessions,
        ...executionParams
      } = params ?? {};
      void _statefulSessions;
      /* Drop any model-supplied `runtime_session_hint` from the raw args: the
       * hint must only come from ToolNode's injected `_runtime_session_hint`
       * (below), never from the tool call itself. */
      const {
        command,
        runtime_session_hint: _ignoredModelHint,
        ...rest
      } = rawInput as {
        command: string;
        runtime_session_hint?: unknown;
        args?: string[];
      };
      void _ignoredModelHint;
      const { session_id, _injected_files, _runtime_session_hint } =
        (config.toolCall ?? {}) as {
          session_id?: string;
          _injected_files?: t.CodeEnvFile[];
          _runtime_session_hint?: string;
        };

      const postData: Record<string, unknown> = {
        lang: 'bash',
        code: command,
        ...rest,
        ...executionParams,
      };

      if (
        typeof _runtime_session_hint === 'string' &&
        _runtime_session_hint !== ''
      ) {
        postData.runtime_session_hint = _runtime_session_hint;
      }

      /* See `CodeExecutor.ts` for the rationale — `/files/<session_id>`
       * HTTP fallback was removed because codeapi's sessionAuth requires
       * kind/id query params unavailable at this point. */
      if (_injected_files && _injected_files.length > 0) {
        postData.files = _injected_files;
      } else if (
        session_id != null &&
        session_id.length > 0 &&
        !Array.isArray(postData.files)
      ) {
        // eslint-disable-next-line no-console
        console.debug(
          `[BashExecutor] No injected files for session_id=${session_id} — exec will run without input files`
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
          command
        );
        const hasFiles = result.files != null && result.files.length > 0;
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
          (error as Error | undefined)?.message ?? '',
          command
        );
        throw new Error(`Execution error:\n\n${messageWithReminder}`);
      }
    },
    {
      name: BashExecutionToolName,
      description: buildBashExecutionToolDescription({
        statefulSessions: params?.statefulSessions,
      }),
      schema: buildBashExecutionToolSchema(params ?? undefined),
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export { createBashExecutionTool };
