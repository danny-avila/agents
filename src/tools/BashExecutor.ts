import { spawn } from 'child_process';
import { tool, DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import { Constants } from '@/common';

const DEFAULT_TIMEOUT = 30000;
const emptyOutputMessage =
  'stdout: Empty. Ensure you\'re writing output explicitly.\n';

export const BashExecutionToolSchema = {
  type: 'object',
  properties: {
    command: {
      type: 'string',
      description: `The bash command or script to execute locally.
- The environment is your local machine; exercise caution with destructive operations.
- Standard Unix utilities and installed programs are available.
- Input code **IS ALREADY** displayed to the user, so **DO NOT** repeat it in your response.
- Output **IS NOT** displayed to the user, so **DO** write all desired output explicitly.
- Use \`echo\`, \`printf\`, or \`cat\` for all outputs.`,
    },
    args: {
      type: 'array',
      items: { type: 'string' },
      description:
        'Additional arguments passed to the script via positional parameters ($1, $2, etc.).',
    },
    timeout: {
      type: 'integer',
      minimum: 1000,
      maximum: 300000,
      default: DEFAULT_TIMEOUT,
      description:
        'Maximum execution time in milliseconds. Default: 30 seconds. Max: 5 minutes.',
    },
  },
  required: ['command'],
} as const;

export const BashExecutionToolDescription = `
Execute bash commands or scripts locally and return stdout/stderr output.

Usage:
- Runs on the local machine with access to the local filesystem and environment.
- Standard Unix utilities and installed programs are available.
- NEVER use this tool to execute malicious or destructive commands.
`.trim();

export const BashExecutionToolName = Constants.EXECUTE_BASH;

export const BashExecutionToolDefinition = {
  name: BashExecutionToolName,
  description: BashExecutionToolDescription,
  schema: BashExecutionToolSchema,
} as const;

function runBashProcess(
  command: string,
  options: {
    bashPath: string;
    timeout: number;
    workDir?: string;
    args?: string[];
  }
): Promise<{ stdout: string; stderr: string; exitCode: number | null }> {
  return new Promise((resolve, reject) => {
    const shellArgs = ['-c', command];
    if (options.args && options.args.length > 0) {
      shellArgs.push('--', ...options.args);
    }

    const proc = spawn(options.bashPath, shellArgs, {
      timeout: options.timeout,
      cwd: options.workDir,
      env: { ...process.env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data: Buffer) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on('error', (err: Error) => {
      reject(new Error(`Failed to start bash process: ${err.message}`));
    });

    proc.on('close', (code: number | null, signal: string | null) => {
      if (signal != null) {
        stderr += `\nProcess terminated by signal: ${signal}\n`;
      }
      resolve({ stdout, stderr, exitCode: code });
    });
  });
}

export function createBashExecutionTool(
  params: t.BashExecutionToolParams = {}
): DynamicStructuredTool {
  const bashPath = params.bashPath ?? 'bash';
  const defaultTimeout = params.defaultTimeout ?? DEFAULT_TIMEOUT;
  const workDir = params.workDir;

  return tool(
    async (rawInput) => {
      const input = rawInput as {
        command: string;
        args?: string[];
        timeout?: number;
      };
      const { command, args, timeout = defaultTimeout } = input;

      try {
        const { stdout, stderr, exitCode } = await runBashProcess(command, {
          bashPath,
          timeout,
          workDir,
          args,
        });

        let formatted = '';
        if (stdout) {
          formatted += `stdout:\n${stdout}\n`;
        } else {
          formatted += emptyOutputMessage;
        }
        if (stderr) {
          formatted += `stderr:\n${stderr}\n`;
        }
        if (exitCode !== 0 && exitCode !== null) {
          formatted += `exit code: ${exitCode}\n`;
        }

        return formatted.trim();
      } catch (error) {
        throw new Error(`Bash execution error:\n\n${(error as Error).message}`);
      }
    },
    {
      name: BashExecutionToolName,
      description: BashExecutionToolDescription,
      schema: BashExecutionToolSchema,
    }
  );
}
