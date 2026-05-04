/**
 * `compile_check` — a thin LLM-callable wrapper around the project's
 * standard typecheck/lint command. Lets the agent answer "did my
 * change break anything?" without us shipping a real LSP client.
 *
 * Auto-detection priority (first hit wins):
 *
 *   1. `local.compileCheck.command`            — explicit override
 *   2. `tsconfig.json`                          → `npx --no-install tsc --noEmit`
 *   3. `package.json` with a typescript dep    → same as 2
 *   4. `pyproject.toml` or `setup.py` / `setup.cfg`
 *      with a dev dep on mypy                  → `python3 -m mypy .`
 *      else                                    → `python3 -m py_compile <every .py>`
 *      (bounded by find-walk so node_modules
 *      and `.venv` don't blow up)
 *   5. `Cargo.toml`                            → `cargo check --message-format=short`
 *   6. `go.mod`                                → `go vet ./...`
 *   7. otherwise                               → tells the agent there's
 *                                                 no detected toolchain.
 *
 * Output is the spawn process's stdout/stderr passed through
 * `truncateLocalOutput` so a 10MB tsc dump can't blow context. The
 * exit code is reported.
 */

import { readFile, stat } from 'fs/promises';
import { resolve } from 'path';
import { tool } from '@langchain/core/tools';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  getLocalCwd,
  spawnLocalProcess,
  truncateLocalOutput,
  validateBashCommand,
} from './LocalExecutionEngine';
import { Constants } from '@/common';

/** Back-compat alias; canonical name lives on `Constants.COMPILE_CHECK`. */
export const CompileCheckToolName = Constants.COMPILE_CHECK;

const CompileCheckSchema: t.JsonSchemaType = {
  type: 'object',
  properties: {
    command: {
      type: 'string',
      description:
        'Optional explicit command to run instead of the auto-detected one. Runs verbatim from the local engine cwd; honours the standard sandbox/AST gate.',
    },
    timeout_ms: {
      type: 'integer',
      description:
        'Optional timeout in milliseconds. Defaults to 120000 (2 min).',
    },
  },
};

type DetectedKind =
  | 'typescript'
  | 'python-mypy'
  | 'python-compile'
  | 'rust'
  | 'go'
  | 'unknown';

type Detection = {
  kind: DetectedKind;
  command: string;
  reason: string;
};

async function pathExists(p: string): Promise<boolean> {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

async function detect(cwd: string): Promise<Detection> {
  if (await pathExists(resolve(cwd, 'tsconfig.json'))) {
    return {
      kind: 'typescript',
      command: 'npx --no-install tsc --noEmit',
      reason: 'tsconfig.json present',
    };
  }
  if (await pathExists(resolve(cwd, 'package.json'))) {
    const pkgRaw = await readFile(resolve(cwd, 'package.json'), 'utf8').catch(
      () => ''
    );
    if (pkgRaw.includes('"typescript"')) {
      return {
        kind: 'typescript',
        command: 'npx --no-install tsc --noEmit',
        reason: 'package.json declares typescript',
      };
    }
  }
  if (await pathExists(resolve(cwd, 'Cargo.toml'))) {
    return {
      kind: 'rust',
      command: 'cargo check --message-format=short',
      reason: 'Cargo.toml present',
    };
  }
  if (await pathExists(resolve(cwd, 'go.mod'))) {
    return {
      kind: 'go',
      command: 'go vet ./...',
      reason: 'go.mod present',
    };
  }
  if (
    (await pathExists(resolve(cwd, 'pyproject.toml'))) ||
    (await pathExists(resolve(cwd, 'setup.py'))) ||
    (await pathExists(resolve(cwd, 'setup.cfg')))
  ) {
    const pyToml = await readFile(
      resolve(cwd, 'pyproject.toml'),
      'utf8'
    ).catch(() => '');
    if (pyToml.includes('mypy')) {
      return {
        kind: 'python-mypy',
        command: 'python3 -m mypy .',
        reason: 'pyproject.toml declares mypy',
      };
    }
    return {
      kind: 'python-compile',
      command:
        'python3 -c "import compileall, sys; sys.exit(0 if compileall.compile_dir(\'.\', quiet=1, rx=__import__(\'re\').compile(r\'(node_modules|\\.venv|\\.git|build|dist)\')) else 1)"',
      reason: 'Python project (no mypy detected)',
    };
  }
  return {
    kind: 'unknown',
    command: '',
    reason:
      'no recognised project marker (tsconfig.json, package.json[typescript], Cargo.toml, go.mod, pyproject.toml, setup.py)',
  };
}

const DEFAULT_TIMEOUT_MS = 120_000;

export function createCompileCheckTool(
  config: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawInput) => {
      const input = rawInput as {
        command?: string;
        timeout_ms?: number;
      };
      const cwd = getLocalCwd(config);
      const overrideCommand = input.command ?? config.compileCheck?.command;
      let detection: Detection;
      if (overrideCommand != null && overrideCommand.trim() !== '') {
        detection = {
          kind: 'unknown',
          command: overrideCommand,
          reason: 'explicit override',
        };
      } else {
        detection = await detect(cwd);
      }

      if (detection.command === '') {
        const explainer =
          `compile_check: ${detection.reason}. Pass an explicit \`command\` (e.g. \`npm run typecheck\`) to override.`;
        return [
          explainer,
          {
            kind: detection.kind,
            ran: false,
            reason: detection.reason,
            cwd,
          },
        ];
      }

      // Codex P1 #21: route the resolved command through the same
      // safety gates the rest of the local engine uses. Without this
      // a host with `readOnly: true` (or relying on the destructive-
      // command guard) could be bypassed by passing a `command`
      // override to compile_check that performs writes/deletes.
      // Auto-detected commands (tsc/cargo/etc.) pass these gates
      // unchanged — the validation is only blocking for genuinely
      // mutating overrides.
      const validation = await validateBashCommand(detection.command, config);
      if (!validation.valid) {
        const explainer =
          `compile_check refused to run \`${detection.command}\`: ${validation.errors.join('; ')}`;
        return [
          explainer,
          {
            kind: detection.kind,
            ran: false,
            reason: validation.errors.join('; '),
            cwd,
          },
        ];
      }

      const timeoutMs =
        input.timeout_ms ??
        config.compileCheck?.timeoutMs ??
        DEFAULT_TIMEOUT_MS;
      const result = await spawnLocalProcess(
        config.shell ?? (process.platform === 'win32' ? 'bash.exe' : 'bash'),
        ['-lc', detection.command],
        {
          ...config,
          timeoutMs,
          maxOutputChars: config.maxOutputChars ?? 8000,
        }
      );

      const passed =
        result.exitCode === 0 && !result.timedOut;
      const headline = passed
        ? `compile_check (${detection.kind}) PASSED via \`${detection.command}\``
        : `compile_check (${detection.kind}) FAILED via \`${detection.command}\` ` +
          `(exit=${result.exitCode ?? 'unknown'}${result.timedOut ? ', timed_out=true' : ''})`;

      let body = '';
      if (result.stdout !== '') {
        body += `\n\nstdout:\n${truncateLocalOutput(result.stdout, 4000)}`;
      }
      if (result.stderr !== '') {
        body += `\n\nstderr:\n${truncateLocalOutput(result.stderr, 4000)}`;
      }
      if (result.fullOutputPath != null) {
        body += `\n\nfull_output_path: ${result.fullOutputPath}`;
      }
      const summary = `${headline}${body}\n\nworking_directory: ${cwd}\nreason: ${detection.reason}`;

      return [
        summary,
        {
          kind: detection.kind,
          ran: true,
          passed,
          exit_code: result.exitCode,
          timed_out: result.timedOut,
          command: detection.command,
          cwd,
        },
      ];
    },
    {
      name: CompileCheckToolName,
      description:
        'Run the project\'s standard typecheck or lint pass and return its output. Auto-detects from project markers (tsconfig.json/package.json -> tsc; Cargo.toml -> cargo check; go.mod -> go vet; pyproject.toml -> mypy or py_compile). Pass `command` to override.',
      schema: CompileCheckSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createCompileCheckToolDefinition(): t.LCTool {
  return {
    name: CompileCheckToolName,
    description:
      'Run the project\'s standard typecheck or lint pass and return its output.',
    parameters: CompileCheckSchema,
    allowed_callers: ['direct', 'code_execution'],
    responseFormat: Constants.CONTENT_AND_ARTIFACT,
    toolType: 'builtin',
  };
}
