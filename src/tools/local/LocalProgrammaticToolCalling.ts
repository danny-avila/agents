import { randomBytes, randomUUID, timingSafeEqual } from 'crypto';
import { createServer } from 'http';
import { tool } from '@langchain/core/tools';
import type { AddressInfo } from 'net';
import type { IncomingMessage, ServerResponse } from 'http';
import type { DynamicStructuredTool } from '@langchain/core/tools';
import type * as t from '@/types';
import {
  executeTools,
  filterToolsByUsage,
  formatCompletedResponse,
  normalizeToPythonIdentifier,
  ProgrammaticToolCallingName,
  ProgrammaticToolCallingSchema,
  ProgrammaticToolCallingDescription,
} from '@/tools/ProgrammaticToolCalling';
import {
  BashProgrammaticToolCallingSchema,
  BashProgrammaticToolCallingDescription,
  filterBashToolsByUsage,
  normalizeToBashIdentifier,
} from '@/tools/BashProgrammaticToolCalling';
import {
  executeLocalBash,
  executeLocalCode,
  getLocalSessionId,
  shellQuote,
} from './LocalExecutionEngine';
import { Constants } from '@/common';

const DEFAULT_TIMEOUT = 60000;
const LocalProgrammaticToolCallingSchema = {
  ...ProgrammaticToolCallingSchema,
  properties: {
    ...ProgrammaticToolCallingSchema.properties,
    lang: {
      type: 'string',
      enum: ['py', 'python', 'bash', 'sh'],
      default: 'bash',
      description:
        'Local engine runtime for orchestration code. Defaults to bash; use py/python for Python orchestration.',
    },
  },
} as const;

type ToolBridge = {
  url: string;
  token: string;
  close: () => Promise<void>;
};

type ToolRequest = {
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
};

const BRIDGE_AUTH_HEADER = 'x-librechat-bridge-token';

function constantTimeEquals(a: string, b: string): boolean {
  const aBuf = Buffer.from(a, 'utf8');
  const bBuf = Buffer.from(b, 'utf8');
  if (aBuf.length !== bBuf.length) {
    return false;
  }
  return timingSafeEqual(aBuf, bBuf);
}

type LocalProgrammaticRuntime = 'python' | 'bash';

type LocalProgrammaticParams = {
  code: string;
  timeout?: number;
  lang?: string;
  runtime?: string;
  language?: string;
};

type ToolFilter = (toolDefs: t.LCTool[], code: string) => t.LCTool[];

function resolveRuntime(params: LocalProgrammaticParams): LocalProgrammaticRuntime {
  const rawRuntime = params.lang ?? params.runtime ?? params.language ?? 'bash';
  return rawRuntime === 'py' || rawRuntime === 'python' ? 'python' : 'bash';
}

function toSerializable(value: unknown): unknown {
  if (value === undefined) {
    return null;
  }
  return value;
}

async function readRequestBody(req: IncomingMessage): Promise<ToolRequest> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  if (raw === '') {
    return {};
  }
  return JSON.parse(raw) as ToolRequest;
}

function writeJson(res: ServerResponse, status: number, value: unknown): void {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(value));
}

async function createToolBridge(toolMap: t.ToolMap): Promise<ToolBridge> {
  const token = randomBytes(32).toString('hex');
  const server = createServer((req, res) => {
    if (req.method !== 'POST' || req.url !== '/tool') {
      writeJson(res, 404, { error: 'Not found' });
      return;
    }

    const presented = req.headers[BRIDGE_AUTH_HEADER];
    const presentedToken = Array.isArray(presented) ? presented[0] : presented;
    if (
      typeof presentedToken !== 'string' ||
      !constantTimeEquals(presentedToken, token)
    ) {
      writeJson(res, 401, { error: 'Unauthorized' });
      return;
    }

    readRequestBody(req)
      .then(async (body) => {
        if (typeof body.name !== 'string' || body.name === '') {
          writeJson(res, 400, {
            call_id: body.id ?? 'invalid',
            result: null,
            is_error: true,
            error_message: 'Tool request is missing a tool name.',
          });
          return;
        }

        const [result] = await executeTools(
          [
            {
              id: body.id ?? `local_call_${randomUUID()}`,
              name: body.name,
              input: body.input ?? {},
            },
          ],
          toolMap
        );

        writeJson(res, 200, {
          ...result,
          result: toSerializable(result.result),
        });
      })
      .catch((error: Error) => {
        writeJson(res, 500, {
          call_id: 'error',
          result: null,
          is_error: true,
          error_message: error.message,
        });
      });
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });

  const address = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${address.port}/tool`,
    token,
    close: () =>
      new Promise((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      }),
  };
}

function indent(code: string): string {
  return code
    .split('\n')
    .map((line) => `  ${line}`)
    .join('\n');
}

function createPythonProgram(
  code: string,
  toolDefs: t.LCTool[],
  bridgeUrl: string,
  bridgeToken: string
): string {
  const functionDefs = toolDefs
    .map((def) => {
      const pythonName = normalizeToPythonIdentifier(def.name);
      return [
        `async def ${pythonName}(**kwargs):`,
        `  return await __librechat_call_tool(${JSON.stringify(def.name)}, kwargs)`,
      ].join('\n');
    })
    .join('\n\n');

  return `
import asyncio
import json
import urllib.request

__LIBRECHAT_TOOL_BRIDGE = ${JSON.stringify(bridgeUrl)}
__LIBRECHAT_TOOL_TOKEN = ${JSON.stringify(bridgeToken)}

async def __librechat_call_tool(name, payload):
  body = json.dumps({"name": name, "input": payload}).encode("utf-8")
  headers = {
    "Content-Type": "application/json",
    ${JSON.stringify(BRIDGE_AUTH_HEADER)}: __LIBRECHAT_TOOL_TOKEN,
  }

  def request():
    req = urllib.request.Request(__LIBRECHAT_TOOL_BRIDGE, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=300) as response:
      return response.read().decode("utf-8")

  raw = await asyncio.to_thread(request)
  result = json.loads(raw)
  if result.get("is_error"):
    raise RuntimeError(result.get("error_message") or f"Tool {name} failed")
  return result.get("result")

${functionDefs}

async def __librechat_main():
${indent(code)}

asyncio.run(__librechat_main())
`.trimStart();
}

function createBashProgram(
  code: string,
  toolDefs: t.LCTool[],
  bridgeUrl: string,
  bridgeToken: string
): string {
  const functions = toolDefs
    .map((def) => {
      const bashName = normalizeToBashIdentifier(def.name);
      return [
        `${bashName}() {`,
        '  local payload="${1:-}"',
        '  if [ -z "$payload" ]; then payload=\'{}\'; fi',
        `  __librechat_call_tool ${shellQuote(def.name)} "$payload"`,
        '}',
      ].join('\n');
    })
    .join('\n\n');

  return `
__LIBRECHAT_TOOL_BRIDGE=${shellQuote(bridgeUrl)}
__LIBRECHAT_TOOL_HEADER=${shellQuote(BRIDGE_AUTH_HEADER)}
__LIBRECHAT_TOOL_TOKEN=${shellQuote(bridgeToken)}

__librechat_call_tool() {
  local tool_name="$1"
  local payload="$2"
  python3 - "$__LIBRECHAT_TOOL_BRIDGE" "$tool_name" "$payload" "$__LIBRECHAT_TOOL_HEADER" "$__LIBRECHAT_TOOL_TOKEN" <<'PY'
import json
import sys
import urllib.request

url, name, payload, header, token = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
body = json.dumps({"name": name, "input": json.loads(payload)}).encode("utf-8")
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", header: token}, method="POST")
with urllib.request.urlopen(req, timeout=300) as response:
  result = json.loads(response.read().decode("utf-8"))
if result.get("is_error"):
  print(result.get("error_message") or f"Tool {name} failed", file=sys.stderr)
  sys.exit(1)
value = result.get("result")
if isinstance(value, str):
  print(value)
else:
  print(json.dumps(value))
PY
}

${functions}

${code}
`.trimStart();
}

function getProgrammaticContext(config?: {
  toolCall?: unknown;
}): Partial<t.ProgrammaticCache> {
  return (config?.toolCall ?? {}) as Partial<t.ProgrammaticCache>;
}

function createEffectiveToolMap(
  toolMap: t.ToolMap,
  toolDefs: t.LCTool[],
  code: string,
  filterTools: ToolFilter
): { effectiveTools: t.LCTool[]; effectiveMap: t.ToolMap } {
  const effectiveTools = filterTools(toolDefs, code);
  const effectiveMap = new Map<string, t.GenericTool>(
    effectiveTools
      .map((def) => {
        const executable = toolMap.get(def.name);
        return executable == null
          ? undefined
          : ([def.name, executable] as [string, t.GenericTool]);
      })
      .filter((entry): entry is [string, t.GenericTool] => entry != null)
  );

  return { effectiveTools, effectiveMap };
}

async function runLocalProgrammaticTool(args: {
  params: LocalProgrammaticParams;
  config?: { toolCall?: unknown };
  localConfig: t.LocalExecutionConfig;
  runtime: LocalProgrammaticRuntime;
}): Promise<[string, t.ProgrammaticExecutionArtifact]> {
  const { toolMap, toolDefs } = getProgrammaticContext(args.config);

  if (toolMap == null || toolMap.size === 0) {
    throw new Error('No toolMap provided for local programmatic execution.');
  }
  if (toolDefs == null || toolDefs.length === 0) {
    throw new Error('No tool definitions provided for local programmatic execution.');
  }

  const { effectiveTools, effectiveMap } = createEffectiveToolMap(
    toolMap,
    toolDefs,
    args.params.code,
    args.runtime === 'bash' ? filterBashToolsByUsage : filterToolsByUsage
  );
  const bridge = await createToolBridge(effectiveMap);

  try {
    const timeoutMs = args.params.timeout ?? args.localConfig.timeoutMs ?? DEFAULT_TIMEOUT;
    const result =
      args.runtime === 'bash'
        ? await executeLocalBash(
          createBashProgram(args.params.code, effectiveTools, bridge.url, bridge.token),
          { ...args.localConfig, timeoutMs }
        )
        : await executeLocalCode(
          {
            lang: 'py',
            code: createPythonProgram(args.params.code, effectiveTools, bridge.url, bridge.token),
          },
          { ...args.localConfig, timeoutMs }
        );

    if (result.exitCode !== 0 || result.timedOut) {
      throw new Error(
        result.stderr !== ''
          ? result.stderr
          : `Local ${args.runtime} programmatic execution exited with code ${
            result.exitCode ?? 'unknown'
          }`
      );
    }

    return formatCompletedResponse({
      status: 'completed',
      session_id: getLocalSessionId(args.localConfig),
      stdout: result.stdout,
      stderr: result.stderr,
      files: [],
    });
  } finally {
    await bridge.close();
  }
}

export function createLocalProgrammaticToolCallingTool(
  localConfig: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawParams, config) => {
      const params = rawParams as LocalProgrammaticParams;
      return runLocalProgrammaticTool({
        params,
        config,
        localConfig,
        runtime: resolveRuntime(params),
      });
    },
    {
      name: ProgrammaticToolCallingName,
      description: `${ProgrammaticToolCallingDescription}\n\nLocal engine: runs bash by default, or Python when \`lang\` is \`py\` or \`python\`, on the host machine and calls tools through an in-process localhost bridge.`,
      schema: LocalProgrammaticToolCallingSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}

export function createLocalBashProgrammaticToolCallingTool(
  localConfig: t.LocalExecutionConfig = {}
): DynamicStructuredTool {
  return tool(
    async (rawParams, config) => {
      const params = rawParams as LocalProgrammaticParams;
      return runLocalProgrammaticTool({
        params,
        config,
        localConfig,
        runtime: 'bash',
      });
    },
    {
      name: Constants.BASH_PROGRAMMATIC_TOOL_CALLING,
      description: `${BashProgrammaticToolCallingDescription}\n\nLocal engine: runs this bash orchestration code on the host machine and calls tools through an in-process localhost bridge.`,
      schema: BashProgrammaticToolCallingSchema,
      responseFormat: Constants.CONTENT_AND_ARTIFACT,
    }
  );
}
