# Cloudflare Sandbox Tool Execution

`@librechat/agents` can run the built-in coding tools against a
Cloudflare Sandbox by selecting the `cloudflare-sandbox` execution
engine. The sandbox runtime is structural: any object with the required
`exec`, `readFile`, `writeFile`, `mkdir`, `listFiles`, and `deleteFile`
methods can be supplied.

```ts
import {
  CLOUDFLARE_BASH_CODING_TOOL_NAMES,
  createCloudflareBridgeRuntime,
  Providers,
} from '@librechat/agents';
import { getSandbox } from '@cloudflare/sandbox';

const runId = crypto.randomUUID();
const sandbox = getSandbox(env.Sandbox, runId);

const config = {
  llmConfig: {
    provider: Providers.OPENAI,
    model: env.WORKERS_AI_MODEL,
    apiKey: env.CLOUDFLARE_API_TOKEN,
    configuration: {
      baseURL: `https://api.cloudflare.com/client/v4/accounts/${env.CLOUDFLARE_ACCOUNT_ID}/ai/v1`,
    },
  },
  toolExecution: {
    engine: 'cloudflare-sandbox',
    cloudflare: {
      sandbox,
      workspaceRoot: '/workspace',
      env: {
        GITHUB_TOKEN: env.GITHUB_TOKEN,
      },
      timeoutMs: 120000,
      maxOutputChars: 200000,
      includeCodingTools: true,
      codingToolNames: CLOUDFLARE_BASH_CODING_TOOL_NAMES,
      fileCheckpointing: true,
    },
  },
} as const;
```

By default, the Cloudflare backend exposes the same built-in coding
tool names as the local backend:

- `read_file`
- `write_file`
- `edit_file`
- `grep_search`
- `glob_search`
- `list_directory`
- `compile_check`
- `bash_tool`
- `execute_code`
- `run_tools_with_code`
- `run_tools_with_bash`

Review workflows that only want a bash-driven sandbox can set
`codingToolNames` to `CLOUDFLARE_BASH_CODING_TOOL_NAMES`. That exposes
file/search/list/compile tools plus `bash_tool` and `run_tools_with_bash`,
while leaving out `execute_code` and `run_tools_with_code`.

Paths are interpreted inside `workspaceRoot` and are clamped to that
workspace. Command execution uses `sandbox.exec()`; file operations use
the sandbox file APIs. No Workers AI provider is added by this backend:
use `Providers.OPENAI` with an OpenAI-compatible `baseURL` and API key.

## Outside Cloudflare

Code running outside a Worker cannot call `getSandbox(env.Sandbox, id)`
directly because it does not have a Durable Object binding. Deploy a
Worker that wraps your handler with `bridge()` from
`@cloudflare/sandbox/bridge`, set a `SANDBOX_API_KEY` secret, and create
a structural runtime with `createCloudflareBridgeRuntime()`:

```ts
const sandbox = createCloudflareBridgeRuntime({
  baseURL: process.env.CF_SANDBOX_BRIDGE_URL!,
  apiKey: process.env.CF_SANDBOX_API_KEY!,
  sandboxId: runId,
});

const config = {
  toolExecution: {
    engine: 'cloudflare-sandbox',
    cloudflare: {
      sandbox,
      workspaceRoot: '/workspace',
      includeCodingTools: true,
      codingToolNames: CLOUDFLARE_BASH_CODING_TOOL_NAMES,
    },
  },
} as const;
```

The bridge adapter uses `/v1/sandbox/:id/exec` for command execution
and the bridge file endpoints for reads/writes. Operations not exposed
directly by the bridge (`mkdir`, `listFiles`, `deleteFile`) are
implemented with bash commands inside the same sandbox workspace.
