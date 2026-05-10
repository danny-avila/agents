# @librechat/agents

TypeScript utilities for building LibreChat agent workflows. The package provides graph orchestration, streaming event handling, tool execution, provider adapters, and message formatting for single-agent and multi-agent runs.

## Features

- LangGraph-based single-agent and multi-agent workflows
- Streaming content aggregation and run-step event handlers
- Tool calling, tool search, subagent handoffs, and programmatic tool execution
- Provider adapters for Anthropic, Bedrock, Vertex AI, OpenAI-compatible providers, Google, Mistral, DeepSeek, and xAI
- Message formatting, context pruning, summarization, and cache-control helpers

## Installation

```bash
npm install @librechat/agents
```

## Basic Usage

```typescript
import { HumanMessage } from '@langchain/core/messages';
import { Providers, Run } from '@librechat/agents';

const run = await Run.create({
  runId: crypto.randomUUID(),
  graphConfig: {
    type: 'standard',
    instructions: 'You are a helpful assistant.',
    llmConfig: {
      provider: Providers.OPENAI,
      model: 'gpt-4o-mini',
      apiKey: process.env.OPENAI_API_KEY,
    },
  },
  returnContent: true,
});

const content = await run.processStream(
  { messages: [new HumanMessage('Hello')] },
  {
    runId: crypto.randomUUID(),
    streamMode: 'values',
    version: 'v2',
  }
);
```

## Programmatic Sessions

For scripts, CI, and programmatic integrations, use the session facade. It
keeps a JSONL session tree by default, so runs can be resumed, cloned, forked,
branched in place, compacted, and inspected later.

```typescript
import { Providers, createAgentSession } from '@librechat/agents';

const session = await createAgentSession({
  graphConfig: {
    type: 'standard',
    instructions: 'You are a concise coding assistant.',
    llmConfig: {
      provider: Providers.OPENAI,
      model: 'gpt-4o-mini',
      apiKey: process.env.OPENAI_API_KEY,
    },
  },
});

const result = await session.run('Summarize this repository.');
console.log(result.text);
console.log(session.sessionPath); // durable .jsonl session file
```

Sessions expose tree operations inspired by Pi-style workflows:

```typescript
const store = session.getSessionStore();
const forkPoint = store?.getForkPoints()[0];

if (forkPoint) {
  const forked = await session.fork(forkPoint.id, { position: 'before' });
  await forked.run('Try a different approach from here.');
}

const cloned = await session.clone();
await cloned.compact({ instructions: 'Keep only implementation decisions.' });
```

`session.stream()` projects the SDK's existing graph events, and
`session.compact()` uses the same summarization node, hooks, and provider
logic as normal runs. JSONL is the durable journal; the graph remains the
execution engine.

OpenAI-compatible streaming helpers are available as experimental subpaths:

```typescript
import { composeEventHandlers } from '@librechat/agents';
import { createOpenAIHandlers } from '@librechat/agents/openai';
import { createResponsesEventHandlers } from '@librechat/agents/responses';

const customHandlers = composeEventHandlers(
  createOpenAIHandlers(openAIConfig),
  createResponsesEventHandlers(responsesConfig),
  hostHandlers
);
```

## Development

```bash
npm ci
npm run build
npm test
npx tsc --noEmit
npx eslint src/
```

## Documentation

- [Multi-agent patterns](./docs/multi-agent-patterns.md)
- [Summarization behavior](./docs/summarization-behavior.md)

## License

MIT
