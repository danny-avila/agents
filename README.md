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
