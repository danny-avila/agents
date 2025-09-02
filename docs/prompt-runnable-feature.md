# Prompt Runnable Feature

## Overview

The `promptInstructions` feature allows agents to dynamically append a human message after all state messages, ensuring AI agents respond appropriately in complex multi-agent scenarios.

## Problem Solved

In parallel multi-agent systems, when multiple AI agents send messages to a downstream agent (like a summarizer), the downstream agent may not respond because it only sees AI messages without a human prompt. This feature adds a dynamic human message to trigger a response.

## Implementation

### 1. Agent Configuration

Add `promptInstructions` to your agent configuration:

```typescript
const agent: AgentInputs = {
  agentId: 'summarizer',
  provider: Providers.ANTHROPIC,
  instructions: 'You are a summary expert...',
  // Static prompt
  promptInstructions: 'Please provide a summary of the above analyses.',
  // OR dynamic prompt based on message state
  promptInstructions: (messages) => {
    const analystCount = messages.filter(
      (msg) => msg._getType() === 'ai' && msg.name?.includes('analyst')
    ).length;

    if (analystCount >= 3) {
      return 'Please synthesize the three analyses above.';
    }
    return undefined; // No prompt yet
  },
};
```

### 2. How It Works

1. The `promptInstructions` is converted to a `promptRunnable` during agent initialization
2. This runnable is applied after all message processing, including system messages
3. If the instructions are a function, it receives the current messages and can decide whether to add a prompt
4. If a prompt is returned, it's appended as a `HumanMessage`

### 3. Use Cases

#### Parallel Processing Aggregation

When multiple agents process in parallel and send results to an aggregator:

```typescript
promptInstructions: (messages) => {
  const inputCount = countInputMessages(messages);
  if (inputCount >= expectedCount) {
    return 'Based on all inputs above, please provide your synthesis.';
  }
  return undefined;
};
```

#### Conditional Prompting

Add prompts only under certain conditions:

```typescript
promptInstructions: (messages) => {
  const lastMessage = messages[messages.length - 1];
  if (needsFollowUp(lastMessage)) {
    return 'Please elaborate on your previous response.';
  }
  return undefined;
};
```

#### Static Prompts

For simple cases where you always want to add a prompt:

```typescript
promptInstructions: 'Please provide your analysis based on the conversation above.';
```

## Benefits

1. **Flexibility**: Use static strings or dynamic functions
2. **Context Awareness**: Functions can inspect the full message history
3. **Clean Separation**: Keeps prompt logic separate from agent instructions
4. **Runnable Pattern**: Consistent with existing `systemRunnable` pattern
5. **Composability**: Runnables can be chained and composed

## Technical Details

- Implemented as a `RunnableLambda` that processes messages
- Applied in `Graph.createCallModel()` after message pruning
- Async to support dynamic imports
- Returns messages unchanged if no prompt is needed
