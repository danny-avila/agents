# Multi-Agent Patterns in Agentus

This document explains the different multi-agent patterns supported by the `MultiAgentGraph` class.

## Edge Types

The `MultiAgentGraph` supports two types of edges between agents:

### 1. Handoff Edges (Dynamic Routing)

**Use Case**: When an agent needs to dynamically decide which agent to call next based on the conversation context.

**How it works**: Creates transfer tools that agents can use to explicitly hand off control to another agent.

**Example**:

```typescript
const edges: t.GraphEdge[] = [
  {
    from: 'classifier',
    to: ['technical_expert', 'business_expert', 'general_assistant'],
    description: 'Route to appropriate expert based on query type',
    edgeType: 'handoff', // Optional - this is the default for conditional edges
    condition: (state) => {
      // Dynamic routing logic
      if (state.messages[0].content.includes('technical')) {
        return 'technical_expert';
      }
      // ... more logic
    },
  },
];
```

**Default behavior**:

- Single-to-single edges default to handoff
- Edges with conditions are always handoff
- Edges with `edgeType: 'handoff'` are handoff

### 2. Parallel Edges (Automatic Fan-out/Fan-in)

**Use Case**: When you want multiple agents to process simultaneously without explicit handoff logic.

**How it works**: Creates direct graph edges that cause automatic parallel execution.

**Example**:

```typescript
const edges: t.GraphEdge[] = [
  {
    from: 'researcher',
    to: ['analyst1', 'analyst2', 'analyst3'], // Fan-out
    description: 'Distribute to all analysts for parallel processing',
    edgeType: 'direct', // Explicit parallel execution
  },
  {
    from: ['analyst1', 'analyst2', 'analyst3'], // Fan-in
    to: 'summarizer',
    description: 'Aggregate results from all analysts',
    edgeType: 'direct',
  },
];
```

**Default behavior**:

- Single-to-multiple edges default to parallel (fan-out)
- Multiple-to-single edges should explicitly set `edgeType: 'direct'` for fan-in

## Common Patterns

### 1. Sequential Handoffs

```typescript
// Flight assistant can transfer to hotel assistant and vice versa
const edges = [
  { from: 'flight_assistant', to: 'hotel_assistant' },
  { from: 'hotel_assistant', to: 'flight_assistant' },
];
```

### 2. Supervisor Pattern (Handoff)

```typescript
// Supervisor decides which expert to route to
const edges = [
  {
    from: 'supervisor',
    to: ['expert1', 'expert2', 'expert3'],
    condition: (state) => decideExpert(state),
  },
  { from: 'expert1', to: 'supervisor' },
  { from: 'expert2', to: 'supervisor' },
  { from: 'expert3', to: 'supervisor' },
];
```

### 3. Map-Reduce Pattern (Parallel)

```typescript
// Distribute work and aggregate results
const edges = [
  {
    from: 'coordinator',
    to: ['worker1', 'worker2', 'worker3'],
    edgeType: 'direct', // Fan-out
  },
  {
    from: ['worker1', 'worker2', 'worker3'],
    to: 'aggregator',
    edgeType: 'direct', // Fan-in
  },
];
```

### 4. Hybrid Pattern

```typescript
// Mix of handoff and parallel
const edges = [
  // Classifier uses handoff to route
  {
    from: 'classifier',
    to: ['path_a', 'path_b'],
    condition: (state) => choosePath(state),
  },
  // Path A uses parallel processing
  {
    from: 'path_a',
    to: ['processor1', 'processor2'],
    edgeType: 'direct',
  },
  // Processors converge
  {
    from: ['processor1', 'processor2'],
    to: 'finalizer',
    edgeType: 'direct',
  },
];
```

## Important Notes

1. **Event Streaming**: When using parallel edges, you may see "Run ID not found in run map" errors in the console. These are harmless and can be ignored - they occur because LangGraph creates new run IDs for parallel executions that the event stream handler doesn't track.

2. **State Management**: All agents share the same state (messages). Parallel agents see the same state snapshot and their updates are merged.

3. **Tool Creation**:
   - Handoff edges create transfer tools (e.g., `transfer_to_agent_name`)
   - Parallel edges create direct graph connections (no tools)

4. **Performance**: Parallel execution can significantly speed up processing when agents perform independent work.
