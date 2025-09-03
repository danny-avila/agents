# Multi-Agent Test Scripts - Example Output

## multi-agent-test.ts

```
Testing Multi-Agent Handoff System...

Invoking multi-agent graph...

====== ON_RUN_STEP ======
{ name: 'flight_assistant', type: 'agent', ... }
[flight_assistant] Starting...

====== CHAT_MODEL_STREAM ======
I'll help you book a flight from Boston to New York...

====== TOOL_START ======
{ name: 'transfer_to_hotel_assistant', ... }

====== ON_RUN_STEP_COMPLETED ======
[flight_assistant] Completed

====== ON_RUN_STEP ======
{ name: 'hotel_assistant', type: 'agent', ... }
[hotel_assistant] Starting...

====== CHAT_MODEL_STREAM ======
Great! I'll help you find a hotel near Times Square...

Final content parts:
[
  { type: 'text', text: "I'll help you book a flight..." },
  { type: 'tool_use', name: 'transfer_to_hotel_assistant', ... },
  { type: 'text', text: "Great! I'll help you find a hotel..." }
]
```

## multi-agent-parallel.ts

```
Testing Parallel Multi-Agent System (Fan-in/Fan-out)...

Invoking parallel multi-agent graph...

====== ON_RUN_STEP ======
[researcher] Starting analysis...

====== ON_RUN_STEP ======
[analyst1] Starting analysis...
[analyst2] Starting analysis...
[analyst3] Starting analysis...

====== CHAT_MODEL_STREAM ======
[analyst1] From a financial perspective...
[analyst2] From a technical perspective...
[analyst3] From a market perspective...

====== ON_RUN_STEP_COMPLETED ======
[analyst1] Completed analysis
[analyst2] Completed analysis
[analyst3] Completed analysis

====== ON_RUN_STEP ======
[summarizer] Starting analysis...

Final content parts: 5 parts
```

## multi-agent-conditional.ts

```
Testing Conditional Multi-Agent System...

--- Processing: "How do I implement a binary search tree in Python?" ---

====== ON_RUN_STEP ======
{ name: 'classifier', type: 'agent', ... }

====== ON_RUN_STEP ======
{ name: 'technical_expert', type: 'agent', ... }
Routing to: technical_expert

====== CHAT_MODEL_STREAM ======
To implement a binary search tree in Python, you'll need to create a Node class...

Expert used: technical_expert
Content parts: 2
---

--- Processing: "What are the key strategies for market expansion?" ---

Routing to: business_expert
...

--- Processing: "What is the capital of France?" ---

Routing to: general_assistant
...
```

## Key Features of Updated Scripts

1. **Proper Event Handling**: All GraphEvents are properly handled with custom handlers
2. **Content Aggregation**: Using `createContentAggregator()` to collect all content parts
3. **Stream Output**: Real-time token streaming via `ChatModelStreamHandler`
4. **Debug Logging**: Comprehensive event logging for debugging multi-agent flows
5. **Conversation History**: Proper tracking of messages across agent interactions
6. **Return Content**: `returnContent: true` ensures content parts are returned

The scripts now follow the same patterns as `simple.ts` and `tools.ts`, providing consistent behavior and comprehensive event handling across all multi-agent scenarios.
