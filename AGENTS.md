# @librechat/agents

## Project Overview

`@librechat/agents` is a TypeScript library for LLM agent orchestration — tool calling, multi-agent graphs, message formatting, streaming, and provider abstraction (Anthropic, Bedrock, VertexAI, OpenAI, Google). Published as `@librechat/agents` on npm. This is a major backend dependency of [LibreChat](../LibreChat/CLAUDE.md) (same team).

| Path            | Purpose                                             |
| --------------- | --------------------------------------------------- |
| `src/messages/` | Message formatting, caching, content processing     |
| `src/graphs/`   | LangGraph-based agent graphs (single + multi-agent) |
| `src/llm/`      | Provider-specific LLM wrappers and utilities        |
| `src/tools/`    | Tool definitions and search                         |
| `src/agents/`   | Agent definitions and handoff logic                 |
| `src/types/`    | Shared TypeScript types                             |
| `src/common/`   | Enums, constants                                    |
| `src/run.ts`    | Main run orchestration                              |
| `src/stream.ts` | Streaming logic                                     |

---

## Code Style

### Structure and Clarity

- **Never-nesting**: early returns, flat code, minimal indentation. Break complex operations into well-named helpers.
- **Functional first**: pure functions, immutable data, `map`/`filter`/`reduce` over imperative loops. Only reach for OOP when it clearly improves domain modeling or state encapsulation.
- **No dynamic imports** unless absolutely necessary.

### DRY

- Extract repeated logic into utility functions.
- Parameterized helpers instead of near-duplicate functions.
- Constants for repeated values; configuration objects over duplicated init code.
- Shared validators, centralized error handling, single source of truth for business rules.
- Shared typing system with interfaces/types extending common base definitions.
- Abstraction layers for external API interactions.

### Iteration and Performance

- **Minimize looping** — especially over shared data structures like message arrays, which are iterated frequently throughout the codebase. Every additional pass adds up at scale.
- Consolidate sequential O(n) operations into a single pass whenever possible; never loop over the same collection twice if the work can be combined.
- Choose data structures that reduce the need to iterate (e.g., `Map`/`Set` for lookups instead of `Array.find`/`Array.includes`).
- Avoid unnecessary object creation; consider space-time tradeoffs.
- Prevent memory leaks: careful with closures, dispose resources/event listeners, no circular references.

### Type Safety

- **Never use `any`**. Explicit types for all parameters, return values, and variables.
- **Limit `unknown`** — avoid `unknown`, `Record<string, unknown>`, and `as unknown as T` assertions. A `Record<string, unknown>` almost always signals a missing explicit type definition.
- **Don't duplicate types** — before defining a new type, check whether it already exists in `src/types/`. Reuse and extend existing types (`MessageContentComplex`, `ExtendedMessageContent`, `TMessage`, etc.) rather than creating redundant definitions.
- Use union types, generics, and interfaces appropriately.
- All TypeScript and ESLint warnings/errors must be addressed — do not leave unresolved diagnostics.

### Comments and Documentation

- Write self-documenting code; no inline comments narrating what code does.
- JSDoc only for complex/non-obvious logic or intellisense on public APIs.
- Single-line JSDoc for brief docs, multi-line for complex cases.
- Avoid standalone `//` comments unless absolutely necessary.

### Import Order

Imports are organized into three sections:

1. **Package imports** — sorted shortest to longest line length.
2. **`import type` imports** — sorted longest to shortest (package types first, then local types; length resets between sub-groups).
3. **Local/project imports** — sorted longest to shortest.

Multi-line imports count total character length across all lines. Consolidate value imports from the same module. Always use standalone `import type { ... }` — never inline `type` inside value imports.

### JS/TS Loop Preferences

- **Limit looping as much as possible.** Prefer single-pass transformations and avoid re-iterating the same data.
- `for (let i = 0; ...)` for performance-critical or index-dependent operations.
- `for...of` for simple array iteration.
- `for...in` only for object property enumeration.

---

## Development Commands

| Command              | Purpose                            |
| -------------------- | ---------------------------------- |
| `npm run build`      | Build CJS + ESM + types via Rollup |
| `npx jest <pattern>` | Run tests matching pattern         |
| `npx tsc --noEmit`   | Type-check without emitting        |
| `npx eslint src/`    | Lint source files                  |

- Package manager: Yarn (`.yarn/` directory, `yarn.lock`)
- TypeScript with path aliases: `@/*` → `src/*`
- Test framework: Jest with `ts-jest`

---

## Testing

- Framework: **Jest**, run from project root: `npx jest <pattern>`.
- Test files: `*.test.ts` and `*.spec.ts` under `src/`.

### Philosophy

- **Real logic over mocks.** Exercise actual code paths with real dependencies. Mocking is a last resort.
- **Spies over mocks.** Assert that real functions are called with expected arguments and frequency without replacing underlying logic.
- **MCP**: use real `@modelcontextprotocol/sdk` exports for servers, transports, and tool definitions. Mirror real scenarios, don't stub SDK internals.
- Only mock what you cannot control: external HTTP APIs, rate-limited services, non-deterministic system calls.
- Heavy mocking is a code smell, not a testing strategy.

---

## Formatting

Fix all formatting lint errors (trailing spaces, tabs, newlines, indentation) using auto-fix when available. All TypeScript/ESLint warnings and errors **must** be resolved.
