# ADR 0001: Reuse Exact Context Token Counts Conservatively

## Status

Accepted

## Context

Context-pressure measurement creates a new meter for each provider request.
Within agent tool loops, most retained messages are unchanged, but each new
meter previously tokenized the entire retained history again. Provider
attribution, calibration, and overflow decisions must remain request-scoped,
and message objects can contain mutable structured content or metadata.
Callers may also provide custom token counters with semantics the library does
not know.

## Decision

`AgentContext` retains a weakly referenced cache of exact per-message counts
for the built-in tokenizer factory and explicitly compatible host counters. A
request-scoped context-pressure meter uses this as a second-level cache while
preserving its existing local cache and all request-specific accounting.

A count is reusable only for a non-proxy message with string content and a
stable token-relevant surface. The cache compares content, message type, role,
assistant tool-call state, legacy function-call state, and tool provider type
before every reuse. Structured content, accessors, proxies, and tool calls
remain on the exact recount path. Custom token counters are uncached unless a
host explicitly marks them compatible with the built-in deterministic surface
contract.

## Consequences

- Retained string messages are tokenized once per `AgentContext`, reducing
  repeated tokenizer work in multi-step model/tool loops.
- Mutation, replacement, reordering, compaction, and provider projection keep
  the same measured results because aggregate accounting is never persisted.
- The weak map does not keep discarded messages alive.
- Hosts that reconstruct graphs between HTTP requests need a separate,
  serialization-safe cache keyed by tokenizer version and message content to
  reuse counts across those requests.
- Hosts can opt a wrapper around the built-in counting semantics into reuse;
  marking a stateful or wider custom counter would violate the cache contract.
- New token-relevant fields in the built-in counter must be added to the stable
  surface before affected messages can remain cacheable.
