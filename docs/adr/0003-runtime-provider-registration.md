# ADR 0003: Resolve Built-In and Host Providers Through One Registry

## Status

Accepted

## Context

Adding a chat-model provider required editing and publishing the static
provider constructor map. A host could not supply a provider at runtime, and
provider behavior was split across constructor lookup, family predicates,
manual tool-stream handling, and strict message-alternation sets. Exposing only
a second constructor map would let those rules drift and would still require
unsafe assertions for host-specific options.

## Decision

A **Provider Registration** owns a provider's constructor, family, manual
tool-stream behavior, and strict-alternation requirement. Built-ins register
through the same module as host providers. Constructor lookup keeps its
existing `getChatModelClass` interface, while hosts add registrations through
`registerProvider` before constructing agent configuration.

Provider names are process-local and unique. Duplicate registrations fail
closed. Registration returns an ownership-scoped disposer so tests and hot
reload can remove only the binding they created. The registry uses a versioned
global symbol so ESM and CommonJS package graphs in the same process resolve
the same host bindings. Built-ins remain module-graph-local so provider wrapper
identity checks use the matching ESM or CommonJS constructors. Constructors are
validated as constructible, and model initialization reports a
provider-specific error when tools are requested from a model without
`bindTools`.

Host applications declaration-merge `CustomProviderOptionsMap` in
`@librechat/agents/provider-registration` to carry their configuration type
through agent and fallback configuration. Unaugmented host names remain
accepted at runtime but receive only the built-in option union at compile time.

## Consequences

- Hosts can add compatible providers without an agents package release.
- Provider-family message shaping and streaming traits stay local to one
  registration instead of being recomputed across callers.
- Built-in provider types remain precise, while typed host options require an
  explicit declaration merge.
- Registration must run in every process before agent configuration is used;
  bindings are not durable and do not cross workers automatically.
- Exact built-in behavior remains keyed to its provider identity unless a
  registration trait deliberately generalizes that behavior.
