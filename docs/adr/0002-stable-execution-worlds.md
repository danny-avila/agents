# ADR 0002: Keep Execution Worlds Stable Across Tool Bindings

## Status

Accepted

## Context

Coding tools use one filesystem namespace and one subprocess launcher. The
existing local execution seam represented those capabilities as optional
fields, and Cloudflare rebuilt both adapters whenever an agent binding rebuilt
its tool bundle. Capability caches correctly key by subprocess identity, but a
new adapter function on every binding made a warm Cloudflare runtime look like
a new backend and repeated remote availability probes.

## Decision

An **Execution World** is the complete filesystem, subprocess, and sandbox
identity used by a coding-tool bundle. The Node host exposes one default world.
Cloudflare retains one world per execution configuration and reuses it across
tool bindings. The existing partial `local.exec` shape and legacy top-level
`local.spawn` option remain supported at the public configuration boundary.

Capability caches continue to key by the world's subprocess function and
environment. Reusing a world therefore reuses only backend capability facts;
command output, timeout state, cancellation, and tool results remain scoped to
each invocation.

## Consequences

- Rebuilt Cloudflare tool bundles avoid repeated remote capability probes.
- Filesystem and subprocess adapters are explicitly paired as one namespace.
- Local and Cloudflare output, error, timeout, cleanup, and security behavior
  remain in their existing adapters.
- Changes to captured Cloudflare workspace, timeout, or sandbox settings rebuild
  the cached world so its filesystem and subprocess capabilities stay paired.
- Non-bash local `execute_code` staging still uses the host temporary directory;
  moving temporary lifecycle operations into the world is a separate change.
