# ADR 0006: Align Compaction with Provider Cache Prefixes

## Status

Accepted

## Context

Normal agent requests already place provider-specific cache breakpoints on the
stable tool prefix. The summarizer bound the same logical tools independently,
so Anthropic, OpenRouter, and Bedrock compaction requests could omit or resolve
those breakpoints differently. Compaction also placed its message breakpoint
after the unique summarization instruction, preventing the retained history
from matching a reusable prefix.

An earlier design attempted to predict whether the normal and compaction
requests had identical cache identities. It fingerprinted provider options,
environment fallbacks, formatted tools, and message projections before the
request was sent. That approach required a closed allowlist over provider SDK
behavior and repeatedly admitted false eligibility when a new route input or
resolution rule was discovered.

## Decision

Normal and summarization model construction use one live tool projection. It
resolves local or Cloudflare execution tools, the discovered deferred-tool set,
and provider cache preparation at invocation time. It preserves the existing
static-before-deferred partition and applies the same cache marker and TTL
rules for Anthropic, OpenRouter, and cache-capable Bedrock Claude models. Other
providers and Bedrock model families receive their original tools unchanged.

When self-summarization uses prompt caching, retained history receives the
provider-specific tail breakpoint before the unique compaction instruction is
appended. Bedrock history TTL resolution uses the configured model family even
when the serving route is an opaque application inference profile. Fallback
providers receive no marker intended for another provider.

Cache reuse is established by provider-reported cache-creation and cache-read
usage. The SDK does not retain request snapshots or predict eligibility from
configuration and environment inputs. A live before-and-after benchmark primes
the normal tool prefix, invokes the former unmarked compaction shape, and then
invokes the aligned shape while recording cache reads and latency.

## Consequences

- Normal request preparation performs the same provider-specific tool work as
  before; the logic moves behind a shared function rather than adding another
  pass to the hot path.
- Summarization performs one provider tool-prefix preparation pass only when
  compaction runs and can reuse both stable tools and retained history.
- Provider usage remains the authoritative cache receipt, so SDK routing and
  serialization changes cannot silently make an eligibility predictor lie.
- Exact serialized-request comparison remains available as a future diagnostic
  seam, but is not required for cache reuse and is not part of this change.
