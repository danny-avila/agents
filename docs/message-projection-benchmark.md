# Provider-request projection benchmark

This benchmark measures the synchronous provider-facing message derivation that
occurs after graph-state shaping and before the model adapter serializes a
request. It is a baseline for the log-first projection investigation: it does
not call a provider or include network latency.

Run it with:

```sh
npm run bench:provider-projection
```

The benchmark uses stable text-only and tool-result histories for OpenAI and
Anthropic projections. It reports median wall-clock time over seven samples and
checks that each sample produces the same number of provider messages. Each
scenario reports the default `off` path and the opt-in `observe` path, which
adds the read-only provenance invariant scan over valid source-backed messages.
It isolates synchronous projection and scan cost; it does not include LangChain
callback dispatch or delivery of a warning for an invalid batch.

Use the tool-result case when evaluating a new projection seam. That path
performs meaningful provider normalization; a text-only history mostly measures
the cost of the read-only scans that preserve an unchanged message array.

The default mode is `off`; it does not create a callback handler or run the
invariant scan. The observed measurements quantify rollout cost and are not a
production default.

## Representative result

On Node 24.16.0 / Apple Silicon, 250 projections produced these medians:

| Scenario | Provider | Off | Observe | Added per request |
| --- | --- | ---: | ---: | ---: |
| text-100 | OpenAI | 0.71 ms | 1.26 ms | 2.20 µs |
| text-100 | Anthropic | 0.34 ms | 0.81 ms | 1.88 µs |
| text-500 | OpenAI | 3.13 ms | 6.26 ms | 12.52 µs |
| text-500 | Anthropic | 1.54 ms | 4.71 ms | 12.68 µs |
| tools-100 | OpenAI | 19.57 ms | 22.02 ms | 9.80 µs |
| tools-100 | Anthropic | 10.93 ms | 12.76 ms | 7.32 µs |

The measured observation scan added 1.88–12.68 µs per request. Timing varies
by host, so rerun the benchmark when changing the invariant or projection path.
