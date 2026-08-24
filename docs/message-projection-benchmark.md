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
checks that each sample produces the same number of provider messages.

Use the tool-result case when evaluating a new projection seam. That path
performs meaningful provider normalization; a text-only history mostly measures
the cost of the read-only scans that preserve an unchanged message array.
