# Eager tool readiness parsing

On Node v24.16.0 / macOS, compared main `945119d4584ca2ee7bfac9f50b68eb5995b3a21a` with the seal-first readiness change. The same opt-in benchmark test was copied into the baseline checkout. Four separate Jest processes ran sequentially in candidate / baseline / baseline / candidate order after the other local checks completed.

Each process reports the median of seven samples after two warmups. Each sample streams deterministic JSON arguments and a subsequent tool index through the real `ChatModelStreamHandler`, exercising argument reconciliation, readiness, and eager dispatch. Graph bookkeeping and external dispatch use the existing test fixtures. No model, network, production provider parsing, or full LibreChat request is measured. Each sample asserts that the first call prestarted.

Ranges below are the two process medians, in milliseconds. Input sizes refer to SQL string content before its JSON envelope.

| Input | Chunk size | Baseline elapsed | Candidate elapsed | Baseline CPU | Candidate CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 B | 64 B | 0.213–0.293 | 0.172–0.192 | 0.381–0.445 | 0.338–0.402 |
| 128 B | 256 B | 0.081–0.124 | 0.076–0.085 | 0.083–0.155 | 0.077–0.094 |
| 8 KiB | 64 B | 3.488–3.552 | 2.151–2.226 | 4.483–4.625 | 3.195–3.488 |
| 8 KiB | 256 B | 0.927–0.930 | 0.559–0.668 | 0.932–0.935 | 0.559–0.686 |
| 48 KiB | 64 B | 46.579–47.605 | 23.229–25.351 | 53.479–54.837 | 33.493–37.595 |
| 48 KiB | 256 B | 11.864–12.156 | 6.106–6.679 | 11.934–12.243 | 6.221–6.753 |

The larger fixtures show about 28–51% lower handler elapsed time and 22–49% lower process CPU across these ranges. Tiny cases are dominated by timing variability. These results exceed the proposed 10% local-handler improvement gate, but do not establish an end-to-end user latency percentage, concurrent-load performance, or a retained-memory reduction. Process CPU includes runtime/GC work.

Reproduce on each checkout with the identical benchmark fixture:

```sh
BENCH_EAGER_READINESS=1 npx jest src/__tests__/stream.eagerArgsDivergence.test.ts --runInBand -t 'benchmarks tool stream'
```

The behavioral regression test separately asserts that readiness does not parse unsealed arguments and parses a sealed argument string once before dispatch. Provider fragment reconciliation still parses its own buffers; that compatibility-sensitive behavior is unchanged.
