# Execution-world benchmark

The benchmark measures repeated coding-tool bindings against one warm remote
runtime. Each binding performs a JavaScript syntax check. Before stable world
identity, every rebuilt adapter misses the backend-keyed Node availability
cache and performs both a capability probe and the syntax check. With a stable
world, only the first binding probes Node.

Run it with:

```sh
npm run bench:execution-world
```

Result on 2026-08-23 with ten bindings and 10 ms simulated remote latency:

| Metric | Recreated world | Stable world |
| --- | ---: | ---: |
| Remote process calls | 20 | 11 |
| Elapsed time | 237.58 ms | 133.08 ms |
| Relative speed | 1.00x | 1.79x |

The exact elapsed time depends on the machine and transport latency. The
deterministic signal is the 45% reduction in remote process calls. A real
remote runtime generally has higher per-call latency than this benchmark's
10 ms, so avoiding nine calls also removes nine network/runtime round trips.
