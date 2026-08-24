# Compaction range benchmark

The benchmark models a common long-running agent shape: one user request
followed by many assistant tool calls and large tool results. The prior
turn-boundary selector treated that entire history as one indivisible turn, so
it exposed no compactable head even after context pressure triggered.

Run it with:

```sh
npm run bench:compaction-range
```

The before case uses the previous user-turn-only selection. The after case
keeps the same turn preference, then falls back to a token-priced boundary only
after closed tool-call/result units. The benchmark uses the SDK's deterministic
`o200k_base` token counter and fails if any tool pair crosses the boundary.

Results on 2026-08-24:

| Scenario       | Total tokens | Compactable before | Compactable after | Retained after |
| -------------- | -----------: | -----------------: | ----------------: | -------------: |
| 20 tool steps  |       50,824 |                  0 |    40,570 (79.8%) |         10,254 |
| 50 tool steps  |      126,541 |                  0 |   103,861 (82.1%) |         22,680 |
| 100 tool steps |      253,006 |                  0 |   209,968 (83.0%) |         43,038 |

This is a provider-runtime optimization rather than a local CPU microbenchmark:
the selection adds one bounded linear scan only when compaction has already
fired. The decision metric is how much repeated provider input becomes
replaceable by one checkpoint while preserving recent raw evidence.
