# Provider context-overflow signatures

What each provider actually does when the input exceeds what it can accept,
and how the SDK turns that into a forced compaction instead of an error the
caller sees.

Everything in the tables below was captured by sending a real over-limit
prompt to a live endpoint with
[`src/scripts/context-overflow-probe.ts`](../src/scripts/context-overflow-probe.ts).
The same payloads are frozen as test fixtures in
[`contextOverflowSignatures.ts`](../src/utils/__tests__/fixtures/contextOverflowSignatures.ts),
so a matcher can never drift away from what a provider really sends.

## Why this is not a one-line check

The obvious implementation — look for "context length" in the message — fails
on most of the providers this SDK supports:

- **OpenAI usually reports overflow as a `429`, not a `400`.** When a prompt is
  large enough to overrun the account's tokens-per-minute allowance, it is
  rejected on the rate-limit path before context validation runs. The error
  says `rate_limit_exceeded`, and any matcher that excludes rate limits
  discards the single most common OpenAI symptom.
- **Bedrock can report overflow inside an HTTP `200`.** For Nova and Llama, the
  failure arrives mid-stream, so `$metadata.httpStatusCode` is `200` and the
  thrown value is a bare `Error`.
- **Bedrock's wording changes per upstream model** — three different sentences
  across four models, only one of which reports token counts.
- **Vertex AI discards the reason entirely.** Its gaxios path throws
  `Google request failed with status code 400` with no body, indistinguishable
  from any other 400.
- **xAI says "maximum prompt length"**, not "context length", so LangChain's own
  `ContextOverflowError` wrapper does not catch it.
- **OpenRouter rejects on an inflated estimate.** It counted 56,827 tokens for a
  prompt the underlying tokenizer scores at ~42,600 — about 1.33× — so a budget
  set to its stated ceiling still overflows.
- **DeepSeek accepted a 227k-token prompt** that was assumed over-limit;
  `deepseek-v4-flash` actually reports a 1,048,565-token window. Configured
  limits are not reliable; the provider's own number is.

## Captured signatures

| Provider     | Model probed                                                                           | Thrown as                          | HTTP    | Message                                                                                                    |
| ------------ | -------------------------------------------------------------------------------------- | ---------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `anthropic`  | `claude-haiku-4-5`                                                                     | `ContextOverflowError` (LangChain) | 400     | `prompt is too long: 274468 tokens > 200000 maximum`                                                       |
| `openAI`     | `gpt-4o-mini`                                                                          | `ContextOverflowError` (LangChain) | 400     | `This model's maximum context length is 128000 tokens. However, your messages resulted in 149767 tokens.`  |
| `openAI`     | `gpt-5-nano`, `gpt-4`, `gpt-4o`, `gpt-4.1-nano`                                        | `RateLimitError`                   | **429** | `Request too large … on tokens per min (TPM): Limit 200000, Requested 480002`                              |
| `bedrock`    | `us.anthropic.claude-haiku-4-5`                                                        | `ValidationException`              | 400     | `The model returned the following errors: prompt is too long: 207848 tokens > 200000 maximum`              |
| `bedrock`    | `us.anthropic.claude-sonnet-4-5`                                                       | `ValidationException`              | 400     | `The model returned the following errors: Input is too long for requested model.`                          |
| `bedrock`    | `us.amazon.nova-lite`                                                                  | `Error`                            | **200** | `… Input Tokens Exceeded: Number of input tokens exceeds maximum length.`                                  |
| `bedrock`    | `us.meta.llama3-1-70b`                                                                 | `Error`                            | **200** | `… This model's maximum context length is 131072 tokens.`                                                  |
| `google`     | `gemini-3.1-flash-image`, `gemini-omni-flash-preview`                                  | `GoogleGenerativeAIFetchError`     | 400     | `[400 Bad Request] The input token count exceeds the maximum number of tokens allowed (65536).`            |
| `vertexai`   | `gemini-2.5-flash-lite`                                                                | `Error`                            | 400     | `Google request failed with status code 400` — **no reason text**                                          |
| `openrouter` | 9 upstreams (qwen, mistral, meta, openai, anthropic, deepseek, moonshot, google, x-ai) | `ContextOverflowError` (LangChain) | 400     | `This endpoint's maximum context length is 32768 tokens. However, you requested about 56827 tokens …`      |
| `deepseek`   | `deepseek-v4-flash`                                                                    | `ContextOverflowError` (LangChain) | 400     | `This model's maximum context length is 1048565 tokens. However, you requested 1179668 tokens …`           |
| `xai`        | `grok-build-0.1`                                                                       | `BadRequestError`                  | 400     | `This model's maximum prompt length is 256000 but the request contains 332986 tokens.`                     |
| `mistral`    | `mistral-tiny-latest`                                                                  | `SDKError`                         | 400     | `Prompt contains 170397 tokens and 0 draft tokens, too large for model with 131072 maximum context length` |

`azureOpenAI`, `moonshot`, and `mistralai` are not probed separately: they
reuse the OpenAI and Mistral clients verbatim, so the captured signatures for
those clients apply unchanged.

### Not reproduced

- **`vertexai` with a reason** — the native-fetch path in
  `@langchain/google-common` does include the response body
  (`Google request failed with status code 400: {…}`), but the service-account
  (gaxios) path used here does not. The matcher handles both.
- **`moonshot` direct** — no `MOONSHOT_API_KEY` was available. `ChatMoonshot`
  extends `ChatOpenAI`, and the OpenRouter probe of `moonshotai/kimi-k2`
  behaved like every other OpenAI-compatible endpoint.

## What the SDK does with this

Detection lives in [`src/utils/errors.ts`](../src/utils/errors.ts):
`getContextOverflowInfo(error, context)` returns the kind of overflow plus
whatever numbers the provider disclosed, or `null` for anything compaction
cannot fix.

Two kinds are distinguished:

- `context_window` — the input exceeded the model's window.
- `request_too_large` — a single request exceeded a per-minute token
  allowance. Waiting cannot help, because the request can never fit the
  bucket; only a smaller prompt can. This is separated from ordinary
  throttling numerically: it counts only when `Requested >= Limit`. When
  `Requested < Limit`, the account was merely busy and a retry is correct.

Deliberately excluded, with fixtures pinning each: genuine RPM/TPM throttling,
quota and billing failures, authentication failures, output-token-cap errors
(`max_tokens` too large), and Bedrock's invalid-model and legacy-model errors.

Vertex AI's reasonless 400 is classified only when the caller can corroborate
it — the SDK passes its own estimate of the prompt it just sent, and the
signature counts as overflow only when that estimate had reached 80% of the
budget. Without corroboration the error propagates untouched.

## Recovery

[`src/llm/contextOverflowRecovery.ts`](../src/llm/contextOverflowRecovery.ts)
turns a detection into a budget for the retry:

1. **Believe the provider over the config.** If the error names a ceiling, that
   becomes the new budget — this is how a `maxContextTokens` that was simply
   wrong gets corrected mid-run.
2. **Reserve what the completion will cost.** Several providers count the
   requested `max_tokens` against the same ceiling and quote a combined total —
   OpenRouter's `you requested about 56827 tokens (56811 of text input, 16 in
the output)`, DeepSeek's `(1179652 in the messages, 16 in the completion)`,
   and OpenAI's 429 `Requested 480002`. The retry budget governs the prompt
   only, so a known completion allowance comes off the ceiling first;
   otherwise a large `max_tokens` keeps the request over the limit no matter
   how far the prompt is compacted.
3. **Convert units.** If the error names the size it attributed to _the prompt_,
   the ratio against our own estimate is the conversion factor between the two
   tokenizers. Targeting a raw ceiling from a provider that counts 1.33× our
   tokens would just overflow again. This uses `promptTokens`, never
   `requestedTokens` — calibrating on a completion-inclusive total would invent
   a ratio and shrink the prompt far past what the overflow calls for, so
   providers that quote only a total (OpenAI's 429, xAI) contribute no ratio.
4. **Shrink blindly when nothing was disclosed** — 70% of the prompt that was
   rejected.
5. **Bound it.** Two forced compactions per agent per run, reset at the start of
   each turn. After that the error propagates, so a model that rejects
   everything cannot make a run compact forever — and a single overflow cannot
   permanently shrink the budget for every later turn.

In [`Graph.ts`](../src/graphs/Graph.ts), a failed model call that classifies as
overflow applies the corrected budget and returns a summarization request
instead of throwing. The graph routes to the summarize node, compacts, and
comes back to the agent node, which re-prunes against the corrected budget and
retries — the caller sees a slightly longer turn, not an error.

Fallback providers still run for every other failure, and for an overflow whose
recovery budget is spent. They are skipped on the first overflow on purpose:
an overflow is caused by the payload, not by the provider being unavailable, so
re-sending the same oversized prompt elsewhere is unlikely to help.

When `summarizationEnabled` is `false`, the summarize node performs no model
call — the corrected budget plus a re-prune is the entire fix, and the caller
never pays for a summarization they opted out of.

## Re-running the probe

```bash
DOTENV_CONFIG_PATH=./.env node --loader ./tsconfig-paths-bootstrap.mjs --experimental-specifier-resolution=node ./src/scripts/context-overflow-probe.ts --list
```

Drop `--list` to execute. Useful flags: `--only <provider,…>`, `--model
<substring>`, `--tier full|confirm`, `--mode stream|invoke|both`, `--factor <n>`,
`--out <path>`.

Over-limit requests are rejected at validation, so providers do not bill the
prompt. Two cautions:

- A payload that lands _under_ the limit **is** billed in full. The probe sizes
  payloads in single-token words so the word count is a guaranteed lower bound
  on tokens, and the `confirm` tier overshoots 2× so a stale context-window
  figure cannot cause an accidental accept.
- `--factor` bypasses the safety floor. It exists because on accounts whose TPM
  ceiling sits below the context window, the only way to observe the true
  context rejection is to land the payload between the two.
