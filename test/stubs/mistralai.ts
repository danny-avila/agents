// test/stubs/mistralai.ts
/**
 * Jest stand-in for `@langchain/mistralai`.
 *
 * The real package requires `@mistralai/mistralai`, which is published
 * `"type": "module"` with no CommonJS build. Jest's CJS runtime cannot load
 * it, and transforming it does not help — Node decides module kind from the
 * package's `type` field, not from what a transform emits. So any suite that
 * transitively imports `src/llm/providers.ts` — which pulls in `ChatMistralAI`
 * only to populate the provider constructor map — died at collection, whether
 * or not the test had anything to do with Mistral.
 *
 * Stubbing is safe here because nothing under test constructs this class: the
 * `mistral` entry in `llmConfigs` routes through the OpenAI-compatible client,
 * and `ChatMistralAI` appears solely as a map value.
 *
 * It throws rather than no-ops so the assumption above stays honest — a test
 * that ever does exercise Mistral fails loudly here instead of silently
 * passing against a hollow double.
 */
export class ChatMistralAI {
  constructor() {
    throw new Error(
      'ChatMistralAI is stubbed under Jest (test/stubs/mistralai.ts) because ' +
        '@mistralai/mistralai ships ESM-only and cannot load in the CJS test ' +
        'runtime. If a test genuinely needs Mistral, run it outside Jest or ' +
        'give this stub a real implementation.'
    );
  }
}
