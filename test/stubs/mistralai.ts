// test/stubs/mistralai.ts
/**
 * Jest stand-in for `@langchain/mistralai`.
 *
 * The real package requires `@mistralai/mistralai`, which is published
 * `"type": "module"` with no CommonJS build. Jest's CJS runtime cannot load
 * it, and transforming it does not help — Node decides module kind from the
 * package's `type` field, not from what a transform emits. So any suite that
 * transitively imports `src/llm/providers.ts` — which pulls in the Mistral
 * class only to populate the provider constructor map — died at collection,
 * whether or not the test had anything to do with Mistral.
 *
 * The stub is a minimal functional double: the constructor records fields so
 * `CustomChatMistralAI` (which layers stream smoothing on top) can be
 * constructed and unit-tested. Any test that actually reaches the network
 * layer fails loudly in `_streamResponseChunks` instead of silently passing
 * against a hollow double — suites that need real Mistral streaming replace
 * that method (e.g. via `jest.spyOn`) or run outside Jest.
 */
export class ChatMistralAI {
  model?: string;
  apiKey?: string;

  constructor(fields?: { model?: string; apiKey?: string }) {
    this.model = fields?.model;
    this.apiKey = fields?.apiKey;
  }

  // eslint-disable-next-line require-yield
  async *_streamResponseChunks(): AsyncGenerator<never> {
    throw new Error(
      'ChatMistralAI is stubbed under Jest (test/stubs/mistralai.ts) because ' +
        '@mistralai/mistralai ships ESM-only and cannot load in the CJS test ' +
        'runtime. Replace `_streamResponseChunks` (jest.spyOn) or run outside ' +
        'Jest to exercise real Mistral streaming.'
    );
  }
}
