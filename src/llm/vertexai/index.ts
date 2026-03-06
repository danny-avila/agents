import { ChatGoogle } from '@langchain/google-gauth';
import { ChatConnection } from '@langchain/google-common';
import type {
  GeminiContent,
  GeminiRequest,
  GoogleAIModelRequestParams,
  GoogleAbstractedClient,
} from '@langchain/google-common';
import type { BaseMessage } from '@langchain/core/messages';
import { isAIMessage } from '@langchain/core/messages';
import type { GoogleThinkingConfig, VertexAIClientOptions } from '@/types';

type AdditionalKwargs =
  | undefined
  | (BaseMessage['additional_kwargs'] & {
      signatures?: Array<string | undefined>;
    });

/**
 * Fixes thought signatures on functionCall parts in the formatted Gemini request.
 *
 * `@langchain/google-common` stores signatures as a flat array in
 * `additional_kwargs.signatures` (one per response part) and re-attaches them
 * by index only when `signatures.length === parts.length`. This fails when:
 * - The API omits a signature (length mismatch)
 * - Streaming chunks merge with different part counts
 * - The signature for a functionCall part is an empty string
 *
 * This function correlates each "model" content block in the formatted request
 * back to its originating AI message, then re-attaches non-empty signatures
 * that the library failed to apply.
 */
function fixThoughtSignatures(
  contents: GeminiContent[],
  input: BaseMessage[]
): void {
  // Collect AI messages that have signatures, in order
  const aiMessages = input.filter(
    (msg) =>
      isAIMessage(msg) &&
      Array.isArray((msg.additional_kwargs as AdditionalKwargs)?.signatures) &&
      (msg.additional_kwargs.signatures as string[]).length > 0
  );

  // Collect "model" content blocks from the formatted request, in order
  const modelContents = contents.filter((c) => c.role === 'model');

  // They should correspond 1:1 in order (both derived from the same input sequence)
  const count = Math.min(aiMessages.length, modelContents.length);
  for (let i = 0; i < count; i++) {
    const msg = aiMessages[i];
    const content = modelContents[i];
    const signatures = (msg.additional_kwargs as AdditionalKwargs)?.signatures;

    // Collect non-empty signatures that aren't already attached to any part
    const attachedSignatures = new Set(
      content.parts
        .map((p) => p.thoughtSignature)
        .filter((s): s is string => s != null && s !== '')
    );
    const availableSignatures = signatures?.filter(
      (s) => s != null && s !== '' && !attachedSignatures.has(s)
    );

    // Assign available signatures to functionCall parts missing one, in order
    let sigIdx = 0;
    for (const part of content.parts) {
      if (
        'functionCall' in part &&
        (part.thoughtSignature == null || part.thoughtSignature === '') &&
        sigIdx < (availableSignatures?.length ?? 0)
      ) {
        part.thoughtSignature = availableSignatures?.[sigIdx];
        sigIdx++;
      }
    }
  }
}

class CustomChatConnection extends ChatConnection<VertexAIClientOptions> {
  thinkingConfig?: GoogleThinkingConfig;

  async formatData(
    input: BaseMessage[],
    parameters: GoogleAIModelRequestParams
  ): Promise<unknown> {
    const formattedData = (await super.formatData(
      input,
      parameters
    )) as GeminiRequest;
    if (formattedData.generationConfig?.thinkingConfig?.thinkingBudget === -1) {
      // -1 means "let the model decide" - delete the property so the API doesn't receive an invalid value
      if (
        formattedData.generationConfig.thinkingConfig.includeThoughts === false
      ) {
        formattedData.generationConfig.thinkingConfig.includeThoughts = true;
      }
      delete formattedData.generationConfig.thinkingConfig.thinkingBudget;
    }
    if (
      this.thinkingConfig?.thinkingLevel != null &&
      this.thinkingConfig.thinkingLevel !== ''
    ) {
      formattedData.generationConfig ??= {};
      // thinkingLevel and thinkingBudget cannot coexist — the API rejects the request.
      // Remove thinkingBudget when thinkingLevel is set.
      const { thinkingBudget: _, ...existingThinkingConfig } =
        (formattedData.generationConfig.thinkingConfig as
          | Record<string, unknown>
          | undefined) ?? {};
      (
        formattedData.generationConfig as Record<string, unknown>
      ).thinkingConfig = {
        ...existingThinkingConfig,
        thinkingLevel: this.thinkingConfig.thinkingLevel,
        ...(this.thinkingConfig.includeThoughts != null && {
          includeThoughts: this.thinkingConfig.includeThoughts,
        }),
      };
    }
    if (formattedData.contents) {
      fixThoughtSignatures(formattedData.contents, input);
      // gemini-3.1+ models reject role="function"; convert to role="user"
      for (const content of formattedData.contents) {
        if (content.role === 'function') {
          (content as { role: string }).role = 'user';
        }
      }
    }
    return formattedData;
  }
}

/**
 * Integration with Google Vertex AI chat models.
 *
 * Setup:
 * Install `@langchain/google-vertexai` and set your stringified
 * Vertex AI credentials as an environment variable named `GOOGLE_APPLICATION_CREDENTIALS`.
 *
 * ```bash
 * npm install @langchain/google-vertexai
 * export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials"
 * ```
 *
 * ## [Constructor args](https://api.js.langchain.com/classes/_langchain_google_vertexai.index.ChatVertexAI.html#constructor.new_ChatVertexAI)
 *
 * ## [Runtime args](https://api.js.langchain.com/interfaces/langchain_google_common_types.GoogleAIBaseLanguageModelCallOptions.html)
 *
 * Runtime args can be passed as the second argument to any of the base runnable methods `.invoke`. `.stream`, `.batch`, etc.
 * They can also be passed via `.withConfig`, or the second arg in `.bindTools`, like shown in the examples below:
 *
 * ```typescript
 * // When calling `.withConfig`, call options should be passed via the first argument
 * const llmWithArgsBound = llm.withConfig({
 *   stop: ["\n"],
 *   tools: [...],
 * });
 *
 * // When calling `.bindTools`, call options should be passed via the second argument
 * const llmWithTools = llm.bindTools(
 *   [...],
 *   {
 *     tool_choice: "auto",
 *   }
 * );
 * ```
 *
 * ## Examples
 *
 * <details open>
 * <summary><strong>Instantiate</strong></summary>
 *
 * ```typescript
 * import { ChatVertexAI } from '@langchain/google-vertexai';
 *
 * const llm = new ChatVertexAI({
 *   model: "gemini-1.5-pro",
 *   temperature: 0,
 *   // other params...
 * });
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Invoking</strong></summary>
 *
 * ```typescript
 * const input = `Translate "I love programming" into French.`;
 *
 * // Models also accept a list of chat messages or a formatted prompt
 * const result = await llm.invoke(input);
 * console.log(result);
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "\"J'adore programmer\" \n\nHere's why this is the best translation:\n\n* **J'adore** means \"I love\" and conveys a strong passion.\n* **Programmer** is the French verb for \"to program.\"\n\nThis translation is natural and idiomatic in French. \n",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": [],
 *   "usage_metadata": {
 *     "input_tokens": 9,
 *     "output_tokens": 63,
 *     "total_tokens": 72
 *   }
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Streaming Chunks</strong></summary>
 *
 * ```typescript
 * for await (const chunk of await llm.stream(input)) {
 *   console.log(chunk);
 * }
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "\"",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": []
 * }
 * AIMessageChunk {
 *   "content": "J'adore programmer\" \n",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": []
 * }
 * AIMessageChunk {
 *   "content": "",
 *   "additional_kwargs": {},
 *   "response_metadata": {},
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": []
 * }
 * AIMessageChunk {
 *   "content": "",
 *   "additional_kwargs": {},
 *   "response_metadata": {
 *     "finishReason": "stop"
 *   },
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": [],
 *   "usage_metadata": {
 *     "input_tokens": 9,
 *     "output_tokens": 8,
 *     "total_tokens": 17
 *   }
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Aggregate Streamed Chunks</strong></summary>
 *
 * ```typescript
 * import { AIMessageChunk } from '@langchain/core/messages';
 * import { concat } from '@langchain/core/utils/stream';
 *
 * const stream = await llm.stream(input);
 * let full: AIMessageChunk | undefined;
 * for await (const chunk of stream) {
 *   full = !full ? chunk : concat(full, chunk);
 * }
 * console.log(full);
 * ```
 *
 * ```txt
 * AIMessageChunk {
 *   "content": "\"J'adore programmer\" \n",
 *   "additional_kwargs": {},
 *   "response_metadata": {
 *     "finishReason": "stop"
 *   },
 *   "tool_calls": [],
 *   "tool_call_chunks": [],
 *   "invalid_tool_calls": [],
 *   "usage_metadata": {
 *     "input_tokens": 9,
 *     "output_tokens": 8,
 *     "total_tokens": 17
 *   }
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Bind tools</strong></summary>
 *
 * ```typescript
 * import { z } from 'zod';
 *
 * const GetWeather = {
 *   name: "GetWeather",
 *   description: "Get the current weather in a given location",
 *   schema: z.object({
 *     location: z.string().describe("The city and state, e.g. San Francisco, CA")
 *   }),
 * }
 *
 * const GetPopulation = {
 *   name: "GetPopulation",
 *   description: "Get the current population in a given location",
 *   schema: z.object({
 *     location: z.string().describe("The city and state, e.g. San Francisco, CA")
 *   }),
 * }
 *
 * const llmWithTools = llm.bindTools([GetWeather, GetPopulation]);
 * const aiMsg = await llmWithTools.invoke(
 *   "Which city is hotter today and which is bigger: LA or NY?"
 * );
 * console.log(aiMsg.tool_calls);
 * ```
 *
 * ```txt
 * [
 *   {
 *     name: 'GetPopulation',
 *     args: { location: 'New York City, NY' },
 *     id: '33c1c1f47e2f492799c77d2800a43912',
 *     type: 'tool_call'
 *   }
 * ]
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Structured Output</strong></summary>
 *
 * ```typescript
 * import { z } from 'zod';
 *
 * const Joke = z.object({
 *   setup: z.string().describe("The setup of the joke"),
 *   punchline: z.string().describe("The punchline to the joke"),
 *   rating: z.number().optional().describe("How funny the joke is, from 1 to 10")
 * }).describe('Joke to tell user.');
 *
 * const structuredLlm = llm.withStructuredOutput(Joke, { name: "Joke" });
 * const jokeResult = await structuredLlm.invoke("Tell me a joke about cats");
 * console.log(jokeResult);
 * ```
 *
 * ```txt
 * {
 *   setup: 'What do you call a cat that loves to bowl?',
 *   punchline: 'An alley cat!'
 * }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Usage Metadata</strong></summary>
 *
 * ```typescript
 * const aiMsgForMetadata = await llm.invoke(input);
 * console.log(aiMsgForMetadata.usage_metadata);
 * ```
 *
 * ```txt
 * { input_tokens: 9, output_tokens: 8, total_tokens: 17 }
 * ```
 * </details>
 *
 * <br />
 *
 * <details>
 * <summary><strong>Stream Usage Metadata</strong></summary>
 *
 * ```typescript
 * const streamForMetadata = await llm.stream(
 *   input,
 *   {
 *     streamUsage: true
 *   }
 * );
 * let fullForMetadata: AIMessageChunk | undefined;
 * for await (const chunk of streamForMetadata) {
 *   fullForMetadata = !fullForMetadata ? chunk : concat(fullForMetadata, chunk);
 * }
 * console.log(fullForMetadata?.usage_metadata);
 * ```
 *
 * ```txt
 * { input_tokens: 9, output_tokens: 8, total_tokens: 17 }
 * ```
 * </details>
 *
 * <br />
 */
export class ChatVertexAI extends ChatGoogle {
  lc_namespace = ['langchain', 'chat_models', 'vertexai'];
  dynamicThinkingBudget = false;
  thinkingConfig?: GoogleThinkingConfig;

  static lc_name(): 'LibreChatVertexAI' {
    return 'LibreChatVertexAI';
  }

  constructor(fields?: VertexAIClientOptions) {
    const dynamicThinkingBudget = fields?.thinkingBudget === -1;
    super({
      ...fields,
      platformType: 'gcp',
    });
    this.dynamicThinkingBudget = dynamicThinkingBudget;
    this.thinkingConfig = fields?.thinkingConfig;
  }
  invocationParams(
    options?: this['ParsedCallOptions'] | undefined
  ): GoogleAIModelRequestParams {
    const params = super.invocationParams(options);
    if (this.dynamicThinkingBudget) {
      params.maxReasoningTokens = -1;
    }
    return params;
  }
  buildConnection(
    fields: VertexAIClientOptions | undefined,
    client: GoogleAbstractedClient
  ): void {
    // Note: buildConnection is called from super() BEFORE this.thinkingConfig is set,
    // so we must read thinkingConfig from `fields` directly.
    const thinkingConfig = fields?.thinkingConfig ?? this.thinkingConfig;

    const connection = new CustomChatConnection(
      { ...fields, ...this },
      this.caller,
      client,
      false
    );
    connection.thinkingConfig = thinkingConfig;
    this.connection = connection;

    const streamedConnection = new CustomChatConnection(
      { ...fields, ...this },
      this.caller,
      client,
      true
    );
    streamedConnection.thinkingConfig = thinkingConfig;
    this.streamedConnection = streamedConnection;
  }
}
