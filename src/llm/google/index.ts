/* eslint-disable @typescript-eslint/ban-ts-comment */
import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { getEnvironmentVariable } from '@langchain/core/utils/env';
import { GoogleGenerativeAI as GenerativeAI } from '@google/generative-ai';
import type {
  GenerateContentRequest,
  SafetySetting,
} from '@google/generative-ai';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';
import type { BaseMessage, UsageMetadata } from '@langchain/core/messages';
import type { GeminiGenerationConfig } from '@langchain/google-common';
import type { GeminiApiUsageMetadata } from './types';
import type { GoogleClientOptions } from '@/types';
import {
  convertResponseContentToChatGenerationChunk,
  convertBaseMessagesToContent,
} from './utils/common';

export class CustomChatGoogleGenerativeAI extends ChatGoogleGenerativeAI {
  thinkingConfig?: GeminiGenerationConfig['thinkingConfig'];
  constructor(fields: GoogleClientOptions) {
    super(fields);

    this.model = fields.model.replace(/^models\//, '');

    this.maxOutputTokens = fields.maxOutputTokens ?? this.maxOutputTokens;

    if (this.maxOutputTokens != null && this.maxOutputTokens < 0) {
      throw new Error('`maxOutputTokens` must be a positive integer');
    }

    this.temperature = fields.temperature ?? this.temperature;
    if (
      this.temperature != null &&
      (this.temperature < 0 || this.temperature > 2)
    ) {
      throw new Error('`temperature` must be in the range of [0.0,2.0]');
    }

    this.topP = fields.topP ?? this.topP;
    if (this.topP != null && this.topP < 0) {
      throw new Error('`topP` must be a positive integer');
    }

    if (this.topP != null && this.topP > 1) {
      throw new Error('`topP` must be below 1.');
    }

    this.topK = fields.topK ?? this.topK;
    if (this.topK != null && this.topK < 0) {
      throw new Error('`topK` must be a positive integer');
    }

    this.stopSequences = fields.stopSequences ?? this.stopSequences;

    this.apiKey = fields.apiKey ?? getEnvironmentVariable('GOOGLE_API_KEY');
    if (this.apiKey == null || this.apiKey === '') {
      throw new Error(
        'Please set an API key for Google GenerativeAI ' +
          'in the environment variable GOOGLE_API_KEY ' +
          'or in the `apiKey` field of the ' +
          'ChatGoogleGenerativeAI constructor'
      );
    }

    this.safetySettings = fields.safetySettings ?? this.safetySettings;
    if (this.safetySettings && this.safetySettings.length > 0) {
      const safetySettingsSet = new Set(
        this.safetySettings.map((s) => s.category)
      );
      if (safetySettingsSet.size !== this.safetySettings.length) {
        throw new Error(
          'The categories in `safetySettings` array must be unique'
        );
      }
    }

    this.thinkingConfig = fields.thinkingConfig ?? this.thinkingConfig;

    this.streaming = fields.streaming ?? this.streaming;
    this.json = fields.json;

    // @ts-ignore - Accessing private property from parent class
    this.client = new GenerativeAI(this.apiKey).getGenerativeModel(
      {
        model: this.model,
        safetySettings: this.safetySettings as SafetySetting[],
        generationConfig: {
          stopSequences: this.stopSequences,
          maxOutputTokens: this.maxOutputTokens,
          temperature: this.temperature,
          topP: this.topP,
          topK: this.topK,
          ...(this.json != null
            ? { responseMimeType: 'application/json' }
            : {}),
        },
      },
      {
        apiVersion: fields.apiVersion,
        baseUrl: fields.baseUrl,
        customHeaders: fields.customHeaders,
      }
    );
    this.streamUsage = fields.streamUsage ?? this.streamUsage;
  }

  static lc_name(): 'LibreChatGoogleGenerativeAI' {
    return 'LibreChatGoogleGenerativeAI';
  }

  invocationParams(
    options?: this['ParsedCallOptions']
  ): Omit<GenerateContentRequest, 'contents'> {
    const params = super.invocationParams(options);
    if (this.thinkingConfig) {
      /** @ts-ignore */
      this.client.generationConfig = {
        /** @ts-ignore */
        ...this.client.generationConfig,
        /** @ts-ignore */
        thinkingConfig: this.thinkingConfig,
      };
    }
    return params;
  }

  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const prompt = convertBaseMessagesToContent(
      messages,
      this._isMultimodalModel,
      this.useSystemInstruction
    );
    let actualPrompt = prompt;
    if (prompt?.[0].role === 'system') {
      const [systemInstruction] = prompt;
      /** @ts-ignore */
      this.client.systemInstruction = systemInstruction;
      actualPrompt = prompt.slice(1);
    }
    const parameters = this.invocationParams(options);
    const request = {
      ...parameters,
      contents: actualPrompt,
    };
    const stream = await this.caller.callWithOptions(
      { signal: options.signal },
      async () => {
        /** @ts-ignore */
        const { stream } = await this.client.generateContentStream(request);
        return stream;
      }
    );

    let index = 0;
    let lastUsageMetadata: UsageMetadata | undefined;
    for await (const response of stream) {
      if (
        'usageMetadata' in response &&
        this.streamUsage !== false &&
        options.streamUsage !== false
      ) {
        const genAIUsageMetadata = response.usageMetadata as
          | GeminiApiUsageMetadata
          | undefined;

        const output_tokens =
          (genAIUsageMetadata?.candidatesTokenCount ?? 0) +
          (genAIUsageMetadata?.thoughtsTokenCount ?? 0);
        lastUsageMetadata = {
          input_tokens: genAIUsageMetadata?.promptTokenCount ?? 0,
          output_tokens,
          total_tokens: genAIUsageMetadata?.totalTokenCount ?? 0,
        };
      }

      const chunk = convertResponseContentToChatGenerationChunk(response, {
        usageMetadata: undefined,
        index,
      });
      index += 1;
      if (!chunk) {
        continue;
      }

      yield chunk;
      await runManager?.handleLLMNewToken(
        chunk.text || '',
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk }
      );
    }

    if (lastUsageMetadata) {
      const finalChunk = new ChatGenerationChunk({
        text: '',
        message: new AIMessageChunk({
          content: '',
          usage_metadata: lastUsageMetadata,
        }),
      });
      yield finalChunk;
      await runManager?.handleLLMNewToken(
        finalChunk.text || '',
        undefined,
        undefined,
        undefined,
        undefined,
        { chunk: finalChunk }
      );
    }
  }
}
