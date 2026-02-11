import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableLambda, RunnableSequence } from '@langchain/core/runnables';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type { AIMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes } from '@/common';

const defaultTitlePrompt = `Analyze this conversation and provide:
1. The detected language of the conversation
2. A concise title in the detected language (5 words or less, no punctuation or quotation)

{convo}`;

const titleSchema = {
  type: 'object',
  properties: {
    title: {
      type: 'string',
      description:
        'A concise title for the conversation in 5 words or less, without punctuation or quotation',
    },
  },
  required: ['title'],
} as const;

const combinedSchema = {
  type: 'object',
  properties: {
    language: {
      type: 'string',
      description: 'The detected language of the conversation',
    },
    title: {
      type: 'string',
      description:
        'A concise title for the conversation in 5 words or less, without punctuation or quotation',
    },
  },
  required: ['language', 'title'],
} as const;

export const createTitleRunnable = async (
  model: t.ChatModelInstance,
  _titlePrompt?: string
): Promise<Runnable> => {
  // Disabled since this works fine
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  /* @ts-ignore */
  const titleLLM = model.withStructuredOutput(titleSchema);
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  /* @ts-ignore */
  const combinedLLM = model.withStructuredOutput(combinedSchema);

  const titlePrompt = ChatPromptTemplate.fromTemplate(
    _titlePrompt ?? defaultTitlePrompt
  ).withConfig({ runName: 'TitlePrompt' });

  const titleOnlyInnerChain = RunnableSequence.from([titlePrompt, titleLLM]);
  const combinedInnerChain = RunnableSequence.from([titlePrompt, combinedLLM]);

  /** Wrap titleOnlyChain in RunnableLambda to create parent span */
  const titleOnlyChain = new RunnableLambda({
    func: async (
      input: { convo: string },
      config?: Partial<RunnableConfig>
    ): Promise<{ title: string }> => {
      const result = await titleOnlyInnerChain.invoke(input, config);
      return result as { title: string };
    },
  }).withConfig({ runName: 'TitleOnlyChain' });

  /** Wrap combinedChain in RunnableLambda to create parent span */
  const combinedChain = new RunnableLambda({
    func: async (
      input: { convo: string },
      config?: Partial<RunnableConfig>
    ): Promise<{ language: string; title: string }> => {
      const result = await combinedInnerChain.invoke(input, config);
      return result as { language: string; title: string };
    },
  }).withConfig({ runName: 'TitleLanguageChain' });

  /** Runnable to add default values if needed */
  const addDefaults = new RunnableLambda({
    func: (
      result: { language: string; title: string } | undefined
    ): { language: string; title: string } => ({
      language: result?.language ?? 'English',
      title: result?.title ?? '',
    }),
  }).withConfig({ runName: 'AddDefaults' });

  const combinedChainInner = RunnableSequence.from([
    combinedChain,
    addDefaults,
  ]);

  /** Wrap combinedChainWithDefaults in RunnableLambda to create parent span */
  const combinedChainWithDefaults = new RunnableLambda({
    func: async (
      input: { convo: string },
      config?: Partial<RunnableConfig>
    ): Promise<{ language: string; title: string }> => {
      return await combinedChainInner.invoke(input, config);
    },
  }).withConfig({ runName: 'CombinedChainWithDefaults' });

  return new RunnableLambda({
    func: async (
      input: {
        convo: string;
        inputText: string;
        skipLanguage: boolean;
      },
      config?: Partial<RunnableConfig>
    ): Promise<{ language: string; title: string } | { title: string }> => {
      const invokeInput = { convo: input.convo };

      if (input.skipLanguage) {
        return (await titleOnlyChain.invoke(invokeInput, config)) as {
          title: string;
        };
      }

      return await combinedChainWithDefaults.invoke(invokeInput, config);
    },
  }).withConfig({ runName: 'TitleGenerator' });
};

const defaultCompletionPrompt = `Provide a concise, 5-word-or-less title for the conversation, using title case conventions. Only return the title itself.

Conversation:
{convo}`;

export const createCompletionTitleRunnable = async (
  model: t.ChatModelInstance,
  titlePrompt?: string
): Promise<Runnable> => {
  const completionPrompt = ChatPromptTemplate.fromTemplate(
    titlePrompt ?? defaultCompletionPrompt
  ).withConfig({ runName: 'CompletionTitlePrompt' });

  /** Runnable to extract content from model response */
  const extractContent = new RunnableLambda({
    func: (response: AIMessage): { title: string } => {
      let content = '';
      if (typeof response.content === 'string') {
        content = response.content;
      } else if (Array.isArray(response.content)) {
        content = response.content
          .filter(
            (part): part is { type: ContentTypes.TEXT; text: string } =>
              part.type === ContentTypes.TEXT
          )
          .map((part) => part.text)
          .join('');
      }
      return { title: content.trim() };
    },
  }).withConfig({ runName: 'ExtractTitle' });

  const innerChain = RunnableSequence.from([
    completionPrompt,
    model,
    extractContent,
  ]);

  /** Wrap in RunnableLambda to create a parent span for LangFuse */
  return new RunnableLambda({
    func: async (
      input: { convo: string },
      config?: Partial<RunnableConfig>
    ): Promise<{ title: string }> => {
      return await innerChain.invoke(input, config);
    },
  }).withConfig({ runName: 'CompletionTitleChain' });
};
