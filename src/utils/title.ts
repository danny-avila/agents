import { z } from 'zod';
import { RunnableLambda } from '@langchain/core/runnables';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import type { Runnable, RunnableConfig } from '@langchain/core/runnables';
import type { AIMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ContentTypes } from '@/common';

const defaultTitlePrompt = `Analyze this conversation and provide:
1. The detected language of the conversation
2. A concise title in the detected language (5 words or less, no punctuation or quotation)

{convo}`;

const titleSchema = z.object({
  title: z
    .string()
    .describe(
      'A concise title for the conversation in 5 words or less, without punctuation or quotation'
    ),
});

const combinedSchema = z.object({
  language: z.string().describe('The detected language of the conversation'),
  title: z
    .string()
    .describe(
      'A concise title for the conversation in 5 words or less, without punctuation or quotation'
    ),
});

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
  );

  return new RunnableLambda({
    func: async (
      input: {
        convo: string;
        inputText: string;
        skipLanguage: boolean;
      },
      config?: Partial<RunnableConfig>
    ): Promise<{ language: string; title: string } | { title: string }> => {
      if (input.skipLanguage) {
        return (await titlePrompt.pipe(titleLLM).invoke(
          {
            convo: input.convo,
          },
          config
        )) as { title: string };
      }

      const result = (await titlePrompt.pipe(combinedLLM).invoke(
        {
          convo: input.convo,
        },
        config
      )) as { language: string; title: string } | undefined;

      return {
        language: result?.language ?? 'English',
        title: result?.title ?? '',
      };
    },
  });
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
  );

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
  });

  const chain = completionPrompt.pipe(model).pipe(extractContent);

  return new RunnableLambda({
    func: async (
      input: {
        convo: string;
        inputText: string;
        skipLanguage: boolean;
      },
      config?: Partial<RunnableConfig>
    ): Promise<{ title: string }> => {
      return await chain.invoke({ convo: input.convo }, config);
    },
  });
};
