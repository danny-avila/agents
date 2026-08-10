import {
  getAssistantTextPhase,
  getMessageCreationContentMetadata,
  splitAssistantTextContentByPhase,
} from './assistantPhase';
import { ContentTypes } from '@/common';

describe('assistant phase metadata', () => {
  it('reads provider-native Open Responses phase metadata', () => {
    const part = {
      type: ContentTypes.TEXT,
      text: '',
      phase: 'commentary' as const,
    };

    expect(getAssistantTextPhase(part)).toBe('commentary');
    expect(getMessageCreationContentMetadata([part])).toEqual({
      content_type: ContentTypes.TEXT,
      phase: 'commentary',
    });
  });

  it('reads LangChain standard-content extras', () => {
    const part = {
      type: ContentTypes.TEXT,
      text: '',
      extras: { phase: 'final_answer' as const },
    };

    expect(getMessageCreationContentMetadata([part])).toEqual({
      content_type: ContentTypes.TEXT,
      phase: 'final_answer',
    });
  });

  it('announces reasoning without inventing a text phase', () => {
    expect(
      getMessageCreationContentMetadata([
        { type: ContentTypes.THINK, think: 'checking' },
      ])
    ).toEqual({ content_type: ContentTypes.THINK });
  });

  it('preserves every provider phase in a multi-block text chunk', () => {
    const commentary = {
      type: ContentTypes.TEXT,
      text: 'I will inspect the session path.',
      phase: 'commentary' as const,
    };
    const moreCommentary = {
      type: ContentTypes.TEXT,
      text: 'The refresh branch is stale.',
      phase: 'commentary' as const,
    };
    const finalAnswer = {
      type: ContentTypes.TEXT,
      text: 'The refresh path is fixed.',
      phase: 'final_answer' as const,
    };

    const groups = splitAssistantTextContentByPhase([
      commentary,
      moreCommentary,
      finalAnswer,
    ]);

    expect(groups).toEqual([[commentary, moreCommentary], [finalAnswer]]);
    expect(
      groups.map((group) => getMessageCreationContentMetadata(group))
    ).toEqual([
      { content_type: ContentTypes.TEXT, phase: 'commentary' },
      { content_type: ContentTypes.TEXT, phase: 'final_answer' },
    ]);
  });
});
