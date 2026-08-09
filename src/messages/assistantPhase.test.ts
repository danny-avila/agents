import {
  getAssistantTextPhase,
  getMessageCreationContentMetadata,
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
});
