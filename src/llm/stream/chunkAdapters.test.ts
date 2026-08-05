import { AIMessageChunk } from '@langchain/core/messages';
import { ChatGenerationChunk } from '@langchain/core/outputs';
import {
  toGenerationSmoothItem,
  cloneGenerationChunkPiece,
} from './chunkAdapters';

type MessageChunkFields = {
  usage_metadata?: AIMessageChunk['usage_metadata'];
  additional_kwargs?: AIMessageChunk['additional_kwargs'];
  response_metadata?: AIMessageChunk['response_metadata'];
};

function textChunk(
  text: string,
  extra: MessageChunkFields = {}
): ChatGenerationChunk {
  return new ChatGenerationChunk({
    text,
    message: new AIMessageChunk({ content: text, ...extra }),
  });
}

describe('toGenerationSmoothItem classification', () => {
  it('splits plain string-content text chunks', () => {
    const item = toGenerationSmoothItem(textChunk('alpha beta gamma'));
    expect(item.smooth).toBe(true);
    expect(item.atomic).toBeUndefined();
    expect(item.text).toBe('alpha beta gamma');
  });

  it('keeps chunks carrying reasoning_content kwargs atomic', () => {
    const chunk = textChunk('visible text here', {
      additional_kwargs: { reasoning_content: 'hidden thought' },
    });
    const item = toGenerationSmoothItem(chunk);
    expect(item.smooth).toBe(true);
    expect(item.atomic).toBe(true);
    expect(item.emit({ text: item.text, isFirst: true, isLast: true })).toBe(
      chunk
    );
  });

  it('keeps chunks carrying a reasoning summary object atomic', () => {
    const chunk = textChunk('visible text here', {
      additional_kwargs: { reasoning: { summary: [{ text: 'thought' }] } },
    });
    expect(toGenerationSmoothItem(chunk).atomic).toBe(true);
  });

  it('keeps chunks carrying OpenRouter reasoning_details atomic', () => {
    const chunk = textChunk('visible text here', {
      additional_kwargs: {
        reasoning_details: [{ type: 'reasoning.text', text: 'thought' }],
      },
    });
    expect(toGenerationSmoothItem(chunk).atomic).toBe(true);
  });

  it('keeps chunks carrying camelCase finishReason atomic', () => {
    const chunk = new ChatGenerationChunk({
      text: 'final text with several words here',
      generationInfo: { finishReason: 'STOP' },
      message: new AIMessageChunk({
        content: 'final text with several words here',
      }),
    });
    const item = toGenerationSmoothItem(chunk);
    expect(item.atomic).toBe(true);
    expect(item.emit({ text: item.text, isFirst: true, isLast: true })).toBe(
      chunk
    );
  });

  it('classifies usage-only chunks as passthrough', () => {
    const chunk = new ChatGenerationChunk({
      text: '',
      message: new AIMessageChunk({
        content: '',
        usage_metadata: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
      }),
    });
    const item = toGenerationSmoothItem(chunk);
    expect(item.smooth).toBe(false);
    expect(item.text).toBe('');
  });
});

describe('cloneGenerationChunkPiece metadata scoping', () => {
  const chunk = textChunk('alpha beta gamma', {
    usage_metadata: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
    additional_kwargs: { annotation: 'once' },
    response_metadata: { model_name: 'test-model' },
  });

  it('keeps kwargs, response metadata and usage on the first piece only', () => {
    const first = cloneGenerationChunkPiece(chunk, {
      text: 'alpha ',
      isFirst: true,
      isLast: false,
    });
    const later = cloneGenerationChunkPiece(chunk, {
      text: 'beta ',
      isFirst: false,
      isLast: false,
    });

    const firstMessage = first.message as AIMessageChunk;
    const laterMessage = later.message as AIMessageChunk;
    expect(firstMessage.additional_kwargs).toEqual({ annotation: 'once' });
    expect(firstMessage.response_metadata).toEqual({
      model_name: 'test-model',
    });
    expect(firstMessage.usage_metadata).toBeDefined();
    expect(laterMessage.additional_kwargs).toEqual({});
    expect(laterMessage.response_metadata).toEqual({});
    expect(laterMessage.usage_metadata).toBeUndefined();
  });

  it('returns the original chunk for unsplit pieces', () => {
    expect(
      cloneGenerationChunkPiece(chunk, {
        text: 'alpha beta gamma',
        isFirst: true,
        isLast: true,
      })
    ).toBe(chunk);
  });
});
