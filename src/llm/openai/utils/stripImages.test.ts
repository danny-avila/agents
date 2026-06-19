import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { stripImagesFromMessages } from './index';

const imagePart = {
  type: 'image_url' as const,
  image_url: { url: 'data:image/png;base64,AAAA' },
};

describe('stripImagesFromMessages', () => {
  it('returns the input unchanged when visionCapable is true', () => {
    const messages = [
      new HumanMessage({ content: [{ type: 'text', text: 'hi' }, imagePart] }),
    ];
    expect(stripImagesFromMessages(messages, true)).toBe(messages);
  });

  it('strips image_url parts but keeps text for non-vision models', () => {
    const messages = [
      new HumanMessage({
        content: [{ type: 'text', text: 'describe this' }, imagePart],
      }),
    ];
    const result = stripImagesFromMessages(messages, false);
    const content = result[0].content as Array<{ type: string }>;
    expect(content.some((p) => p.type === 'image_url')).toBe(false);
    expect(content.some((p) => p.type === 'text')).toBe(true);
  });

  it('inserts a text placeholder when an image-only message is emptied', () => {
    const messages = [new HumanMessage({ content: [imagePart] })];
    const result = stripImagesFromMessages(messages, false);
    const content = result[0].content as Array<{ type: string }>;
    expect(content.some((p) => p.type === 'image_url')).toBe(false);
    expect(content.length).toBeGreaterThan(0);
    expect(content[0].type).toBe('text');
  });

  it('leaves string content and image-free messages untouched', () => {
    const stringMsg = new HumanMessage({ content: 'plain text' });
    const noImageMsg = new AIMessage({
      content: [{ type: 'text', text: 'ok' }],
    });
    const result = stripImagesFromMessages([stringMsg, noImageMsg], false);
    expect(result[0]).toBe(stringMsg);
    expect(result[1]).toBe(noImageMsg);
  });

  it('does not mutate the original message when stripping', () => {
    const original = new HumanMessage({
      content: [{ type: 'text', text: 'q' }, imagePart],
    });
    stripImagesFromMessages([original], false);
    const content = original.content as Array<{ type: string }>;
    expect(content.some((p) => p.type === 'image_url')).toBe(true);
  });
});
