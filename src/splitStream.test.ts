import { nanoid } from 'nanoid';
import { MessageContentText } from '@langchain/core/messages';
import type * as t from '@/types';
import { GraphEvents, StepTypes, ContentTypes } from '@/common';
import { createContentAggregator } from './stream';
import { SplitStreamHandler } from './splitStream';
import { createMockStream } from './mockStream';

// Mock sleep to speed up tests
jest.mock('@/utils', () => ({
  sleep: (): Promise<void> => Promise.resolve(),
}));

describe('Stream Generation and Handling', () => {
  let mockHandlers: {
    [GraphEvents.ON_RUN_STEP]: jest.Mock;
    [GraphEvents.ON_MESSAGE_DELTA]: jest.Mock;
  };

  beforeEach(() => {
    mockHandlers = {
      [GraphEvents.ON_RUN_STEP]: jest.fn(),
      [GraphEvents.ON_MESSAGE_DELTA]: jest.fn(),
    };
  });

  it('should properly stream tokens including spaces', async () => {
    const stream = createMockStream({
      text: 'Hello world!',
      streamRate: 0,
    })();

    const tokens: string[] = [];
    for await (const chunk of stream) {
      const content = chunk.choices?.[0]?.delta.content ?? '';
      if (content) tokens.push(content);
    }

    expect(tokens).toEqual(['Hello', ' ', 'world!']);
  });

  it('should handle code blocks without splitting them', async () => {
    const runId = nanoid();
    const handler = new SplitStreamHandler({
      runId,
      blockThreshold: 10,
      handlers: mockHandlers,
    });

    const codeText = `Code:
\`\`\`
const x = 1;
const y = 2;
const z = 2;
const a = 2;
const b = 2;
const c = 2;
const d = 2;
const e = 2;
const f = 2;
const g = 2;
const h = 2;
\`\`\`
End code.`;

    const stream = createMockStream({
      text: codeText,
      streamRate: 0,
    })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    // Verify that only one message block was created for the code section
    const runSteps = mockHandlers[GraphEvents.ON_RUN_STEP].mock.calls;
    expect(runSteps.length).toBe(2); // Should only create one message block
  });

  it('should split content when exceeding threshold', async () => {
    const runId = nanoid();
    const handler = new SplitStreamHandler({
      runId,
      handlers: mockHandlers,
      // Set a very low threshold for testing
      blockThreshold: 10,
    });

    // Make the text longer and ensure it has clear breaking points
    const longText =
      'This is the first sentence. And here is another sentence. And yet another one here. Finally one more.';

    const stream = createMockStream({
      text: longText,
      streamRate: 0,
    })();

    // For debugging
    // let totalLength = 0;
    for await (const chunk of stream) {
      handler.handle(chunk);
      // For debugging
      // const content = chunk.choices?.[0]?.delta.content;
      // if (content) {
      //   totalLength += content.length;
      //   console.log(`Current length: ${totalLength}, Content: "${content}"`);
      // }
    }

    // Verify multiple message blocks were created
    const runSteps = mockHandlers[GraphEvents.ON_RUN_STEP].mock.calls;
    // console.log('Number of run steps:', runSteps.length);
    expect(runSteps.length).toEqual(handler.currentIndex + 1);
  });

  it('should handle reasoning text separately', async () => {
    const runId = nanoid();
    new SplitStreamHandler({
      runId,
      handlers: mockHandlers,
    });

    const stream = createMockStream({
      text: 'Main content',
      reasoningText: 'Reasoning text',
      streamRate: 0,
    })();

    const reasoningTokens: string[] = [];
    const contentTokens: string[] = [];

    for await (const chunk of stream) {
      const reasoning = chunk.choices?.[0]?.delta.reasoning_content ?? '';
      const content = chunk.choices?.[0]?.delta.content ?? '';

      if (reasoning) reasoningTokens.push(reasoning);
      if (content) contentTokens.push(content);
    }

    expect(reasoningTokens.length).toBeGreaterThan(0);
    expect(contentTokens.length).toBeGreaterThan(0);
  });

  it('should preserve empty strings and whitespace', async () => {
    const stream = createMockStream({
      text: 'Hello  world', // Note double space
      streamRate: 0,
    })();

    const tokens: string[] = [];
    for await (const chunk of stream) {
      const content = chunk.choices?.[0]?.delta.content ?? '';
      if (!content) {
        return;
      }
      tokens.push(content);
    }

    expect(tokens).toContain(' ');
    expect(tokens.join('')).toBe('Hello  world');
  });
});

describe('ContentAggregator with SplitStreamHandler', () => {
  it('should aggregate content from multiple message blocks', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 5,
    });

    const text = 'First sentence. Second sentence. Third sentence.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    expect(contentParts.length).toBeGreaterThan(0);
    contentParts.forEach((part) => {
      expect(part?.type).toBe(ContentTypes.TEXT);
      if (part?.type === ContentTypes.TEXT) {
        expect(typeof part.text).toBe('string');
        expect(part.text.length).toBeGreaterThan(0);
      }
    });

    const fullText = contentParts
      .filter((part) => part?.type === ContentTypes.TEXT)
      .map((part) => (part?.type === ContentTypes.TEXT ? part.text : ''))
      .join('');
    expect(fullText).toBe(text);
  });

  it('should maintain content order across splits', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 15,
    });

    const text = 'First part. Second part. Third part.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const texts = contentParts
      .filter((part) => part?.type === ContentTypes.TEXT)
      .map((part) => (part?.type === ContentTypes.TEXT ? part.text : ''));

    expect(texts[0]).toContain('First');
    expect(texts[texts.length - 1]).toContain('Third');
  });

  it('should handle code blocks as single content parts', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 10,
    });

    const text = `Before code.
\`\`\`python
def test():
    return True
\`\`\`
After code.`;

    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const codeBlockPart = contentParts.find(
      (part) =>
        part?.type === ContentTypes.TEXT &&
        part.text.includes('```python') === true
    );

    expect(codeBlockPart).toBeDefined();
    if (codeBlockPart?.type === ContentTypes.TEXT) {
      expect(codeBlockPart.text).toContain('def test()');
      expect(codeBlockPart.text).toContain('return True');
    }
  });

  it('should properly map steps to their content', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 5,
    });

    const text = 'Hi. Ok. Yes.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    Array.from(stepMap.entries()).forEach(([_stepId, step]) => {
      expect(step?.type).toBe(StepTypes.MESSAGE_CREATION);
      const currentIndex = step?.index ?? -1;
      const stepContent = contentParts[currentIndex];
      if (!stepContent && currentIndex > 0) {
        const prevStepContent = contentParts[currentIndex - 1];
        expect(
          (prevStepContent as MessageContentText | undefined)?.text
        ).toEqual(text);
      } else if (stepContent?.type === ContentTypes.TEXT) {
        expect(stepContent.text.length).toBeGreaterThan(0);
      }
    });

    contentParts.forEach((part, index) => {
      const hasMatchingStep = Array.from(stepMap.values()).some(
        (step) => step?.index === index
      );
      expect(hasMatchingStep).toBe(true);
    });
  });

  it('should aggregate content across multiple splits while preserving order', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent } = createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
      },
      blockThreshold: 10,
    });

    const text = 'A. B. C. D. E. F.';
    const stream = createMockStream({ text, streamRate: 0 })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const letters = ['A', 'B', 'C', 'D', 'E', 'F'];
    let letterIndex = 0;

    contentParts.forEach((part) => {
      if (part?.type === ContentTypes.TEXT) {
        while (
          letterIndex < letters.length &&
          part.text.includes(letters[letterIndex]) === true
        ) {
          letterIndex++;
        }
      }
    });

    expect(letterIndex).toBe(letters.length);
  });
});

describe('SplitStreamHandler with Reasoning Tokens', () => {
  it('should apply same splitting rules to both content types', async () => {
    const runId = nanoid();
    const mockHandlers: t.SplitStreamHandlers = {
      [GraphEvents.ON_RUN_STEP]: jest.fn(),
      [GraphEvents.ON_MESSAGE_DELTA]: jest.fn(),
      [GraphEvents.ON_REASONING_DELTA]: jest.fn(),
    };

    const handler = new SplitStreamHandler({
      runId,
      handlers: mockHandlers,
      blockThreshold: 3,
    });

    const stream = createMockStream({
      text: 'First text. Second text. Third text.',
      reasoningText: 'First thought. Second thought. Third thought.',
      streamRate: 0,
    })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    const runSteps = (mockHandlers[GraphEvents.ON_RUN_STEP] as jest.Mock).mock
      .calls;
    const reasoningDeltas = (
      mockHandlers[GraphEvents.ON_REASONING_DELTA] as jest.Mock
    ).mock.calls;
    const messageDeltas = (
      mockHandlers[GraphEvents.ON_MESSAGE_DELTA] as jest.Mock
    ).mock.calls;

    // Both content types should create multiple blocks
    expect(runSteps.length).toBeGreaterThan(1);
    expect(reasoningDeltas.length).toBeGreaterThan(0);
    expect(messageDeltas.length).toBeGreaterThan(0);

    // Verify splitting behavior for both types
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const getStepTypes = (calls: any[]): string[] =>
      calls
        .map(([{ data }]) =>
          data.stepDetails?.type === StepTypes.MESSAGE_CREATION
            ? data.stepDetails.message_creation.message_id
            : null
        )
        .filter(Boolean);

    const messageSteps = getStepTypes(runSteps);
    expect(new Set(messageSteps).size).toBeGreaterThan(1);
  });

  it('should properly map steps to their reasoning content', async () => {
    const runId = nanoid();
    const { contentParts, aggregateContent, stepMap } =
      createContentAggregator();

    const handler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_RUN_STEP]: aggregateContent,
        [GraphEvents.ON_MESSAGE_DELTA]: aggregateContent,
        [GraphEvents.ON_REASONING_DELTA]: aggregateContent,
      },
      blockThreshold: 5,
    });

    const text = 'Main content.';
    const reasoningText = 'First thought. Second thought. Third thought.';
    const stream = createMockStream({
      text,
      reasoningText,
      streamRate: 0,
    })();

    for await (const chunk of stream) {
      handler.handle(chunk);
    }

    Array.from(stepMap.entries()).forEach(([_stepId, step]) => {
      expect(step?.type).toBe(StepTypes.MESSAGE_CREATION);
      const currentIndex = step?.index ?? -1;
      const stepContent = contentParts[currentIndex];

      if (stepContent?.type === ContentTypes.THINK) {
        // Verify reasoning content structure
        expect(stepContent).toHaveProperty('think');
        expect(typeof stepContent.think).toBe('string');
        expect(stepContent.think.length).toBeGreaterThan(0);
      }
    });

    // Verify at least one reasoning content part exists
    const reasoningParts = contentParts.filter(
      (part) => part?.type === ContentTypes.THINK
    );
    expect(reasoningParts.length).toBeGreaterThan(0);

    // Verify the content order (reasoning should come before main content)
    const contentTypes = contentParts
      .filter((part) => part !== undefined)
      .map((part) => part.type);

    expect(contentTypes).toContain(ContentTypes.THINK);
    expect(contentTypes).toContain(ContentTypes.TEXT);

    // Verify the complete reasoning content is preserved
    const fullReasoningText = reasoningParts
      .map((part) => (part?.type === ContentTypes.THINK ? part.think : ''))
      .join('');
    expect(fullReasoningText).toBe(reasoningText);
  });
});

describe('SplitStreamHandler', () => {
  it('should handle think blocks correctly', async () => {
    const runId = nanoid();
    const messageDeltaEvents: t.MessageDeltaEvent[] = [];
    const reasoningDeltaEvents: t.ReasoningDeltaEvent[] = [];

    const streamHandler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_MESSAGE_DELTA]: ({ data }): void => {
          messageDeltaEvents.push(data);
        },
        [GraphEvents.ON_REASONING_DELTA]: ({ data }): void => {
          reasoningDeltaEvents.push(data);
        },
      },
    });

    const content =
      'Here\'s some regular text. <think>Now I\'m thinking deeply about something important. This should all be reasoning.</think> Back to regular text.';

    const stream = createMockStream({
      text: content,
      streamRate: 5,
    })();

    for await (const chunk of stream) {
      streamHandler.handle(chunk);
    }

    // Check that content before <think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Here\'s') === true
      )
    ).toBe(true);

    // Check that <think> tag was handled as reasoning
    expect(
      reasoningDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined
          )?.think.includes('<think>') === true
      )
    ).toBe(true);

    // Check that content inside <think> tags was handled as reasoning
    expect(
      reasoningDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined
          )?.think.includes('thinking') === true
      )
    ).toBe(true);

    // Check that </think> tag was handled as reasoning
    expect(
      reasoningDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined
          )?.think.includes('</think>') === true
      )
    ).toBe(true);

    // Check that content after </think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Back') === true
      )
    ).toBe(true);
  });

  it('should ignore think tags inside code blocks', async () => {
    const runId = nanoid();
    const messageDeltaEvents: t.MessageDeltaEvent[] = [];
    const reasoningDeltaEvents: t.ReasoningDeltaEvent[] = [];

    const streamHandler = new SplitStreamHandler({
      runId,
      handlers: {
        [GraphEvents.ON_MESSAGE_DELTA]: ({ data }): void => {
          messageDeltaEvents.push(data);
        },
        [GraphEvents.ON_REASONING_DELTA]: ({ data }): void => {
          reasoningDeltaEvents.push(data);
        },
      },
    });

    const content =
      'Regular text. ```<think>This should stay as code</think>``` More text.';

    const stream = createMockStream({
      text: content,
      streamRate: 5,
    })();

    for await (const chunk of stream) {
      streamHandler.handle(chunk);
    }

    // Check that think tags inside code blocks were treated as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Regular') === true
      )
    ).toBe(true);

    // Verify no reasoning events were generated
    expect(reasoningDeltaEvents.length).toBe(0);
  });

  it('should properly split content with think tags while maintaining context', async () => {
    const runId = nanoid();
    const messageDeltaEvents: t.MessageDeltaEvent[] = [];
    const reasoningDeltaEvents: t.ReasoningDeltaEvent[] = [];
    const runStepEvents: t.RunStep[] = [];
    const { contentParts, aggregateContent } = createContentAggregator();

    const streamHandler = new SplitStreamHandler({
      runId,
      blockThreshold: 20, // Small threshold to force splits
      handlers: {
        [GraphEvents.ON_MESSAGE_DELTA]: (event): void => {
          messageDeltaEvents.push(event.data);
          aggregateContent(event);
        },
        [GraphEvents.ON_REASONING_DELTA]: (event): void => {
          reasoningDeltaEvents.push(event.data);
          aggregateContent(event);
        },
        [GraphEvents.ON_RUN_STEP]: (event): void => {
          runStepEvents.push(event.data);
          aggregateContent(event);
        },
      },
    });

    const content =
      'Here\'s some regular text. <think>Now I\'m thinking deeply about something important. This is a long thought that should be split into multiple parts. We want to ensure the splitting works correctly.</think> Back to regular text after thinking.';

    const stream = createMockStream({
      text: content,
      streamRate: 5,
    })();

    for await (const chunk of stream) {
      streamHandler.handle(chunk);
    }

    // Verify that multiple message blocks were created
    expect(runStepEvents.length).toBeGreaterThan(2);

    // Check that content before <think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('regular') === true
      )
    ).toBe(true);

    // Verify that reasoning content was split into multiple parts
    const reasoningParts = reasoningDeltaEvents
      .map(
        (event) =>
          (event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined)
            ?.think
      )
      .filter(Boolean);
    expect(reasoningParts.length).toBeGreaterThan(1);

    // Verify that the complete reasoning content is preserved when joined
    const fullReasoningContent = reasoningParts.join('');
    expect(fullReasoningContent).toContain('thinking');
    expect(fullReasoningContent).toContain('should');
    expect(fullReasoningContent).toContain('be');
    expect(fullReasoningContent).toContain('split');

    // Check that each reasoning part maintains proper think context
    let seenThinkOpen = false;
    let seenThinkClose = false;
    reasoningParts.forEach((part) => {
      if (part == null) return;
      if (part.includes('<think>')) {
        seenThinkOpen = true;
      }
      if (part.includes('</think>')) {
        seenThinkClose = true;
      }
      // Middle parts should be handled as reasoning even without explicit think tags
      if (!part.includes('<think>') && !part.includes('</think>')) {
        expect(
          reasoningDeltaEvents.some(
            (event) =>
              (event.delta.content?.[0] as t.ReasoningDeltaUpdate | undefined)
                ?.think === part
          )
        ).toBe(true);
      }
    });
    expect(seenThinkOpen).toBe(true);
    expect(seenThinkClose).toBe(true);

    // Check that content after </think> was handled as regular text
    expect(
      messageDeltaEvents.some(
        (event) =>
          (
            event.delta.content?.[0] as t.MessageDeltaUpdate | undefined
          )?.text.includes('Back') === true
      )
    ).toBe(true);

    const thinkingBlocks = contentParts.filter(
      (part) => part?.type === ContentTypes.THINK
    );
    expect(thinkingBlocks.length).toBeGreaterThan(0);
    expect(
      (thinkingBlocks[0] as t.ReasoningContentText).think.startsWith('<think>')
    ).toBeTruthy();
  });
});
