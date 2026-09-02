import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import type { BaseMessage } from '@langchain/core/messages';
import type { TokenCounter } from '@/types/run';
import {
  calculateMaskedResultMaxChars,
  maskConsumedToolResults,
  preFlightTruncateToolResults,
  resolveFadingBudgetTokens,
  createPruneMessages,
} from '@/messages/prune';
import { calculateMaxToolResultChars } from '@/utils/truncation';

const tokenCounter: TokenCounter = (message) => {
  const content = message.content;
  const text = typeof content === 'string' ? content : JSON.stringify(content);
  return Math.ceil(text.length / 4);
};

function serialize(message: BaseMessage): string {
  const content = message.content;
  return typeof content === 'string' ? content : JSON.stringify(content);
}

function toolCall(id: string): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [{ id, name: 'fetch', args: {}, type: 'tool_call' }],
  });
}

function toolResult(id: string, chars: number): ToolMessage {
  return new ToolMessage({
    content: `${id}:${'x'.repeat(chars)}`,
    tool_call_id: id,
    name: 'fetch',
  });
}

/** One user turn: question, tool call, tool result, answer. */
function round(index: number, resultChars: number): BaseMessage[] {
  const id = `tc-${index}`;
  return [
    new HumanMessage(`question ${index}`),
    toolCall(id),
    toolResult(id, resultChars),
    new AIMessage(`answer ${index}`),
  ];
}

function conversation(rounds: number[]): BaseMessage[] {
  return rounds.flatMap((chars, index) => round(index, chars));
}

function countMap(
  messages: BaseMessage[]
): Record<string, number | undefined> {
  const map: Record<string, number | undefined> = {};
  for (let i = 0; i < messages.length; i++) {
    map[i] = tokenCounter(messages[i]);
  }
  return map;
}

describe('resolveFadingBudgetTokens', () => {
  it('scales the fixed context window by the discrete pressure band only', () => {
    expect(resolveFadingBudgetTokens(100_000, 0.3)).toBe(100_000);
    expect(resolveFadingBudgetTokens(100_000, 0.8)).toBe(100_000);
    expect(resolveFadingBudgetTokens(100_000, 0.849)).toBe(100_000);
    expect(resolveFadingBudgetTokens(100_000, 0.85)).toBe(50_000);
    expect(resolveFadingBudgetTokens(100_000, 0.9)).toBe(20_000);
    expect(resolveFadingBudgetTokens(100_000, 0.99)).toBe(5_000);
    expect(resolveFadingBudgetTokens(2_000, 0.99)).toBe(1_024);
  });
});

describe('calculateMaskedResultMaxChars', () => {
  it('keeps a fixed fraction of the fresh-result cap with a floor', () => {
    expect(calculateMaskedResultMaxChars(100_000)).toBe(
      Math.floor(calculateMaxToolResultChars(100_000) * 0.1)
    );
    expect(calculateMaskedResultMaxChars(1_024)).toBe(300);
  });
});

describe('preFlightTruncateToolResults stability', () => {
  it('truncates a historical result to the same bytes however many results follow it', () => {
    const snapshots = [0, 1, 4, 20].map((later) => {
      const messages = conversation([6_000, 6_000, ...Array(later).fill(50)]);
      const indexTokenCountMap = countMap(messages);
      preFlightTruncateToolResults({
        messages,
        maxContextTokens: 1_000,
        indexTokenCountMap,
        tokenCounter,
      });
      return [serialize(messages[2]), serialize(messages[6])];
    });

    expect(snapshots[0][0]).toContain('truncated');
    expect(snapshots[0][1]).toContain('truncated');
    for (const snapshot of snapshots) {
      expect(snapshot).toEqual(snapshots[0]);
    }
  });
});

describe('maskConsumedToolResults stability', () => {
  it('masks a consumed result to the same bytes however many consumed results exist', () => {
    const snapshots = [1, 3, 12].map((rounds) => {
      const messages = conversation(Array(rounds).fill(6_000));
      const indexTokenCountMap = countMap(messages);
      const masked = maskConsumedToolResults({
        messages,
        indexTokenCountMap,
        tokenCounter,
        maxChars: 2_000,
      });
      expect(masked).toBe(rounds);
      return serialize(messages[2]);
    });

    expect(snapshots[0].length).toBeLessThanOrEqual(2_000);
    expect(snapshots[0]).toContain('truncated');
    for (const snapshot of snapshots) {
      expect(snapshot).toBe(snapshots[0]);
    }
  });

  it('never masks below the placeholder floor', () => {
    const messages = conversation([6_000]);
    maskConsumedToolResults({
      messages,
      indexTokenCountMap: countMap(messages),
      tokenCounter,
      maxChars: 10,
    });
    expect(serialize(messages[2]).length).toBeLessThanOrEqual(300);
    expect(serialize(messages[2]).length).toBeGreaterThan(10);
  });
});

describe('pruner keeps historical tool results byte-stable across turns', () => {
  const maxTokens = 40_000;

  /** Mirrors a host that rebuilds full-content messages and a fresh pruner every run. */
  function runTurn(
    messages: BaseMessage[],
    options: { summarizationEnabled: boolean; calibrationRatio: number }
  ): { early: string; pressure: number } {
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: messages.length,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: options.summarizationEnabled,
      calibrationRatio: options.calibrationRatio,
    });
    const result = pruneMessages({ messages });
    return { early: serialize(messages[2]), pressure: result.contextPressure ?? 0 };
  }

  it('fit-to-budget truncation (summarization on) ignores calibration drift and growth', () => {
    const first = runTurn(conversation([60_000]), {
      summarizationEnabled: true,
      calibrationRatio: 1,
    });
    expect(first.early).toContain('truncated');
    expect(first.early.length).toBeLessThanOrEqual(
      calculateMaxToolResultChars(maxTokens)
    );

    const ratios = [1.03, 1.12, 1.25, 1.4];
    ratios.forEach((calibrationRatio, index) => {
      const later = runTurn(
        conversation([60_000, ...Array(index + 1).fill(400)]),
        { summarizationEnabled: true, calibrationRatio }
      );
      expect(later.pressure).toBeLessThan(0.8);
      expect(later.early).toBe(first.early);
    });
  });

  it('masking under pressure (summarization off) is stable within a band', () => {
    const history = [60_000, 48_000, 16_000];
    const first = runTurn(conversation(history), {
      summarizationEnabled: false,
      calibrationRatio: 1,
    });
    expect(first.pressure).toBeGreaterThanOrEqual(0.8);
    expect(first.pressure).toBeLessThan(0.85);
    expect(first.early).toContain('truncated');
    expect(first.early.length).toBeLessThanOrEqual(
      calculateMaskedResultMaxChars(maxTokens)
    );

    const ratios = [1.01, 1.02, 1.03];
    ratios.forEach((calibrationRatio, index) => {
      const later = runTurn(
        conversation([...history, ...Array(index + 1).fill(200)]),
        { summarizationEnabled: false, calibrationRatio }
      );
      expect(later.pressure).toBeGreaterThanOrEqual(0.8);
      expect(later.pressure).toBeLessThan(0.85);
      expect(later.early).toBe(first.early);
    });
  });

  it('holds bytes stable within one run while provider usage recalibrates', () => {
    const messages = conversation([60_000]);
    const pruneMessages = createPruneMessages({
      maxTokens,
      startIndex: 0,
      tokenCounter,
      indexTokenCountMap: countMap(messages),
      summarizationEnabled: true,
    });
    const first = pruneMessages({ messages });
    const early = serialize(messages[2]);
    expect(early).toContain('truncated');

    let previousRatio = first.calibrationRatio ?? 1;
    for (let turn = 1; turn <= 3; turn++) {
      messages.push(...round(turn, 400));
      const result = pruneMessages({
        messages,
        usageMetadata: { input_tokens: 13_500 + 500 * turn, output_tokens: 10 },
        totalTokensFresh: true,
      });
      expect(result.calibrationRatio).not.toBe(previousRatio);
      previousRatio = result.calibrationRatio ?? previousRatio;
      expect(serialize(messages[2])).toBe(early);
    }
  });
});
