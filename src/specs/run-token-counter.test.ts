import { HumanMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { createTokenCounter, encodingOfTokenCounter } from '@/utils/tokens';
import { Providers } from '@/common';
import { Run } from '@/run';

/** Korean is the widest measured gap between the two encodings, so a counter
 *  built for the wrong one undercounts here by the largest factor. */
const CJK_MESSAGE = new HumanMessage(
  '사용자가 인증 미들웨어를 리팩터링하여 속도 제한 전에 토큰 검증이 이루어지도록 요청했습니다.'
);

/**
 * A host that supplies `indexTokenCountMap` without a counter delegates the
 * choice of tokenizer to `Run.create`, and every message the run measures is
 * denominated in whatever it picks. OpenRouter is the provider here because it
 * serves Claude alongside everything else: the model string is the only signal,
 * so a run that fails to read it has nothing to fall back on but `o200k_base`.
 */
async function deriveRunTokenCounter(
  clientOptions: Record<string, unknown>
): Promise<t.TokenCounter> {
  const config: t.RunConfig = {
    runId: 'run-token-counter',
    graphConfig: {
      type: 'standard',
      llmConfig: {
        provider: Providers.OPENROUTER,
        streaming: false,
        streamUsage: false,
      },
      clientOptions,
      instructions: 'Be concise.',
    },
    indexTokenCountMap: { 0: 10 },
  };
  await Run.create<t.IState>(config);
  if (config.tokenCounter == null) {
    throw new Error('Run.create did not derive a token counter');
  }
  return config.tokenCounter;
}

describe('Run.create token counter derivation', () => {
  it('reads the model through the modelName alias', async () => {
    const counter = await deriveRunTokenCounter({
      modelName: 'anthropic/claude-sonnet-4',
    });

    expect(encodingOfTokenCounter(counter)).toBe('claude');
    /* Guards the consequence rather than the label: the two encodings must
     * disagree on this text, or picking the wrong one would cost nothing. */
    const openaiCounter = await createTokenCounter('o200k_base');
    expect(counter(CJK_MESSAGE)).toBeGreaterThan(openaiCounter(CJK_MESSAGE));
  });

  it('still reads the canonical model key', async () => {
    const counter = await deriveRunTokenCounter({
      model: 'anthropic/claude-sonnet-4',
    });

    expect(encodingOfTokenCounter(counter)).toBe('claude');
  });

  it('falls back to o200k_base when no model is configured', async () => {
    const counter = await deriveRunTokenCounter({});

    expect(encodingOfTokenCounter(counter)).toBe('o200k_base');
  });
});
