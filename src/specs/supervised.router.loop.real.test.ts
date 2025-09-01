import { config } from 'dotenv';
config();
import { Calculator } from '@langchain/community/tools/calculator';
import { HumanMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import type * as t from '@/types';

describe('Supervised router loop with toolEnd=false (real provider)', () => {
  jest.setTimeout(60000);

  test('TOOLS -> ROUTER -> AGENT loop occurs (two agent turns)', async () => {
    const provider = Providers.OPENAI; // use OpenAI for reliable tool-calling
    const llmConfig = getLLMConfig(provider);

    let agentEnds = 0;
    const run = await Run.create<t.IState>({
      runId: 'supervised-loop-real',
      graphConfig: {
        type: 'supervised',
        llmConfig,
        routerEnabled: true,
        // toolEnd defaults to false, so TOOLS should route back to ROUTER
        tools: [new Calculator()],
        instructions: 'Use the calculator if needed, then continue.',
      },
      returnContent: false,
      customHandlers: {
        // Count agent completions
        on_chat_model_end: {
          handle: () => {
            agentEnds += 1;
          },
        } as unknown as t.EventHandler,
      },
    });

    const inputs = {
      messages: [
        new HumanMessage(
          'Use the calculator tool to compute 2+2, then summarize in one short sentence.'
        ),
      ],
    };
    const config_ = {
      configurable: { provider },
      streamMode: 'values' as const,
      version: 'v2' as const,
    };

    await run.processStream(inputs, config_);

    expect(agentEnds).toBeGreaterThanOrEqual(2);
  });

  test('TOOLS -> END when toolEnd=true (single agent turn)', async () => {
    const provider = Providers.OPENAI;
    const llmConfig = getLLMConfig(provider);

    let agentEnds = 0;
    const run = await Run.create<t.IState>({
      runId: 'supervised-end-real',
      graphConfig: {
        type: 'supervised',
        llmConfig,
        routerEnabled: true,
        toolEnd: true,
        tools: [new Calculator()],
        instructions: 'Use the calculator if needed, then stop.',
      },
      returnContent: false,
      customHandlers: {
        on_chat_model_end: {
          handle: () => {
            agentEnds += 1;
          },
        } as unknown as t.EventHandler,
      },
    });

    const inputs = {
      messages: [
        new HumanMessage('Compute 3+5 with the calculator, then stop.'),
      ],
    };
    const config_ = {
      configurable: { provider },
      streamMode: 'values' as const,
      version: 'v2' as const,
    };

    await run.processStream(inputs, config_);

    expect(agentEnds).toBe(1);
  });
});
