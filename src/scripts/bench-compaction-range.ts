/* eslint-disable no-console */
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';

import type { BaseMessage } from '@langchain/core/messages';
import { createTokenCounter } from '@/utils/tokens';
import {
  resolveIntraTurnRetainTokens,
  splitAtRecencyBoundary,
} from '@/messages/recency';

interface Scenario {
  name: string;
  toolSteps: number;
  toolOutputChars: number;
  maxContextTokens: number;
}

function createToolOutput(step: number, charCount: number): string {
  let state = (step + 1) * 2_654_435_761;
  let output = `step=${step}; identifier=VALUE_${step}\n`;
  while (output.length < charCount) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    output += `${(state >>> 0).toString(16).padStart(8, '0')} `;
  }
  return output.slice(0, charCount);
}

function createToolHistory({
  toolSteps,
  toolOutputChars,
}: Scenario): BaseMessage[] {
  const messages: BaseMessage[] = [
    new HumanMessage(
      'Inspect the repository, fix the issue, test it, and open a PR.'
    ),
  ];
  for (let index = 0; index < toolSteps; index++) {
    const callId = `call_${index}`;
    messages.push(
      new AIMessage({
        content: `Inspecting repository evidence for step ${index}.`,
        tool_calls: [
          {
            id: callId,
            name: 'exec_command',
            args: {
              intent: `Inspecting repository evidence for step ${index}`,
              cmd: `rg pattern_${index}`,
            },
          },
        ],
      }),
      new ToolMessage({
        content: createToolOutput(index, toolOutputChars),
        tool_call_id: callId,
        name: 'exec_command',
      })
    );
  }
  messages.push(
    new AIMessage('I have enough evidence and am preparing the implementation.')
  );
  return messages;
}

function sumTokens(
  messages: BaseMessage[],
  tokenCounter: (message: BaseMessage) => number
): number {
  let total = 0;
  for (const message of messages) {
    total += tokenCounter(message);
  }
  return total;
}

function assertPairingPreserved(
  head: BaseMessage[],
  tail: BaseMessage[]
): void {
  const sideByCallId = new Map<string, 'head' | 'tail'>();
  for (const [side, messages] of [
    ['head', head],
    ['tail', tail],
  ] as const) {
    for (const message of messages) {
      if (message.getType() === 'ai') {
        for (const call of (message as AIMessage).tool_calls ?? []) {
          if (call.id != null) {
            sideByCallId.set(call.id, side);
          }
        }
        continue;
      }
      if (message.getType() !== 'tool') {
        continue;
      }
      const callId = (message as ToolMessage).tool_call_id;
      if (sideByCallId.get(callId) !== side) {
        throw new Error(`Compaction split tool pair ${callId}`);
      }
    }
  }
}

const tokenCounter = await createTokenCounter();
const scenarios: Scenario[] = [
  {
    name: '20 tool steps',
    toolSteps: 20,
    toolOutputChars: 4_096,
    maxContextTokens: 64_000,
  },
  {
    name: '50 tool steps',
    toolSteps: 50,
    toolOutputChars: 4_096,
    maxContextTokens: 128_000,
  },
  {
    name: '100 tool steps',
    toolSteps: 100,
    toolOutputChars: 4_096,
    maxContextTokens: 256_000,
  },
];

for (const scenario of scenarios) {
  const messages = createToolHistory(scenario);
  const before = splitAtRecencyBoundary(messages, {
    turns: 2,
    tokenCounter,
  });
  const after = splitAtRecencyBoundary(messages, {
    turns: 2,
    tokenCounter,
    intraTurnTokens: resolveIntraTurnRetainTokens({
      maxContextTokens: scenario.maxContextTokens,
    }),
  });
  assertPairingPreserved(after.head, after.tail);

  const totalTokens = sumTokens(messages, tokenCounter);
  const beforeCompactableTokens = sumTokens(before.head, tokenCounter);
  const afterCompactableTokens = sumTokens(after.head, tokenCounter);
  const retainedTokens = sumTokens(after.tail, tokenCounter);
  console.log(
    JSON.stringify({
      scenario: scenario.name,
      totalTokens,
      beforeCompactableTokens,
      afterCompactableTokens,
      compactablePercent: Number(
        ((afterCompactableTokens / totalTokens) * 100).toFixed(1)
      ),
      retainedTokens,
      toolPairingPreserved: true,
    })
  );
}
