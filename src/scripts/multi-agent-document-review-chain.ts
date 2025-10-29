import { config } from 'dotenv';
config();

import {
  HumanMessage,
  BaseMessage,
  getBufferString,
} from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

/**
 * Create edges for a document review chain where each agent adds their expertise
 */
function createDocumentReviewChain(agentIds: string[]): t.GraphEdge[] {
  const edges: t.GraphEdge[] = [];

  for (let i = 0; i < agentIds.length - 1; i++) {
    edges.push({
      from: agentIds[i],
      to: agentIds[i + 1],
      edgeType: 'direct',
      prompt: (messages: BaseMessage[], startIndex: number) => {
        const runMessages = messages.slice(startIndex);
        const bufferString = getBufferString(runMessages);

        // Custom prompt that maintains context of document review
        return `You are reviewing a document. Here is the analysis so far from previous reviewers:\n\n${bufferString}\n\nPlease add your specific review based on your expertise. Build upon previous insights without repeating them.`;
      },
      excludeResults: true,
    });
  }

  return edges;
}

async function testDocumentReviewChain() {
  console.log('Testing Document Review Chain...\n');

  const { contentParts, aggregateContent } = createContentAggregator();

  // Define specialized document reviewers
  const agents: t.AgentInputs[] = [
    {
      agentId: 'grammar_checker',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a Grammar and Style Checker.
      Focus on:
      - Grammar errors
      - Spelling mistakes
      - Sentence structure
      - Writing clarity
      
      Start with "GRAMMAR & STYLE CHECK:" and list issues found.`,
      maxContextTokens: 4000,
    },
    {
      agentId: 'fact_checker',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a Fact Checker.
      Focus on:
      - Accuracy of claims
      - Data verification needs
      - Source requirements
      - Logical consistency
      
      Start with "FACT CHECK:" and note any claims that need verification.`,
      maxContextTokens: 4000,
    },
    {
      agentId: 'tone_reviewer',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are a Tone and Audience Reviewer.
      Focus on:
      - Appropriate tone for target audience
      - Consistency of voice
      - Engagement level
      - Cultural sensitivity
      
      Start with "TONE & AUDIENCE REVIEW:" and provide specific feedback.`,
      maxContextTokens: 4000,
    },
    {
      agentId: 'final_editor',
      provider: Providers.ANTHROPIC,
      clientOptions: {
        modelName: 'claude-haiku-4-5',
        apiKey: process.env.ANTHROPIC_API_KEY,
      },
      instructions: `You are the Final Editor.
      Based on all previous reviews:
      1. Summarize key issues found
      2. Prioritize changes needed
      3. Provide final recommendation (approve/revise/reject)
      
      Start with "FINAL EDITORIAL DECISION:" and be decisive.`,
      maxContextTokens: 4000,
    },
  ];

  const agentIds = agents.map((a) => a.agentId);
  const edges = createDocumentReviewChain(agentIds);

  // Custom handlers (simplified)
  const customHandlers = {
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        const runStepData = data as any;
        if (runStepData?.name) {
          console.log(`\n✍️  ${runStepData.name} reviewing...`);
        }
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
  };

  const runConfig: t.RunConfig = {
    runId: `doc-review-chain-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents,
      edges,
    },
    customHandlers,
    returnContent: true,
  };

  try {
    const run = await Run.create(runConfig);

    // Sample document to review
    const documentToReview = `
    The Impact of Artificial Intelligence on Modern Business

    Artificial Intelligence (AI) is revolutionizing how businesses operate in 2024. Studies show that 85% of companies have implemented some form of AI, leading to average productivity gains of 40%.

    Key benefits includes:
    - Automated decision-making reducing human error by 90%
    - Cost savings of up to $1 billion annually for large corporations
    - Enhanced customer experiences through 24/7 AI support

    However, challenges remain. Many organizations struggles with data privacy concerns and the need for specialized talent. The future of AI in business looks bright, but companies must carefully navigate these obstacles to fully realize it's potential.
    `;

    const userMessage = `Please review this document:\n\n${documentToReview}`;
    const messages = [new HumanMessage(userMessage)];

    console.log('Document submitted for review...\n');

    const config = {
      configurable: { thread_id: 'doc-review-1' },
      streamMode: 'values',
      version: 'v2' as const,
    };

    await run.processStream({ messages }, config);

    console.log('\n=== Review Chain Complete ===');

    // Show how each reviewer built upon previous feedback
    const runMessages = run.getRunMessages();
    if (runMessages) {
      console.log('\nReview progression:');
      runMessages.forEach((msg, i) => {
        if (msg._getType() === 'ai') {
          console.log(`\nStep ${i + 1}: ${agentIds[Math.floor(i / 2)]}`);
          console.log('---');
          console.log(msg.content.toString().slice(0, 200) + '...');
        }
      });
    }
  } catch (error) {
    console.error('Error in document review chain:', error);
  }
}

// Run the test
// testDocumentReviewChain();
