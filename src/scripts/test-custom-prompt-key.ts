#!/usr/bin/env bun

import { config } from 'dotenv';
config();

import { HumanMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers } from '@/common';
import type * as t from '@/types';

/**
 * Test the custom promptKey feature for handoff edges
 * This demonstrates how to use custom parameter names instead of "instructions"
 */
async function testCustomPromptKey() {
  console.log('Testing Custom Prompt Key Feature...\n');

  const runConfig: t.RunConfig = {
    runId: `test-custom-prompt-key-${Date.now()}`,
    graphConfig: {
      type: 'multi-agent',
      agents: [
        {
          agentId: 'supervisor',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-haiku-4-5',
            apiKey: process.env.ANTHROPIC_API_KEY,
          },
          instructions: `You are a Task Supervisor managing different agents:
          
          1. transfer_to_researcher - For research tasks (uses "query" parameter)
          2. transfer_to_designer - For design tasks (uses "requirements" parameter)
          3. transfer_to_coder - For coding tasks (uses "specification" parameter)
          
          Each agent expects different parameter names in their handoff tools.
          Pay attention to the parameter names when calling each tool.`,
          maxContextTokens: 8000,
        },
        {
          agentId: 'researcher',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-haiku-4-5',
            apiKey: process.env.ANTHROPIC_API_KEY,
          },
          instructions: `You are a Research Agent. You receive research queries to investigate.
          Look for the "Query:" field in the transfer message.`,
          maxContextTokens: 8000,
        },
        {
          agentId: 'designer',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-haiku-4-5',
            apiKey: process.env.ANTHROPIC_API_KEY,
          },
          instructions: `You are a Design Agent. You receive design requirements to implement.
          Look for the "Requirements:" field in the transfer message.`,
          maxContextTokens: 8000,
        },
        {
          agentId: 'coder',
          provider: Providers.ANTHROPIC,
          clientOptions: {
            modelName: 'claude-haiku-4-5',
            apiKey: process.env.ANTHROPIC_API_KEY,
          },
          instructions: `You are a Coding Agent. You receive technical specifications to implement.
          Look for the "Specification:" field in the transfer message.`,
          maxContextTokens: 8000,
        },
      ],
      edges: [
        {
          from: 'supervisor',
          to: 'researcher',
          edgeType: 'handoff',
          // Custom parameter name: "query"
          prompt: 'The research question or topic to investigate',
          promptKey: 'query',
        },
        {
          from: 'supervisor',
          to: 'designer',
          edgeType: 'handoff',
          // Custom parameter name: "requirements"
          prompt: 'The design requirements and constraints',
          promptKey: 'requirements',
        },
        {
          from: 'supervisor',
          to: 'coder',
          edgeType: 'handoff',
          // Custom parameter name: "specification"
          prompt: 'The technical specification for the code to implement',
          promptKey: 'specification',
        },
      ],
    },
  };

  const run = await Run.create(runConfig);

  // Test queries for different agents
  const testQueries = [
    // 'Research the latest trends in sustainable energy storage technologies',
    'Design a mobile app interface for a fitness tracking application',
    // 'Write a Python function that calculates the Fibonacci sequence recursively',
  ];

  const config = {
    configurable: {
      thread_id: 'custom-prompt-key-test-1',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  for (const query of testQueries) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`USER QUERY: "${query}"`);
    console.log('='.repeat(60));

    const inputs = {
      messages: [new HumanMessage(query)],
    };

    await run.processStream(inputs, config);

    console.log(`\n${'─'.repeat(60)}`);
    console.log('Each agent receives instructions via their custom parameter:');
    console.log('- Researcher expects "query"');
    console.log('- Designer expects "requirements"');
    console.log('- Coder expects "specification"');
    console.log('─'.repeat(60));
  }

  console.log('\n\nDemonstration complete!');
  console.log('The promptKey feature allows for more semantic parameter names');
  console.log('that better match the domain and purpose of each agent.');
}

// Run the test
testCustomPromptKey().catch(console.error);
