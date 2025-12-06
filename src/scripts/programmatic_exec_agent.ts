// src/scripts/programmatic_exec_agent.ts
/**
 * Test script for Programmatic Tool Calling (PTC) with agent integration.
 * Run with: npm run programmatic_exec_agent
 *
 * Demonstrates:
 * 1. Tool classification with allowed_callers:
 *    - direct: Tool bound to LLM (can be called directly)
 *    - code_execution: Tool available for PTC (not bound to LLM)
 *    - Both: Tool bound to LLM AND available for PTC
 * 2. Deferred loading with defer_loading: true (for tool search)
 * 3. Agent-level tool configuration via toolRegistry
 * 4. ToolNode runtime injection of programmatic tools
 *
 * This shows the real-world integration pattern with agents.
 */
import { config } from 'dotenv';
config();

import type { StructuredToolInterface } from '@langchain/core/tools';
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type { RunnableConfig } from '@langchain/core/runnables';
import type * as t from '@/types';
import { createCodeExecutionTool } from '@/tools/CodeExecutor';
import { createProgrammaticToolCallingTool } from '@/tools/ProgrammaticToolCalling';
import { createToolSearchRegexTool } from '@/tools/ToolSearchRegex';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';
import {
  createGetTeamMembersTool,
  createGetExpensesTool,
  createGetWeatherTool,
  createProgrammaticToolRegistry,
} from '@/test/mockTools';

// ============================================================================
// Tool Registry (Metadata)
// ============================================================================

/**
 * Tool registry only needs business logic tools that require filtering.
 * Special tools (execute_code, programmatic_code_execution, tool_search_regex)
 * are always bound directly to the LLM and don't need registry entries.
 */
function createAgentToolRegistry(): t.LCToolRegistry {
  // Use shared programmatic tool registry (get_team_members, get_expenses, etc.)
  return createProgrammaticToolRegistry();
}

// ============================================================================
// Main
// ============================================================================

const conversationHistory: BaseMessage[] = [];

async function main(): Promise<void> {
  console.log('Programmatic Tool Calling - Agent Integration Test');
  console.log('===================================================\n');

  const { userName, location, provider } = await getArgs();
  const llmConfig = getLLMConfig(provider);

  // Create all tool instances
  const mockTools: StructuredToolInterface[] = [
    createGetTeamMembersTool(),
    createGetExpensesTool(),
    createGetWeatherTool(),
  ];

  const toolMap = new Map(mockTools.map((t) => [t.name, t]));

  // Create special tools (PTC, code execution, tool search)
  const codeExecTool = createCodeExecutionTool();
  const ptcTool = createProgrammaticToolCallingTool();
  const toolSearchTool = createToolSearchRegexTool();

  // Build complete tool list and map
  const allTools = [...mockTools, codeExecTool, ptcTool, toolSearchTool];
  const completeToolMap = new Map(allTools.map((t) => [t.name, t]));

  // Create tool registry with allowed_callers configuration
  // Only includes business logic tools (not special tools like execute_code, PTC, tool_search)
  const toolRegistry = createAgentToolRegistry();

  console.log('Tool configuration:');
  console.log('- Total tools:', allTools.length);
  console.log(
    '- Programmatic-allowed:',
    Array.from(toolRegistry.values())
      .filter((t) => t.allowed_callers?.includes('code_execution'))
      .map((t) => t.name)
      .join(', ')
  );
  console.log(
    '- Direct-only:',
    Array.from(toolRegistry.values())
      .filter(
        (t) =>
          !t.allowed_callers ||
          (t.allowed_callers.includes('direct') &&
            !t.allowed_callers.includes('code_execution'))
      )
      .map((t) => t.name)
      .join(', ')
  );
  console.log(
    '- Both:',
    Array.from(toolRegistry.values())
      .filter(
        (t) =>
          t.allowed_callers?.includes('direct') &&
          t.allowed_callers?.includes('code_execution')
      )
      .map((t) => t.name)
      .join(', ')
  );

  // Create run with toolRegistry configuration
  const run = await Run.create<t.IState>({
    runId: 'ptc-agent-test',
    graphConfig: {
      type: 'standard',
      llmConfig,
      agents: [
        {
          agentId: 'default',
          provider: llmConfig.provider,
          clientOptions: llmConfig,
          tools: allTools,
          toolMap: completeToolMap,
          toolRegistry, // Pass tool registry for programmatic/deferred tool config
          instructions:
            'You are an AI assistant with access to programmatic tool calling. ' +
            'When you need to process multiple items or perform complex data operations, ' +
            'use the programmatic_code_execution tool to write Python code that calls tools efficiently.',
        },
      ],
    },
    returnContent: true,
  });

  const config: Partial<RunnableConfig> & {
    version: 'v1' | 'v2';
    streamMode: string;
  } = {
    configurable: {
      provider,
      thread_id: 'ptc-test-conversation',
    },
    streamMode: 'values',
    version: 'v2',
  };

  console.log('\n' + '='.repeat(70));
  console.log('Test: Process team expenses using PTC');
  console.log('='.repeat(70) + '\n');

  const userMessage = new HumanMessage(
    `Hi! I need you to analyze our team's expenses. Please:
1. Get the list of team members
2. For each member, get their expense records
3. Calculate the total expenses per member
4. Identify anyone who spent more than $300
5. Show me the results in a nice format

Use the programmatic_code_execution tool to do this efficiently - don't call each tool separately!`
  );

  conversationHistory.push(userMessage);

  const inputs = {
    messages: conversationHistory,
  };

  console.log('Running agent with PTC capability...\n');

  const finalContentParts = await run.processStream(inputs, config);
  const finalMessages = run.getRunMessages();

  if (finalMessages) {
    conversationHistory.push(...finalMessages);
  }

  console.log('\n' + '='.repeat(70));
  console.log('Agent Response:');
  console.log('='.repeat(70));

  if (finalContentParts) {
    for (const part of finalContentParts) {
      if (part?.type === 'text' && part.text) {
        console.log(part.text);
      }
    }
  }

  console.log('\n' + '='.repeat(70));
  console.log('Test completed successfully!');
  console.log('='.repeat(70));
  console.log('\nKey observations:');
  console.log(
    '1. LLM only sees tools with allowed_callers including "direct" (get_weather, execute_code, programmatic_code_execution, tool_search_regex)'
  );
  console.log(
    '2. When PTC is invoked, ToolNode automatically injects programmatic tools (get_team_members, get_expenses, get_weather)'
  );
  console.log(
    '3. No need to manually configure runtime toolMap - handled by agent context'
  );
  console.log(
    '4. Tool filtering based on allowed_callers happens automatically\n'
  );
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

main().catch((err) => {
  console.error('Fatal error:', err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  process.exit(1);
});
