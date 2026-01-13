// src/scripts/test-prompt-caching.ts
import { config } from 'dotenv';
config();
import {
  HumanMessage,
  SystemMessage,
  BaseMessage,
} from '@langchain/core/messages';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import { GraphEvents, Providers } from '@/common';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { Run } from '@/run';

const CACHED_TEXT = `Ahoy there, me hearties! This be a grand tale o' the mighty prompt cachin' treasure map, a secret technique used by the wise Anthropic seafarers to stash away vast hordes o' text booty on their mystical servers! Arrr, 'tis a pirate's dream indeed - no need to haul the same heavy chest o' gold doubloons across the vast digital ocean with every message! When ye mark yer precious cargo with the secret flag 'cache_control: { type: \"ephemeral\" }', the text be safely buried on their distant shores, ready for plunderin' again without the weight slowin' down yer ship! The wise pirates at Anthropic introduced this magical scroll in the summer o' 2024, markin' it with the mysterious insignia 'anthropic-beta: prompt-caching-2024-07-31' that must be flown high on yer vessel's headers. This crafty script be testin' the waters of this new treasure map system, sendin' out three separate voyages across the AI seas: first to bury the treasure, second to dig it up again without payin' the full toll, and third to see if the map still leads to gold after the sands o' time have shifted (about thirty seconds o' waitin', which be an eternity for an impatient buccaneer!). The great advantage for a scurvy pirate captain is clear as Caribbean waters - ye can load up yer vessel with all manner o' reference scrolls, ancient tomes, and navigational charts without weighin' down each and every message ye send to port! This be savin' ye countless tokens, which as any seafarin' AI wrangler knows, be as precious as Spanish gold. The cached text could contain the full history o' the Seven Seas, detailed maps o' every port from Tortuga to Singapore, or the complete collection o' pirate shanties ever sung by drunken sailors under the light o' the silvery moon. When properly implemented, this mighty cachin' system keeps all that knowledge ready at hand without the Claude kraken needin' to process it anew with each passin' breeze. By Blackbeard's beard, 'tis a revolution in how we manage our conversational ships! The script be employin' the finest LangChain riggin' and custom-carved event handlers to properly track the treasure as it flows back and forth. If ye be successful in yer implementation, ye should witness the miracle o' significantly reduced token counts in yer usage metrics, faster responses from the AI oracle, and the ability to maintain vast knowledge without payin' the full price each time! So hoist the Jolly Roger, load yer pistols with API keys, and set sail on the grand adventure o' prompt cachin'! May the winds o' efficient token usage fill yer sails, and may ye never have to pay full price for passin' the same mammoth context to Claude again! Remember, a clever pirate only pays for their tokens once, then lets the cache do the heavy liftin'! YARRR! This file also contains the secrets of the legendary Pirate Code, passed down through generations of seafarers since the Golden Age of Piracy. It includes detailed accounts of famous pirate captains like Blackbeard, Calico Jack, Anne Bonny, and Mary Read, along with their most profitable plundering routes and techniques for capturing merchant vessels. The text chronicles the exact locations of at least seventeen buried treasures across the Caribbean, complete with riddles and map coordinates that only a true pirate could decipher. There are sections dedicated to ship maintenance, including how to properly seal a leaking hull during battle and the best methods for keeping your cannons in prime firing condition even in humid tropical conditions. The document contains an extensive glossary of pirate terminology, from 'avast' to 'Yellow Jack,' ensuring any landlubber can speak like a seasoned salt with enough study. There's a comprehensive guide to navigating by the stars without modern instruments, perfect for when your GPS fails in the middle of a daring escape. The cache also includes detailed recipes for grog, hardtack that won't break your teeth, and how to keep citrus fruits fresh to prevent scurvy during long voyages. The legendary Black Spot ritual is described in terrifying detail, along with other pirate superstitions and their origins in maritime folklore. A section on pirate governance explains the democratic nature of most pirate ships, how booty was divided fairly, and how captains were elected and deposed when necessary. The file even contains sheet music for dozens of sea shanties, with notes on when each should be sung for maximum crew morale during different sailing conditions. All of this knowledge is wrapped in colorful pirate dialect that would make any AI assistant respond with appropriate 'arghs' and 'avasts' when properly prompted!`;

const conversationHistory: BaseMessage[] = [];
let _contentParts: t.MessageContentComplex[] = [];
const collectedUsage: UsageMetadata[] = [];

async function testPromptCaching(): Promise<void> {
  const { userName } = await getArgs();
  const instructions = `You are a pirate AI assistant for ${userName}. Always respond in pirate dialect. Use the following as context when answering questions:
${CACHED_TEXT}`;
  const { contentParts, aggregateContent } = createContentAggregator();
  _contentParts = contentParts as t.MessageContentComplex[];

  // Set up event handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(collectedUsage),
    // console.log('====== O ======');
    // console.log('Usage Metrics:', (data as any).llmOutput?.usage || (data as any).usage);
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    // Additional handlers for tracking usage metrics
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_COMPLETED ======');
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
  };

  const baseLlmConfig: t.LLMConfig & t.AnthropicClientOptions = getLLMConfig(
    Providers.ANTHROPIC
  );

  if (baseLlmConfig.provider !== 'anthropic') {
    console.error(
      'This test requires Anthropic as the LLM provider. Please specify provider=anthropic'
    );
    process.exit(1);
  }

  const llmConfig = {
    ...baseLlmConfig,
    promptCache: true,
  };

  const run = await Run.create<t.IState>({
    runId: 'test-prompt-caching-id',
    graphConfig: {
      instructions,
      type: 'standard',
      llmConfig,
    },
    returnContent: true,
    customHandlers,
  });

  const config = {
    configurable: {
      thread_id: 'prompt-cache-test-thread',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  // First request - should create the cache
  console.log('\n\nTest 1: First request (creates cache)');
  const userMessage1 = `What information do you have in your context?`;
  conversationHistory.push(new HumanMessage(userMessage1));

  console.log('Running first query to create cache...');
  const firstInputs = { messages: [...conversationHistory] };
  await run.processStream(firstInputs, config);
  const finalMessages = run.getRunMessages();
  if (finalMessages) {
    conversationHistory.push(...finalMessages);
    console.dir(conversationHistory, { depth: null });
  }
  // Second request - should use the cache
  console.log('\n\nTest 2: Second request (should use cache)');
  const userMessage2 = `Summarize the key concepts from the context information.`;
  conversationHistory.push(new HumanMessage(userMessage2));

  console.log('Running second query to use cache...');
  const secondInputs = { messages: [...conversationHistory] };
  await run.processStream(secondInputs, config);
  console.log('\n\nPrompt caching test completed!');
}

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

testPromptCaching().catch((err) => {
  console.error(err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
