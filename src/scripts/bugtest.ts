// src/scripts/bugtest.ts
import { config } from 'dotenv';
config();
import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import type * as t from '@/types';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import {
  ToolEndHandler,
  ModelEndHandler,
  createMetadataAggregator,
} from '@/events';
import { getLLMConfig } from '@/utils/llmConfig';
import { getArgs } from '@/scripts/args';
import { GraphEvents, Providers } from '@/common';
import { Run } from '@/run';

const conversationHistory: BaseMessage[] = [];
let _contentParts: t.MessageContentComplex[] = [];

async function testGoogleTokenBug(): Promise<void> {
  console.log('🔍 Testing Google Streaming Token Calculation Bug...\\n');

  // Force Google provider for this test
  const provider = Providers.GOOGLE;
  const llmConfig = getLLMConfig(provider) as any;

  // Override to use the problematic model
  llmConfig.provider = provider;
  llmConfig.model = 'gemini-2.5-pro';
  llmConfig.streaming = true;
  llmConfig.streamUsage = true;

  console.log(`📋 Test Configuration:`);
  console.log(`   Provider: ${provider}`);
  console.log(`   Model: ${llmConfig.model}`);
  console.log(`   Streaming: ${llmConfig.streaming}`);
  console.log(`   Stream Usage: ${llmConfig.streamUsage}`);
  console.log('');

  const { contentParts, aggregateContent } = createContentAggregator();
  _contentParts = contentParts as t.MessageContentComplex[];

  // Enhanced handlers with detailed token logging
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (event: string, data: t.StreamEventData): void => {
        console.log('====== CHAT_MODEL_END ======');
        console.log('🔢 FINAL USAGE DATA:', data);
        const handler = new ModelEndHandler();
        handler.handle(event, data as any);
      },
    },
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
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
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP ======');
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_RUN_STEP_DELTA ======');
        aggregateContent({ event, data: data as t.RunStepDeltaEvent });
      },
    },
    // [GraphEvents.ON_MESSAGE_DELTA]: {
    //   handle: (
    //     event: GraphEvents.ON_MESSAGE_DELTA,
    //     data: t.StreamEventData
    //   ): void => {
    //     console.log('====== ON_MESSAGE_DELTA ======');
    //     // Log token information if available
    //     const messageDelta = data as t.MessageDeltaEvent;
    //     if ((messageDelta.delta as any)?.usage_metadata) {
    //       console.log(
    //         '🔢 TOKEN UPDATE:',
    //         (messageDelta.delta as any).usage_metadata
    //       );
    //     }
    //     aggregateContent({ event, data: data as t.MessageDeltaEvent });
    //   },
    // },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: (
        event: GraphEvents.ON_REASONING_DELTA,
        data: t.StreamEventData
      ): void => {
        console.log('====== ON_REASONING_DELTA ======');
        aggregateContent({ event, data: data as t.ReasoningDeltaEvent });
      },
    },
  };

  const run = await Run.create<t.IState>({
    runId: 'google-token-bug-test',
    graphConfig: {
      type: 'standard',
      llmConfig,
      reasoningKey: 'reasoning',
      instructions:
        'You are a helpful AI assistant. Provide detailed responses.',
    },
    returnContent: true,
    customHandlers,
  });

  const config = {
    configurable: {
      thread_id: 'google-token-bug-test-thread',
    },
    streamMode: 'values',
    version: 'v2' as const,
  };

  const problemMessage = `Please provide a detailed technical specification for an Android Xposed module called "SimChassis" that automatically switches mobile data between two SIM cards based on signal quality analysis. The app should include both a UI settings application and backend Xposed module functionality. Include all technical requirements, architecture components, and code delivery format specifications.`;

  console.log('📤 Sending problem message to Google API...');
  console.log(`Message length: ${problemMessage.length} characters`);
  console.log('');

  conversationHistory.push(new HumanMessage(problemMessage));

  const inputs = {
    messages: conversationHistory,
  };

  console.log('🔄 Processing stream with detailed token logging...\\n');

  const finalContentParts = await run.processStream(inputs, config);
  const finalMessages = run.getRunMessages();

  if (finalMessages) {
    conversationHistory.push(...finalMessages);
  }

  console.log('\\n📊 FINAL RESULTS:');
  console.log('==================');
  console.log(`Total messages in conversation: ${conversationHistory.length}`);
  console.log(`Content parts generated: ${contentParts.length}`);

  // Analyze the final message for token counts
  const lastMessage = conversationHistory[conversationHistory.length - 1];
  if (lastMessage && (lastMessage as any).usage_metadata) {
    const usageMetadata = (lastMessage as any).usage_metadata;
    console.log('\\n🔍 USAGE METADATA ANALYSIS:');
    console.log('============================');
    console.log(`Input tokens: ${usageMetadata.input_tokens}`);
    console.log(`Output tokens: ${usageMetadata.output_tokens}`);
    console.log(`Total tokens: ${usageMetadata.total_tokens}`);

    // Check for extreme values
    if (usageMetadata.output_tokens > 1000000) {
      console.log('\\n🚨 EXTREME TOKEN COUNT DETECTED!');
      console.log('==================================');
      console.log(`Output tokens: ${usageMetadata.output_tokens} (EXTREME!)`);
      console.log(
        'This indicates the streaming token calculation bug is still present.'
      );
    } else {
      console.log('\\n✅ Token counts appear reasonable.');
    }
  } else {
    console.log('\\n⚠️  No usage metadata found in final message.');
  }

  console.log('\\n📝 CONTENT PARTS SUMMARY:');
  console.log('==========================');
  contentParts.forEach((part, index) => {
    if (part && part.type === 'text' && 'text' in part) {
      console.log(
        `Part ${index + 1}: TEXT (${(part as any).text?.length || 0} chars)`
      );
    } else if (part && part.type === 'thinking' && 'thinking' in part) {
      console.log(
        `Part ${index + 1}: THINKING (${(part as any).thinking?.length || 0} chars)`
      );
    } else if (part) {
      console.log(
        `Part ${index + 1}: ${part.type} (${JSON.stringify(part).length} chars)`
      );
    }
  });

  // Create metadata aggregator to get final token counts
  const { handleLLMEnd, collected } = createMetadataAggregator();
  const titleOptions: t.RunTitleOptions = {
    provider,
    inputText: problemMessage,
    contentParts,
    chainOptions: {
      callbacks: [
        {
          handleLLMEnd,
        },
      ],
    },
  };

  const titleResult = await run.generateTitle(titleOptions);
  console.log('\\n📈 METADATA COLLECTION:');
  console.log('========================');
  console.log('Generated Title:', titleResult);
  console.log('Collected metadata:', collected);
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

testGoogleTokenBug().catch((err) => {
  console.error('Test failed:', err);
  console.log('Conversation history:');
  console.dir(conversationHistory, { depth: null });
  console.log('Content parts:');
  console.dir(_contentParts, { depth: null });
  process.exit(1);
});
