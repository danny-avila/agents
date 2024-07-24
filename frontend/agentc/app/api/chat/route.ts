import { NextResponse } from 'next/server';
import { Processor } from '../../../../src/utils/processor';
import { getLLMConfig } from '../../../../src/utils/llmConfig';

export async function POST(req: Request) {
  console.log("API route called"); // Debug log
  try {
    const { message } = await req.json();
    console.log("Received message:", message); // Debug log

    // Initialize the Processor
    const llmConfig = getLLMConfig('anthropic'); // Or whichever provider you're using
    console.log("LLM Config:", llmConfig); // Debug log

    const processor = await Processor.create({
      graphConfig: {
        type: 'standard',
        llmConfig,
        tools: [], // Add any tools you want to use here
      },
      customHandlers: {}, // Add any custom handlers if needed
    });

    // Process the message
    const processorInput = {
      messages: [{ content: message, role: 'user' }],
    };

    const sessionConfig = {
      configurable: {
        provider: 'anthropic', // Or whichever provider you're using
        thread_id: `web-session-${Date.now()}`,
        instructions: "You are a helpful AI assistant.",
        additional_instructions: "Provide concise and accurate responses.",
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    console.log("Processing message with config:", sessionConfig); // Debug log
    const aiResponse = await processor.processStream(processorInput, sessionConfig);
    console.log("AI Response:", aiResponse); // Debug log

    if (aiResponse && typeof aiResponse.content === 'string') {
      return NextResponse.json({ message: aiResponse.content });
    } else {
      throw new Error('Invalid response from AI');
    }
  } catch (error: unknown) {
    console.error("Error in API route:", error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    return NextResponse.json({ error: 'Failed to process message', details: errorMessage }, { status: 500 });
  }
}