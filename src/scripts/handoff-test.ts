import { config } from 'dotenv';
config();

import { z } from 'zod';
import { tool } from '@langchain/core/tools';
import { ChatAnthropic } from '@langchain/anthropic';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import {
  StateGraph,
  MessagesAnnotation,
  Command,
  START,
  getCurrentTaskInput,
  END,
} from '@langchain/langgraph';
import { ToolMessage } from '@langchain/core/messages';

interface CreateHandoffToolParams {
  agentName: string;
  description?: string;
}

const createHandoffTool = ({
  agentName,
  description,
}: CreateHandoffToolParams) => {
  const toolName = `transfer_to_${agentName}`;
  const toolDescription = description || `Ask agent '${agentName}' for help`;

  const handoffTool = tool(
    async (_, config) => {
      const toolMessage = new ToolMessage({
        content: `Successfully transferred to ${agentName}`,
        name: toolName,
        tool_call_id: config.toolCall.id,
      });

      // inject the current agent state
      const state =
        getCurrentTaskInput() as (typeof MessagesAnnotation)['State'];
      return new Command({
        goto: agentName,
        update: { messages: state.messages.concat(toolMessage) },
        graph: Command.PARENT,
      });
    },
    {
      name: toolName,
      schema: z.object({}),
      description: toolDescription,
    }
  );

  return handoffTool;
};

const bookHotel = tool(
  async (input: { hotel_name: string }) => {
    return `Successfully booked a stay at ${input.hotel_name}.`;
  },
  {
    name: 'book_hotel',
    description: 'Book a hotel',
    schema: z.object({
      hotel_name: z.string().describe('The name of the hotel to book'),
    }),
  }
);

const bookFlight = tool(
  async (input: { from_airport: string; to_airport: string }) => {
    return `Successfully booked a flight from ${input.from_airport} to ${input.to_airport}.`;
  },
  {
    name: 'book_flight',
    description: 'Book a flight',
    schema: z.object({
      from_airport: z.string().describe('The departure airport code'),
      to_airport: z.string().describe('The arrival airport code'),
    }),
  }
);

const transferToHotelAssistant = createHandoffTool({
  agentName: 'hotel_assistant',
  description: 'Transfer user to the hotel-booking assistant.',
});

const transferToFlightAssistant = createHandoffTool({
  agentName: 'flight_assistant',
  description: 'Transfer user to the flight-booking assistant.',
});

const llm = new ChatAnthropic({
  modelName: 'claude-3-5-sonnet-latest',
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const flightAssistant = createReactAgent({
  llm,
  tools: [bookFlight, transferToHotelAssistant],
  prompt: 'You are a flight booking assistant',
  name: 'flight_assistant',
});

const hotelAssistant = createReactAgent({
  llm,
  tools: [bookHotel, transferToFlightAssistant],
  prompt: 'You are a hotel booking assistant',
  name: 'hotel_assistant',
});

const multiAgentGraph = new StateGraph(MessagesAnnotation)
  .addNode('flight_assistant', flightAssistant, {
    ends: ['hotel_assistant', END],
  })
  .addNode('hotel_assistant', hotelAssistant, {
    ends: ['flight_assistant', END],
  })
  .addEdge(START, 'flight_assistant')
  .compile();

const stream = await multiAgentGraph.stream({
  messages: [
    {
      role: 'user',
      content: 'book a flight from BOS to JFK and a stay at McKittrick Hotel',
    },
  ],
});

for await (const chunk of stream) {
  console.log(chunk);
  console.log('\n');
}
