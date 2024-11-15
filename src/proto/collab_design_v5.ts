// src/collab_design_v5.ts
import 'dotenv/config';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';
import { END, StateGraphArgs, START, StateGraph, MemorySaver } from '@langchain/langgraph';
import { AgentExecutor, createOpenAIToolsAgent } from 'langchain/agents';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import { Runnable, RunnableConfig } from '@langchain/core/runnables';
import { JsonOutputToolsParser } from 'langchain/output_parsers';
import { HandlerRegistry } from '@/events';
import { ChatOpenAI } from '@langchain/openai';
import { ChatBedrockConverse } from '@langchain/aws';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatMistralAI } from '@langchain/mistralai';
import { ChatVertexAI } from '@langchain/google-vertexai';
import { BedrockChat } from '@langchain/community/chat_models/bedrock/web';
import { supervisorPrompt } from '@/prompts/collab';
import type * as t from '@/types';
import { Providers } from '@/common';

interface AgentStateChannels {
  messages: BaseMessage[];
  next: string;
}

export interface Member {
  name: string;
  systemPrompt: string;
  tools: any[];
  llmConfig: LLMConfig;
}

interface LLMConfig {
  provider: Providers;
  [key: string]: any;
}

interface SupervisorConfig {
  systemPrompt?: string;
  llmConfig: LLMConfig;
}

const llmProviders: Record<Providers, any> = {
  [Providers.OPENAI]: ChatOpenAI,
  [Providers.VERTEXAI]: ChatVertexAI,
  [Providers.BEDROCK_LEGACY]: BedrockChat,
  [Providers.MISTRALAI]: ChatMistralAI,
  [Providers.BEDROCK]: ChatBedrockConverse,
  [Providers.ANTHROPIC]: ChatAnthropic,
};

export class CollaborativeProcessor {
  graph: t.CompiledWorkflow | null = null;
  private handlerRegistry: HandlerRegistry;
  private members: Member[];
  private supervisorConfig: SupervisorConfig;

  constructor(
    members: Member[],
    supervisorConfig: SupervisorConfig,
    customHandlers?: Record<string, any>
  ) {
    this.members = members;
    this.supervisorConfig = supervisorConfig;
    this.handlerRegistry = new HandlerRegistry();
    if (customHandlers) {
      for (const [eventType, handler] of Object.entries(customHandlers)) {
        this.handlerRegistry.register(eventType, handler);
      }
    }
  }

  async initialize(): Promise<void> {
    this.graph = await this.createGraph();
  }

  private async createGraph(): Promise<t.CompiledWorkflow> {
    const agentStateChannels: StateGraphArgs<AgentStateChannels>['channels'] = {
      messages: {
        value: (x?: BaseMessage[], y?: BaseMessage[]) => (x ?? []).concat(y ?? []),
        default: () => [],
      },
      next: {
        value: (x?: string, y?: string) => y ?? x ?? END,
        default: () => END,
      },
    };

    async function createAgent(
      llmConfig: LLMConfig,
      tools: any[],
      systemPrompt: string
    ): Promise<AgentExecutor> {
      const { provider, ...clientOptions } = llmConfig;
      const LLMClass = llmProviders[provider];
      if (!LLMClass) {
        throw new Error(`Unsupported LLM provider: ${provider}`);
      }
      const llm = new LLMClass(clientOptions);

      const prompt = await ChatPromptTemplate.fromMessages([
        ['system', systemPrompt],
        new MessagesPlaceholder('messages'),
        new MessagesPlaceholder('agent_scratchpad'),
      ]);
      const agent = await createOpenAIToolsAgent({ llm, tools, prompt });
      return new AgentExecutor({ agent, tools });
    }

    const memberNames = this.members.map(member => member.name);

    const systemPrompt = this.supervisorConfig.systemPrompt || supervisorPrompt;
    const options = [END, ...memberNames];

    const functionDef = {
      name: 'route',
      description: 'Select the next role.',
      parameters: {
        title: 'routeSchema',
        type: 'object',
        properties: {
          next: {
            title: 'Next',
            anyOf: [
              { enum: options },
            ],
          },
        },
        required: ['next'],
      },
    };

    const toolDef = {
      type: 'function',
      function: functionDef,
    } as const;

    const prompt = ChatPromptTemplate.fromMessages([
      ['system', systemPrompt],
      new MessagesPlaceholder('messages'),
      [
        'system',
        'Given the conversation above, who should act next?' +
        ' Or should we FINISH? Select one of: {options}',
      ],
    ]);

    const formattedPrompt = await prompt.partial({
      options: options.join(', '),
      members: memberNames.join(', '),
    });

    const { provider, ...clientOptions } = this.supervisorConfig.llmConfig;
    const LLMClass = llmProviders[provider];
    if (!LLMClass) {
      throw new Error(`Unsupported LLM provider for supervisor: ${provider}`);
    }
    const llm = new LLMClass(clientOptions);

    const supervisorChain = formattedPrompt
      .pipe(llm.bindTools(
        [toolDef],
        {
          tool_choice: { 'type': 'function', 'function': { 'name': 'route' } },
        },
      ))
      .pipe(new JsonOutputToolsParser())
      .pipe((x) => (x[0].args));

    const workflow = new StateGraph({
      channels: agentStateChannels,
    });

    // Dynamically create agents and add nodes for each member
    for (const member of this.members) {
      const agent = await createAgent(member.llmConfig, member.tools, member.systemPrompt);
      const node = async (
        state: AgentStateChannels,
        config?: RunnableConfig,
      ) => {
        const agentPromise = agent.invoke(state, config);

        // Store the promise in the state
        await this.graph?.updateState(config, {
          [`${member.name}Promise`]: agentPromise,
        });

        const result = await agentPromise;
        return {
          messages: [
            new HumanMessage({ content: result.output, name: member.name }),
          ],
        };
      };
      workflow.addNode(member.name, node);
      workflow.addEdge(member.name, 'supervisor');
    }

    const supervisorNode = async (
      state: AgentStateChannels,
      config?: RunnableConfig,
    ) => {
      // Get the current state
      const currentState = await this.graph?.getState(config);

      // Wait for all member promises to resolve
      const memberPromises = this.members.map(member => currentState[`${member.name}Promise`]);
      await Promise.all(memberPromises);

      // Clear the promises for the next iteration
      for (const member of this.members) {
        await this.graph?.updateState(config, {
          [`${member.name}Promise`]: undefined,
        });
      }

      const result = await supervisorChain.invoke(state, config);
      return result;
    };

    workflow.addNode('supervisor', supervisorNode);

    workflow.addConditionalEdges(
      'supervisor',
      (x: AgentStateChannels) => x.next,
    );

    workflow.addEdge(START, 'supervisor');

    const memory = new MemorySaver();
    return workflow.compile({ checkpointer: memory });
  }

  async processStream(
    inputs: { messages: BaseMessage[] },
    config: Partial<RunnableConfig> & { version: 'v1' | 'v2' },
  ) {
    if (!this.graph) {
      throw new Error('CollaborativeProcessor not initialized. Call initialize() first.');
    }
    const stream = this.graph.streamEvents(inputs, config);
    for await (const event of stream) {
      const handler = this.handlerRegistry.getHandler(event.event);
      if (handler) {
        handler.handle(event.event, event.data);
      }
    }
  }
}
