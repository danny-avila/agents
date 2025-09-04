import { config } from 'dotenv';
config();

import { HumanMessage, BaseMessage } from '@langchain/core/messages';
import { Run } from '@/run';
import { Providers, GraphEvents } from '@/common';
import { ChatModelStreamHandler, createContentAggregator } from '@/stream';
import { ToolEndHandler, ModelEndHandler } from '@/events';
import type * as t from '@/types';

const conversationHistory: BaseMessage[] = [];

/**
 * Example of supervisor-based multi-agent system
 *
 * The supervisor has handoff tools for 5 different specialists.
 * To demonstrate the concept while respecting LangGraph constraints,
 * we show two approaches:
 *
 * 1. All 5 specialists exist but share the same adaptive configuration
 * 2. Only 2 agents total by using a single adaptive specialist (requires workaround)
 */
async function testSupervisorMultiAgent() {
  console.log('Testing Supervisor-Based Multi-Agent System...\n');

  // NOTE: To truly have only 2 agents with 5 handoff tools, you would need:
  // 1. Custom tool implementation (see multi-agent-supervisor-mock.ts)
  // 2. Or modify MultiAgentGraph to support "virtual" agents
  // 3. Or use a single conditional edge with role parameter
  //
  // This example shows the concept using 6 agents that share configuration

  // Set up content aggregator
  const { contentParts, aggregateContent } = createContentAggregator();

  // Define configurations for all possible specialists
  const specialistConfigs = {
    data_analyst: {
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Data Analyst specialist. Your expertise includes:
      - Statistical analysis and data visualization
      - SQL queries and database optimization
      - Python/R for data science
      - Machine learning model evaluation
      - A/B testing and experiment design
      
      Follow the supervisor's specific instructions carefully.`,
      maxContextTokens: 8000,
    },
    security_expert: {
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Security Expert. Your expertise includes:
      - Cybersecurity best practices
      - Vulnerability assessment and penetration testing
      - Security architecture and threat modeling
      - Compliance (GDPR, HIPAA, SOC2, etc.)
      - Incident response and forensics
      
      Follow the supervisor's specific instructions carefully.`,
      maxContextTokens: 8000,
    },
    product_designer: {
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Product Designer. Your expertise includes:
      - User experience (UX) design principles
      - User interface (UI) design and prototyping
      - Design systems and component libraries
      - User research and usability testing
      - Accessibility and inclusive design
      
      Follow the supervisor's specific instructions carefully.`,
      maxContextTokens: 8000,
    },
    devops_engineer: {
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a DevOps Engineer. Your expertise includes:
      - CI/CD pipeline design and optimization
      - Infrastructure as Code (Terraform, CloudFormation)
      - Container orchestration (Kubernetes, Docker)
      - Cloud platforms (AWS, GCP, Azure)
      - Monitoring, logging, and observability
      
      Follow the supervisor's specific instructions carefully.`,
      maxContextTokens: 8000,
    },
    legal_advisor: {
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are a Legal Advisor specializing in technology. Your expertise includes:
      - Software licensing and open source compliance
      - Data privacy and protection laws
      - Intellectual property and patents
      - Contract review and negotiation
      - Regulatory compliance for tech companies
      
      Follow the supervisor's specific instructions carefully.`,
      maxContextTokens: 8000,
    },
  };

  // Track which specialist role was selected
  let selectedRole = '';
  let roleInstructions = '';

  // Create custom handlers
  const customHandlers = {
    [GraphEvents.TOOL_END]: new ToolEndHandler(),
    [GraphEvents.CHAT_MODEL_END]: new ModelEndHandler(),
    [GraphEvents.CHAT_MODEL_STREAM]: new ChatModelStreamHandler(),
    [GraphEvents.ON_RUN_STEP]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP,
        data: t.StreamEventData
      ): void => {
        const runStepData = data as any;
        if (runStepData?.name) {
          console.log(`\n[${runStepData.name}] Processing...`);
        }
        aggregateContent({ event, data: data as t.RunStep });
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: (
        event: GraphEvents.ON_RUN_STEP_COMPLETED,
        data: t.StreamEventData
      ): void => {
        aggregateContent({
          event,
          data: data as unknown as { result: t.ToolEndEvent },
        });
      },
    },
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: (
        event: GraphEvents.ON_MESSAGE_DELTA,
        data: t.StreamEventData
      ): void => {
        console.dir(data, { depth: null });
        aggregateContent({ event, data: data as t.MessageDeltaEvent });
      },
    },
    [GraphEvents.TOOL_START]: {
      handle: (
        _event: string,
        data: t.StreamEventData,
        metadata?: Record<string, unknown>
      ): void => {
        const toolData = data as any;
        if (toolData?.name?.includes('transfer_to_')) {
          const specialist = toolData.name.replace('transfer_to_', '');
          console.log(`\n🔀 Transferring to ${specialist}...`);
          selectedRole = specialist;
        }
      },
    },
  };

  // Function to create the graph with supervisor having multiple handoff options
  function createSupervisorGraph(): t.RunConfig {
    console.log(`\nCreating graph with supervisor and 5 specialist agents.`);
    console.log('All specialists share the same adaptive configuration.\n');

    // Define the adaptive specialist configuration that will be reused
    const specialistConfig = {
      provider: Providers.OPENAI,
      clientOptions: {
        modelName: 'gpt-4.1',
        apiKey: process.env.OPENAI_API_KEY,
      },
      instructions: `You are an Adaptive Specialist. Your agent ID indicates your role:
      
      - data_analyst: Focus on statistical analysis, metrics, ML evaluation, A/B testing
      - security_expert: Focus on cybersecurity, vulnerability assessment, compliance  
      - product_designer: Focus on UX/UI design, user research, accessibility
      - devops_engineer: Focus on CI/CD, infrastructure, cloud platforms, monitoring
      - legal_advisor: Focus on licensing, privacy laws, contracts, regulatory compliance
      
      The supervisor will provide specific instructions. Follow them while maintaining your expert perspective.`,
      maxContextTokens: 8000,
    };

    // Create the graph with supervisor and all 5 specialists
    // All specialists share the same adaptive configuration
    const agents: t.AgentInputs[] = [
      {
        agentId: 'supervisor',
        provider: Providers.OPENAI,
        clientOptions: {
          modelName: 'gpt-4.1-mini',
          apiKey: process.env.OPENAI_API_KEY,
        },
        instructions: `You are a Task Supervisor with access to 5 specialist agents:
        1. transfer_to_data_analyst - For statistical analysis and metrics
        2. transfer_to_security_expert - For cybersecurity and vulnerability assessment  
        3. transfer_to_product_designer - For UX/UI design
        4. transfer_to_devops_engineer - For infrastructure and deployment
        5. transfer_to_legal_advisor - For compliance and licensing
        
        Your role is to:
        1. Analyze the incoming request
        2. Decide which specialist is best suited
        3. Use the appropriate transfer tool (e.g., transfer_to_data_analyst)
        4. Provide specific instructions to guide their work
        
        Be specific about what you need from the specialist.`,
        maxContextTokens: 8000,
      },
      // Include all 5 specialists with the same adaptive configuration
      {
        agentId: 'data_analyst',
        ...specialistConfig,
      },
      {
        agentId: 'security_expert',
        ...specialistConfig,
      },
      {
        agentId: 'product_designer',
        ...specialistConfig,
      },
      {
        agentId: 'devops_engineer',
        ...specialistConfig,
      },
      {
        agentId: 'legal_advisor',
        ...specialistConfig,
      },
    ];

    // Create edges from supervisor to all 5 specialists
    const edges: t.GraphEdge[] = [
      {
        from: 'supervisor',
        to: 'data_analyst',
        description:
          'Transfer to data analyst for statistical analysis and metrics',
        edgeType: 'handoff',
      },
      {
        from: 'supervisor',
        to: 'security_expert',
        description: 'Transfer to security expert for cybersecurity assessment',
        edgeType: 'handoff',
      },
      {
        from: 'supervisor',
        to: 'product_designer',
        description: 'Transfer to product designer for UX/UI design',
        edgeType: 'handoff',
      },
      {
        from: 'supervisor',
        to: 'devops_engineer',
        description:
          'Transfer to DevOps engineer for infrastructure and deployment',
        edgeType: 'handoff',
      },
      {
        from: 'supervisor',
        to: 'legal_advisor',
        description: 'Transfer to legal advisor for compliance and licensing',
        edgeType: 'handoff',
      },
    ];

    return {
      runId: `supervisor-multi-agent-${Date.now()}`,
      graphConfig: {
        type: 'multi-agent',
        agents,
        edges,
      },
      customHandlers,
      returnContent: true,
    };
  }

  try {
    // Test with different queries
    const testQueries = [
      'How can we analyze user engagement metrics to improve our product?',
      // 'What security measures should we implement for our new API?',
      // 'Can you help design a better onboarding flow for our mobile app?',
      // 'We need to set up a CI/CD pipeline for our microservices.',
      // 'What are the legal implications of using GPL-licensed code in our product?',
    ];

    const config = {
      configurable: {
        thread_id: 'supervisor-conversation-1',
      },
      streamMode: 'values',
      version: 'v2' as const,
    };

    for (const query of testQueries) {
      console.log(`\n${'='.repeat(60)}`);
      console.log(`USER QUERY: "${query}"`);
      console.log('='.repeat(60));

      // Reset conversation
      conversationHistory.length = 0;
      conversationHistory.push(new HumanMessage(query));

      // Create graph with supervisor having 5 handoff tools to 1 adaptive specialist
      const runConfig = createSupervisorGraph();
      const run = await Run.create(runConfig);

      console.log('Processing request...');

      // Process with streaming
      const inputs = {
        messages: conversationHistory,
      };

      const finalContentParts = await run.processStream(inputs, config);
      const finalMessages = run.getRunMessages();

      if (finalMessages) {
        conversationHistory.push(...finalMessages);
      }

      // Show summary
      console.log(`\n${'─'.repeat(60)}`);
      console.log(`Agents in graph: 6 total (supervisor + 5 specialists)`);
      console.log(`All specialists share the same adaptive configuration`);
      console.log(
        `Supervisor tools: transfer_to_data_analyst, transfer_to_security_expert,`
      );
      console.log(
        `                 transfer_to_product_designer, transfer_to_devops_engineer,`
      );
      console.log(`                 transfer_to_legal_advisor`);
      console.log('─'.repeat(60));
    }
  } catch (error) {
    console.error('Error in supervisor multi-agent test:', error);
  }
}

// Run the test
testSupervisorMultiAgent();
