import fs from 'fs/promises';
import { pull } from "langchain/hub";
import { ChatOpenAI } from "@langchain/openai";
import type { ChatPromptTemplate } from "@langchain/core/prompts";
import { AgentExecutor, createOpenAIFunctionsAgent, AgentStep } from "langchain/agents";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import type { RunLogPatch } from "@langchain/core/tracers/log_stream";
import dotenv from 'dotenv';

type ExtractedJSONPatchOperation = Pick<RunLogPatch, 'ops'>;
type OperationType = ExtractedJSONPatchOperation extends { ops: (infer T)[] } ? T : never;


// Load environment variables from .env file
dotenv.config();

// Define the tools the agent will have access to.
const tools = [new TavilySearchResults({})];

const llm = new ChatOpenAI({
  model: "gpt-3.5-turbo-1106",
  temperature: 0,
  streaming: true,
});

// Get the prompt to use - you can modify this!
// If you want to see the prompt in full, you can at:
// https://smith.langchain.com/hub/hwchase17/openai-functions-agent
const prompt = await pull<ChatPromptTemplate>(
  "hwchase17/openai-functions-agent"
);

const agent = await createOpenAIFunctionsAgent({
  llm,
  tools,
  prompt,
});

const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

const logStream = await agentExecutor.streamLog({
  input: "what are the current US election polls 2024. today is 7/6/24",
});

const finalState: RunLogPatch[] = [];
const outputs: RunLogPatch[] = [];
let accumulatedOutput = '';
let accumulatedArguments = '';

let functionName: string | undefined = undefined;

function processStreamedOutput(op: any) {
  let output = '';
  if (op.value.text !== undefined) {
    output += op.value.text;
  }
  if (op.value.message && op.value.message.kwargs) {
    const kwargs = op.value.message.kwargs;
    if (kwargs.content) {
      output += kwargs.content;
    }
  }
  if (output) {
    accumulatedOutput += output;
    process.stdout.write(output);
  }
}

// A helper function to handle the event pattern for logged arguments
function handleLoggedArgument(loggedArgument: any) {
  if (loggedArgument.value?.message?.additional_kwargs?.function_call) {
    const functionCall = loggedArgument.value.message.additional_kwargs.function_call;

    if (functionCall.name) {
      functionName = functionCall.name;
      process.stdout.write(`Logged Function Name:
        ${JSON.stringify(functionCall, null, 2)}
      `);
    }

    if (functionCall.arguments) {
      accumulatedArguments += functionCall.arguments;
      // Print the part of the argument as it comes
      // process.stdout.write(`Logged Argument: { "arguments": "${functionCall.arguments}" }\n`);
      process.stdout.write(`Logged Argument:\n${JSON.stringify(functionCall, null, 2)}`);
    }

    // Check if the full arguments string has been accumulated
    if (accumulatedArguments.startsWith("{") && accumulatedArguments.endsWith("}")) {
      // Build the final logged argument string
      const completeArguments = accumulatedArguments;
      const namePart = functionName ? `"name": "${functionName}", ` : '';

      console.log(`\nLogged Argument: {\n  ${namePart}"arguments": ${completeArguments}\n}\n`);

      // Reset accumulators
      accumulatedArguments = '';
      functionName = undefined;
    }
  }
}

for await (const chunk of logStream) {
  finalState.push(chunk);
  outputs.push(chunk);

  if (!chunk.ops) continue;

  for (const op of chunk.ops) {
    if (isStreamedOutput(op)) {
      processStreamedOutput(op);
      if (hasFunctionCall(op)) {
        handleLoggedArgument(op);
      }
    } else if (isFinalOutput(op)) {
      printFinalOutput(op);
    }
  }
}

function isStreamedOutput(op: OperationType) {
  return op.op === 'add' && (
    op.path.includes('/streamed_output/-') || 
    op.path.includes('/streamed_output_str/-')
  );
}

function hasFunctionCall(op: OperationType) {
  return op?.['value']?.message?.additional_kwargs?.function_call;
}

function isFinalOutput(op: OperationType) {
  return op.op === 'add' && 
         op.value?.output && 
         op.path?.startsWith('/logs/') && 
         op.path?.endsWith('final_output') && 
         !op.path?.includes('Runnable');
}

function printFinalOutput(op: OperationType) {
  process.stdout.write(JSON.stringify(op, null, 2));
  process.stdout.write(`

########################_START_##########################
        ${JSON.stringify(op?.['value']?.output, null, 2)}
########################__END__##########################

        `);
}

// Define types for the final output structure
interface FinalOutput {
  id: string;
  streamed_output: Array<{
    intermediateSteps?: AgentStep[];
    output?: string;
  }>;
  final_output?: {
    output: string;
  };
  logs: Record<string, any>;
}

// Process finalState to create FinalOutput
const finalOutput: FinalOutput = {
  id: '',
  streamed_output: [],
  logs: {},
};

for (const patch of finalState) {
  if (patch.ops) {
    for (const op of patch.ops) {
      if (op.op === 'add' || op.op === 'replace') {
        if (op.path === '/id') {
          finalOutput.id = op.value;
        } else if (op.path === '/streamed_output/-') {
          finalOutput.streamed_output.push(op.value);
        } else if (op.path === '/final_output') {
          finalOutput.final_output = op.value;
        } else if (op.path.startsWith('/logs/')) {
          const logKey = op.path.split('/')[2];
          finalOutput.logs[logKey] = op.value;
        }
      }
    }
  }
}

// Save outputs to a JSON file
await fs.writeFile('outputs.json', JSON.stringify(outputs, null, 2));
console.log("\n\nOutputs have been saved to outputs.json");

// Save the final state separately
await fs.writeFile('final_output.json', JSON.stringify(finalOutput, null, 2));
console.log("\n\nFinal output has been saved to final_output.json");

// Save the cleaned-up accumulated output
await fs.writeFile('cleaned_output.txt', accumulatedOutput);
console.log("\n\nCleaned output has been saved to cleaned_output.txt");
