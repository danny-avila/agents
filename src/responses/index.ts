import { GraphEvents } from '@/common';
import type { UsageMetadata } from '@langchain/core/messages';
import type * as t from '@/types';

export interface ResponsesCompatibleWriter {
  write(data: string): void | Promise<void>;
}

export type ResponseStatus =
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'incomplete';
export type ItemStatus = 'in_progress' | 'incomplete' | 'completed';

export interface ResponseContext {
  responseId: string;
  model: string;
  createdAt: number;
  previousResponseId?: string;
  instructions?: string;
}

export interface ResponseOutputTextContent {
  type: 'output_text';
  text: string;
  annotations: [];
  logprobs: [];
}

export interface ResponseMessageItem {
  type: 'message';
  id: string;
  role: 'assistant';
  status: ItemStatus;
  content: ResponseOutputTextContent[];
}

export interface ResponseFunctionCallItem {
  type: 'function_call';
  id: string;
  call_id: string;
  name: string;
  arguments: string;
  status: ItemStatus;
}

export interface ResponseReasoningItem {
  type: 'reasoning';
  id: string;
  status: ItemStatus;
  content: Array<{ type: 'reasoning_text'; text: string }>;
  summary: [];
}

export type ResponseOutputItem =
  | ResponseMessageItem
  | ResponseFunctionCallItem
  | ResponseReasoningItem;

export interface ResponseObject {
  id: string;
  object: 'response';
  created_at: number;
  completed_at: number | null;
  status: ResponseStatus;
  model: string;
  previous_response_id: string | null;
  instructions: string | null;
  output: ResponseOutputItem[];
  error: { type: string; message: string; code?: string } | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  } | null;
}

export interface ResponseEvent {
  type: string;
  sequence_number: number;
  [key: string]: unknown;
}

export interface ResponseTracker {
  sequenceNumber: number;
  items: ResponseOutputItem[];
  message: ResponseMessageItem | undefined;
  reasoning: ResponseReasoningItem | undefined;
  functionCalls: Map<string, ResponseFunctionCallItem>;
  functionCallsByStep: Map<string, Array<ResponseFunctionCallItem | undefined>>;
  responseCreated: boolean;
  usage: {
    inputTokens: number;
    outputTokens: number;
  };
  nextSequence(): number;
}

export interface ResponsesHandlerConfig {
  writer: ResponsesCompatibleWriter;
  context: ResponseContext;
  tracker: ResponseTracker;
}

let responseItemId = 0;

function createItemId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}${(responseItemId++).toString(36)}`;
}

export function createResponseTracker(): ResponseTracker {
  const tracker: ResponseTracker = {
    sequenceNumber: 0,
    items: [],
    message: undefined,
    reasoning: undefined,
    functionCalls: new Map(),
    functionCallsByStep: new Map(),
    responseCreated: false,
    usage: {
      inputTokens: 0,
      outputTokens: 0,
    },
    nextSequence: () => tracker.sequenceNumber++,
  };
  return tracker;
}

function getTokenCount(value: number | null | undefined): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

interface ResponseToolCallFragment {
  index?: number;
  id?: string;
  name?: string;
  args?: string | object;
  function?: {
    name?: string;
    arguments?: string | object;
  };
}

function getToolCallIndex(
  toolCall: ResponseToolCallFragment,
  fallbackIndex: number
): number {
  return typeof toolCall.index === 'number' ? toolCall.index : fallbackIndex;
}

function getToolCallPositionKey(
  stepId: string,
  toolCall: ResponseToolCallFragment,
  fallbackIndex: number
): string {
  const index = getToolCallIndex(toolCall, fallbackIndex);
  if (stepId !== '') {
    return `${stepId}:${index}`;
  }
  return `tool:${index}`;
}

function getToolCallIdKey(
  toolCall: ResponseToolCallFragment
): string | undefined {
  return toolCall.id != null && toolCall.id !== '' ? toolCall.id : undefined;
}

function getExistingFunctionCall(
  config: ResponsesHandlerConfig,
  idKey: string | undefined,
  positionKey: string | undefined,
  stepId: string,
  toolCall: ResponseToolCallFragment
): ResponseFunctionCallItem | undefined {
  const keyed =
    idKey == null
      ? undefined
      : (config.tracker.functionCalls.get(idKey) ??
        (positionKey == null
          ? undefined
          : config.tracker.functionCalls.get(positionKey)));
  if (keyed != null) {
    return keyed;
  }
  const positioned =
    positionKey == null
      ? undefined
      : config.tracker.functionCalls.get(positionKey);
  if (positioned != null) {
    return positioned;
  }
  if (stepId === '') {
    return undefined;
  }
  return findFunctionCallByStep(config, stepId, toolCall);
}

function findFunctionCallByStep(
  config: ResponsesHandlerConfig,
  stepId: string,
  toolCall: ResponseToolCallFragment
): ResponseFunctionCallItem | undefined {
  const items = config.tracker.functionCallsByStep.get(stepId);
  if (items == null) {
    return undefined;
  }
  const index = typeof toolCall.index === 'number' ? toolCall.index : undefined;
  if (index != null) {
    return items[index];
  }
  const name = getToolCallName(toolCall);
  let fallback: ResponseFunctionCallItem | undefined;
  for (const item of items) {
    if (item == null || item.status === 'completed') {
      continue;
    }
    if (name !== '' && item.name !== name) {
      continue;
    }
    if (fallback != null) {
      return undefined;
    }
    fallback = item;
  }
  return fallback;
}

function trackFunctionCallKeys(
  config: ResponsesHandlerConfig,
  item: ResponseFunctionCallItem,
  idKey: string | undefined,
  positionKey: string
): void {
  config.tracker.functionCalls.set(positionKey, item);
  if (idKey != null) {
    config.tracker.functionCalls.set(idKey, item);
  }
}

function trackFunctionCallStep(
  config: ResponsesHandlerConfig,
  stepId: string,
  item: ResponseFunctionCallItem,
  fallbackIndex: number
): void {
  if (stepId === '') {
    return;
  }
  const items = config.tracker.functionCallsByStep.get(stepId) ?? [];
  items[fallbackIndex] = item;
  config.tracker.functionCallsByStep.set(stepId, items);
}

function getToolCallName(toolCall: ResponseToolCallFragment): string {
  return toolCall.name ?? toolCall.function?.name ?? '';
}

function getToolCallArguments(toolCall: ResponseToolCallFragment): string {
  const args = toolCall.args ?? toolCall.function?.arguments;
  if (args == null) {
    return '';
  }
  if (typeof args === 'string') {
    return args;
  }
  return JSON.stringify(args);
}

async function ensureFunctionCall(
  config: ResponsesHandlerConfig,
  stepId: string,
  toolCall: ResponseToolCallFragment,
  fallbackIndex: number
): Promise<ResponseFunctionCallItem> {
  await ensureResponseCreated(config);
  const positionKey = getToolCallPositionKey(stepId, toolCall, fallbackIndex);
  const idKey = getToolCallIdKey(toolCall);
  const existing = getExistingFunctionCall(
    config,
    idKey,
    positionKey,
    stepId,
    toolCall
  );
  const toolCallIndex = getToolCallIndex(toolCall, fallbackIndex);
  const name = getToolCallName(toolCall);
  if (existing) {
    trackFunctionCallKeys(config, existing, idKey, positionKey);
    trackFunctionCallStep(config, stepId, existing, toolCallIndex);
    if (idKey != null) {
      existing.call_id = idKey;
    }
    if (name !== '') {
      existing.name = name;
    }
    return existing;
  }
  const item: ResponseFunctionCallItem = {
    type: 'function_call',
    id: createItemId('fc'),
    call_id: idKey ?? positionKey,
    name,
    arguments: '',
    status: 'in_progress',
  };
  trackFunctionCallKeys(config, item, idKey, positionKey);
  trackFunctionCallStep(config, stepId, item, toolCallIndex);
  config.tracker.items.push(item);
  await writeResponseEvent(config.writer, {
    type: 'response.output_item.added',
    sequence_number: config.tracker.nextSequence(),
    output_index: config.tracker.items.length - 1,
    item,
  });
  return item;
}

async function emitFunctionCallArgumentsDelta(params: {
  config: ResponsesHandlerConfig;
  item: ResponseFunctionCallItem;
  delta: string;
}): Promise<void> {
  if (params.delta === '') {
    return;
  }
  params.item.arguments += params.delta;
  await writeResponseEvent(params.config.writer, {
    type: 'response.function_call_arguments.delta',
    sequence_number: params.config.tracker.nextSequence(),
    item_id: params.item.id,
    output_index: params.config.tracker.items.indexOf(params.item),
    call_id: params.item.call_id,
    delta: params.delta,
  });
}

async function emitFunctionCallArgumentsDone(
  config: ResponsesHandlerConfig,
  item: ResponseFunctionCallItem
): Promise<void> {
  await writeResponseEvent(config.writer, {
    type: 'response.function_call_arguments.done',
    sequence_number: config.tracker.nextSequence(),
    item_id: item.id,
    output_index: config.tracker.items.indexOf(item),
    call_id: item.call_id,
    name: item.name,
    arguments: item.arguments,
  });
}

async function completeFunctionCall(
  config: ResponsesHandlerConfig,
  stepId: string,
  toolCall: ResponseToolCallFragment
): Promise<void> {
  const fallbackIndex =
    typeof toolCall.index === 'number' ? toolCall.index : undefined;
  const positionKey =
    fallbackIndex == null
      ? undefined
      : getToolCallPositionKey(stepId, toolCall, fallbackIndex);
  const item =
    getExistingFunctionCall(
      config,
      getToolCallIdKey(toolCall),
      positionKey,
      stepId,
      toolCall
    ) ??
    (await ensureFunctionCall(config, stepId, toolCall, fallbackIndex ?? 0));
  if (item.status === 'completed') {
    return;
  }
  const finalArguments = getToolCallArguments(toolCall);
  if (
    finalArguments !== '' &&
    finalArguments !== item.arguments &&
    finalArguments.startsWith(item.arguments)
  ) {
    await emitFunctionCallArgumentsDelta({
      config,
      item,
      delta: finalArguments.slice(item.arguments.length),
    });
  } else if (finalArguments !== '') {
    item.arguments = finalArguments;
  }
  await emitFunctionCallArgumentsDone(config, item);
  item.status = 'completed';
  await writeResponseEvent(config.writer, {
    type: 'response.output_item.done',
    sequence_number: config.tracker.nextSequence(),
    output_index: config.tracker.items.indexOf(item),
    item,
  });
}

export function buildResponse(
  context: ResponseContext,
  tracker: ResponseTracker,
  status: ResponseStatus = 'in_progress'
): ResponseObject {
  const completed = status === 'completed';
  return {
    id: context.responseId,
    object: 'response',
    created_at: context.createdAt,
    completed_at: completed ? Math.floor(Date.now() / 1000) : null,
    status,
    model: context.model,
    previous_response_id: context.previousResponseId ?? null,
    instructions: context.instructions ?? null,
    output: tracker.items,
    error: null,
    usage: completed
      ? {
        input_tokens: tracker.usage.inputTokens,
        output_tokens: tracker.usage.outputTokens,
        total_tokens: tracker.usage.inputTokens + tracker.usage.outputTokens,
      }
      : null,
  };
}

export async function writeResponseEvent(
  writer: ResponsesCompatibleWriter,
  event: ResponseEvent
): Promise<void> {
  await writer.write(`event: ${event.type}\n`);
  await writer.write(`data: ${JSON.stringify(event)}\n\n`);
}

export async function writeResponsesDone(
  writer: ResponsesCompatibleWriter
): Promise<void> {
  await writer.write('data: [DONE]\n\n');
}

export async function ensureResponseCreated(
  config: ResponsesHandlerConfig
): Promise<void> {
  if (config.tracker.responseCreated) {
    return;
  }
  config.tracker.responseCreated = true;
  await writeResponseEvent(config.writer, {
    type: 'response.created',
    sequence_number: config.tracker.nextSequence(),
    response: buildResponse(config.context, config.tracker),
  });
}

async function emitOutputContentDone(
  config: ResponsesHandlerConfig,
  item: ResponseMessageItem | ResponseReasoningItem
): Promise<void> {
  const outputIndex = config.tracker.items.indexOf(item);
  if (item.type === 'message') {
    await writeResponseEvent(config.writer, {
      type: 'response.output_text.done',
      sequence_number: config.tracker.nextSequence(),
      item_id: item.id,
      output_index: outputIndex,
      content_index: 0,
      text: item.content[0].text,
    });
    return;
  }
  await writeResponseEvent(config.writer, {
    type: 'response.reasoning_text.done',
    sequence_number: config.tracker.nextSequence(),
    item_id: item.id,
    output_index: outputIndex,
    content_index: 0,
    text: item.content[0].text,
  });
}

async function ensureMessage(
  config: ResponsesHandlerConfig
): Promise<ResponseMessageItem> {
  await ensureResponseCreated(config);
  if (config.tracker.message) {
    return config.tracker.message;
  }
  const item: ResponseMessageItem = {
    type: 'message',
    id: createItemId('msg'),
    role: 'assistant',
    status: 'in_progress',
    content: [{ type: 'output_text', text: '', annotations: [], logprobs: [] }],
  };
  config.tracker.message = item;
  config.tracker.items.push(item);
  await writeResponseEvent(config.writer, {
    type: 'response.output_item.added',
    sequence_number: config.tracker.nextSequence(),
    output_index: config.tracker.items.length - 1,
    item,
  });
  return item;
}

async function ensureReasoning(
  config: ResponsesHandlerConfig
): Promise<ResponseReasoningItem> {
  await ensureResponseCreated(config);
  if (config.tracker.reasoning) {
    return config.tracker.reasoning;
  }
  const item: ResponseReasoningItem = {
    type: 'reasoning',
    id: createItemId('reason'),
    status: 'in_progress',
    content: [{ type: 'reasoning_text', text: '' }],
    summary: [],
  };
  config.tracker.reasoning = item;
  config.tracker.items.push(item);
  await writeResponseEvent(config.writer, {
    type: 'response.output_item.added',
    sequence_number: config.tracker.nextSequence(),
    output_index: config.tracker.items.length - 1,
    item,
  });
  return item;
}

export function createResponsesEventHandlers(
  config: ResponsesHandlerConfig
): Record<string, t.EventHandler> {
  return {
    [GraphEvents.ON_MESSAGE_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        const item = await ensureMessage(config);
        for (const part of (data as t.MessageDeltaEvent).delta.content ?? []) {
          if (!('text' in part) || typeof part.text !== 'string') {
            continue;
          }
          item.content[0].text += part.text;
          await writeResponseEvent(config.writer, {
            type: 'response.output_text.delta',
            sequence_number: config.tracker.nextSequence(),
            item_id: item.id,
            output_index: config.tracker.items.indexOf(item),
            content_index: 0,
            delta: part.text,
          });
        }
      },
    },
    [GraphEvents.ON_REASONING_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        const item = await ensureReasoning(config);
        for (const part of (data as t.ReasoningDeltaEvent).delta.content ??
          []) {
          let text: string | undefined;
          if ('think' in part && typeof part.think === 'string') {
            text = part.think;
          } else if ('text' in part && typeof part.text === 'string') {
            text = part.text;
          }
          if (typeof text !== 'string') {
            continue;
          }
          item.content[0].text += text;
          await writeResponseEvent(config.writer, {
            type: 'response.reasoning_text.delta',
            sequence_number: config.tracker.nextSequence(),
            item_id: item.id,
            output_index: config.tracker.items.indexOf(item),
            content_index: 0,
            delta: text,
          });
        }
      },
    },
    [GraphEvents.ON_RUN_STEP]: {
      handle: async (_event, data): Promise<void> => {
        const runStep = data as t.RunStep;
        if (runStep.stepDetails.type !== 'tool_calls') {
          return;
        }
        const toolCalls = runStep.stepDetails.tool_calls ?? [];
        for (let index = 0; index < toolCalls.length; index++) {
          await ensureFunctionCall(config, runStep.id, toolCalls[index], index);
        }
      },
    },
    [GraphEvents.ON_RUN_STEP_DELTA]: {
      handle: async (_event, data): Promise<void> => {
        const runStepDelta = data as t.RunStepDeltaEvent;
        if (runStepDelta.delta.type !== 'tool_calls') {
          return;
        }
        const toolCalls = runStepDelta.delta.tool_calls ?? [];
        for (let index = 0; index < toolCalls.length; index++) {
          const item = await ensureFunctionCall(
            config,
            runStepDelta.id,
            toolCalls[index],
            index
          );
          await emitFunctionCallArgumentsDelta({
            config,
            item,
            delta: getToolCallArguments(toolCalls[index]),
          });
        }
      },
    },
    [GraphEvents.ON_RUN_STEP_COMPLETED]: {
      handle: async (_event, data): Promise<void> => {
        const completed = data as {
          result?: {
            id?: string;
            index?: number;
            type?: string;
            tool_call?: ResponseToolCallFragment;
          };
        };
        if (!completed.result?.tool_call) {
          return;
        }
        await completeFunctionCall(
          config,
          completed.result.id ?? '',
          completed.result.tool_call
        );
      },
    },
    [GraphEvents.CHAT_MODEL_END]: {
      handle: (_event, data): void => {
        const usage = (data as t.ModelEndData)?.output?.usage_metadata as
          | Partial<UsageMetadata>
          | undefined;
        if (!usage) {
          return;
        }
        config.tracker.usage.inputTokens += getTokenCount(usage.input_tokens);
        config.tracker.usage.outputTokens += getTokenCount(usage.output_tokens);
      },
    },
  };
}

export async function emitResponseCompleted(
  config: ResponsesHandlerConfig
): Promise<void> {
  await ensureResponseCreated(config);
  for (const item of config.tracker.items) {
    if (item.status === 'completed') {
      continue;
    }
    if (item.type === 'message' || item.type === 'reasoning') {
      await emitOutputContentDone(config, item);
    } else {
      await emitFunctionCallArgumentsDone(config, item);
    }
    item.status = 'completed';
    await writeResponseEvent(config.writer, {
      type: 'response.output_item.done',
      sequence_number: config.tracker.nextSequence(),
      output_index: config.tracker.items.indexOf(item),
      item,
    });
  }
  await writeResponseEvent(config.writer, {
    type: 'response.completed',
    sequence_number: config.tracker.nextSequence(),
    response: buildResponse(config.context, config.tracker, 'completed'),
  });
  await writeResponsesDone(config.writer);
}
