import { CallbackHandler } from '@langfuse/langchain';
import {
  getLangfuseTracerProvider,
  propagateAttributes,
} from '@langfuse/tracing';
import type { BaseMessage, MessageContent } from '@langchain/core/messages';
import type { Generation, LLMResult } from '@langchain/core/outputs';
import type { PropagateAttributesParams } from '@langfuse/tracing';
import type * as t from '@/types';
import { isPresent } from '@/utils/misc';
import { ContentTypes } from '@/common';

const TRACE_METADATA_MAX_LENGTH = 200;
const LANGFUSE_FORCE_FLUSH_ON_DISPOSE = 'LANGFUSE_FORCE_FLUSH_ON_DISPOSE';

export type LangfuseTraceMetadata = Record<string, string>;
type LangfuseMetadata = NonNullable<t.LangfuseConfig['metadata']>;

type LangfuseHandlerParams = {
  userId?: string;
  sessionId?: string;
  traceMetadata?: LangfuseTraceMetadata;
  tags?: string[];
};

type AgentLangfuseHandlerParams = LangfuseHandlerParams & {
  langfuse?: t.LangfuseConfig;
};

type LangfuseAttributeParams = AgentLangfuseHandlerParams & {
  traceName?: string;
};

type FlushableTracerProvider = {
  forceFlush?: () => Promise<void> | void;
};

type ReasoningSummary = {
  summary?: Array<{ text?: string | null }>;
};

type ReasoningDetail = {
  type?: string;
  text?: string | null;
};

type MessageWithReasoning = BaseMessage & {
  additional_kwargs?: BaseMessage['additional_kwargs'] & {
    reasoning?: string | ReasoningSummary | null;
    reasoning_content?: string | ReasoningSummary | null;
    reasoning_details?: ReasoningDetail[] | null;
  };
};

type ChatGenerationWithMessage = Generation & {
  message: MessageWithReasoning;
};

function parseBooleanEnv(value?: string): boolean {
  if (value == null) {
    return false;
  }
  return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function getReasoningText(
  value: string | ReasoningSummary | null | undefined
): string | undefined {
  if (typeof value === 'string') {
    return isPresent(value) ? value : undefined;
  }
  const summaryText = value?.summary
    ?.map((summary) => summary.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return summaryText != null && summaryText !== '' ? summaryText : undefined;
}

function getReasoningDetailsText(
  value: ReasoningDetail[] | null | undefined
): string | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  const reasoningText = value
    .filter((detail) => detail.type === 'reasoning.text')
    .map((detail) => detail.text ?? '')
    .filter((text) => text !== '')
    .join('');
  return reasoningText !== '' ? reasoningText : undefined;
}

function getMessageReasoningText(
  message: MessageWithReasoning
): string | undefined {
  const additionalKwargs = message.additional_kwargs;
  if (additionalKwargs == null) {
    return undefined;
  }

  return (
    getReasoningText(additionalKwargs.reasoning_content) ??
    getReasoningText(additionalKwargs.reasoning) ??
    getReasoningDetailsText(additionalKwargs.reasoning_details)
  );
}

function hasReasoningContentPart(content: MessageContent): boolean {
  return (
    Array.isArray(content) &&
    content.some((part) => {
      if (!isRecord(part) || typeof part.type !== 'string') {
        return false;
      }
      return (
        part.type === ContentTypes.THINK ||
        part.type === ContentTypes.THINKING ||
        part.type === ContentTypes.REASONING ||
        part.type === ContentTypes.REASONING_CONTENT ||
        part.type === 'redacted_thinking'
      );
    })
  );
}

function withReasoningContent(
  content: MessageContent,
  reasoningText: string
): MessageContent {
  if (hasReasoningContentPart(content)) {
    return content;
  }

  const reasoningPart = {
    type: ContentTypes.THINK,
    think: reasoningText,
  };
  return Array.isArray(content)
    ? [reasoningPart, ...content]
    : [reasoningPart, { type: ContentTypes.TEXT, text: content }];
}

function hasMessage(
  generation: Generation
): generation is ChatGenerationWithMessage {
  return (
    'message' in generation &&
    isRecord(generation.message) &&
    'additional_kwargs' in generation.message &&
    'content' in generation.message
  );
}

function cloneMessageWithContent(
  message: MessageWithReasoning,
  content: MessageContent
): MessageWithReasoning {
  return Object.assign(Object.create(Object.getPrototypeOf(message)), message, {
    content,
  });
}

function withReasoningForLangfuseOutput(output: LLMResult): LLMResult {
  let changed = false;
  const generations = output.generations.map((generationList) => {
    return generationList.map((generation) => {
      if (!hasMessage(generation)) {
        return generation;
      }
      const reasoningText = getMessageReasoningText(generation.message);
      if (!isPresent(reasoningText)) {
        return generation;
      }

      const originalContent = generation.message.content;
      const nextContent = withReasoningContent(originalContent, reasoningText);
      if (nextContent === originalContent) {
        return generation;
      }

      changed = true;
      return {
        ...generation,
        message: cloneMessageWithContent(generation.message, nextContent),
      };
    });
  });

  if (!changed) {
    return output;
  }
  return {
    ...output,
    generations,
  };
}

class ReasoningAwareLangfuseCallbackHandler extends CallbackHandler {
  override async handleLLMEnd(
    output: LLMResult,
    runId: string,
    parentRunId?: string | undefined
  ): Promise<void> {
    await super.handleLLMEnd(
      withReasoningForLangfuseOutput(output),
      runId,
      parentRunId
    );
  }
}

function hasLangfuseTracingConfig(langfuse?: t.LangfuseConfig): boolean {
  return (
    langfuse?.toolNodeTracing != null || langfuse?.toolOutputTracing != null
  );
}

function hasLangfuseTraceAttributes(langfuse?: t.LangfuseConfig): boolean {
  return (
    Object.keys(createTraceMetadata(langfuse?.metadata ?? {})).length > 0 ||
    (mergeLangfuseTags(undefined, langfuse?.tags)?.length ?? 0) > 0
  );
}

export function hasLangfuseConfigCredentials(
  langfuse?: t.LangfuseConfig
): langfuse is t.LangfuseConfig & {
  publicKey: string;
  secretKey: string;
} {
  return (
    langfuse != null &&
    isPresent(langfuse.publicKey) &&
    isPresent(langfuse.secretKey)
  );
}

function hasLangfuseConfigBaseUrl(langfuse?: t.LangfuseConfig): boolean {
  return isPresent(langfuse?.baseUrl);
}

export function isExplicitLangfuseConfig(langfuse?: t.LangfuseConfig): boolean {
  return (
    langfuse?.enabled != null ||
    isPresent(langfuse?.publicKey) ||
    isPresent(langfuse?.secretKey) ||
    isPresent(langfuse?.baseUrl) ||
    hasLangfuseTraceAttributes(langfuse) ||
    hasLangfuseTracingConfig(langfuse)
  );
}

function createTraceMetadata(
  metadata: Record<string, unknown>
): LangfuseTraceMetadata {
  const traceMetadata: LangfuseTraceMetadata = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (value == null) {
      continue;
    }
    const stringValue = typeof value === 'string' ? value : String(value);
    if (
      stringValue.trim() === '' ||
      stringValue.length > TRACE_METADATA_MAX_LENGTH
    ) {
      continue;
    }
    traceMetadata[key] = stringValue;
  }
  return traceMetadata;
}

export function createLangfuseTraceMetadata({
  messageId,
  parentMessageId,
  agentId,
  agentName,
}: {
  messageId?: unknown;
  parentMessageId?: unknown;
  agentId?: unknown;
  agentName?: unknown;
}): LangfuseTraceMetadata {
  return createTraceMetadata({
    messageId,
    parentMessageId,
    agentId,
    agentName,
  });
}

function mergeLangfuseTraceMetadata(
  traceMetadata?: LangfuseTraceMetadata,
  metadata?: LangfuseMetadata
): LangfuseTraceMetadata | undefined {
  const merged = createTraceMetadata({
    ...(metadata ?? {}),
    ...(traceMetadata ?? {}),
  });
  return Object.keys(merged).length > 0 ? merged : undefined;
}

function mergeLangfuseTags(
  tags?: string[],
  configTags?: string[]
): string[] | undefined {
  const merged = [...(tags ?? []), ...(configTags ?? [])].filter(
    (tag) => tag.trim() !== ''
  );
  return merged.length > 0 ? [...new Set(merged)] : undefined;
}

export function getLangfuseTraceName(
  traceMetadata?: LangfuseTraceMetadata,
  fallback: string = 'LibreChat Agent'
): string {
  const agentName = traceMetadata?.agentName;
  return isPresent(agentName) ? `${fallback}: ${agentName}` : fallback;
}

export function hasLangfuseEnvConfig(): boolean {
  return hasLangfuseEnvCredentials();
}

export function hasLangfuseEnvCredentials(): boolean {
  return (
    isPresent(process.env.LANGFUSE_SECRET_KEY) &&
    isPresent(process.env.LANGFUSE_PUBLIC_KEY)
  );
}

export function shouldCreateLangfuseHandler(
  langfuse?: t.LangfuseConfig
): boolean {
  if (langfuse?.enabled === false) {
    return false;
  }
  return (
    hasLangfuseEnvConfig() ||
    hasLangfuseConfigCredentials(langfuse) ||
    (hasLangfuseConfigBaseUrl(langfuse) && hasLangfuseEnvCredentials())
  );
}

export function createLegacyLangfuseHandler(
  params: LangfuseHandlerParams
): CallbackHandler {
  return new ReasoningAwareLangfuseCallbackHandler(params);
}

export function createLangfuseHandler({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
  tags,
}: AgentLangfuseHandlerParams): CallbackHandler | undefined {
  if (!shouldCreateLangfuseHandler(langfuse)) {
    return undefined;
  }
  return new ReasoningAwareLangfuseCallbackHandler({
    userId,
    sessionId,
    traceMetadata: mergeLangfuseTraceMetadata(
      traceMetadata,
      langfuse?.metadata
    ),
    tags: mergeLangfuseTags(tags, langfuse?.tags),
  });
}

function createPropagateAttributeParams({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
  traceName,
  tags,
}: LangfuseAttributeParams): PropagateAttributesParams {
  return {
    userId,
    sessionId,
    traceName,
    tags: mergeLangfuseTags(tags, langfuse?.tags),
    metadata: mergeLangfuseTraceMetadata(traceMetadata, langfuse?.metadata),
  };
}

export function withLangfuseAttributes<T>(
  params: LangfuseAttributeParams,
  action: () => T
): T {
  if (!shouldCreateLangfuseHandler(params.langfuse)) {
    return action();
  }
  return propagateAttributes(createPropagateAttributeParams(params), action);
}

export function hasExplicitLangfuseConfig(
  contexts: Iterable<{ langfuse?: t.LangfuseConfig }>
): boolean {
  for (const context of contexts) {
    if (isExplicitLangfuseConfig(context.langfuse)) {
      return true;
    }
  }
  return false;
}

export function isLangfuseCallbackHandler(value: unknown): boolean {
  return value instanceof CallbackHandler;
}

export async function disposeLangfuseHandler(value: unknown): Promise<void> {
  if (
    value == null ||
    !parseBooleanEnv(process.env[LANGFUSE_FORCE_FLUSH_ON_DISPOSE])
  ) {
    return;
  }
  const provider = getLangfuseTracerProvider() as FlushableTracerProvider;
  await provider.forceFlush?.();
}
