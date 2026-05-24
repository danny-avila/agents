import { CallbackHandler } from '@langfuse/langchain';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import {
  createObservationAttributes,
  createTraceAttributes,
} from '@langfuse/tracing';
import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { SpanStatusCode } from '@opentelemetry/api';
import type { Serialized } from '@langchain/core/load/serializable';
import type { BaseMessage } from '@langchain/core/messages';
import type { LLMResult } from '@langchain/core/outputs';
import type { Span } from '@opentelemetry/api';
import type * as t from '@/types';
import { isPresent } from '@/utils/misc';

type TraceMetadata = Record<string, unknown>;

type LangfuseHandlerParams = {
  userId?: string;
  sessionId?: string;
  traceMetadata?: TraceMetadata;
};

type AgentLangfuseHandlerParams = LangfuseHandlerParams & {
  langfuse?: t.LangfuseConfig;
};

type ResolvedLangfuseConfig = t.LangfuseConfig & {
  enabled: true;
  publicKey: string;
  secretKey: string;
};

function getModelName(serialized: Serialized): string {
  const serializedRecord = serialized as unknown as Record<string, unknown>;
  const kwargs = serializedRecord.kwargs as Record<string, unknown> | undefined;
  const modelName =
    kwargs?.model ??
    kwargs?.model_name ??
    kwargs?.modelName ??
    kwargs?.model_id ??
    kwargs?.modelId ??
    serializedRecord.name;

  if (typeof modelName === 'string' && modelName.trim() !== '') {
    return modelName;
  }

  if (Array.isArray(serializedRecord.id) && serializedRecord.id.length > 0) {
    return String(serializedRecord.id[serializedRecord.id.length - 1]);
  }

  return 'ChatModel';
}

function getModelParameters(
  extraParams?: Record<string, unknown>
): Record<string, string | number> {
  const invocationParams = extraParams?.invocation_params;
  const params =
    invocationParams != null && typeof invocationParams === 'object'
      ? (invocationParams as Record<string, unknown>)
      : (extraParams ?? {});

  return Object.fromEntries(
    Object.entries(params).filter(([, value]) => {
      return typeof value === 'string' || typeof value === 'number';
    })
  ) as Record<string, string | number>;
}

function getOutput(output: LLMResult): unknown {
  return output.generations.map((generation) =>
    generation.map((item) => {
      if ('message' in item && item.message != null) {
        return (item.message as { content?: unknown }).content;
      }
      return item.text;
    })
  );
}

function getUsageDetails(
  output: LLMResult
): Record<string, number> | undefined {
  const llmOutput = output.llmOutput as Record<string, unknown> | undefined;
  const usage = llmOutput?.tokenUsage ?? llmOutput?.usage;
  if (usage == null || typeof usage !== 'object') {
    return undefined;
  }

  const usageEntries = Object.entries(usage as Record<string, unknown>).filter(
    ([, value]) => typeof value === 'number'
  );

  return usageEntries.length > 0
    ? (Object.fromEntries(usageEntries) as Record<string, number>)
    : undefined;
}

function getTraceName(traceMetadata?: TraceMetadata): string {
  const agentName = traceMetadata?.agentName;
  return typeof agentName === 'string' && agentName.trim() !== ''
    ? `LibreChat Agent: ${agentName}`
    : 'LibreChat Agent';
}

export class LangfuseAgentCallbackHandler extends BaseCallbackHandler {
  name = 'librechat_langfuse_agent_handler';

  private readonly provider: BasicTracerProvider;
  private readonly processor: LangfuseSpanProcessor;
  private readonly userId?: string;
  private readonly sessionId?: string;
  private readonly traceMetadata?: TraceMetadata;
  private readonly spans = new Map<string, Span>();

  constructor({
    langfuse,
    userId,
    sessionId,
    traceMetadata,
  }: LangfuseHandlerParams & { langfuse: ResolvedLangfuseConfig }) {
    super();
    this.userId = userId;
    this.sessionId = sessionId;
    this.traceMetadata = traceMetadata;
    this.processor = new LangfuseSpanProcessor({
      publicKey: langfuse.publicKey,
      secretKey: langfuse.secretKey,
      ...(isPresent(langfuse.baseUrl) ? { baseUrl: langfuse.baseUrl } : {}),
      environment:
        process.env.LANGFUSE_TRACING_ENVIRONMENT ??
        process.env.NODE_ENV ??
        'development',
      exportMode: 'immediate',
    });
    this.provider = new BasicTracerProvider({
      spanProcessors: [this.processor],
    });
  }

  private startGenerationSpan({
    llm,
    input,
    runId,
    extraParams,
    metadata,
    name,
  }: {
    llm: Serialized;
    input: unknown;
    runId: string;
    extraParams?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
    name?: string;
  }): void {
    if (this.spans.has(runId)) {
      return;
    }

    const tracer = this.provider.getTracer('librechat-agents-langfuse');
    const spanName =
      typeof name === 'string' && name.trim() !== '' ? name : getModelName(llm);
    const span = tracer.startSpan(spanName, {
      attributes: {
        ...createTraceAttributes({
          name: getTraceName(this.traceMetadata),
          userId: this.userId,
          sessionId: this.sessionId,
          metadata: this.traceMetadata,
        }),
        ...createObservationAttributes('generation', {
          input,
          model: getModelName(llm),
          modelParameters: getModelParameters(extraParams),
          metadata: {
            ...metadata,
            ...this.traceMetadata,
          },
        }),
      },
    });
    this.spans.set(runId, span);
  }

  async handleChatModelStart(
    llm: Serialized,
    messages: BaseMessage[][],
    runId: string,
    _parentRunId?: string,
    extraParams?: Record<string, unknown>,
    _tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string
  ): Promise<void> {
    this.startGenerationSpan({
      llm,
      input: messages,
      runId,
      extraParams,
      metadata,
      name,
    });
  }

  async handleLLMStart(
    llm: Serialized,
    prompts: string[],
    runId: string,
    _parentRunId?: string,
    extraParams?: Record<string, unknown>,
    _tags?: string[],
    metadata?: Record<string, unknown>,
    name?: string
  ): Promise<void> {
    this.startGenerationSpan({
      llm,
      input: prompts,
      runId,
      extraParams,
      metadata,
      name,
    });
  }

  async handleLLMEnd(output: LLMResult, runId: string): Promise<void> {
    const span = this.spans.get(runId);
    if (!span) {
      return;
    }

    span.setAttributes(
      createObservationAttributes('generation', {
        output: getOutput(output),
        usageDetails: getUsageDetails(output),
      })
    );
    span.end();
    this.spans.delete(runId);
    await this.flush();
  }

  async handleLLMError(err: unknown, runId: string): Promise<void> {
    const span = this.spans.get(runId);
    if (!span) {
      return;
    }

    const message = err instanceof Error ? err.message : String(err);
    span.setStatus({ code: SpanStatusCode.ERROR, message });
    span.setAttributes(
      createObservationAttributes('generation', {
        level: 'ERROR',
        statusMessage: message,
      })
    );
    span.end();
    this.spans.delete(runId);
    await this.flush();
  }

  private async flush(): Promise<void> {
    try {
      await this.provider.forceFlush();
    } catch (error) {
      process.emitWarning(
        `[LangfuseAgentCallbackHandler] Failed to flush Langfuse spans: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  async dispose(): Promise<void> {
    for (const span of this.spans.values()) {
      span.end();
    }
    this.spans.clear();
    await this.flush();
    try {
      await this.provider.shutdown();
    } catch (error) {
      process.emitWarning(
        `[LangfuseAgentCallbackHandler] Failed to shut down Langfuse provider: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }
}

function hasRequiredLangfuseConfig(
  langfuse?: t.LangfuseConfig
): langfuse is ResolvedLangfuseConfig {
  return (
    langfuse?.enabled === true &&
    isPresent(langfuse.publicKey) &&
    isPresent(langfuse.secretKey)
  );
}

export function createLegacyLangfuseHandler(
  params: LangfuseHandlerParams
): CallbackHandler {
  return new CallbackHandler(params);
}

export function createLangfuseHandler({
  langfuse,
  userId,
  sessionId,
  traceMetadata,
}: AgentLangfuseHandlerParams): LangfuseAgentCallbackHandler | undefined {
  if (!hasRequiredLangfuseConfig(langfuse)) {
    return undefined;
  }

  return new LangfuseAgentCallbackHandler({
    langfuse,
    userId,
    sessionId,
    traceMetadata,
  });
}

export function hasExplicitLangfuseConfig(
  contexts: Iterable<{ langfuse?: t.LangfuseConfig }>
): boolean {
  for (const context of contexts) {
    if (context.langfuse != null) {
      return true;
    }
  }
  return false;
}

export function hasLangfuseEnvConfig(): boolean {
  return (
    isPresent(process.env.LANGFUSE_SECRET_KEY) &&
    isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
    isPresent(process.env.LANGFUSE_BASE_URL)
  );
}

export function isLangfuseCallbackHandler(value: unknown): boolean {
  return (
    value instanceof CallbackHandler ||
    value instanceof LangfuseAgentCallbackHandler
  );
}

export async function disposeLangfuseHandler(value: unknown): Promise<void> {
  if (value instanceof LangfuseAgentCallbackHandler) {
    await value.dispose();
  }
}
