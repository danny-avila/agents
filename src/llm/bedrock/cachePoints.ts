// Vendored from @langchain/aws@1.4.2 (utils/message_inputs.ts) because the
// upstream `applyCachePointsToConversePayload` export is internal.
import type {
  BedrockPromptCacheControl,
  ConverseCommandParams,
} from '@langchain/aws';
import type * as Bedrock from '@aws-sdk/client-bedrock-runtime';

function isConverseCachePoint(block: unknown): boolean {
  return Boolean(
    typeof block === 'object' &&
      block !== null &&
      'cachePoint' in block &&
      block.cachePoint &&
      typeof block.cachePoint === 'object' &&
      block.cachePoint !== null &&
      'type' in block.cachePoint
  );
}

function createConverseCachePointBlock(
  cacheControl: BedrockPromptCacheControl,
  isNovaModel: boolean
): { cachePoint: { type: 'default'; ttl?: '1h' } } {
  const ttl =
    !isNovaModel && cacheControl.ttl && cacheControl.ttl !== '5m'
      ? cacheControl.ttl
      : undefined;
  return {
    cachePoint: {
      type: 'default',
      ...(ttl ? { ttl } : {}),
    },
  };
}

export function applyCachePointsToConversePayload(fields: {
  cacheControl?: BedrockPromptCacheControl;
  system: Bedrock.SystemContentBlock[];
  messages: Bedrock.Message[];
  params?: Partial<ConverseCommandParams>;
  modelId: string;
}): void {
  const { cacheControl, system, messages, params, modelId } = fields;
  if (!cacheControl) {
    return;
  }

  const isNovaModel = modelId.toLowerCase().includes('amazon.nova');
  const cacheBlock = createConverseCachePointBlock(cacheControl, isNovaModel);

  if (
    system.length > 0 &&
    !system.some((block) => isConverseCachePoint(block))
  ) {
    system.push(cacheBlock);
  }

  const lastMessage = messages[messages.length - 1];
  const lastContent = lastMessage.content;
  if (Array.isArray(lastContent)) {
    const hasNovaToolBlock =
      isNovaModel &&
      lastContent.some(
        (block) =>
          typeof block === 'object' &&
          block !== null &&
          ('toolResult' in block || 'toolUse' in block)
      );
    if (
      !hasNovaToolBlock &&
      !lastContent.some((block) => isConverseCachePoint(block))
    ) {
      lastContent.push(cacheBlock);
    }
  }

  const tools = params?.toolConfig?.tools;
  if (
    !isNovaModel &&
    Array.isArray(tools) &&
    !tools.some((tool) => isConverseCachePoint(tool))
  ) {
    tools.push(cacheBlock as unknown as Bedrock.Tool);
  }
}
