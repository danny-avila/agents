import { NodeSDK } from '@opentelemetry/sdk-node';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { isPresent } from '@/utils/misc';

if (
  isPresent(process.env.LANGFUSE_SECRET_KEY) &&
  isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
  isPresent(process.env.LANGFUSE_BASE_URL)
) {
  const langfuseSpanProcessor = new LangfuseSpanProcessor({
    publicKey: process.env.LANGFUSE_PUBLIC_KEY,
    secretKey: process.env.LANGFUSE_SECRET_KEY,
    baseUrl: process.env.LANGFUSE_BASE_URL,
    environment: process.env.LANGFUSE_TRACING_ENVIRONMENT ?? process.env.NODE_ENV ?? 'development',
  });

  const sdk = new NodeSDK({
    spanProcessors: [langfuseSpanProcessor],
  });

  sdk.start();
}
