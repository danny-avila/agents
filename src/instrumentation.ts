import { NodeSDK } from '@opentelemetry/sdk-node';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { isPresent } from '@/utils/misc';

if (
  isPresent(process.env.LANGFUSE_SECRET_KEY) &&
  isPresent(process.env.LANGFUSE_PUBLIC_KEY) &&
  isPresent(process.env.LANGFUSE_BASE_URL ?? process.env.LANGFUSE_BASEURL)
) {
  const langfuseSpanProcessor = new LangfuseSpanProcessor();

  const sdk = new NodeSDK({
    spanProcessors: [langfuseSpanProcessor],
  });

  sdk.start();
}
