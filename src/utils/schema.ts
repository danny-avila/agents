// src/utils/schema.ts
import { zodToJsonSchema } from 'zod-to-json-schema';
import type { ZodTypeAny } from 'zod';

/** Checks if a schema is a Zod schema by looking for the _def property */
export function isZodSchema(schema: unknown): schema is ZodTypeAny {
  return (
    schema != null && typeof schema === 'object' && '_def' in (schema as object)
  );
}

/**
 * Converts a schema to JSON schema format.
 * Handles both Zod schemas (converts) and JSON schemas (passthrough).
 */
export function toJsonSchema(
  schema: unknown,
  name?: string,
  description?: string
): Record<string, unknown> {
  if (isZodSchema(schema)) {
    const zodSchema = schema as ZodTypeAny & {
      describe: (desc: string) => ZodTypeAny;
    };
    const described =
      description != null && description !== ''
        ? zodSchema.describe(description)
        : schema;
    return zodToJsonSchema(
      described as Parameters<typeof zodToJsonSchema>[0],
      name ?? ''
    );
  }
  return schema as Record<string, unknown>;
}
