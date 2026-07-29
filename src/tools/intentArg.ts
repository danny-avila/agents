/**
 * @fileoverview Tool intent labels.
 *
 * Lets a tool declare, as the FIRST property of its input schema, an `intent`
 * string: one model-authored sentence stating what that specific call is about
 * to do ("Searching for OAuth handling in the callback router"). Because the
 * property is first, it is the first key providers stream in the tool-call
 * args, so a host UI can render it as the call's live status label before the
 * rest of the args exist. When the call settles, {@link applyOutcome} edits
 * the sentence in place into its outcome form — a tool-supplied replacement
 * (`outcome`), a tool-supplied span edit (`outcome_patch`), or a mechanical
 * present-progressive→past-tense transform of the leading verb.
 *
 * The arg is always optional (never listed in `required`): the same schemas
 * are callable from programmatic tool calling, where no UI renders a label
 * and forcing generated code to fabricate one would be pure cost. Tool bodies
 * must call {@link stripIntent} before using their args so no tool receives a
 * parameter it did not declare.
 */

import type { JsonSchemaType, OutcomePatch } from '@/types';

/** Argument carrying the model-authored label for a tool call. */
export const INTENT_ARG = 'intent';

/** Model-facing instruction for the injected `intent` property. */
export const INTENT_DESCRIPTION =
  'ALWAYS write this field FIRST, before any other argument. One short sentence, ' +
  'present progressive, stating what this specific call is about to do: ' +
  '"Searching for OAuth handling in the callback router". It is shown to the user ' +
  'as the live status label for this call while it runs, so write it for a human ' +
  'reading a progress line. Do not restate the tool name. Do not exceed one sentence. ' +
  'When you make several calls to the same tool in one turn, each intent must ' +
  'distinguish that call from its siblings.';

/**
 * Canonical (frozen) shape of the injected property. Always embed a COPY
 * (`{ ...INTENT_PROPERTY }`): LangChain's JSON-schema validator stamps a
 * `__absolute_uri__` marker onto every subschema it dereferences, which
 * throws on a frozen object — and a single shared instance would be stamped
 * with one schema's URI while embedded in many.
 */
export const INTENT_PROPERTY: JsonSchemaType = Object.freeze<JsonSchemaType>({
  type: 'string',
  description: INTENT_DESCRIPTION,
});

/**
 * Discriminates the intent LABEL property from a tool's own business
 * parameter that merely shares the name: the label contract always opens
 * with the same instruction. Removal/sanitize passes must never strip a
 * parameter the tool actually needs.
 */
export function isIntentLabelProperty(property: unknown): boolean {
  if (property == null || typeof property !== 'object') {
    return false;
  }
  const record = property as { type?: unknown; description?: unknown };
  return (
    record.type === 'string' &&
    typeof record.description === 'string' &&
    record.description.startsWith('ALWAYS write this field FIRST')
  );
}

/**
 * Returns a copy of the parameters schema with `intent` prepended as the
 * FIRST property (object key order is insertion order and every provider
 * serializer preserves it — first key in the schema means first key in the
 * streamed input). Never mutates the input; no-op when the schema already
 * declares `intent`. The property is not added to `required`.
 */
export function withIntent(parameters?: JsonSchemaType): JsonSchemaType {
  const existingProps = parameters?.properties ?? {};
  if (INTENT_ARG in existingProps) {
    return parameters as JsonSchemaType;
  }
  return {
    ...parameters,
    type: 'object',
    properties: { [INTENT_ARG]: { ...INTENT_PROPERTY }, ...existingProps },
  };
}

/**
 * Coerces tool-call args to an object, parsing a stringified JSON object
 * (some providers deliver args as a string). Returns undefined otherwise.
 */
function coerceArgsObject(args: unknown): Record<string, unknown> | undefined {
  if (typeof args === 'object' && args !== null && !Array.isArray(args)) {
    return args as Record<string, unknown>;
  }
  if (typeof args === 'string' && args.trim().startsWith('{')) {
    try {
      const parsed = JSON.parse(args) as unknown;
      if (parsed != null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return undefined;
    }
  }
  return undefined;
}

/**
 * Reads the model-authored intent from tool-call args (handles stringified
 * args). Returns undefined when absent, empty, or not a string.
 */
export function readIntent(args: unknown): string | undefined {
  const value = coerceArgsObject(args)?.[INTENT_ARG];
  if (typeof value !== 'string') {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed === '' ? undefined : trimmed;
}

/**
 * Returns the args without the `intent` key so downstream consumers that did
 * not declare it never receive it. Parses stringified JSON object args;
 * returns the value unchanged when the key is absent.
 */
export function stripIntent(args: unknown): unknown {
  const obj = coerceArgsObject(args);
  if (!obj || !(INTENT_ARG in obj)) {
    return args;
  }
  const { [INTENT_ARG]: _omit, ...rest } = obj;
  return rest;
}

/**
 * Leading-verb map for the mechanical outcome transform, keyed by the
 * lowercased first word of the intent. Deliberately small: an unknown leading
 * word leaves the intent unchanged rather than mangling it.
 */
const OUTCOME_VERB_MAP: ReadonlyMap<string, string> = new Map([
  ['searching', 'Searched'],
  ['reading', 'Read'],
  ['writing', 'Wrote'],
  ['editing', 'Edited'],
  ['running', 'Ran'],
  ['creating', 'Created'],
  ['checking', 'Checked'],
  ['fetching', 'Fetched'],
  ['listing', 'Listed'],
  ['looking', 'Looked'],
  ['building', 'Built'],
  ['deleting', 'Deleted'],
  ['updating', 'Updated'],
  ['adding', 'Added'],
  ['removing', 'Removed'],
  ['verifying', 'Verified'],
  ['analyzing', 'Analyzed'],
  ['generating', 'Generated'],
  ['delegating', 'Delegated'],
  ['spawning', 'Spawned'],
  ['compiling', 'Compiled'],
  ['grepping', 'Grepped'],
]);

function matchLeadingCase(replacement: string, original: string): string {
  if (original.charAt(0) === original.charAt(0).toLowerCase()) {
    return replacement.charAt(0).toLowerCase() + replacement.slice(1);
  }
  return replacement;
}

function transformLeadingVerb(intent: string): string {
  const spaceIdx = intent.search(/\s/);
  const leading = spaceIdx === -1 ? intent : intent.slice(0, spaceIdx);
  const mapped = OUTCOME_VERB_MAP.get(leading.toLowerCase());
  if (mapped == null) {
    return intent;
  }
  return matchLeadingCase(mapped, leading) + intent.slice(leading.length);
}

/**
 * Resolves the settled label for a call from its model-authored `intent` and
 * the tool's result fields, in precedence order:
 *
 *  1. `outcome` — full replacement authored by the tool.
 *  2. `outcome_patch` — first occurrence of `from` in the intent replaced
 *     with `to` (case-sensitive); no-op when `from` is absent or empty.
 *  3. Mechanical transform — the leading word mapped present-progressive →
 *     past tense; an unknown leading word leaves the intent unchanged.
 *
 * Returns undefined when there is neither an intent nor an outcome, so
 * callers fall back to their default label. Pure and dependency-free — host
 * UIs needing identical logic can import or mirror it.
 */
export function applyOutcome(
  intent: string | undefined,
  result?: { outcome?: string; outcome_patch?: OutcomePatch },
): string | undefined {
  const outcome = result?.outcome;
  if (typeof outcome === 'string' && outcome.trim() !== '') {
    return outcome;
  }
  if (intent == null || intent === '') {
    return undefined;
  }
  const patch = result?.outcome_patch;
  if (patch != null && patch.from !== '' && intent.includes(patch.from)) {
    /** Replacement callback keeps `to` verbatim — a direct string second
     *  argument would interpret `$&`/`$'`-style tokens in tool-authored
     *  text (e.g. labels derived from shell syntax). */
    return intent.replace(patch.from, () => patch.to);
  }
  return transformLeadingVerb(intent);
}

/**
 * Hard cap on an emitted outcome label. The label is a single progress line
 * in UI chrome; a tool that derives it from data (or a malformed patch)
 * must not be able to inflate completion events or persisted parts.
 */
const MAX_OUTCOME_CHARS = 256;

function boundOutcomeLabel(label: string | undefined): string | undefined {
  if (label == null) {
    return undefined;
  }
  const singleLine = label.replace(/\s+/g, ' ').trim();
  if (singleLine === '') {
    return undefined;
  }
  if (singleLine.length <= MAX_OUTCOME_CHARS) {
    return singleLine;
  }
  return `${singleLine.slice(0, MAX_OUTCOME_CHARS - 1)}…`;
}

/**
 * Resolves the settled label to emit on a completion event: only when the
 * tool actually authored `outcome`/`outcome_patch` fields. Returns undefined
 * otherwise — the mechanical transform of a bare intent is left to the host
 * so the wire never carries a label the host can derive itself. The result
 * is collapsed to a bounded single line before emission.
 *
 * For failed calls (`isError`), only tool-AUTHORED text may label the call:
 * an explicit `outcome`, or a patch whose `from` actually matches the
 * intent. An unmatched patch must not fall through to the mechanical
 * past-tense transform — wording drift in a failure patch would otherwise
 * render a success-looking label for an error.
 */
export function resolveToolOutcome(
  args: unknown,
  fields?: { outcome?: string; outcome_patch?: OutcomePatch } | null,
  options?: { isError?: boolean },
): string | undefined {
  if (fields == null || (fields.outcome == null && fields.outcome_patch == null)) {
    return undefined;
  }
  if (options?.isError !== true) {
    return boundOutcomeLabel(applyOutcome(readIntent(args), fields));
  }
  const outcome = fields.outcome;
  if (typeof outcome === 'string' && outcome.trim() !== '') {
    return boundOutcomeLabel(outcome);
  }
  const intent = readIntent(args);
  const patch = fields.outcome_patch;
  if (
    intent != null &&
    patch != null &&
    patch.from !== '' &&
    intent.includes(patch.from)
  ) {
    return boundOutcomeLabel(intent.replace(patch.from, () => patch.to));
  }
  return undefined;
}

/**
 * Reads the outcome fields off a tool-execution result: the typed
 * `outcome`/`outcome_patch` fields when present, else the artifact channel
 * (see {@link readOutcomeFields}) — so a `content_and_artifact` tool authors
 * its label the same way on the direct and event-driven paths.
 */
export function outcomeFieldsFromResult(result: {
  outcome?: string;
  outcome_patch?: OutcomePatch;
  artifact?: unknown;
}): { outcome?: string; outcome_patch?: OutcomePatch } | undefined {
  if (result.outcome != null || result.outcome_patch != null) {
    return result;
  }
  return readOutcomeFields(result.artifact);
}

/**
 * Extracts validated `outcome`/`outcome_patch` fields from an arbitrary
 * value — the artifact channel through which an in-process
 * `content_and_artifact` tool authors its settled label. Returns undefined
 * when neither field is usable.
 */
export function readOutcomeFields(
  source: unknown,
): { outcome?: string; outcome_patch?: OutcomePatch } | undefined {
  if (source == null || typeof source !== 'object' || Array.isArray(source)) {
    return undefined;
  }
  const record = source as Record<string, unknown>;
  const outcome =
    typeof record.outcome === 'string' && record.outcome.trim() !== ''
      ? record.outcome
      : undefined;
  let outcome_patch: OutcomePatch | undefined;
  const rawPatch = record.outcome_patch;
  if (rawPatch != null && typeof rawPatch === 'object' && !Array.isArray(rawPatch)) {
    const patch = rawPatch as Record<string, unknown>;
    if (typeof patch.from === 'string' && typeof patch.to === 'string') {
      outcome_patch = { from: patch.from, to: patch.to };
    }
  }
  if (outcome == null && outcome_patch == null) {
    return undefined;
  }
  return { outcome, outcome_patch };
}
