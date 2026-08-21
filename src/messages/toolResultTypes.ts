import { isProxy } from 'node:util/types';

export type ProviderToolCallKind =
  | 'tool'
  | 'server'
  | 'anthropic-server'
  | 'mcp'
  | 'google'
  | 'bedrock';

export interface ProviderToolCallPartDescriptor {
  readonly callId: string;
  readonly kind: ProviderToolCallKind;
  readonly name?: string;
  readonly sourceType: string;
}

export interface ProviderToolResultPartDescriptor {
  readonly type: string;
  readonly compatibleCallKinds: readonly ProviderToolCallKind[];
  readonly toolCallId?: string;
  readonly expectedToolNames?: readonly string[];
  readonly requiresPreviousExecutableCode?: boolean;
  readonly allowHumanMessagePairing?: boolean;
}

export interface ProviderToolCallIndexEntry {
  readonly descriptor: ProviderToolCallPartDescriptor;
  secondarySourceType?: string;
}

export type ProviderToolCallIndex = Map<
  string,
  ProviderToolCallIndexEntry | null
>;

type DataProperty =
  | { readonly found: true; readonly value: unknown }
  | { readonly found: false };

interface ValidationBudget {
  remaining: number;
}

export const PROVIDER_TOOL_RESULT_MAX_ARRAY_ENTRIES = 256;
export const PROVIDER_TOOL_RESULT_MAX_VALIDATION_ENTRIES = 4096;
export const PROVIDER_TOOL_PAIRING_MAX_IDENTIFIER_CHARS = 512;
export const PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS = 4096;

const TOOL_CALL_KINDS = ['tool'] as const;
const SERVER_CALL_KINDS = ['server'] as const;
const ANTHROPIC_SERVER_CALL_KINDS = ['anthropic-server'] as const;
const MCP_CALL_KINDS = ['mcp'] as const;
const GOOGLE_CALL_KINDS = ['google'] as const;
const BEDROCK_CALL_KINDS = ['bedrock', 'tool'] as const;

const ADVISOR_TOOL_NAMES = ['advisor'] as const;
const BASH_TOOL_NAMES = ['bash_code_execution'] as const;
const CODE_EXECUTION_TOOL_NAMES = ['code_execution'] as const;
const TEXT_EDITOR_TOOL_NAMES = ['text_editor_code_execution'] as const;
const TOOL_SEARCH_TOOL_NAMES = [
  'tool_search_tool_regex',
  'tool_search_tool_bm25',
] as const;
const WEB_FETCH_TOOL_NAMES = ['web_fetch'] as const;
const WEB_SEARCH_TOOL_NAMES = ['web_search'] as const;
const ANTHROPIC_SERVER_TOOL_ID_PREFIX = 'srvtoolu_';

function stringSet(values: readonly string[]): ReadonlySet<string> {
  return new Set(values);
}

const GOOGLE_CODE_EXECUTION_OUTCOMES = stringSet([
  'OUTCOME_DEADLINE_EXCEEDED',
  'OUTCOME_FAILED',
  'OUTCOME_OK',
  'OUTCOME_UNSPECIFIED',
  'outcome_deadline_exceeded',
  'outcome_failed',
  'outcome_ok',
  'outcome_unspecified',
]);
const PROVIDER_TOOL_CALL_KINDS = stringSet([
  'anthropic-server',
  'bedrock',
  'google',
  'mcp',
  'server',
  'tool',
]);
const ANTHROPIC_SERVER_TOOL_NAMES = stringSet([
  ...ADVISOR_TOOL_NAMES,
  ...BASH_TOOL_NAMES,
  ...CODE_EXECUTION_TOOL_NAMES,
  ...TEXT_EDITOR_TOOL_NAMES,
  ...TOOL_SEARCH_TOOL_NAMES,
  ...WEB_FETCH_TOOL_NAMES,
  ...WEB_SEARCH_TOOL_NAMES,
]);

function getRecord(value: unknown): Record<string, unknown> | undefined {
  if (value == null || typeof value !== 'object') {
    return undefined;
  }
  try {
    if (isProxy(value) || Array.isArray(value)) {
      return undefined;
    }
    const prototype = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null
      ? (value as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

function getPropertyContainer(
  value: unknown
): Record<string, unknown> | undefined {
  if (value == null || typeof value !== 'object') {
    return undefined;
  }
  try {
    return !isProxy(value) && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

function getArray(value: unknown): readonly unknown[] | undefined {
  if (value == null || typeof value !== 'object') {
    return undefined;
  }
  try {
    return !isProxy(value) && Array.isArray(value)
      ? (value as readonly unknown[])
      : undefined;
  } catch {
    return undefined;
  }
}

function readDataProperty(
  record: Record<string, unknown>,
  key: string
): DataProperty {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(record, key);
    if (
      descriptor == null ||
      descriptor.enumerable !== true ||
      !('value' in descriptor)
    ) {
      return { found: false };
    }
    return { found: true, value: descriptor.value };
  } catch {
    return { found: false };
  }
}

function hasOnlyOwnDataProperties(
  record: Record<string, unknown>,
  allowed: readonly string[]
): boolean {
  try {
    const keys = Reflect.ownKeys(record);
    if (keys.length > allowed.length) {
      return false;
    }
    for (let index = 0; index < keys.length; index++) {
      const key = keys[index];
      if (typeof key !== 'string' || !allowed.includes(key)) {
        return false;
      }
      const descriptor = Object.getOwnPropertyDescriptor(record, key);
      if (
        descriptor == null ||
        descriptor.enumerable !== true ||
        !('value' in descriptor)
      ) {
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

function getBoundedDataArray(
  value: unknown,
  maxEntries: number,
  budget?: ValidationBudget
): readonly unknown[] | undefined {
  const array = getArray(value);
  if (
    array == null ||
    array.length > maxEntries ||
    (budget != null && array.length > budget.remaining)
  ) {
    return undefined;
  }
  if (budget != null) {
    budget.remaining -= array.length;
  }
  for (let index = 0; index < array.length; index++) {
    let descriptor: PropertyDescriptor | undefined;
    try {
      descriptor = Object.getOwnPropertyDescriptor(array, String(index));
    } catch {
      return undefined;
    }
    if (
      descriptor == null ||
      descriptor.enumerable !== true ||
      !('value' in descriptor)
    ) {
      return undefined;
    }
  }
  return array;
}

export function getBoundedProviderPairingArray(
  value: unknown
): readonly unknown[] | undefined {
  return getBoundedDataArray(value, PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS);
}

export function getBoundedProviderPairingArrayProperty(
  value: unknown,
  key: string
): readonly unknown[] | undefined {
  const record = getPropertyContainer(value);
  if (record == null) {
    return undefined;
  }
  const property = readDataProperty(record, key);
  return property.found
    ? getBoundedProviderPairingArray(property.value)
    : undefined;
}

function readString(
  record: Record<string, unknown>,
  key: string,
  allowEmpty = false
): string | undefined {
  const property = readDataProperty(record, key);
  return property.found &&
    typeof property.value === 'string' &&
    (allowEmpty || property.value.length > 0)
    ? property.value
    : undefined;
}

export function isBoundedProviderPairingString(
  value: unknown,
  allowEmpty = false
): value is string {
  return (
    typeof value === 'string' &&
    (allowEmpty || value.length > 0) &&
    value.length <= PROVIDER_TOOL_PAIRING_MAX_IDENTIFIER_CHARS
  );
}

function readPairingString(
  record: Record<string, unknown>,
  key: string,
  allowEmpty = false
): string | undefined {
  const property = readDataProperty(record, key);
  return property.found &&
    isBoundedProviderPairingString(property.value, allowEmpty)
    ? property.value
    : undefined;
}

function hasRequiredDataProperties(
  record: Record<string, unknown>,
  keys: readonly string[] | undefined
): boolean {
  if (keys == null) {
    return true;
  }
  for (let index = 0; index < keys.length; index++) {
    const property = readDataProperty(record, keys[index]);
    if (!property.found || property.value === undefined) {
      return false;
    }
  }
  return true;
}

function hasOptionalNullableString(
  record: Record<string, unknown>,
  key: string
): boolean {
  const property = readDataProperty(record, key);
  return (
    !property.found ||
    property.value === null ||
    typeof property.value === 'string'
  );
}

function isCacheControl(value: unknown): boolean {
  if (value == null) {
    return true;
  }
  const record = getRecord(value);
  if (
    record == null ||
    !hasOnlyOwnDataProperties(record, ['type', 'ttl']) ||
    readString(record, 'type') !== 'ephemeral'
  ) {
    return false;
  }
  const ttl = readDataProperty(record, 'ttl');
  return (
    !ttl.found ||
    ttl.value === null ||
    ttl.value === '5m' ||
    ttl.value === '1h'
  );
}

function hasValidCacheControlProperty(
  record: Record<string, unknown>
): boolean {
  const property = readDataProperty(record, 'cache_control');
  return !property.found || isCacheControl(property.value);
}

function isServerToolCaller(value: unknown): boolean {
  const record = getRecord(value);
  if (record == null) {
    return false;
  }
  const type = readString(record, 'type');
  if (type === 'direct') {
    return hasOnlyOwnDataProperties(record, ['type']);
  }
  return (
    (type === 'code_execution_20250825' ||
      type === 'code_execution_20260120') &&
    hasOnlyOwnDataProperties(record, ['type', 'tool_id']) &&
    readPairingString(record, 'tool_id') != null
  );
}

function hasValidCallerProperty(record: Record<string, unknown>): boolean {
  const property = readDataProperty(record, 'caller');
  return !property.found || isServerToolCaller(property.value);
}

function hasValidProviderMetadata(
  record: Record<string, unknown>
): boolean {
  const id = readDataProperty(record, 'id');
  if (id.found && !isBoundedProviderPairingString(id.value)) {
    return false;
  }
  const agentId = readDataProperty(record, 'agentId');
  if (agentId.found && !isBoundedProviderPairingString(agentId.value)) {
    return false;
  }
  const index = readDataProperty(record, 'index');
  if (
    index.found &&
    !(
      (typeof index.value === 'number' && Number.isSafeInteger(index.value)) ||
      isBoundedProviderPairingString(index.value)
    )
  ) {
    return false;
  }
  const groupId = readDataProperty(record, 'groupId');
  if (
    groupId.found &&
    !(typeof groupId.value === 'number' && Number.isFinite(groupId.value))
  ) {
    return false;
  }
  const thought = readDataProperty(record, 'thought');
  if (thought.found && typeof thought.value !== 'boolean') {
    return false;
  }
  const signature = readDataProperty(record, 'thoughtSignature');
  return (
    !signature.found ||
    isBoundedProviderPairingString(signature.value, true)
  );
}

const ANTHROPIC_STRUCTURED_RESULT_TOOL_NAMES: Readonly<
  Partial<Record<string, readonly string[]>>
> = {
  advisor_tool_result: ADVISOR_TOOL_NAMES,
  bash_code_execution_tool_result: BASH_TOOL_NAMES,
  code_execution_tool_result: CODE_EXECUTION_TOOL_NAMES,
  text_editor_code_execution_tool_result: TEXT_EDITOR_TOOL_NAMES,
  tool_search_tool_result: TOOL_SEARCH_TOOL_NAMES,
  web_fetch_tool_result: WEB_FETCH_TOOL_NAMES,
};

/** Designated provider output is opaque once its exact envelope and matching
 * call establish authorship. Only its container size/proxy identity is read;
 * nested values are deliberately not walked on the formatting hot path. */
function isOpaqueBoundedArray(value: unknown): boolean {
  const array = getArray(value);
  return (
    array != null && array.length <= PROVIDER_TOOL_RESULT_MAX_ARRAY_ENTRIES
  );
}

function isOpaqueStringOrBoundedArray(value: unknown): boolean {
  return typeof value === 'string' || isOpaqueBoundedArray(value);
}

function isWebSearchResult(value: unknown): boolean {
  const record = getRecord(value);
  return (
    record != null &&
    hasOnlyOwnDataProperties(record, [
      'type',
      'encrypted_content',
      'title',
      'url',
      'page_age',
    ]) &&
    readString(record, 'type') === 'web_search_result' &&
    readString(record, 'encrypted_content', true) != null &&
    readString(record, 'title', true) != null &&
    readString(record, 'url') != null &&
    hasOptionalNullableString(record, 'page_age')
  );
}

function isWireWebSearchContent(
  value: unknown,
  budget: ValidationBudget
): boolean {
  const error = getRecord(value);
  if (error != null) {
    return (
      hasOnlyOwnDataProperties(error, ['type', 'error_code']) &&
      readString(error, 'type') === 'web_search_tool_result_error' &&
      readString(error, 'error_code') != null
    );
  }
  const results = getBoundedDataArray(
    value,
    PROVIDER_TOOL_RESULT_MAX_ARRAY_ENTRIES,
    budget
  );
  if (results == null) {
    return false;
  }
  for (let index = 0; index < results.length; index++) {
    if (!isWebSearchResult(results[index])) {
      return false;
    }
  }
  return true;
}

export function hasStructurallyValidAnthropicWebSearchResultContent(
  part: unknown
): boolean {
  const record = getRecord(part);
  if (record == null || readPairingString(record, 'tool_use_id') == null) {
    return false;
  }
  const content = readDataProperty(record, 'content');
  return (
    content.found &&
    isWireWebSearchContent(content.value, {
      remaining: PROVIDER_TOOL_RESULT_MAX_VALIDATION_ENTRIES,
    })
  );
}

function isGoogleCodeExecutionResult(
  record: Record<string, unknown>
): boolean {
  const nested = readDataProperty(record, 'codeExecutionResult');
  const result = nested.found ? getRecord(nested.value) : undefined;
  const outcome = result == null ? undefined : readString(result, 'outcome');
  return (
    result != null &&
    hasOnlyOwnDataProperties(result, ['outcome', 'output']) &&
    outcome != null &&
    GOOGLE_CODE_EXECUTION_OUTCOMES.has(outcome) &&
    readString(result, 'output', true) != null
  );
}

function getGoogleToolResponseDescriptor(
  record: Record<string, unknown>,
  type: string
): ProviderToolResultPartDescriptor | undefined {
  const nested = readDataProperty(record, 'toolResponse');
  const response = nested.found ? getRecord(nested.value) : undefined;
  if (response == null) {
    return undefined;
  }
  const toolCallId = readPairingString(response, 'id');
  const responsePayload = readDataProperty(response, 'response');
  const resultPayload = readDataProperty(response, 'result');
  if (toolCallId == null) {
    return undefined;
  }
  if (
    responsePayload.found &&
    responsePayload.value !== undefined &&
    !resultPayload.found &&
    hasOnlyOwnDataProperties(response, ['id', 'name', 'response'])
  ) {
    const name = readPairingString(response, 'name');
    return name == null
      ? undefined
      : {
        type,
        toolCallId,
        compatibleCallKinds: GOOGLE_CALL_KINDS,
        expectedToolNames: [name],
      };
  }
  if (
    resultPayload.found &&
    resultPayload.value !== undefined &&
    !responsePayload.found &&
    hasOnlyOwnDataProperties(response, ['id', 'toolType', 'result']) &&
    readPairingString(response, 'toolType') != null
  ) {
    return { type, toolCallId, compatibleCallKinds: GOOGLE_CALL_KINDS };
  }
  return undefined;
}

function getBedrockDescriptor(
  record: Record<string, unknown>,
  type: string
): ProviderToolResultPartDescriptor | undefined {
  const nested = readDataProperty(record, 'toolResult');
  const result = nested.found ? getRecord(nested.value) : undefined;
  if (
    result == null ||
    !hasOnlyOwnDataProperties(result, [
      'toolUseId',
      'content',
      'status',
      'type',
    ])
  ) {
    return undefined;
  }
  const toolCallId = readPairingString(result, 'toolUseId');
  const content = readDataProperty(result, 'content');
  const status = readDataProperty(result, 'status');
  const nestedType = readDataProperty(result, 'type');
  if (
    toolCallId == null ||
    !content.found ||
    !isOpaqueBoundedArray(content.value) ||
    (status.found && status.value !== 'success' && status.value !== 'error') ||
    (nestedType.found && typeof nestedType.value !== 'string')
  ) {
    return undefined;
  }
  return {
    type,
    toolCallId,
    compatibleCallKinds: BEDROCK_CALL_KINDS,
    allowHumanMessagePairing: true,
  };
}

const ANTHROPIC_RESULT_FIELDS = [
  'type',
  'tool_use_id',
  'content',
  'cache_control',
] as const;
const ANTHROPIC_GENERIC_RESULT_FIELDS = [
  ...ANTHROPIC_RESULT_FIELDS,
  'is_error',
] as const;
const ANTHROPIC_SERVER_RESULT_FIELDS = [
  ...ANTHROPIC_RESULT_FIELDS,
  'caller',
] as const;
const LOCAL_RESULT_METADATA_FIELDS = [
  'id',
  'index',
  'agentId',
  'groupId',
] as const;
const TOOL_RESULT_ENVELOPE_FIELDS: Readonly<
  Record<string, readonly string[]>
> = {
  advisor_tool_result: ANTHROPIC_RESULT_FIELDS,
  bash_code_execution_tool_result: ANTHROPIC_RESULT_FIELDS,
  code_execution_tool_result: ANTHROPIC_RESULT_FIELDS,
  mcp_tool_result: ANTHROPIC_GENERIC_RESULT_FIELDS,
  text_editor_code_execution_tool_result: ANTHROPIC_RESULT_FIELDS,
  tool_search_tool_result: ANTHROPIC_RESULT_FIELDS,
  tool_result: ANTHROPIC_GENERIC_RESULT_FIELDS,
  web_fetch_tool_result: ANTHROPIC_SERVER_RESULT_FIELDS,
  web_search_tool_result: ANTHROPIC_SERVER_RESULT_FIELDS,
  server_tool_call_result: [
    'type',
    'toolCallId',
    'status',
    'output',
    ...LOCAL_RESULT_METADATA_FIELDS,
  ],
  server_tool_result: [
    'type',
    'tool_call_id',
    'status',
    'output',
    ...LOCAL_RESULT_METADATA_FIELDS,
  ],
  codeExecutionResult: [
    'type',
    'codeExecutionResult',
    'agentId',
    'groupId',
  ],
  toolResponse: [
    'type',
    'toolResponse',
    'thought',
    'thoughtSignature',
    'agentId',
    'groupId',
  ],
  toolResult: ['type', 'toolResult', 'agentId', 'groupId'],
};

function anthropicDescriptor(
  record: Record<string, unknown>,
  type: string,
  expectedToolNames: readonly string[]
): ProviderToolResultPartDescriptor | undefined {
  const toolCallId = readPairingString(record, 'tool_use_id');
  const content = readDataProperty(record, 'content');
  return toolCallId != null &&
    content.found &&
    getRecord(content.value) != null
    ? {
      type,
      toolCallId,
      compatibleCallKinds: ANTHROPIC_SERVER_CALL_KINDS,
      expectedToolNames,
    }
    : undefined;
}

export function getProviderToolResultPartDescriptor(
  part: unknown
): ProviderToolResultPartDescriptor | undefined {
  const record = getRecord(part);
  if (record == null) {
    return undefined;
  }
  const type = readString(record, 'type');
  const allowedFields = type == null ? undefined : TOOL_RESULT_ENVELOPE_FIELDS[type];
  if (
    type == null ||
    allowedFields == null ||
    !hasOnlyOwnDataProperties(record, allowedFields) ||
    !hasValidProviderMetadata(record) ||
    !hasValidCacheControlProperty(record) ||
    !hasValidCallerProperty(record)
  ) {
    return undefined;
  }
  const structuredToolNames = ANTHROPIC_STRUCTURED_RESULT_TOOL_NAMES[type];
  if (structuredToolNames != null) {
    return anthropicDescriptor(
      record,
      type,
      structuredToolNames
    );
  }
  if (type === 'web_search_tool_result') {
    const toolCallId = readPairingString(record, 'tool_use_id');
    const content = readDataProperty(record, 'content');
    return toolCallId != null &&
      content.found &&
      (getRecord(content.value) != null ||
        isOpaqueBoundedArray(content.value))
      ? {
        type,
        toolCallId,
        compatibleCallKinds: ANTHROPIC_SERVER_CALL_KINDS,
        expectedToolNames: WEB_SEARCH_TOOL_NAMES,
      }
      : undefined;
  }
  if (type === 'mcp_tool_result') {
    const toolCallId = readPairingString(record, 'tool_use_id');
    const content = readDataProperty(record, 'content');
    const isError = readDataProperty(record, 'is_error');
    return toolCallId != null &&
      content.found &&
      isOpaqueStringOrBoundedArray(content.value) &&
      isError.found &&
      typeof isError.value === 'boolean'
      ? { type, toolCallId, compatibleCallKinds: MCP_CALL_KINDS }
      : undefined;
  }
  if (type === 'server_tool_call_result' || type === 'server_tool_result') {
    const toolCallId = readPairingString(
      record,
      type === 'server_tool_call_result' ? 'toolCallId' : 'tool_call_id'
    );
    const status = readString(record, 'status');
    const output = readDataProperty(record, 'output');
    return toolCallId != null &&
      (status === 'success' || status === 'error') &&
      output.found &&
      output.value !== undefined
      ? { type, toolCallId, compatibleCallKinds: SERVER_CALL_KINDS }
      : undefined;
  }
  if (type === 'codeExecutionResult') {
    return isGoogleCodeExecutionResult(record)
      ? {
        type,
        compatibleCallKinds: GOOGLE_CALL_KINDS,
        requiresPreviousExecutableCode: true,
      }
      : undefined;
  }
  if (type === 'toolResponse') {
    return getGoogleToolResponseDescriptor(record, type);
  }
  if (type === 'toolResult') {
    return getBedrockDescriptor(record, type);
  }
  if (type === 'tool_result') {
    const toolCallId = readPairingString(record, 'tool_use_id');
    const content = readDataProperty(record, 'content');
    const isError = readDataProperty(record, 'is_error');
    return toolCallId != null &&
      content.found &&
      isOpaqueStringOrBoundedArray(content.value) &&
      (!isError.found || typeof isError.value === 'boolean')
      ? {
        type,
        toolCallId,
        compatibleCallKinds: TOOL_CALL_KINDS,
        allowHumanMessagePairing: true,
      }
      : undefined;
  }
  return undefined;
}

function optionalName(
  record: Record<string, unknown>
): { readonly name?: string } | undefined {
  const property = readDataProperty(record, 'name');
  if (!property.found || property.value === undefined) {
    return {};
  }
  return isBoundedProviderPairingString(property.value)
    ? { name: property.value }
    : undefined;
}

function hasExactCallShape(
  record: Record<string, unknown>,
  fields: readonly string[],
  required: readonly string[]
): boolean {
  return (
    hasOnlyOwnDataProperties(record, fields) &&
    hasRequiredDataProperties(record, required) &&
    hasValidProviderMetadata(record)
  );
}

function hasOptionalCallType(
  record: Record<string, unknown>,
  expected: string
): boolean {
  const type = readDataProperty(record, 'type');
  return !type.found || type.value === expected;
}

function isAnthropicServerToolCall(
  callId: string,
  name: string | undefined
): boolean {
  return (
    callId.startsWith(ANTHROPIC_SERVER_TOOL_ID_PREFIX) &&
    name != null &&
    ANTHROPIC_SERVER_TOOL_NAMES.has(name)
  );
}

function getAnthropicCallKind(
  type: 'tool_use' | 'server_tool_use' | 'mcp_tool_use',
  callId: string,
  name: string | undefined
): ProviderToolCallKind {
  if (type === 'mcp_tool_use') {
    return 'mcp';
  }
  if (
    type === 'server_tool_use' ||
    isAnthropicServerToolCall(callId, name)
  ) {
    return 'anthropic-server';
  }
  return 'tool';
}

const AI_TOOL_CALL_FIELDS = ['type', 'id', 'name', 'args'] as const;
const LOCAL_CALL_METADATA_FIELDS = ['agentId', 'groupId'] as const;
const STANDARD_CALL_FIELDS = [
  'type',
  'id',
  'name',
  'args',
  'index',
  ...LOCAL_CALL_METADATA_FIELDS,
] as const;
const LIBRECHAT_CALL_FIELDS = [
  'type',
  'id',
  'name',
  'args',
  'output',
  'outcome',
  'auth',
  'expires_at',
] as const;
const ANTHROPIC_CALL_FIELDS = [
  'type',
  'id',
  'name',
  'input',
  'caller',
  'cache_control',
  'index',
  ...LOCAL_CALL_METADATA_FIELDS,
] as const;
const ANTHROPIC_MCP_CALL_FIELDS = [
  'type',
  'id',
  'name',
  'input',
  'server_name',
  'cache_control',
  'index',
  ...LOCAL_CALL_METADATA_FIELDS,
] as const;
const GOOGLE_CALL_FIELDS = [
  'type',
  'toolCall',
  'thought',
  'thoughtSignature',
  ...LOCAL_CALL_METADATA_FIELDS,
] as const;

export function getProviderAIMessageToolCallDescriptor(
  toolCall: unknown
): ProviderToolCallPartDescriptor | undefined {
  const record = getRecord(toolCall);
  if (
    record == null ||
    !hasExactCallShape(
      record,
      AI_TOOL_CALL_FIELDS,
      ['id', 'name', 'args']
    ) ||
    !hasOptionalCallType(record, 'tool_call')
  ) {
    return undefined;
  }
  const callId = readPairingString(record, 'id');
  const name = optionalName(record);
  return callId == null || name == null
    ? undefined
    : {
      callId,
      kind: isAnthropicServerToolCall(callId, name.name)
        ? 'anthropic-server'
        : 'tool',
      sourceType: 'ai_tool_calls',
      ...name,
    };
}

export function getProviderToolCallPartDescriptor(
  part: unknown
): ProviderToolCallPartDescriptor | undefined {
  const record = getRecord(part);
  if (record == null) {
    return undefined;
  }
  const type = readString(record, 'type');
  if (type === 'tool_call') {
    const nested = readDataProperty(record, 'tool_call');
    const call = nested.found ? getRecord(nested.value) : record;
    const outerIsValid = nested.found
      ? hasExactCallShape(
        record,
        ['type', 'tool_call', 'agentId', 'groupId'],
        ['tool_call']
      )
      : hasExactCallShape(
        record,
        STANDARD_CALL_FIELDS,
        ['id', 'name', 'args']
      );
    const callIsValid = nested.found
      ? call != null &&
        hasExactCallShape(
          call,
          LIBRECHAT_CALL_FIELDS,
          ['id', 'name', 'args']
        ) &&
        hasOptionalCallType(call, 'tool_call')
      : call != null;
    if (!outerIsValid || !callIsValid) {
      return undefined;
    }
    const callId = call == null ? undefined : readPairingString(call, 'id');
    const name = call == null ? undefined : optionalName(call);
    return callId == null || name == null
      ? undefined
      : {
        callId,
        kind: isAnthropicServerToolCall(callId, name.name)
          ? 'anthropic-server'
          : 'tool',
        sourceType: type,
        ...name,
      };
  }
  if (
    type === 'tool_use' ||
    type === 'server_tool_use' ||
    type === 'mcp_tool_use'
  ) {
    const fields =
      type === 'mcp_tool_use'
        ? ANTHROPIC_MCP_CALL_FIELDS
        : ANTHROPIC_CALL_FIELDS;
    const required =
      type === 'mcp_tool_use'
        ? ['id', 'name', 'input', 'server_name']
        : ['id', 'name', 'input'];
    if (
      !hasExactCallShape(record, fields, required) ||
      !hasValidCacheControlProperty(record) ||
      !hasValidCallerProperty(record) ||
      (type === 'mcp_tool_use' &&
        readPairingString(record, 'server_name') == null)
    ) {
      return undefined;
    }
    const callId = readPairingString(record, 'id');
    const name = optionalName(record);
    if (
      type === 'server_tool_use' &&
      (name?.name == null || !ANTHROPIC_SERVER_TOOL_NAMES.has(name.name))
    ) {
      return undefined;
    }
    return callId == null || name == null
      ? undefined
      : {
        callId,
        kind: getAnthropicCallKind(type, callId, name.name),
        sourceType: type,
        ...name,
      };
  }
  if (type === 'server_tool_call') {
    if (
      !hasExactCallShape(
        record,
        STANDARD_CALL_FIELDS,
        ['id', 'name', 'args']
      )
    ) {
      return undefined;
    }
    const callId = readPairingString(record, 'id');
    const name = optionalName(record);
    return callId == null || name == null
      ? undefined
      : {
        callId,
        kind: isAnthropicServerToolCall(callId, name.name)
          ? 'anthropic-server'
          : 'server',
        sourceType: type,
        ...name,
      };
  }
  if (type === 'toolCall' || type === 'toolUse') {
    const outerFields =
      type === 'toolCall'
        ? GOOGLE_CALL_FIELDS
        : ['type', 'toolUse', 'agentId', 'groupId'];
    if (
      !hasExactCallShape(record, outerFields, [type]) ||
      !hasValidProviderMetadata(record)
    ) {
      return undefined;
    }
    const nested = readDataProperty(record, type);
    const call = nested.found ? getRecord(nested.value) : undefined;
    if (call == null) {
      return undefined;
    }
    if (type === 'toolCall') {
      const isNamedVariant = hasExactCallShape(
        call,
        ['id', 'name', 'args'],
        ['id', 'name', 'args']
      );
      const isTypedVariant = hasExactCallShape(
        call,
        ['id', 'toolType', 'args'],
        ['id', 'toolType', 'args']
      );
      if (
        (!isNamedVariant && !isTypedVariant) ||
        (isTypedVariant && readPairingString(call, 'toolType') == null)
      ) {
        return undefined;
      }
    } else {
      const nestedType = readDataProperty(call, 'type');
      if (
        !hasExactCallShape(
          call,
          ['toolUseId', 'name', 'input', 'type'],
          ['toolUseId', 'name', 'input']
        ) ||
        (nestedType.found && typeof nestedType.value !== 'string')
      ) {
        return undefined;
      }
    }
    const callId = readPairingString(
      call,
      type === 'toolCall' ? 'id' : 'toolUseId'
    );
    const name = optionalName(call);
    return callId == null || name == null
      ? undefined
      : {
        callId,
        kind: type === 'toolCall' ? 'google' : 'bedrock',
        sourceType: type,
        ...name,
      };
  }
  return undefined;
}

export function appendProviderToolCallDescriptor(
  index: ProviderToolCallIndex,
  descriptor: ProviderToolCallPartDescriptor
): void {
  if (
    !PROVIDER_TOOL_CALL_KINDS.has(descriptor.kind) ||
    !isBoundedProviderPairingString(descriptor.callId) ||
    !isBoundedProviderPairingString(descriptor.sourceType) ||
    (descriptor.name != null &&
      !isBoundedProviderPairingString(descriptor.name))
  ) {
    return;
  }
  const existing = index.get(descriptor.callId);
  if (existing === undefined) {
    if (index.size >= PROVIDER_TOOL_PAIRING_MAX_OUTER_PARTS) {
      return;
    }
    index.set(descriptor.callId, { descriptor });
    return;
  }
  if (existing === null) {
    return;
  }
  const isDualRepresentation =
    existing.secondarySourceType == null &&
    existing.descriptor.sourceType !== descriptor.sourceType &&
    (existing.descriptor.sourceType === 'ai_tool_calls' ||
      descriptor.sourceType === 'ai_tool_calls');
  if (
    existing.descriptor.kind === descriptor.kind &&
    existing.descriptor.name === descriptor.name &&
    isDualRepresentation
  ) {
    existing.secondarySourceType = descriptor.sourceType;
    return;
  }
  index.set(descriptor.callId, null);
}

function isExecutableCodePart(part: unknown): boolean {
  const record = getRecord(part);
  if (
    record == null ||
    readString(record, 'type') !== 'executableCode' ||
    !hasOnlyOwnDataProperties(record, [
      'type',
      'executableCode',
      'agentId',
      'groupId',
    ]) ||
    !hasValidProviderMetadata(record)
  ) {
    return false;
  }
  const nested = readDataProperty(record, 'executableCode');
  const executable = nested.found ? getRecord(nested.value) : undefined;
  return (
    executable != null &&
    hasOnlyOwnDataProperties(executable, ['language', 'code']) &&
    readString(executable, 'language') != null &&
    readString(executable, 'code', true) != null
  );
}

export function consumeProviderToolResultPair(
  descriptor: ProviderToolResultPartDescriptor,
  calls: ProviderToolCallIndex,
  previousPart?: unknown
): boolean {
  if (descriptor.requiresPreviousExecutableCode === true) {
    return isExecutableCodePart(previousPart);
  }
  if (
    descriptor.toolCallId == null ||
    !isBoundedProviderPairingString(descriptor.toolCallId)
  ) {
    return false;
  }
  const entry = calls.get(descriptor.toolCallId);
  if (entry == null) {
    return false;
  }
  const candidate = entry.descriptor;
  if (!descriptor.compatibleCallKinds.includes(candidate.kind)) {
    return false;
  }
  if (
    descriptor.expectedToolNames != null &&
    (candidate.name == null ||
      !descriptor.expectedToolNames.includes(candidate.name))
  ) {
    return false;
  }
  calls.delete(descriptor.toolCallId);
  return true;
}
