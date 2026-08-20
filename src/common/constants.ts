/**
 * Anthropic direct API tool schema overhead multiplier.
 * Empirically calibrated against real MCP tool sets (29 tools).
 * Accounts for Anthropic's internal XML-like tool encoding plus
 * a ~300-token hidden tool-system preamble.
 */
export const ANTHROPIC_TOOL_TOKEN_MULTIPLIER = 2.6;

/**
 * Default tool schema overhead multiplier for all non-Anthropic providers.
 * Covers OpenAI function-calling format, Bedrock, and other providers.
 * Empirically calibrated at ~1.4× the raw JSON token count.
 */
export const DEFAULT_TOOL_TOKEN_MULTIPLIER = 1.4;

/**
 * Default ceiling on cooperative stream seals per run. Each seal costs one
 * extra superstep, so this also sizes the recursion-limit headroom a
 * preemption-enabled run reserves.
 */
export const DEFAULT_MAX_SEALS = 8;

/**
 * Per-hook timeout for `PreemptBoundary`, deliberately far above
 * `DEFAULT_HOOK_TIMEOUT_MS`. A host drain has already popped its queue and
 * persisted content by the time this fires, so a timeout does not cancel the
 * work — it only loses the messages the run was about to inject.
 */
export const PREEMPT_BOUNDARY_HOOK_TIMEOUT_MS = 120_000;

/**
 * Default LangGraph recursion limit for a run. Callers may override it via
 * the stream config; a preemption-enabled run adds its seal budget on top.
 */
export const DEFAULT_RECURSION_LIMIT = 50;

/** Stable runtime names used for agent workflow observations. */
export const STANDARD_GRAPH_RUN_NAME = 'AgentGraph';
export const MULTI_AGENT_GRAPH_RUN_NAME = 'MultiAgentGraph';
export const AGENT_MODEL_CALL_RUN_NAME = 'AgentModelCall';
export const ACTIVITY_LABEL_RUN_NAME = 'StepLabel';
export const REASONING_LABEL_RUN_NAME = 'ReasoningLabel';
export const ACTIVITY_PHASE_RUN_NAME = 'MultiStepLabel';
export const ACTIVITY_PHASE_LABEL_RUN_NAME = 'MultiStepLabelGeneration';

/**
 * Whether this build enforces `LangfuseConfig.privacy` (content redaction
 * and media suppression on exported traces). Hosts gate on it to fail
 * closed: with an older runtime that ignores the field, a privacy mode must
 * disable export rather than silently send unmasked content.
 */
export const LANGFUSE_PRIVACY_ENFORCEMENT_SUPPORTED = true;
