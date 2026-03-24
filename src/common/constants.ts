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
