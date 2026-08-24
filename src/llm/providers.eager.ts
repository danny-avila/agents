import * as openai from '@/llm/openai';
import * as google from '@/llm/google';
import * as bedrock from '@/llm/bedrock';
import * as mistral from '@/llm/mistral';
import * as vertexai from '@/llm/vertexai';
import * as anthropic from '@/llm/anthropic';
import * as openrouter from '@/llm/openrouter';
import { registerSourceModeModules } from '@/lazyRequire';

/**
 * Source-mode counterpart of the lazy provider loading the built package uses: commands
 * that run the TypeScript directly import this module first, so every lazily loadable
 * provider module is loaded through the active ESM loader — one module graph, one set of
 * class identities — and registered with the lazy-require seam.
 */
registerSourceModeModules({
  'llm/openai/index': openai,
  'llm/google/index': google,
  'llm/bedrock/index': bedrock,
  'llm/mistral/index': mistral,
  'llm/vertexai/index': vertexai,
  'llm/anthropic/index': anthropic,
  'llm/openrouter/index': openrouter,
});
