import {
  CodeExecutionTool,
  FunctionDeclarationsTool as GoogleGenerativeAIFunctionDeclarationsTool,
  GoogleSearchRetrievalTool,
} from '@google/generative-ai';
import { BindToolsInput } from '@langchain/core/language_models/chat_models';

/** New GoogleSearch tool for Gemini 2.0+ models */
export interface GoogleSearchTool {
  googleSearch: Record<string, never>;
}

export type GoogleGenerativeAIToolType =
  | BindToolsInput
  | GoogleGenerativeAIFunctionDeclarationsTool
  | CodeExecutionTool
  | GoogleSearchRetrievalTool
  | GoogleSearchTool;

/** Enum for content modality types */
enum Modality {
  MODALITY_UNSPECIFIED = 'MODALITY_UNSPECIFIED',
  TEXT = 'TEXT',
  IMAGE = 'IMAGE',
  VIDEO = 'VIDEO',
  AUDIO = 'AUDIO',
  DOCUMENT = 'DOCUMENT',
}

/** Interface for modality token count */
interface ModalityTokenCount {
  modality: Modality;
  tokenCount: number;
}

/** Interface for input token details with cache and tier tracking */
export interface InputTokenDetails {
  cache_read?: number;
  over_200k?: number;
  cache_read_over_200k?: number;
}

/** Main interface for Gemini API usage metadata */
export interface GeminiApiUsageMetadata {
  promptTokenCount?: number;
  totalTokenCount?: number;
  thoughtsTokenCount?: number;
  candidatesTokenCount?: number;
  toolUsePromptTokenCount?: number;
  cachedContentTokenCount?: number;
  promptTokensDetails: ModalityTokenCount[];
  candidatesTokensDetails?: ModalityTokenCount[];
  cacheTokensDetails?: ModalityTokenCount[];
  toolUsePromptTokensDetails?: ModalityTokenCount[];
  trafficType?: string;
}
