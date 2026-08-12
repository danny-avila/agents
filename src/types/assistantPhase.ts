/**
 * Semantic phase attached to assistant text by Open Responses-compatible
 * providers. Commentary is intermediate agent narration; final_answer marks
 * the user-facing answer that closes the preceding activity phase.
 */
export type AssistantTextPhase = 'commentary' | 'final_answer';
