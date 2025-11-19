// src/types/tools.ts
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableToolLike } from '@langchain/core/runnables';
import type { ToolCall } from '@langchain/core/messages/tool';
import type { ToolErrorData } from './stream';
import { EnvVar } from '@/common';

/** Replacement type for `import type { ToolCall } from '@langchain/core/messages/tool'` in order to have stringified args typed */
export type CustomToolCall = {
  name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  args: string | Record<string, any>;
  id?: string;
  type?: 'tool_call';
  output?: string;
};

export type GenericTool = StructuredToolInterface | RunnableToolLike;

export type ToolMap = Map<string, GenericTool>;
export type ToolRefs = {
  tools: GenericTool[];
  toolMap?: ToolMap;
};

export type ToolRefGenerator = (tool_calls: ToolCall[]) => ToolRefs;

export type ToolNodeOptions = {
  name?: string;
  tags?: string[];
  handleToolErrors?: boolean;
  loadRuntimeTools?: ToolRefGenerator;
  toolCallStepIds?: Map<string, string>;
  errorHandler?: (
    data: ToolErrorData,
    metadata?: Record<string, unknown>
  ) => Promise<void>;
};

export type ToolNodeConstructorParams = ToolRefs & ToolNodeOptions;

export type ToolEndEvent = {
  /** The Step Id of the Tool Call */
  id: string;
  /** The Completed Tool Call */
  tool_call: ToolCall;
  /** The content index of the tool call */
  index: number;
};

export type CodeEnvFile = {
  id: string;
  name: string;
  session_id: string;
};

export type CodeExecutionToolParams =
  | undefined
  | {
      session_id?: string;
      user_id?: string;
      apiKey?: string;
      files?: CodeEnvFile[];
      [EnvVar.CODE_API_KEY]?: string;
    };

export type FileRef = {
  id: string;
  name: string;
  path?: string;
};

export type FileRefs = FileRef[];

export type ExecuteResult = {
  session_id: string;
  stdout: string;
  stderr: string;
  files?: FileRefs;
};
