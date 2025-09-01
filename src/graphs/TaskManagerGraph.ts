// src/graphs/TaskManagerGraph.ts
import type * as t from '@/types';
import { StandardGraph } from '@/graphs/Graph';

export class TaskManagerGraph extends StandardGraph {
  members?: t.Member[];
  supervisorConfig?: { systemPrompt?: string; llmConfig: t.LLMConfig };

  constructor({
    members,
    supervisorConfig,
    ...standardInput
  }: t.StandardGraphInput & {
    members?: t.Member[];
    supervisorConfig?: { systemPrompt?: string; llmConfig: t.LLMConfig };
  }) {
    super(standardInput);
    this.members = members;
    this.supervisorConfig = supervisorConfig;
  }
}
