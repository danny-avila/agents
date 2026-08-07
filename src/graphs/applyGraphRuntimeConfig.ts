import type { StandardGraph } from './Graph';

export type GraphRuntimeConfig = Pick<
  StandardGraph,
  | 'hookRegistry'
  | 'humanInTheLoop'
  | 'toolOutputReferences'
  | 'eagerEventToolExecution'
  | 'codeSessionToolNames'
  | 'interruptingToolNames'
  | 'toolExecution'
>;

export function applyGraphRuntimeConfig(
  graph: StandardGraph,
  config: GraphRuntimeConfig
): void {
  graph.hookRegistry = config.hookRegistry;
  graph.humanInTheLoop = config.humanInTheLoop;
  graph.toolOutputReferences = config.toolOutputReferences;
  graph.eagerEventToolExecution = config.eagerEventToolExecution;
  graph.codeSessionToolNames = config.codeSessionToolNames;
  graph.interruptingToolNames = config.interruptingToolNames;
  graph.toolExecution = config.toolExecution;
}
